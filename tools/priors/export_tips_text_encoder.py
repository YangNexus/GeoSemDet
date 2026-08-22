#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    min_timescale: int = 1
    max_timescale: int = 10_000

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, seq_length: int, device: torch.device) -> torch.Tensor:
        position = torch.arange(seq_length, dtype=torch.float32, device=device)[None, :]
        num_timescales = self.embedding_dim // 2
        log_timescale_increment = torch.log(
            torch.tensor(float(self.max_timescale) / float(self.min_timescale), device=device)
        ) / torch.maximum(
            torch.tensor(num_timescales, dtype=torch.float32, device=device) - 1,
            torch.tensor(1.0, device=device),
        )
        inv_timescales = self.min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32, device=device) * -log_timescale_increment
        )
        scaled_time = position[:, :, None] * inv_timescales[None, None, :]
        signal = torch.cat((torch.sin(scaled_time), torch.cos(scaled_time)), dim=2)
        return F.pad(signal, (0, self.embedding_dim % 2, 0, 0, 0, 0))


class MlpBlockWithMask(nn.Module):
    def __init__(self, mlp_dim: int, d_model: int):
        super().__init__()
        self.c_fc = nn.Linear(d_model, mlp_dim)
        self.c_proj = nn.Linear(mlp_dim, d_model)

    def forward(self, inputs: torch.Tensor, mlp_mask: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(inputs)
        x = F.relu(x)
        x = x * mlp_mask[..., None]
        x = self.c_proj(x)
        x = x * mlp_mask[..., None]
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, mlp_dim: int):
        super().__init__()
        self.n_head = n_head
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = MlpBlockWithMask(mlp_dim, d_model)
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_mask = mask[:, None, None, :].repeat(1, self.n_head, x.shape[0], 1).flatten(0, 1)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf")).masked_fill(attn_mask == 1, 0)
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attention(self.ln_1(x), mask.permute(1, 0))
        x = x + self.mlp(self.ln_2(x), mask)
        return x, mask


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, mlp_dim: int):
        super().__init__()
        self.resblocks = nn.ModuleList(
            [ResidualAttentionBlock(width, heads, mlp_dim) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            x, mask = block(x, mask)
        return x


class TextEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, num_heads: int, mlp_dim: int, vocab_size: int):
        super().__init__()
        self.embedding_dim = hidden_size
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedder = PositionalEmbedding(hidden_size)
        self.transformer = Transformer(hidden_size, num_layers, num_heads, mlp_dim)
        self.ln_final = nn.LayerNorm(hidden_size)

    def forward(self, ids: torch.Tensor, paddings: torch.Tensor) -> torch.Tensor:
        _, seq_length = ids.shape
        mask = (paddings == 0).to(torch.float32).permute(1, 0)
        x = self.token_embedding(ids)
        x = x * (self.embedding_dim ** 0.5)
        x = x + self.pos_embedder(seq_length, x.device)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, mask)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        valid = 1.0 - paddings.to(x.dtype)
        x = x * valid[:, :, None]
        return x.sum(dim=1) / (valid.sum(dim=1, keepdim=True) + 1e-8)


def load_npz_state_dict(path: Path) -> dict[str, torch.Tensor]:
    payload = np.load(path)
    state = {}
    for key in payload.files:
        if key == "temperature":
            continue
        state[key] = torch.from_numpy(payload[key])
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Export official TIPSv2 text npz to TorchScript.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--mlp-dim", type=int, default=3072)
    parser.add_argument("--vocab-size", type=int, default=32000)
    args = parser.parse_args()

    model = TextEncoder(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        vocab_size=args.vocab_size,
    ).eval()
    state = load_npz_state_dict(Path(args.checkpoint))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch: missing={missing}, unexpected={unexpected}")

    example_ids = torch.zeros((2, 64), dtype=torch.long)
    example_paddings = torch.ones((2, 64), dtype=torch.int32)
    example_paddings[:, 0] = 0
    with torch.inference_mode():
        traced = torch.jit.trace(model, (example_ids, example_paddings), strict=False)
        traced = torch.jit.freeze(traced)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out))
    with torch.inference_mode():
        output = traced(example_ids, example_paddings)
    print(f"saved={out}")
    print(f"shape={tuple(output.shape)}")


if __name__ == "__main__":
    main()
