#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import sentencepiece as spm
except ImportError as exc:  # pragma: no cover
    raise ImportError("Need `sentencepiece` installed to tokenize TIPS prompts.") from exc


MAX_LEN = 64

DEFAULT_PCB_DESCRIPTIONS = {
    "missing_hole": [
        "a printed circuit board defect where an expected drilled hole or via is absent",
        "a circular pad region on a PCB with a missing dark hole opening",
        "an incomplete PCB pad where the hole area is filled or not drilled",
        "a local PCB defect showing a missing via hole inside the copper pad",
    ],
    "mouse_bite": [
        "a PCB edge or trace defect with small semicircular bite-like notches",
        "a copper trace boundary with repeated small concave bite marks",
        "a local printed circuit board defect resembling tiny mouse bites along copper",
        "an irregular notched copper edge with small rounded missing pieces",
    ],
    "open_circuit": [
        "a PCB copper trace defect where the conductive line is broken and disconnected",
        "a narrow copper path interrupted by a gap causing an open circuit",
        "a discontinuity in a printed circuit board trace with separated copper ends",
        "a broken conductive track on a PCB creating a visible line gap",
    ],
    "short": [
        "a PCB defect where adjacent copper traces are incorrectly connected",
        "an unwanted copper bridge causing a short circuit between neighboring tracks",
        "a local abnormal connection joining two separate printed circuit board traces",
        "a solder-like or copper-like bridge across nearby PCB conductors",
    ],
    "spurious_copper": [
        "an unwanted extra copper blob or island on the printed circuit board surface",
        "a PCB defect with redundant copper residue outside the intended trace pattern",
        "an isolated abnormal copper fragment appearing between normal PCB tracks",
        "extra spurious copper material that should not exist in the circuit pattern",
    ],
    "spur": [
        "a thin unwanted copper protrusion extending from a normal PCB trace",
        "a small branch-like copper spur sticking out from a conductive track",
        "a narrow redundant copper extension attached to a printed circuit board line",
        "a local PCB defect where a copper trace has a sharp extra protruding tail",
    ],
}

DEFAULT_ANCHOR_PROMPTS = [
    "a normal printed circuit board region with regular copper traces, intact pads, and no defect",
    "an abnormal printed circuit board defect region with broken, missing, shorted, bitten, or redundant copper patterns",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build TIPS text bank for PCB defect experiments.")
    parser.add_argument("--ann", required=True, help="COCO annotation json path.")
    parser.add_argument("--out", required=True, help="Output .pth path.")
    parser.add_argument("--model-path", required=True, help="Local TIPS text encoder TorchScript path.")
    parser.add_argument("--tokenizer-path", required=True, help="Local TIPS sentencepiece tokenizer model path.")
    parser.add_argument("--descriptions-json", default=None, help="Optional JSON mapping category name to list of descriptions.")
    parser.add_argument("--anchor-prompts-json", default=None, help="Optional JSON list of 2 anchor prompts: normal, abnormal.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_categories(ann_path: Path):
    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    categories = payload.get("categories", [])
    if not categories:
        raise ValueError(f"No categories found in {ann_path}")
    return sorted(categories, key=lambda item: int(item["id"]))


def load_descriptions(path: str | None):
    if path is None:
        return DEFAULT_PCB_DESCRIPTIONS
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Descriptions JSON must be a dict[str, list[str]].")
    return payload


def load_anchor_prompts(path: str | None):
    if path is None:
        return DEFAULT_ANCHOR_PROMPTS
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 2 or any(not isinstance(item, str) for item in payload):
        raise TypeError("Anchor prompts JSON must be a list[str] with exactly 2 items: normal, abnormal.")
    return payload


class SentencePieceTokenizer:
    def __init__(self, tokenizer_path: str | Path):
        self.processor = spm.SentencePieceProcessor()
        loaded = self.processor.Load(str(tokenizer_path))
        if not loaded:
            raise ValueError(f"Failed to load sentencepiece model from {tokenizer_path}")

    def tokenize(self, texts, max_len=MAX_LEN):
        encoded = self.processor.Encode([text.lower() for text in texts], out_type=int)
        batch_size = len(encoded)
        token_ids = np.zeros((batch_size, max_len), dtype=np.int32)
        for row, token_list in enumerate(encoded):
            truncated = token_list[:max_len]
            if truncated:
                token_ids[row, : len(truncated)] = truncated
        paddings = (token_ids == 0).astype(np.int32)
        return token_ids, paddings


def encode_texts(texts, encoder, tokenizer, device):
    token_ids, paddings = tokenizer.tokenize(texts, max_len=MAX_LEN)
    with torch.inference_mode():
        feats = encoder(
            torch.from_numpy(token_ids).to(device),
            torch.from_numpy(paddings).to(device),
        ).float().cpu()
    return F.normalize(feats, p=2, dim=-1)


def build_text_cache(
    ann_path,
    out_path,
    model_path,
    tokenizer_path,
    descriptions_json=None,
    anchor_prompts_json=None,
    device="cpu",
):
    ann_path = Path(ann_path)
    out_path = Path(out_path)
    device = torch.device(device)
    categories_payload = load_categories(ann_path)
    categories = [item["name"] for item in categories_payload]
    descriptions_map = load_descriptions(descriptions_json)
    anchor_prompts = load_anchor_prompts(anchor_prompts_json)

    class_descriptions = []
    for category in categories:
        descriptions = descriptions_map.get(category)
        if not descriptions:
            descriptions = [f"a printed circuit board defect of type {category}"]
        if not isinstance(descriptions, list) or any(not isinstance(item, str) for item in descriptions):
            raise TypeError(f"Descriptions for category {category!r} must be a list[str].")
        class_descriptions.append(descriptions)

    num_descriptions = max(len(items) for items in class_descriptions)
    padded_descriptions = []
    for category, descriptions in zip(categories, class_descriptions):
        repeated = list(descriptions)
        while len(repeated) < num_descriptions:
            repeated.append(descriptions[-1])
        padded_descriptions.append(repeated[:num_descriptions])

    flat_descriptions = [text for descriptions in padded_descriptions for text in descriptions]
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    encoder = torch.jit.load(str(model_path), map_location=device).eval()
    class_text_feats_all = encode_texts(flat_descriptions, encoder, tokenizer, device)
    class_text_feats_all = class_text_feats_all.view(len(categories), num_descriptions, -1)
    class_text_feats = F.normalize(class_text_feats_all.mean(dim=1), p=2, dim=-1)
    anchor_text_feats = encode_texts(anchor_prompts, encoder, tokenizer, device)
    prompt_template = (
        f"descriptions:{Path(descriptions_json).stem}"
        if descriptions_json is not None
        else "srg_default_pcb_descriptions"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cache_version": "text_bank_v1",
            "text_encoder": "TIPSv2",
            "feature_dim": int(class_text_feats.shape[-1]),
            "num_classes": len(categories),
            "num_anchors": 2,
            "categories": categories,
            "class_descriptions": padded_descriptions,
            "class_text_feats_all": class_text_feats_all,
            "class_text_feats": class_text_feats,
            "anchor_names": ["normal", "abnormal"],
            "anchor_prompts": anchor_prompts,
            "anchor_text_feats": anchor_text_feats,
            "text_feats": class_text_feats,
            "prompts": [items[0] for items in padded_descriptions],
            "prompt_template": prompt_template,
        },
        out_path,
    )
    print(f"saved={out_path}")
    print(f"class_text_feats_all={tuple(class_text_feats_all.shape)}")
    print(f"class_text_feats={tuple(class_text_feats.shape)}")
    print(f"anchor_text_feats={tuple(anchor_text_feats.shape)}")
    print("categories=" + ",".join(categories))


def main():
    args = parse_args()
    build_text_cache(
        ann_path=args.ann,
        out_path=args.out,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        descriptions_json=args.descriptions_json,
        anchor_prompts_json=args.anchor_prompts_json,
        device=args.device,
    )


if __name__ == "__main__":
    main()
