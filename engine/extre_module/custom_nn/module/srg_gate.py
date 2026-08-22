"""
SRG -- Semantic Reliability Gating, encoder-side reference-to-target gate.

Placed on the coarsest encoder level (F5) of the Scale-Fused Pyramid. It reads the two
cached text anchors -- a reference state and a target state -- forms the direction from
one to the other, scores every spatial position against that direction, and folds the
gated response back through a bounded residual branch:

    g       = normalize(W_g * normalize(t_ab - t_no))
    G(x, y) = sigmoid(<F5_hat(x, y), g> / tau_g)
    F5'     = F5 + gamma * DSConv_3x3(F5 * G),   gamma = gamma_max * sigmoid(gamma_raw)

The direction is class agnostic: the same ``g`` is used for every category. ``gamma`` is
bounded by ``gamma_max`` and initialized near zero, so the residual contribution starts
negligible and the initial computation stays close to the unmodified encoder.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.extre_module.ultralytics_nn.conv import Conv, DSConv

__all__ = ["SRGEncoderGate"]


class SRGEncoderGate(nn.Module):
    """Encoder-side reference-to-target gate of SRG."""

    def __init__(
        self,
        c1: int,
        c2: int,
        gc: int = 256,
        tau: float = 0.1,
        gamma_init: float = 1e-3,
        gamma_max: float | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.tau = float(tau)
        self.eps = float(eps)
        self.gamma_max = None if gamma_max is None else float(gamma_max)
        self.image_proj = Conv(c1, c2, k=1, act=False) if c1 != c2 else nn.Identity()
        self.guide_proj = nn.Linear(gc, c2)
        self.refine = DSConv(c1, c2, k=3, s=1, act=False)
        self.shortcut = Conv(c1, c2, k=1, act=False) if c1 != c2 else nn.Identity()
        if self.gamma_max is None:
            self.gamma_raw = nn.Parameter(torch.full((), float(gamma_init)))
        else:
            if self.gamma_max <= 0:
                raise ValueError(f"gamma_max must be positive, got {self.gamma_max}")
            gamma_ratio = min(max(float(gamma_init) / self.gamma_max, 1e-6), 1.0 - 1e-6)
            self.gamma_raw = nn.Parameter(torch.tensor(math.log(gamma_ratio / (1.0 - gamma_ratio))))
        self.last_gate_map = None

    @property
    def gamma(self) -> torch.Tensor:
        if self.gamma_max is None:
            return self.gamma_raw
        return self.gamma_max * torch.sigmoid(self.gamma_raw)

    def forward(self, x: torch.Tensor, anchor_feats: torch.Tensor) -> torch.Tensor:
        if anchor_feats.dim() == 2:
            anchor_feats = anchor_feats.unsqueeze(0).repeat(x.shape[0], 1, 1)
        if anchor_feats.shape[1] != 2:
            raise ValueError(f"SRGEncoderGate expects normal/abnormal anchors with shape [B, 2, C], got {tuple(anchor_feats.shape)}")
        if anchor_feats.shape[0] != x.shape[0]:
            raise ValueError(f"SRGEncoderGate batch mismatch: anchors={anchor_feats.shape[0]}, x={x.shape[0]}")

        normal_feat = anchor_feats[:, 0]
        abnormal_feat = anchor_feats[:, 1]
        guide = F.normalize(abnormal_feat - normal_feat, p=2, dim=-1, eps=self.eps)
        guide = F.normalize(self.guide_proj(guide), p=2, dim=-1, eps=self.eps)

        image_embed = F.normalize(self.image_proj(x), p=2, dim=1, eps=self.eps)
        gate_logits = torch.einsum("bchw,bc->bhw", image_embed, guide) / max(self.tau, self.eps)
        gate = gate_logits.sigmoid().unsqueeze(1)
        self.last_gate_map = gate.detach()

        return self.shortcut(x) + self.gamma.to(x.dtype) * self.refine(x * gate)
