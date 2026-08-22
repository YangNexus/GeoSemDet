"""
DSM -- Directional Strip Mixer.

The block that sits at each of the four merge nodes of the Scale-Fused Pyramid (SFP).
It keeps the gated-convolution structure of MambaOut and expands its single 7x7
depthwise convolution, during training only, into four parallel depthwise branches
(7x7, 1x7, 7x1, 3x3), each followed by its own batch normalization:

    S(X) = BN_7x7(DW_7x7(X)) + BN_1x7(DW_1x7(X))
         + BN_7x1(DW_7x1(X)) + BN_3x3(DW_3x3(X))

The 1x7 and 7x1 strip branches retain directional context for elongated structures
(cracks, formwork seams, conductor traces); the square branches provide isotropic
local context.

All four branches are linear operators followed by batch normalization, so after
training they merge into a single 7x7 depthwise kernel: each branch's normalization
statistics and affine parameters are folded into its kernel and bias, the smaller
kernels are zero-padded around their centres, and all kernels and biases are summed.
The deployed block is therefore structurally identical to plain MambaOut -- the
reparameterization changes the training-time inductive bias, not the inference-time
architecture, and no latency claim follows from it.

Reference: Yu & Wang, "MambaOut: Do We Really Need Mamba for Vision?" (2024).
"""

from functools import partial

import torch.nn as nn

from engine.extre_module.custom_nn.module.mambaout import MambaOut, LayerNormGeneral


class DSMStripConv(nn.Module):
    """Four parallel depthwise branches, each with its own BN, summed elementwise.

    Reparameterizable into a single ``k x k`` depthwise convolution after training.
    """

    def __init__(self, ch, k=7):
        super().__init__()
        self.k = k
        p = k // 2
        self.dw_kk = nn.Conv2d(ch, ch, k, padding=p, groups=ch, bias=False)
        self.dw_1k = nn.Conv2d(ch, ch, (1, k), padding=(0, p), groups=ch, bias=False)
        self.dw_k1 = nn.Conv2d(ch, ch, (k, 1), padding=(p, 0), groups=ch, bias=False)
        self.dw_33 = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.bn_kk = nn.BatchNorm2d(ch)
        self.bn_1k = nn.BatchNorm2d(ch)
        self.bn_k1 = nn.BatchNorm2d(ch)
        self.bn_33 = nn.BatchNorm2d(ch)

    def forward(self, x):
        return (
            self.bn_kk(self.dw_kk(x))
            + self.bn_1k(self.dw_1k(x))
            + self.bn_k1(self.dw_k1(x))
            + self.bn_33(self.dw_33(x))
        )


class DSM(MambaOut):
    """MambaOut gated-convolution block with the directional strip mixer."""

    def __init__(
        self,
        inc,
        dim,
        expansion_ratio=8 / 3,
        kernel_size=7,
        conv_ratio=1.0,
        norm_layer=partial(LayerNormGeneral, eps=1e-6, normalized_dim=(1, 2, 3)),
        act_layer=nn.GELU,
        drop_path=0.0,
        **kwargs,
    ):
        super().__init__(
            inc,
            dim,
            expansion_ratio,
            kernel_size,
            conv_ratio,
            norm_layer,
            act_layer,
            drop_path,
            **kwargs,
        )
        conv_channels = int(conv_ratio * dim)
        self.conv = DSMStripConv(conv_channels, k=kernel_size)
