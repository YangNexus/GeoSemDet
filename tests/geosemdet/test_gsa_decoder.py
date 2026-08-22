import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.deim.gsa_decoder import (  # noqa: E402
    GSATransformer,
    GSADeformableAttention,
)
from engine.deim.dfine_decoder import DFINETransformer, MSDeformableAttention  # noqa: E402


def _value_levels(batch_size=2, num_heads=4, head_dim=8, shapes=((4, 4), (2, 2))):
    return [
        torch.randn(batch_size, num_heads, head_dim, h * w)
        for h, w in shapes
    ]


def test_depth_guided_attention_matches_base_attention_when_lambda_is_near_zero():
    torch.manual_seed(1)
    base = MSDeformableAttention(embed_dim=32, num_heads=4, num_levels=2, num_points=[2, 2])
    dga = GSADeformableAttention(embed_dim=32, num_heads=4, num_levels=2, num_points=[2, 2])
    dga.load_state_dict(base.state_dict(), strict=False)
    dga.depth_lambda_raw.data.fill_(-80.0)

    query = torch.randn(2, 5, 32)
    reference_points = torch.rand(2, 5, 1, 4).clamp(0.1, 0.9)
    value = _value_levels()
    spatial_shapes = [[4, 4], [2, 2]]
    depth_context = dga.build_depth_context(torch.rand(2, 1, 32, 32), spatial_shapes)

    expected = base(query, reference_points, value, spatial_shapes)
    actual = dga(query, reference_points, value, spatial_shapes, depth_context=depth_context)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_depth_guided_attention_sanitizes_invalid_depth_and_tracks_flat_bias_shape():
    torch.manual_seed(2)
    dga = GSADeformableAttention(embed_dim=32, num_heads=4, num_levels=2, num_points=[2, 3])
    query = torch.randn(2, 5, 32)
    reference_points = torch.rand(2, 5, 1, 4).clamp(0.1, 0.9)
    value = _value_levels(shapes=((4, 4), (2, 2)))
    spatial_shapes = [[4, 4], [2, 2]]
    depth = torch.rand(2, 1, 32, 32)
    depth[0, :, 0, 0] = float("nan")
    depth[0, :, 0, 1] = float("inf")
    depth[1, :, 0, 0] = 0.0
    depth_context = dga.build_depth_context(depth, spatial_shapes)

    output = dga(query, reference_points, value, spatial_shapes, depth_context=depth_context)

    assert torch.isfinite(output).all()
    assert dga.last_depth_bias_shape == (2, 5, 4, 5)


def test_depth_guided_transformer_forward_accepts_depth_prior():
    torch.manual_seed(3)
    decoder = GSATransformer(
        num_classes=3,
        hidden_dim=32,
        num_queries=5,
        feat_channels=[32, 32],
        feat_strides=[8, 16],
        num_levels=2,
        num_points=[2, 2],
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        num_denoising=0,
        aux_loss=True,
        reg_max=8,
    )
    decoder.train()

    feats = [
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 32, 4, 4),
    ]
    depth_prior = torch.rand(2, 1, 64, 64)
    depth_prior[0, :, 0, 0] = float("nan")

    outputs = decoder(feats, depth_prior=depth_prior)

    assert outputs["pred_logits"].shape == (2, 5, 3)
    assert outputs["pred_boxes"].shape == (2, 5, 4)
    assert torch.isfinite(outputs["pred_logits"]).all()
    assert torch.isfinite(outputs["pred_boxes"]).all()


def test_depth_guided_transformer_preserves_sr_checkpoint_key_compatibility():
    kwargs = dict(
        num_classes=3,
        hidden_dim=32,
        num_queries=5,
        feat_channels=[32, 32],
        feat_strides=[8, 16],
        num_levels=2,
        num_points=[2, 2],
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        num_denoising=0,
        aux_loss=True,
        reg_max=8,
    )
    base = DFINETransformer(**kwargs)
    dga = GSATransformer(**kwargs)

    missing, unexpected = dga.load_state_dict(base.state_dict(), strict=False)

    assert unexpected == []
    assert set(missing) == {
        "decoder.layers.0.cross_attn.depth_lambda_raw",
        "decoder.layers.0.cross_attn.depth_sigma_raw",
        "decoder.layers.1.cross_attn.depth_lambda_raw",
        "decoder.layers.1.cross_attn.depth_sigma_raw",
    }
