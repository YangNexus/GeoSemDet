import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.deim.deim_criterion import DEIMCriterion
from engine.extre_module.custom_nn.module.srg_gate import SRGEncoderGate
from engine.misc.text_bank import load_text_bank
from engine.deim.geosemdet_decoder import GeoSemDetTransformer
from engine.deim.srg_decoder import SRGTransformer


PCB_CATEGORIES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spurious_copper",
    "spur",
]


def _srg_payload(num_classes=6, num_desc=4, dim=16):
    return {
        "cache_version": "text_bank_v1",
        "text_encoder": "TIPSv2",
        "feature_dim": dim,
        "num_classes": num_classes,
        "num_anchors": 2,
        "categories": PCB_CATEGORIES[:num_classes],
        "class_descriptions": [[f"description {i}-{j}" for j in range(num_desc)] for i in range(num_classes)],
        "class_text_feats_all": torch.randn(num_classes, num_desc, dim),
        "class_text_feats": torch.randn(num_classes, dim),
        "anchor_names": ["normal", "abnormal"],
        "anchor_prompts": ["normal pcb region", "abnormal pcb defect region"],
        "anchor_text_feats": torch.randn(2, dim),
    }


def test_srg_text_cache_loader_validates_and_normalizes():
    cache = load_text_bank(
        _srg_payload(),
        expected_categories=PCB_CATEGORIES,
    )

    assert cache["class_text_feats_all"].shape == (6, 4, 16)
    assert cache["class_text_feats"].shape == (6, 16)
    assert cache["anchor_text_feats"].shape == (2, 16)
    assert torch.allclose(cache["class_text_feats"].norm(dim=-1), torch.ones(6), atol=1e-5)


def test_srg_text_cache_loader_rejects_category_mismatch():
    with pytest.raises(ValueError, match="categories mismatch"):
        load_text_bank(
            _srg_payload(),
            expected_categories=["wrong"] + PCB_CATEGORIES[1:],
        )


def test_srg_residual_gate_preserves_shape_and_starts_near_identity():
    gate = SRGEncoderGate(8, 8, gc=8, gamma_init=1e-3, gamma_max=0.05)
    x = torch.randn(2, 8, 6, 6)
    anchors = torch.randn(2, 2, 8)

    out = gate(x, anchors)

    assert out.shape == x.shape
    assert gate.last_gate_map.shape == (2, 1, 6, 6)
    assert 0 < gate.gamma.item() <= 0.05
    assert torch.mean(torch.abs(out - x)) < 0.1


def test_srg_residual_gate_gamma_is_bounded():
    gate = SRGEncoderGate(8, 8, gc=8, gamma_init=1e-3, gamma_max=0.05)

    assert gate.gamma.item() == pytest.approx(1e-3, rel=1e-4)

    gate.gamma_raw.data.fill_(100.0)

    assert gate.gamma.item() <= 0.050001


def test_srg_dfine_transformer_forward_shapes():
    decoder = SRGTransformer(
        num_classes=6,
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
        text_scale=10.0,
        fuse_text_logits=False,
        alpha_init=0.0,
        alpha_max=0.02,
    )
    decoder.train()

    feats = [torch.randn(2, 32, 8, 8), torch.randn(2, 32, 4, 4)]
    class_text_feats = torch.randn(2, 6, 32)
    outputs = decoder(feats, class_text_feats=class_text_feats)

    assert outputs["pred_logits"].shape == (2, 5, 6)
    assert outputs["pred_logits_closed"].shape == (2, 5, 6)
    assert outputs["pred_text_logits"].shape == (2, 5, 6)
    assert torch.allclose(outputs["pred_logits"], outputs["pred_logits_closed"])
    assert outputs["pred_boxes"].shape == (2, 5, 4)
    assert "srg_alpha" in outputs["srg_stats"]
    assert outputs["srg_stats"]["srg_alpha"].item() <= 0.02
    assert outputs["aux_outputs"][0]["pred_text_logits"].shape == (2, 5, 6)


def test_srg_depth_guided_dfine_transformer_forward_shapes():
    decoder = GeoSemDetTransformer(
        num_classes=6,
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
        text_scale=10.0,
        fuse_text_logits=True,
        fusion_mode="adaptive",
        alpha_init=0.01,
        alpha_max=0.15,
        text_proj_mode="identity",
        depth_sigma_init=0.2,
    )
    decoder.train()

    feats = [torch.randn(2, 32, 8, 8), torch.randn(2, 32, 4, 4)]
    class_text_feats = torch.randn(2, 6, 32)
    depth_prior = torch.rand(2, 1, 64, 64)

    outputs = decoder(feats, class_text_feats=class_text_feats, depth_prior=depth_prior)

    assert outputs["pred_logits"].shape == (2, 5, 6)
    assert outputs["pred_logits_closed"].shape == (2, 5, 6)
    assert outputs["pred_text_logits"].shape == (2, 5, 6)
    assert outputs["pred_boxes"].shape == (2, 5, 4)
    assert "srg_alpha" in outputs["srg_stats"]
    assert outputs["aux_outputs"][0]["pred_text_logits"].shape == (2, 5, 6)
    assert decoder.decoder.layers[0].cross_attn.last_depth_bias_shape is not None


def test_loss_srg_text_only_uses_matched_positive_queries():
    criterion = DEIMCriterion(
        matcher=torch.nn.Identity(),
        weight_dict={"loss_srg_text": 1.0},
        losses=["srg_text"],
        num_classes=3,
    )
    outputs = {
        "pred_text_logits": torch.tensor(
            [[[5.0, -1.0, -1.0], [-1.0, 5.0, -1.0], [9.0, -9.0, -9.0]]],
            requires_grad=True,
        )
    }
    targets = [{"labels": torch.tensor([1], dtype=torch.int64)}]
    indices = [(torch.tensor([1], dtype=torch.int64), torch.tensor([0], dtype=torch.int64))]

    loss = criterion.loss_srg_text(outputs, targets, indices, num_boxes=1.0)["loss_srg_text"]

    assert loss.item() < 0.01
    loss.backward()
    assert outputs["pred_text_logits"].grad[0, 1].abs().sum() > 0
    assert outputs["pred_text_logits"].grad[0, 0].abs().sum() == 0
    assert outputs["pred_text_logits"].grad[0, 2].abs().sum() == 0
