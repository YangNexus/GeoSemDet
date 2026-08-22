import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import engine.extre_module.tasks as tasks
from engine.backbone.hgnetv2 import filter_loadable_state_dict


class _DepthRecordingDecoder(nn.Module):
    supports_depth_prior = True

    def __init__(self):
        super().__init__()
        self.f = [0]
        self.last_depth_prior = None

    def forward(self, feats, targets=None, depth_prior=None):
        self.last_depth_prior = depth_prior
        return {
            "pred_logits": torch.zeros(feats[0].shape[0], 2, 3),
            "pred_boxes": torch.zeros(feats[0].shape[0], 2, 4),
        }


class _SingleInputBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_type = None
        # parse_model attaches these to every layer it builds; DEIMGraph.forward reads them
        self.f = -1
        self.i = 0

    def children(self):
        return iter([self])

    def forward(self, x):
        self.seen_type = type(x)
        return x + 1.0


class _EmptyEncoder(nn.Module):
    """Encoder with no layers: the graph walk must skip straight to the decoder."""

    def children(self):
        return iter([])


def test_filter_loadable_state_dict_keeps_only_matching_keys_and_shapes():
    model = nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3))
    model_state = model.state_dict()
    pretrained_state = {
        "0.weight": torch.randn_like(model_state["0.weight"]),
        "0.bias": torch.randn(5),
        "1.weight": torch.randn_like(model_state["1.weight"]),
        "missing.weight": torch.randn(2, 2),
    }

    filtered_state = filter_loadable_state_dict(model_state, pretrained_state)

    assert set(filtered_state.keys()) == {"0.weight", "1.weight"}
    assert torch.equal(filtered_state["0.weight"], pretrained_state["0.weight"])
    assert torch.equal(filtered_state["1.weight"], pretrained_state["1.weight"])


def test_deim_mg_routes_rgb_to_depth_guided_decoder(monkeypatch):
    decoder = _DepthRecordingDecoder()
    backbone = _SingleInputBackbone()

    def _fake_yaml_load(_):
        return {"backbone": [], "encoder": [], "decoder": []}

    def _fake_parse_model(d, ch, nc, eval_spatial_size, verbose=True):
        return backbone, _EmptyEncoder(), decoder, [0]

    monkeypatch.setattr(tasks, "yaml_load", _fake_yaml_load)
    monkeypatch.setattr(tasks, "parse_model", _fake_parse_model)

    model = tasks.DEIMGraph(yaml_path="dummy.yml")
    samples = {
        "rgb": torch.zeros(2, 3, 8, 8),
        "npy": torch.ones(2, 1, 8, 8),
        "depth_geom": torch.full((2, 1, 8, 8), 2.0),
    }

    outputs = model(samples)

    assert outputs["pred_logits"].shape == (2, 2, 3)
    assert backbone.seen_type is torch.Tensor
    assert decoder.last_depth_prior is samples["depth_geom"]


class _DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = _FeatureLayer()

    def forward(self, x):
        return x


class _FeatureLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = -1
        self.i = 0
        self.seen_type = None

    def forward(self, x):
        self.seen_type = type(x)
        return x


class _IdentityEncoder(nn.Module):
    def forward(self, x):
        return x


class _SRGDepthRecordingDecoder(nn.Module):
    supports_depth_prior = True

    def __init__(self):
        super().__init__()
        self.f = [0]
        self.last_class_text_feats = None
        self.last_depth_prior = None

    def forward(self, feats, targets=None, class_text_feats=None, depth_prior=None):
        self.last_class_text_feats = class_text_feats
        self.last_depth_prior = depth_prior
        return {"pred_logits": torch.zeros(feats[0].shape[0], 2, 3), "pred_boxes": torch.zeros(feats[0].shape[0], 2, 4)}


def test_srgdeim_mg_routes_rgb_and_depth_prior_with_text_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "srg_text_cache.pth"
    categories = ["cat", "dog", "bird"]
    torch.save(
        {
            "cache_version": "text_bank_v1",
            "text_encoder": "test",
            "feature_dim": 16,
            "num_classes": 3,
            "num_anchors": 2,
            "categories": categories,
            "class_descriptions": [[f"{name} desc"] for name in categories],
            "class_text_feats_all": torch.randn(3, 1, 16),
            "class_text_feats": torch.randn(3, 16),
            "anchor_names": ["normal", "abnormal"],
            "anchor_prompts": ["normal region", "abnormal region"],
            "anchor_text_feats": torch.randn(2, 16),
        },
        cache_path,
    )

    decoder = _SRGDepthRecordingDecoder()
    backbone = _DummyBackbone()

    def _fake_yaml_load(_):
        return {"backbone": [], "encoder": [], "decoder": []}

    def _fake_parse_model(d, ch, nc, eval_spatial_size, verbose=True):
        return backbone, _IdentityEncoder(), decoder, [0]

    monkeypatch.setattr(tasks, "yaml_load", _fake_yaml_load)
    monkeypatch.setattr(tasks, "parse_model", _fake_parse_model)

    model = tasks.GeoSemDet(
        yaml_path="dummy.yml",
        img_dim=8,
        text_dim=16,
        text_adapter_layers=1,
        text_cache_file=str(cache_path),
        expected_categories=categories,
    )
    samples = {
        "rgb": torch.randn(2, 3, 8, 8),
        "npy": torch.ones(2, 1, 8, 8),
    }

    outputs = model(samples)

    assert outputs["pred_logits"].shape == (2, 2, 3)
    assert backbone.layer.seen_type is torch.Tensor
    assert model.decoder.last_class_text_feats.shape == (2, 3, 8)
    assert model.decoder.last_depth_prior is samples["npy"]
