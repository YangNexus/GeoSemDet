"""Coherence checks for the four released configs and the layer graph."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.core.yaml_utils import load_config  # noqa: E402

METHOD_CONFIGS = {
    "configs/geosemdet_pcb.yml": 6,
    "configs/geosemdet_concrete12.yml": 12,
}


@pytest.mark.parametrize("config_path,num_classes", sorted(METHOD_CONFIGS.items()))
def test_method_config_wires_model_graph_and_priors(config_path, num_classes):
    cfg = load_config(config_path)

    assert cfg["model"] == "GeoSemDet"
    assert cfg["GeoSemDet"]["yaml_path"] == "configs/arch/geosemdet_n.yaml"
    assert cfg["num_classes"] == num_classes
    assert len(cfg["GeoSemDet"]["expected_categories"]) == num_classes

    # the text bank has to exist and agree with the config's category list
    bank_path = ROOT / cfg["text_cache_file"]
    assert bank_path.is_file(), bank_path
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    assert bank["categories"] == cfg["GeoSemDet"]["expected_categories"]
    assert bank["num_classes"] == num_classes
    assert bank["anchor_names"] == ["normal", "abnormal"]


@pytest.mark.parametrize("config_path", sorted(METHOD_CONFIGS))
def test_method_config_feeds_rgb_and_depth_prior(config_path):
    cfg = load_config(config_path)

    for split in ("train_dataloader", "val_dataloader"):
        dataset = cfg[split]["dataset"]
        assert dataset["type"] == "MultimodalCocoDetection"
        # a prior built for a resized copy of the images must be rejected, not resized
        assert dataset["npy_size_mismatch"] == "error"
        assert dataset["npy_folder"]
        ops = [op["type"] for op in dataset["transforms"]["ops"]]
        assert "NormalizeNPYMinMax" in ops
        assert cfg[split]["collate_fn"]["type"] == "BatchMultimodalCollateFunction"


def test_both_configs_share_one_training_recipe():
    """Only data, class list and text bank may differ between the two benchmarks."""
    pcb = load_config("configs/geosemdet_pcb.yml")
    concrete = load_config("configs/geosemdet_concrete12.yml")

    for key in ("optimizer", "lr_scheduler", "lr_warmup_scheduler", "epoches",
                "lrsheduler", "flat_epoch", "no_aug_epoch", "clip_max_norm",
                "eval_spatial_size", "use_amp", "use_ema"):
        assert pcb[key] == concrete[key], key

    assert pcb["DEIMCriterion"] == concrete["DEIMCriterion"]
    assert pcb["optimizer"]["lr"] == 0.0001
    # the backbone groups inherit the base lr instead of carrying their own
    assert all("lr" not in group for group in pcb["optimizer"]["params"])

    for split in ("train_dataloader", "val_dataloader"):
        assert pcb[split]["collate_fn"] == concrete[split]["collate_fn"], split
        assert pcb[split]["total_batch_size"] == concrete[split]["total_batch_size"]
        pcb_ops = [op["type"] for op in pcb[split]["dataset"]["transforms"]["ops"]]
        con_ops = [op["type"] for op in concrete[split]["dataset"]["transforms"]["ops"]]
        assert pcb_ops == con_ops, split


@pytest.mark.parametrize("config_path", sorted(METHOD_CONFIGS))
def test_srg_text_loss_is_a_bounded_auxiliary_term(config_path):
    cfg = load_config(config_path)
    criterion = cfg["DEIMCriterion"]

    assert "srg_text" in criterion["losses"]
    assert criterion["weight_dict"]["loss_srg_text"] == 0.02
    # applied to the final decoder layer only, not to every auxiliary head
    assert criterion["srg_text_aux_loss"] is False


def test_arch_graph_is_sfp_with_four_dsm_nodes_and_srg_gate():
    cfg = load_config("configs/arch/geosemdet_n.yaml")

    encoder_modules = [layer[1] for layer in cfg["encoder"]]
    # SFP keeps three levels; each of its four merge nodes is a DSM
    assert encoder_modules.count("DSM") == 4
    assert encoder_modules.count("SRGEncoderGate") == 1
    assert encoder_modules[-1] == "SRGEncoderGate", "the SRG gate sits on the coarsest level"

    backbone_modules = [layer[1] for layer in cfg["backbone"]]
    assert backbone_modules[0] == "StemBlock"
    assert backbone_modules.count("HG_Stage") == 4


def test_arch_decoder_bounds_both_priors():
    cfg = load_config("configs/arch/geosemdet_n.yaml")
    decoder = cfg["decoder"][0]

    assert decoder[1] == "GeoSemDetTransformer"
    args = decoder[2]
    assert args["feat_strides"] == [8, 16, 32]
    assert args["num_levels"] == 3
    assert args["num_layers"] == 3
    assert args["num_points"] == [3, 6, 3]
    # GSA: initial bandwidth of the depth-consistency bias
    assert args["depth_sigma_init"] == 0.2
    # SRG: the per-query text correction starts near zero and stays bounded
    assert args["use_text_logits"] is True
    assert args["fuse_text_logits"] is True
    assert args["fusion_mode"] == "adaptive"
    assert args["alpha_init"] == 0.01
    assert args["alpha_max"] == 0.15
