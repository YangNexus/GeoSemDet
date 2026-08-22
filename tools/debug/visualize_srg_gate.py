#!/usr/bin/env python3
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.core import YAMLConfig
from engine.extre_module.custom_nn.module.srg_gate import SRGEncoderGate
from engine.solver.sample_adapter import move_samples_to_device, select_model_input_for_model


def parse_args():
    parser = argparse.ArgumentParser(description="Save SRG encoder gate heatmaps for one validation batch.")
    parser.add_argument("-c", "--config", default="configs/geosemdet_pcb.yml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/srg_gate_maps")
    parser.add_argument("--max-images", type=int, default=2)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def tensor_to_uint8_image(image: torch.Tensor) -> np.ndarray:
    image = image.detach().float().cpu().clamp(0, 1)
    if image.ndim != 3 or image.shape[0] not in (1, 3):
        raise ValueError(f"Expected image tensor [C,H,W], got {tuple(image.shape)}")
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    image = image.permute(1, 2, 0).numpy()
    return (image * 255.0).round().astype(np.uint8)


def save_gate_images(image: torch.Tensor, gate: torch.Tensor, output_prefix: Path):
    base = tensor_to_uint8_image(image)
    gate = gate.detach().float().cpu().clamp(0, 1).numpy()
    raw = (gate * 255.0).round().astype(np.uint8)
    heat = np.zeros_like(base)
    heat[..., 0] = raw
    heat[..., 1] = (raw * 0.15).astype(np.uint8)
    heat[..., 2] = ((255 - raw) * 0.25).astype(np.uint8)
    overlay = (0.65 * base.astype(np.float32) + 0.35 * heat.astype(np.float32)).clip(0, 255).astype(np.uint8)

    Image.fromarray(raw, mode="L").save(output_prefix.with_name(output_prefix.name + "_raw.png"))
    Image.fromarray(overlay, mode="RGB").save(output_prefix.with_name(output_prefix.name + "_overlay.png"))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = YAMLConfig(args.config)
    cfg.yaml_cfg["val_dataloader"]["shuffle"] = False
    cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = min(
        int(cfg.yaml_cfg["val_dataloader"].get("total_batch_size", 2)),
        args.max_images,
    )

    model = cfg.model.to(device).eval()
    gate_modules = [module for module in model.modules() if isinstance(module, SRGEncoderGate)]
    if not gate_modules:
        raise RuntimeError("No SRGEncoderGate modules found in the model.")

    samples, targets = next(iter(cfg.val_dataloader))
    samples = move_samples_to_device(samples, device, non_blocking=False)
    model_inputs = select_model_input_for_model(samples, model=model, key="rgb")

    with torch.inference_mode():
        outputs = model(model_inputs)

    images = model_inputs if torch.is_tensor(model_inputs) else model_inputs["rgb"]
    if not torch.is_tensor(images):
        raise TypeError(f"Expected tensor model inputs for visualization, got {type(images)!r}")
    image_h, image_w = images.shape[-2:]
    max_images = min(args.max_images, images.shape[0])

    saved = 0
    for gate_idx, module in enumerate(gate_modules):
        gate_map = module.last_gate_map
        if gate_map is None:
            raise RuntimeError(f"SRGEncoderGate index {gate_idx} did not record a gate map.")
        gate_map = F.interpolate(gate_map, size=(image_h, image_w), mode="bilinear", align_corners=False)
        for image_idx in range(max_images):
            prefix = output_dir / f"img{image_idx:02d}_gate{gate_idx:02d}"
            save_gate_images(images[image_idx], gate_map[image_idx, 0], prefix)
            saved += 2

    print(f"saved_gate_images={saved}")
    print(f"output_dir={output_dir}")
    print(f"pred_logits_shape={tuple(outputs['pred_logits'].shape)}")


if __name__ == "__main__":
    main()
