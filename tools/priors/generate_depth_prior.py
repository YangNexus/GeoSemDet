#!/usr/bin/env python3
"""
Generate the offline geometric prior consumed by GSA.

Runs the frozen Depth Anything V2 monocular estimator over an image folder and writes
one ``<image-stem>.npz`` per image. The detector uses relative variation only -- no
metric scale is implied, and no depth sensor is involved.

Output contract (enforced by engine/data/dataset/multimodal_coco_dataset.py):
  * one array per file, stored under key ``depth``
  * 2-D, ``float32``, min-max normalized to [0, 1]
  * exactly the resolution of the corresponding RGB image

The size match matters: the dataloader is configured with ``npy_size_mismatch: error``,
so a prior generated for a resized copy of the images will be rejected. Always generate
priors from the same images you train on.

Example
-------
    python tools/priors/generate_depth_prior.py \
        --image-dir data/concrete12/images/train \
        --out-dir   data/depth_concrete12/train/npz \
        --encoder   vitl

Requires the Depth Anything V2 repository on PYTHONPATH and its checkpoint:
    git clone https://github.com/DepthAnything/Depth-Anything-V2
    # checkpoints: depth_anything_v2_{vits,vitb,vitl}.pth
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

MODEL_CONFIGS = {
    "vits": dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384]),
    "vitb": dict(encoder="vitb", features=128, out_channels=[96, 192, 384, 768]),
    "vitl": dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]),
}


def parse_args():
    p = argparse.ArgumentParser(description="Generate depth priors with Depth Anything V2.")
    p.add_argument("--image-dir", required=True, help="Folder of RGB images (searched recursively).")
    p.add_argument("--out-dir", required=True, help="Destination folder for the .npz priors.")
    p.add_argument("--encoder", default="vitl", choices=sorted(MODEL_CONFIGS), help="Backbone size.")
    p.add_argument("--checkpoint", default=None,
                   help="Path to depth_anything_v2_<encoder>.pth. "
                        "Default: checkpoints/depth_anything_v2_<encoder>.pth")
    p.add_argument("--input-size", type=int, default=518,
                   help="Estimator input size; the output is resized back to the RGB resolution.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite", action="store_true", help="Regenerate priors that already exist.")
    return p.parse_args()


def build_model(args):
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError:
        sys.exit(
            "Could not import depth_anything_v2. Clone the repository and put it on PYTHONPATH:\n"
            "  git clone https://github.com/DepthAnything/Depth-Anything-V2\n"
            "  export PYTHONPATH=$PWD/Depth-Anything-V2:$PYTHONPATH"
        )

    ckpt = Path(args.checkpoint) if args.checkpoint else Path(
        f"checkpoints/depth_anything_v2_{args.encoder}.pth")
    if not ckpt.is_file():
        sys.exit(f"Checkpoint not found: {ckpt}")

    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.to(args.device).eval()


def to_prior(depth, width, height):
    """Resize to the RGB resolution and min-max normalize to [0, 1]."""
    depth = torch.from_numpy(np.asarray(depth, dtype=np.float32))[None, None]
    if depth.shape[-2:] != (height, width):
        depth = torch.nn.functional.interpolate(
            depth, size=(height, width), mode="bilinear", align_corners=False)
    depth = depth[0, 0]

    finite = torch.isfinite(depth)
    if not finite.any():
        return None
    lo = depth[finite].min()
    hi = depth[finite].max()
    span = (hi - lo).clamp(min=1e-8)
    depth = ((depth - lo) / span).clamp(0.0, 1.0)
    # non-finite entries become 0; the dataloader's validity mask picks them up as
    # non-positive, and GSA applies no geometric bias where the prior is undefined.
    depth = torch.where(finite, depth, torch.zeros_like(depth))
    return depth.numpy().astype(np.float32, copy=False)


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    if not image_dir.is_dir():
        sys.exit(f"Not a directory: {image_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        sys.exit(f"No images found under {image_dir}")

    stems = {}
    for p in images:
        stems.setdefault(p.stem, []).append(p)
    clashes = {k: v for k, v in stems.items() if len(v) > 1}
    if clashes:
        # priors are looked up by image stem, so duplicates would overwrite each other
        example = next(iter(clashes.items()))
        sys.exit(f"Duplicate image stems found (priors are keyed by stem): {example[0]} -> "
                 f"{[str(x) for x in example[1]]}")

    model = build_model(args)

    written = skipped = failed = 0
    for i, path in enumerate(images, 1):
        out_path = out_dir / f"{path.stem}.npz"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            with Image.open(path) as im:
                width, height = im.size
                rgb = np.array(im.convert("RGB"))[:, :, ::-1]  # infer_image expects BGR
            with torch.no_grad():
                depth = model.infer_image(rgb, args.input_size)
            prior = to_prior(depth, width, height)
            if prior is None:
                print(f"  [skip] non-finite depth for {path}")
                failed += 1
                continue
            np.savez_compressed(out_path, depth=prior)
            written += 1
        except Exception as exc:  # noqa: BLE001 - report and continue over the folder
            print(f"  [fail] {path}: {exc}")
            failed += 1
        if i % 200 == 0:
            print(f"  {i}/{len(images)} processed")

    print(f"done: {written} written, {skipped} already present, {failed} failed -> {out_dir}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
