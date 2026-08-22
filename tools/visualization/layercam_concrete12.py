"""
Qualitative figure: LayerCAM activation maps over the twelve Concrete-12 categories,
comparing a baseline checkpoint against a GeoSemDet checkpoint.

The maps are produced by LayerCAM on the multi-scale encoder outputs, with the summed
confidence of the twenty-five most confident queries as the objective. Fixing the
objective this way also fixes the polarity of the map, which is what makes the panels
comparable across images and across the two models.

The two checkpoints are not part of the repository. Point the script at your own runs
through the environment, e.g.

    LAYERCAM_BASELINE_CKPT=runs/deim_n/best_stg1.pth \
    LAYERCAM_IMPROVED_CKPT=outputs/geosemdet_concrete12/best_stg1.pth \
    python tools/visualization/layercam_concrete12.py

Requires `pytorch-grad-cam`.
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from engine.core import YAMLConfig

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# Reproducing the figure needs two trained checkpoints and the configs they were trained
# with. Only the released GeoSemDet configs ship with this repository, so point these at
# your own runs. Override any of them from the environment.
IMG_DIR = os.environ.get("CONCRETE12_TEST_IMAGES", "data/concrete12/images/test")
PROV = os.environ.get("LAYERCAM_PROVENANCE", "")   # optional JSON: category -> file name
OUT = os.environ.get("LAYERCAM_OUT", "outputs/fig_heatmap_concrete.png")

BASELINE = (os.environ.get("LAYERCAM_BASELINE_CFG", "configs/geosemdet_concrete12.yml"),
            os.environ.get("LAYERCAM_BASELINE_CKPT", "outputs/baseline/best_stg1.pth"))
IMPROVED = (os.environ.get("LAYERCAM_IMPROVED_CFG", "configs/geosemdet_concrete12.yml"),
            os.environ.get("LAYERCAM_IMPROVED_CKPT", "outputs/geosemdet_concrete12/best_stg1.pth"))


def load_model(cfg_path, weight):
    cfg = YAMLConfig(cfg_path, resume=weight)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    ckpt = torch.load(weight, map_location="cpu")
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)
    return cfg.model.to(DEVICE).eval()


class LogitWrap(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["pred_logits"]


class DetectionScore:
    """Scalar objective for gradient-based CAM: the summed confidence of the most
    confident queries. Unlike EigenCAM's principal component, whose sign is arbitrary
    and flips between images, this target fixes the polarity of the map -- warm always
    means "raises the detection score"."""

    def __init__(self, topk=25):
        self.topk = topk

    def __call__(self, logits):          # logits: [num_queries, num_classes]
        conf = logits.sigmoid().max(dim=-1).values
        k = min(self.topk, conf.shape[0])
        return conf.topk(k).values.sum()


def enc_layers(m):
    return list(m.encoder.fpn_blocks) + list(m.encoder.pan_blocks)


print("loading baseline (DEIM-N) ...")
base = load_model(*BASELINE)
print("loading GeoSemDet ...")
impr = load_model(*IMPROVED)
wb, wi = LogitWrap(base), LogitWrap(impr)
lb, li = enc_layers(base), enc_layers(impr)
print("baseline blocks:", [l.__class__.__name__ for l in lb])
print("improved blocks:", [l.__class__.__name__ for l in li])

picks = {e["category"]: e["stem"]
         for e in json.load(open(PROV, encoding="utf-8"))["modality_example_selection"]}
names = list(picks)
tf = T.Compose([T.Resize((640, 640)), T.ToTensor()])

panels = {}
for name in names:
    stem = picks[name]
    path = None
    for ext in (".jpg", ".png", ".jpeg", ".JPG"):
        cand = os.path.join(IMG_DIR, stem + ext)
        if os.path.exists(cand):
            path = cand
            break
    if path is None:
        print("MISSING:", stem)
        continue
    im = Image.open(path).convert("RGB")
    x = tf(im).unsqueeze(0).to(DEVICE)
    rgb = np.array(im.resize((640, 640))) / 255.0
    row = [rgb]
    for wrap, layers in ((wb, lb), (wi, li)):
        cam = LayerCAM(model=wrap, target_layers=layers)
        g = cam(input_tensor=x, targets=[DetectionScore()])[0]
        row.append(show_cam_on_image(rgb, g, use_rgb=True) / 255.0)
        del cam
    panels[name] = row
    print("done:", name)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix"})

names = [n for n in names if n in panels]
ncol = 6
rowlab = ["Input", "DEIM-N\n(baseline)", "GeoSemDet\n(ours)"]
fig, axes = plt.subplots(6, ncol, figsize=(2.55 * ncol, 15.6))
for k, name in enumerate(names):
    blk, col = divmod(k, ncol)
    for r, im in enumerate(panels[name]):
        ax = axes[blk * 3 + r, col]
        ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if r == 0:
            ax.set_title(name, fontsize=10)
        if col == 0:
            ax.set_ylabel(rowlab[r], fontsize=10)
fig.tight_layout(h_pad=0.5, w_pad=0.3)
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
print("saved:", OUT)
