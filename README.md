<h1 align="center">GeoSemDet</h1>

<p align="center">
  <b>Geometry- and Semantics-Guided End-to-End Detection for Visually Confounded Targets</b>
</p>

Targets that resemble their surroundings, or a neighbouring category, are hard to detect.
GeoSemDet is a compact multimodal closed-set detector that combines RGB appearance with an
**estimated depth prior** and **cached category text**, admitting both as *bounded
corrections* at the two decisions confounding actually corrupts: which locations an
attention head samples, and which category a query is assigned.

At capture time the model needs **RGB only**. The depth prior is generated offline from the
image itself, and the text bank is encoded once and cached.

---

## The four components

| Paper | Code | Where |
|---|---|---|
| **SFP** — Scale-Fused Pyramid | the three-level `encoder` graph | `configs/arch/geosemdet_n.yaml` |
| **DSM** — Directional Strip Mixer | `DSM` | `engine/extre_module/custom_nn/module/dsm.py` |
| **GSA** — Geometry-Steered Attention | `GSADeformableAttention`, `GSATransformer` | `engine/deim/gsa_decoder.py` |
| **SRG** — Semantic Reliability Gating | `SRGEncoderGate` | `engine/extre_module/custom_nn/module/srg_gate.py` |
| | `SRGTextScoreHead`, `SRGAdaptiveLogitFusion`, `SRGTransformer` | `engine/deim/srg_decoder.py` |
| the assembled decoder | `GeoSemDetTransformer` | `engine/deim/geosemdet_decoder.py` |
| the assembled model | `GeoSemDet` | `engine/extre_module/tasks.py` |

**SFP** keeps three pyramid levels rather than two, so the stride-8 grid that resolves a
hairline target survives alongside the coarse levels. Each of its four merge nodes is a
**DSM**: during training a 7x7 depthwise convolution is expanded into four parallel
branches (7x7, 1x7, 7x1, 3x3), whose strip branches align the receptive field with
elongated structures; after training they merge back into a single 7x7 kernel, so inference
cost is unchanged. **GSA** adds a clipped depth-consistency bias to the deformable-attention
logits, pulling each query toward evidence on the same physical surface, under a trust
coefficient learned from near zero. **SRG** corrects the classification logits from the
cached category texts, weighted per query by how far visual and textual evidence agree,
leaving the closed label space intact.

## Install

```bash
conda create -n geosemdet python=3.11 -y
```

Then activate it, install PyTorch for your CUDA version
(https://pytorch.org/get-started/locally/), and install the rest:

```bash
pip install -r requirements.txt
```

Backbone weights: `weight/hgnetv2/PPHGNetV2_B0_stage1_MG.pth` (HGNetv2-B0, from D-FINE).

## Data

Both datasets are published on Hugging Face:

- [YangNexus/Concrete-12](https://huggingface.co/datasets/YangNexus/Concrete-12) -- the twelve-class concrete surface-quality benchmark introduced with this paper
- [YangNexus/pcb](https://huggingface.co/datasets/YangNexus/pcb) -- an augmented six-class PCB defect-detection benchmark derived from PKU-Market-PCB and organized for the GeoSemDet workflow

The depth priors are **not** part of either dataset; generate them from the images with
`tools/priors/generate_depth_prior.py` (see below).

Both benchmarks are read in COCO format. `npy_folder` holds the offline depth priors.

```
data/
  pcb/train/images/                       pcb/train/annotations/train.json
  pcb/val/images/                         pcb/val/annotations/val.json
  pcb/test/images/                        pcb/test/annotations/test.json
  depth_pcb/train/npz/                    one .npz per image
  depth_pcb/val/npz/
  depth_pcb/test/npz/

  concrete12/images/train/                concrete12/annotations/instances_train.json
  concrete12/images/val/                  concrete12/annotations/instances_val.json
  concrete12/images/test/                 concrete12/annotations/instances_test.json
  depth_concrete12/train/npz/
  depth_concrete12/val/npz/
  depth_concrete12/test/npz/
```

Paths live in `configs/dataset/pcb.yml` and `configs/dataset/concrete12.yml`. Symlinks work.

## Preparing the two priors (offline, once per dataset)

Neither prior is part of a dataset. Both are derived from its RGB images.

**1. Depth prior** — frozen Depth Anything V2. **The priors are not distributed with this
repository**; generate them yourself from the images with the script below. It writes one
`<image-stem>.npz` per image: a single 2-D `float32` array under key `depth`, at the RGB
resolution. Nothing about the priors is dataset-specific, so the same command works for
either benchmark and for your own data.

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
```

Put that clone on `PYTHONPATH`, place its checkpoint under `checkpoints/`, then:

```bash
python tools/priors/generate_depth_prior.py --image-dir data/concrete12/images/train --out-dir data/depth_concrete12/train/npz
```

> Two constraints, both enforced or assumed downstream:
> - The dataloader runs with `npy_size_mismatch: error`, so a prior generated from a resized
>   copy of the images is rejected. Always generate priors from the images you train on.
> - **Values must lie in `[0, 1]`.** GSA compares depths of sampled points against the query
>   reference and its bandwidth is calibrated for unit-range data, so a raw metric-depth map
>   would distort the bias. The script above normalizes each map before writing it.

**2. Category-text bank** — frozen TIPS text encoder. A few short sentences describe each
category and contrast it with nearby labels; two extra anchors specify the reference and
target states. Everything is embedded once and cached, so the text encoder is not needed at
training or inference time.

The two banks used in the paper ship with this repository
(`text_embedding/text_bank_pcb.pth`, `text_embedding/text_bank_concrete12.pth`), together
with the Concrete-12 sentences (`text_embedding/descriptions_concrete12.json`,
`text_embedding/anchor_prompts_concrete12.json`). To rebuild one, export the encoder:

```bash
python tools/priors/export_tips_text_encoder.py
```

then run the builder:

```bash
python tools/priors/build_text_bank.py --ann-path data/concrete12/annotations/instances_train.json --descriptions-json text_embedding/descriptions_concrete12.json --anchor-prompts-json text_embedding/anchor_prompts_concrete12.json --out-path text_embedding/text_bank_concrete12.pth
```

A bank records its own category list, and the model rejects a bank whose categories do not
match `expected_categories` in the config, so a mismatch fails loudly rather than silently
mislabelling.

## Train

```bash
python train.py -c configs/geosemdet_pcb.yml
```

```bash
python train.py -c configs/geosemdet_concrete12.yml
```

300 epochs at 640x640, total batch size 4, one RTX 4090 (24 GB). Both benchmarks use the
same recipe: it lives in `configs/geosemdet_base.yml`, and the two files above differ only
in the dataset, the class list and the text bank.

## Evaluate

```bash
python train.py -c configs/geosemdet_concrete12.yml --test-only -r path/to/checkpoint.pth
```

The configs validate on the **val** split during training. The paper reports the **held-out
test** split; each dataset config carries a commented-out block that points
`val_dataloader` at it.

## Repository layout

```
configs/          6 files: shared base, the layer graph, two datasets, two runs
engine/deim/      encoder, D-FINE decoder, and the GSA / SRG / GeoSemDet decoders
engine/extre_module/
    tasks.py                       YAML layer graph -> model
    custom_nn/module/dsm.py        DSM
    custom_nn/module/srg_gate.py   SRG encoder gate
engine/data/      COCO and multimodal (RGB + depth) datasets and transforms
engine/solver/    training / evaluation loop
tools/priors/     offline preparation of the depth prior and the text bank
text_embedding/   the cached text banks
```

## Citation

The paper has not yet been published. Formal publication details will be added here when available. In the meantime, please reference this repository as:

```bibtex
@misc{geosemdet,
  author       = {YangNexus},
  title        = {GeoSemDet: Geometry- and Semantics-Guided End-to-End Detection for
                  Visually Confounded Targets},
  howpublished = {GitHub repository},
  url          = {https://github.com/YangNexus/GeoSemDet}
}
```

## License

**AGPL-3.0** (see [LICENSE](LICENSE)). This repository contains code derived from
Ultralytics YOLO, which is AGPL-3.0, so the combined work carries the same copyleft
terms: if you distribute a modified version, or make it available over a network, you
must release your source under AGPL-3.0 as well.

The Apache-2.0 components (DEIM, D-FINE, RT-DETR, MambaOut) are redistributed here under
AGPL-3.0 with their original notices retained. [NOTICE](NOTICE) lists every component and
which files derive from it.

## Acknowledgements

Built on [DEIM](https://github.com/ShihuaHuang95/DEIM),
[D-FINE](https://github.com/Peterande/D-FINE) and
[RT-DETR](https://github.com/lyuwenyu/RT-DETR). DSM builds on
[MambaOut](https://github.com/yuweihao/MambaOut). The priors are produced by frozen
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and
[TIPS](https://github.com/google-deepmind/tips).
