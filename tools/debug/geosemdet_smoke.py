#!/usr/bin/env python3
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.core import YAMLConfig
from engine.solver.sample_adapter import move_samples_to_device, select_model_input_for_model


def parse_args():
    parser = argparse.ArgumentParser(description="Small-batch GeoSemDet debug runner.")
    parser.add_argument("-c", "--config", default="configs/geosemdet_pcb.yml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overfit-steps", type=int, default=2)
    parser.add_argument("--detect-anomaly", action="store_true")
    parser.add_argument("--max-grad-print", type=int, default=20)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def grad_norm_report(model: torch.nn.Module, limit: int):
    rows = []
    total_sq = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        norm = grad.float().norm().item()
        total_sq += norm * norm
        rows.append((name, norm))
    rows.sort(key=lambda item: item[1], reverse=True)
    total_norm = total_sq ** 0.5
    print(f"grad_total_norm={total_norm:.6g}")
    for name, norm in rows[:limit]:
        flag = ""
        if not np.isfinite(norm):
            flag = " NONFINITE"
        elif norm > 100:
            flag = " EXPLODING"
        elif norm < 1e-7:
            flag = " VANISHING"
        print(f"grad_norm {name}: {norm:.6g}{flag}")


def resize_for_eval(inputs, eval_spatial_size):
    """Resize a batch to ``eval_spatial_size``.

    Multi-scale training hands out batches at sizes other than ``eval_spatial_size``, but
    the decoder's eval-mode anchor cache is built for ``eval_spatial_size`` alone. The
    multimodal model is fed a dict (RGB plus the depth prior), so every tensor in it has
    to be resized, not just a bare tensor.
    """
    if eval_spatial_size is None:
        return inputs
    size = tuple(int(v) for v in eval_spatial_size)

    def _resize(value):
        if torch.is_tensor(value) and value.dim() == 4:
            if tuple(value.shape[-2:]) == size:
                return value
            return F.interpolate(value, size=size, mode="bilinear", align_corners=False)
        return value

    if isinstance(inputs, dict):
        return {k: _resize(v) for k, v in inputs.items()}
    return _resize(inputs)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    cfg = YAMLConfig(args.config)
    cfg.yaml_cfg["train_dataloader"]["shuffle"] = False
    cfg.yaml_cfg["train_dataloader"]["drop_last"] = False
    cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = min(
        int(cfg.yaml_cfg["train_dataloader"].get("total_batch_size", 2)),
        2,
    )

    model = cfg.model.to(device)
    criterion = cfg.criterion.to(device)
    optimizer = cfg.optimizer
    model.train()
    criterion.train()

    batch = next(iter(cfg.train_dataloader))
    samples, targets = batch
    samples = move_samples_to_device(samples, device, non_blocking=False)
    targets = move_samples_to_device(targets, device, non_blocking=False)
    model_inputs = select_model_input_for_model(samples, model=model, key="rgb")

    print(f"num_classes={cfg.yaml_cfg['num_classes']}")
    print(f"theoretical_ce_ln_num_classes={np.log(cfg.yaml_cfg['num_classes']):.6f}")
    print(f"overfit_steps={args.overfit_steps}, detect_anomaly={args.detect_anomaly}")

    losses_seen = []
    anomaly_context = torch.autograd.detect_anomaly() if args.detect_anomaly else torch.enable_grad()
    with anomaly_context:
        for step in range(args.overfit_steps):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(model_inputs, targets=targets)
            loss_dict = criterion(outputs, targets, epoch=0, step=step, global_step=step, epoch_step=args.overfit_steps)
            loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {loss_dict}")
            loss.backward()
            grad_norm_report(model, args.max_grad_print)
            optimizer.step()
            losses_seen.append(float(loss.detach().cpu()))
            loss_text = ", ".join(f"{k}={float(v.detach().cpu()):.6g}" for k, v in loss_dict.items())
            print(f"step={step} total_loss={losses_seen[-1]:.6g} {loss_text}")
            if "srg_stats" in outputs:
                stats = {}
                for key, value in outputs["srg_stats"].items():
                    if torch.is_tensor(value) and value.numel() == 1:
                        stats[key] = float(value.detach().cpu())
                print("srg_stats=" + ", ".join(f"{k}={v:.6g}" for k, v in sorted(stats.items())))

    model.eval()
    with torch.inference_mode():
        outputs = model(resize_for_eval(model_inputs, cfg.yaml_cfg.get("eval_spatial_size", [640, 640])))
    model.train()
    print(f"eval_pred_logits_shape={tuple(outputs['pred_logits'].shape)}")
    print(f"eval_pred_boxes_shape={tuple(outputs['pred_boxes'].shape)}")
    if len(losses_seen) >= 2:
        print(f"loss_delta={losses_seen[-1] - losses_seen[0]:.6g}")


if __name__ == "__main__":
    main()
