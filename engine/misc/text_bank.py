from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F


REQUIRED_TEXT_BANK_KEYS = {
    "cache_version",
    "text_encoder",
    "categories",
    "class_descriptions",
    "class_text_feats_all",
    "class_text_feats",
    "anchor_names",
    "anchor_prompts",
    "anchor_text_feats",
}


def _require_tensor(payload: dict, key: str, ndim: int, text_cache_file: str | None) -> torch.Tensor:
    value = payload[key]
    if not isinstance(value, torch.Tensor):
        cache_name = f" {Path(text_cache_file)}" if text_cache_file is not None else ""
        raise TypeError(f"text bank{cache_name} field {key!r} must be a tensor, got {type(value)!r}")
    if value.ndim != ndim:
        cache_name = f" {Path(text_cache_file)}" if text_cache_file is not None else ""
        raise ValueError(f"text bank{cache_name} field {key!r} must have ndim={ndim}, got shape={tuple(value.shape)}")
    return value.float()


def load_text_bank(
    payload: dict,
    text_cache_file: str | None = None,
    expected_categories: Sequence[str] | None = None,
) -> dict:
    if not isinstance(payload, dict):
        raise TypeError(f"text bank must be a dict, got {type(payload)!r}")

    missing_keys = REQUIRED_TEXT_BANK_KEYS - set(payload.keys())
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        cache_name = f" {Path(text_cache_file)}" if text_cache_file is not None else ""
        raise KeyError(f"text bank{cache_name} missing required keys: {missing}")

    categories = payload["categories"]
    if not isinstance(categories, list) or any(not isinstance(category, str) for category in categories):
        raise TypeError("text bank field 'categories' must be a list[str]")

    if expected_categories is not None and list(expected_categories) != categories:
        raise ValueError(
            "text bank categories mismatch: "
            f"expected={list(expected_categories)!r}, got={categories!r}"
        )

    class_descriptions = payload["class_descriptions"]
    if (
        not isinstance(class_descriptions, list)
        or len(class_descriptions) != len(categories)
        or any(not isinstance(items, list) for items in class_descriptions)
        or any(any(not isinstance(item, str) for item in items) for items in class_descriptions)
    ):
        raise TypeError("text bank field 'class_descriptions' must be a list[list[str]] aligned to categories")

    anchor_names = payload["anchor_names"]
    anchor_prompts = payload["anchor_prompts"]
    if anchor_names != ["normal", "abnormal"]:
        raise ValueError(f"text bank anchor_names must be ['normal', 'abnormal'], got {anchor_names!r}")
    if not isinstance(anchor_prompts, list) or len(anchor_prompts) != 2 or any(not isinstance(item, str) for item in anchor_prompts):
        raise TypeError("text bank field 'anchor_prompts' must be a list[str] with length 2")

    class_text_feats_all = _require_tensor(payload, "class_text_feats_all", 3, text_cache_file)
    class_text_feats = _require_tensor(payload, "class_text_feats", 2, text_cache_file)
    anchor_text_feats = _require_tensor(payload, "anchor_text_feats", 2, text_cache_file)

    num_classes = len(categories)
    if class_text_feats_all.shape[0] != num_classes or class_text_feats.shape[0] != num_classes:
        raise ValueError(
            "text bank class feature count mismatch: "
            f"categories={num_classes}, all={tuple(class_text_feats_all.shape)}, pooled={tuple(class_text_feats.shape)}"
        )
    if class_text_feats_all.shape[-1] != class_text_feats.shape[-1] or anchor_text_feats.shape[-1] != class_text_feats.shape[-1]:
        raise ValueError(
            "text bank feature dimension mismatch: "
            f"class_all={tuple(class_text_feats_all.shape)}, class={tuple(class_text_feats.shape)}, "
            f"anchors={tuple(anchor_text_feats.shape)}"
        )
    if anchor_text_feats.shape[0] != 2:
        raise ValueError(f"text bank anchor_text_feats must have shape [2, dim], got {tuple(anchor_text_feats.shape)}")

    return {
        "cache_version": str(payload["cache_version"]),
        "text_encoder": str(payload["text_encoder"]),
        "categories": categories,
        "class_descriptions": class_descriptions,
        "class_text_feats_all": F.normalize(class_text_feats_all, p=2, dim=-1),
        "class_text_feats": F.normalize(class_text_feats, p=2, dim=-1),
        "anchor_names": list(anchor_names),
        "anchor_prompts": list(anchor_prompts),
        "anchor_text_feats": F.normalize(anchor_text_feats, p=2, dim=-1),
        "feature_dim": int(payload.get("feature_dim", class_text_feats.shape[-1])),
        "num_classes": int(payload.get("num_classes", num_classes)),
        "num_anchors": int(payload.get("num_anchors", 2)),
    }


def summarize_text_bank(text_cache: dict, text_cache_file: str | None = None) -> str:
    parts = []
    if text_cache_file is not None:
        parts.append(f"path={Path(text_cache_file)}")
    parts.append(f"cache_version={text_cache['cache_version']}")
    parts.append(f"text_encoder={text_cache['text_encoder']}")
    parts.append(f"categories={text_cache['categories']!r}")
    parts.append(f"class_text_feats_all={tuple(text_cache['class_text_feats_all'].shape)}")
    parts.append(f"class_text_feats={tuple(text_cache['class_text_feats'].shape)}")
    parts.append(f"anchor_text_feats={tuple(text_cache['anchor_text_feats'].shape)}")
    parts.append(f"anchor_names={text_cache['anchor_names']!r}")
    return ", ".join(parts)
