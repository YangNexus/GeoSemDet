import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.matcher import HungarianMatcher


def _make_matcher_epoch_case():
    outputs = {
        "pred_logits": torch.tensor(
            [[[0.0104, 2.0554], [-2.2440, 1.0055]]],
            dtype=torch.float32,
        ),
        "pred_boxes": torch.tensor(
            [[[0.50, 0.50, 0.20, 0.20], [0.50, 0.50, 0.20, 0.20]]],
            dtype=torch.float32,
        ),
    }
    targets = [
        {
            "labels": torch.tensor([0, 1], dtype=torch.int64),
            "boxes": torch.tensor(
                [[0.50, 0.50, 0.20, 0.20], [0.50, 0.50, 0.20, 0.20]],
                dtype=torch.float32,
            ),
        }
    ]
    return outputs, targets


def _matched_columns(matcher, outputs, targets, epoch):
    result = matcher(outputs, targets, epoch=epoch)["indices"][0][1]
    return result.tolist()


@pytest.mark.xfail(
    strict=True,
    reason="HungarianMatcher._use_weighted_focal_cost is never called, so the matching "
           "cost stays weighted at every epoch. no_weight_vfl_epoch affects the loss "
           "only. Behaviour is intentionally unchanged: the reported results were "
           "produced with it. Fixing the gate would alter matching for the first "
           "no_weight_vfl_epoch epochs.",
)
def test_matcher_uses_unweighted_focal_cost_before_no_weight_vfl_epoch():
    outputs, targets = _make_matcher_epoch_case()
    matcher = HungarianMatcher(
        weight_dict={"cost_class": 1, "cost_bbox": 0, "cost_giou": 0},
        use_focal_loss=True,
        alpha=0.25,
        gamma=2.0,
    )
    criterion = DEIMCriterion(
        matcher=matcher,
        weight_dict={},
        losses=["vfl"],
        num_classes=2,
        no_weight_vfl_epoch=10,
    )

    criterion.set_epoch(9)
    assert _matched_columns(matcher, outputs, targets, epoch=9) == [0, 1]


def test_matcher_restores_weighted_focal_cost_at_no_weight_vfl_epoch():
    outputs, targets = _make_matcher_epoch_case()
    matcher = HungarianMatcher(
        weight_dict={"cost_class": 1, "cost_bbox": 0, "cost_giou": 0},
        use_focal_loss=True,
        alpha=0.25,
        gamma=2.0,
    )
    criterion = DEIMCriterion(
        matcher=matcher,
        weight_dict={},
        losses=["vfl"],
        num_classes=2,
        no_weight_vfl_epoch=10,
    )

    criterion.set_epoch(10)
    assert _matched_columns(matcher, outputs, targets, epoch=10) == [1, 0]
