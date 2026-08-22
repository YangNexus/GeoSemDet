import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.solver.baseline_log_monitor import BaselineLogMonitor


def _write_log(path, records):
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def test_baseline_log_monitor_waits_until_start_ratio(tmp_path):
    baseline_path = tmp_path / "log.txt"
    _write_log(
        baseline_path,
        [
            {"epoch": 0, "test_coco_eval_bbox": [0.20]},
            {"epoch": 1, "test_coco_eval_bbox": [0.30]},
        ],
    )
    monitor = BaselineLogMonitor.from_config(
        {
            "enabled": True,
            "path": str(baseline_path),
            "start_ratio": 0.5,
            "min_ap_gap": 0.03,
            "patience": 1,
        },
        total_epochs=4,
    )

    decision = monitor.check({"test_coco_eval_bbox": [0.10], "epoch": 1}, epoch=1)

    assert decision.active is False
    assert decision.should_stop is False
    assert decision.current == pytest.approx(0.10)
    assert decision.baseline == pytest.approx(0.30)
    assert decision.diff == pytest.approx(-0.20)
    assert monitor.bad_epochs == 0


def test_baseline_log_monitor_stops_after_patience_when_ap_gap_is_too_large(tmp_path):
    baseline_path = tmp_path / "log.txt"
    _write_log(
        baseline_path,
        [
            {"epoch": 2, "test_coco_eval_bbox": [0.50]},
            {"epoch": 3, "test_coco_eval_bbox": [0.52]},
        ],
    )
    monitor = BaselineLogMonitor.from_config(
        {
            "enabled": True,
            "path": str(baseline_path),
            "start_ratio": 0.5,
            "min_ap_gap": 0.03,
            "patience": 2,
        },
        total_epochs=4,
    )

    first = monitor.check({"test_coco_eval_bbox": [0.46], "epoch": 2}, epoch=2)
    second = monitor.check({"test_coco_eval_bbox": [0.48], "epoch": 3}, epoch=3)

    assert first.active is True
    assert first.diff == pytest.approx(-0.04)
    assert first.bad_epochs == 1
    assert first.should_stop is False
    assert second.diff == pytest.approx(-0.04)
    assert second.bad_epochs == 2
    assert second.should_stop is True
    assert "baseline AP" in second.reason


def test_baseline_log_monitor_resets_bad_epochs_when_gap_recovers(tmp_path):
    baseline_path = tmp_path / "log.txt"
    _write_log(
        baseline_path,
        [
            {"epoch": 2, "test_coco_eval_bbox": [0.50]},
            {"epoch": 3, "test_coco_eval_bbox": [0.52]},
        ],
    )
    monitor = BaselineLogMonitor.from_config(
        {
            "enabled": True,
            "path": str(baseline_path),
            "start_ratio": 0.5,
            "min_ap_gap": 0.03,
            "patience": 2,
        },
        total_epochs=4,
    )

    first = monitor.check({"test_coco_eval_bbox": [0.46], "epoch": 2}, epoch=2)
    second = monitor.check({"test_coco_eval_bbox": [0.50], "epoch": 3}, epoch=3)

    assert first.bad_epochs == 1
    assert second.bad_epochs == 0
    assert second.should_stop is False
