import sys
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.solver.det_solver import DetSolver


def _build_solver(tmp_path):
    cfg = SimpleNamespace()
    solver = DetSolver(cfg)
    solver.output_dir = tmp_path
    solver.train_dataloader = SimpleNamespace(
        collate_fn=SimpleNamespace(stop_epoch=5, ema_restart_decay=0.9999),
        dataset=SimpleNamespace(
            category2name={0: "obj"},
            remap_mscoco_category=False,
        ),
    )
    return solver


def test_checkpoint_paths_include_last_after_stop_epoch(tmp_path):
    solver = _build_solver(tmp_path)

    checkpoint_paths = solver._build_checkpoint_paths(epoch=6, checkpoint_freq=4)

    assert tmp_path / "last.pth" in checkpoint_paths


def test_load_best_stg1_missing_is_safe(tmp_path):
    solver = _build_solver(tmp_path)

    loaded = solver._load_best_stg1_if_available()

    assert loaded is False


def test_best_stat_roundtrip_in_state_dict(monkeypatch, tmp_path):
    solver = _build_solver(tmp_path)
    solver.best_stat = {
        "epoch": 8,
        "avg_metric": 0.66,
        "coco_eval_bbox": 0.66,
    }

    monkeypatch.setattr("engine.solver.det_solver.BaseSolver.state_dict", lambda self: {"dummy": 1})
    monkeypatch.setattr("engine.solver.det_solver.BaseSolver.load_state_dict", lambda self, state: None)

    state = solver.state_dict()

    assert "best_stat" in state
    assert state["best_stat"]["avg_metric"] == 0.66

    solver2 = _build_solver(tmp_path)
    solver2.load_state_dict(state)

    assert solver2.best_stat["epoch"] == 8
    assert solver2.best_stat["avg_metric"] == 0.66


def test_bootstrap_best_stat_from_eval_only_when_missing(tmp_path):
    solver = _build_solver(tmp_path)
    solver.best_stat = {"epoch": 3, "avg_metric": 0.9, "coco_eval_bbox": 0.9}

    solver._bootstrap_best_stat_from_eval(
        test_stats={"coco_eval_bbox": [0.1]},
        epoch=10,
    )

    assert solver.best_stat["epoch"] == 3
    assert solver.best_stat["avg_metric"] == 0.9


def test_baseline_log_monitor_stop_decision_from_solver_config(tmp_path):
    baseline_path = tmp_path / "baseline_log.txt"
    baseline_path.write_text(
        json.dumps({"epoch": 2, "test_coco_eval_bbox": [0.50]}) + "\n"
        + json.dumps({"epoch": 3, "test_coco_eval_bbox": [0.52]}) + "\n"
    )
    solver = _build_solver(tmp_path)
    solver.cfg.yaml_cfg = {
        "baseline_log_monitor": {
            "enabled": True,
            "path": str(baseline_path),
            "start_ratio": 0.5,
            "min_ap_gap": 0.03,
            "patience": 2,
        }
    }

    solver._build_baseline_log_monitor(total_epochs=4)
    first = solver._check_baseline_log_monitor({"test_coco_eval_bbox": [0.46]}, epoch=2)
    second = solver._check_baseline_log_monitor({"test_coco_eval_bbox": [0.48]}, epoch=3)

    assert first.should_stop is False
    assert second.should_stop is True


def test_baseline_log_monitor_colors_diff_by_gap(monkeypatch, tmp_path):
    baseline_path = tmp_path / "baseline_log.txt"
    baseline_path.write_text(json.dumps({"epoch": 2, "test_coco_eval_bbox": [0.50]}) + "\n")
    solver = _build_solver(tmp_path)
    solver.cfg.yaml_cfg = {
        "baseline_log_monitor": {
            "enabled": True,
            "path": str(baseline_path),
            "start_ratio": 0.5,
            "min_ap_gap": 0.03,
            "patience": 2,
        }
    }
    messages = []
    monkeypatch.setattr("engine.solver.det_solver.logger.info", messages.append)

    solver._build_baseline_log_monitor(total_epochs=4)
    solver._check_baseline_log_monitor({"test_coco_eval_bbox": [0.46]}, epoch=2)

    diff_messages = [message for message in messages if "diff=" in message]
    assert diff_messages
    assert "\033[91m" in diff_messages[-1]
    assert "diff=-0.0400" in diff_messages[-1]
    assert diff_messages[-1].endswith("\033[0m")


def test_baseline_log_monitor_colors_ahead_and_small_gap(monkeypatch, tmp_path):
    baseline_path = tmp_path / "baseline_log.txt"
    baseline_path.write_text(
        json.dumps({"epoch": 2, "test_coco_eval_bbox": [0.50]}) + "\n"
        + json.dumps({"epoch": 3, "test_coco_eval_bbox": [0.50]}) + "\n"
    )
    solver = _build_solver(tmp_path)
    solver.cfg.yaml_cfg = {
        "baseline_log_monitor": {
            "enabled": True,
            "path": str(baseline_path),
            "start_ratio": 0.5,
            "min_ap_gap": 0.03,
            "patience": 2,
        }
    }
    messages = []
    monkeypatch.setattr("engine.solver.det_solver.logger.info", messages.append)

    solver._build_baseline_log_monitor(total_epochs=4)
    solver._check_baseline_log_monitor({"test_coco_eval_bbox": [0.51]}, epoch=2)
    solver._check_baseline_log_monitor({"test_coco_eval_bbox": [0.48]}, epoch=3)

    diff_messages = [message for message in messages if "diff=" in message]
    assert "\033[92m" in diff_messages[-2]
    assert "diff=0.0100" in diff_messages[-2]
    assert "\033[93m" in diff_messages[-1]
    assert "diff=-0.0200" in diff_messages[-1]
