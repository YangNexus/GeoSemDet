import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.optim.lr_scheduler import MetricLrReducer
from engine.solver.det_solver import DetSolver


def _build_solver(yaml_cfg=None):
    cfg = SimpleNamespace()
    if yaml_cfg is not None:
        cfg.yaml_cfg = yaml_cfg
    solver = DetSolver(cfg)
    solver.optimizer = SimpleNamespace(param_groups=[])
    solver.lr_scheduler = None
    solver.self_lr_scheduler = False
    return solver


class _FakeScheduler:
    def __init__(self):
        self.scales = []

    def set_external_lr_scale(self, scale):
        self.scales.append(scale)


class _NoScaleScheduler:
    pass


def test_get_avg_ap_metric_uses_first_ap_value_and_handles_empty_stats():
    solver = _build_solver()

    assert solver._get_avg_ap_metric(None) is None
    assert solver._get_avg_ap_metric({}) is None
    assert solver._get_avg_ap_metric(
        {
            "coco_eval_bbox": [0.40, 0.70],
            "coco_eval_segm": [0.20, 0.60],
            "custom_eval": [0.10, 0.50],
        }
    ) == pytest.approx((0.40 + 0.20 + 0.10) / 3.0)


def test_build_metric_lr_reducer_absent_or_disabled_leaves_reducer_none():
    solver_without_yaml_cfg = _build_solver()
    solver_without_yaml_cfg._build_metric_lr_reducer()
    assert solver_without_yaml_cfg.metric_lr_reducer is None

    solver_disabled = _build_solver({"metric_lr_reducer": {"enabled": False}})
    solver_disabled._pending_metric_lr_reducer_state = {"best_metric": 0.3}
    solver_disabled._build_metric_lr_reducer()

    assert solver_disabled.metric_lr_reducer is None
    assert solver_disabled._pending_metric_lr_reducer_state is None


def test_build_metric_lr_reducer_enabled_uses_config_fields():
    solver = _build_solver(
        {
            "metric_lr_reducer": {
                "enabled": True,
                "threshold": 0.002,
                "patience": 3,
                "factor": 0.2,
                "min_scale": 0.1,
                "cooldown": 4,
                "start_epoch": 5,
            }
        }
    )

    solver._build_metric_lr_reducer()

    assert isinstance(solver.metric_lr_reducer, MetricLrReducer)
    assert solver.metric_lr_reducer.enabled is True
    assert solver.metric_lr_reducer.threshold == pytest.approx(0.002)
    assert solver.metric_lr_reducer.patience == 3
    assert solver.metric_lr_reducer.factor == pytest.approx(0.2)
    assert solver.metric_lr_reducer.min_scale == pytest.approx(0.1)
    assert solver.metric_lr_reducer.cooldown == 4
    assert solver.metric_lr_reducer.start_epoch == 5


def test_metric_lr_reducer_state_round_trip_restores_pending_state(monkeypatch):
    monkeypatch.setattr("engine.solver.det_solver.BaseSolver.state_dict", lambda self: {"dummy": 1})
    monkeypatch.setattr("engine.solver.det_solver.BaseSolver.load_state_dict", lambda self, state: None)

    solver = _build_solver({"metric_lr_reducer": {"enabled": True, "patience": 1, "factor": 0.5}})
    solver._build_metric_lr_reducer()
    solver.metric_lr_reducer.step(0.30, epoch=0)
    solver.metric_lr_reducer.step(0.3005, epoch=1)

    state = solver.state_dict()

    restored = _build_solver({"metric_lr_reducer": {"enabled": True, "patience": 1, "factor": 0.5}})
    restored.self_lr_scheduler = True
    restored.lr_scheduler = _FakeScheduler()

    restored.load_state_dict(state)

    assert restored.metric_lr_reducer is None
    assert restored._pending_metric_lr_reducer_state == state["metric_lr_reducer_state"]

    restored._build_metric_lr_reducer()

    assert restored._pending_metric_lr_reducer_state is None
    assert restored.metric_lr_reducer.lr_scale == pytest.approx(0.5)
    assert restored.metric_lr_reducer.best_metric == pytest.approx(0.30)
    assert restored.metric_lr_reducer.bad_epochs == 0
    assert restored.lr_scheduler.scales == [pytest.approx(0.5)]


def test_metric_lr_reducer_load_state_warns_and_clears_pending_when_disabled(monkeypatch):
    solver = _build_solver({"metric_lr_reducer": {"enabled": False}})
    solver._pending_metric_lr_reducer_state = None
    warnings = []
    monkeypatch.setattr("engine.solver.det_solver.logger.warning", warnings.append)

    solver.load_state_dict({"metric_lr_reducer_state": {"best_metric": 0.3}})
    solver._build_metric_lr_reducer()

    assert solver.metric_lr_reducer is None
    assert solver._pending_metric_lr_reducer_state is None
    assert any("metric_lr_reducer" in message for message in warnings)


def test_apply_metric_lr_reducer_scales_optimizer_lrs_for_normal_scheduler():
    solver = _build_solver(
        {
            "metric_lr_reducer": {
                "enabled": True,
                "patience": 1,
                "threshold": 0.001,
                "factor": 0.5,
            }
        }
    )
    solver.optimizer = SimpleNamespace(
        param_groups=[
            {"lr": 0.1},
            {"lr": 0.01},
        ]
    )
    solver._build_metric_lr_reducer()

    first = solver._apply_metric_lr_reducer({"coco_eval_bbox": [0.3000]}, epoch=0)
    second = solver._apply_metric_lr_reducer({"coco_eval_bbox": [0.3005]}, epoch=1)

    assert first["triggered"] is False
    assert second["triggered"] is True
    assert solver.optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    assert solver.optimizer.param_groups[1]["lr"] == pytest.approx(0.005)


def test_apply_metric_lr_reducer_logs_each_group_lr_change_when_triggered(monkeypatch):
    solver = _build_solver(
        {
            "metric_lr_reducer": {
                "enabled": True,
                "patience": 1,
                "threshold": 0.001,
                "factor": 0.5,
            }
        }
    )
    solver.optimizer = SimpleNamespace(
        param_groups=[
            {"lr": 0.1},
            {"lr": 0.01},
        ]
    )
    messages = []
    monkeypatch.setattr("engine.solver.det_solver.logger.info", messages.append)
    solver._build_metric_lr_reducer()

    solver._apply_metric_lr_reducer({"coco_eval_bbox": [0.3000]}, epoch=0)
    solver._apply_metric_lr_reducer({"coco_eval_bbox": [0.3005]}, epoch=1)

    assert any("param_group[0]" in message and "0.100000" in message and "0.050000" in message for message in messages)
    assert any("param_group[1]" in message and "0.010000" in message and "0.005000" in message for message in messages)


def test_apply_metric_lr_reducer_persists_scaling_into_lambda_lr_base_lrs():
    param = nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD(
        [{"params": [param], "lr": 0.1, "initial_lr": 0.1}],
        lr=0.1,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    solver = _build_solver(
        {
            "metric_lr_reducer": {
                "enabled": True,
                "patience": 1,
                "threshold": 0.001,
                "factor": 0.5,
            }
        }
    )
    solver.optimizer = optimizer
    solver.lr_scheduler = scheduler
    solver._build_metric_lr_reducer()

    solver._apply_metric_lr_reducer({"coco_eval_bbox": [0.3000]}, epoch=0)
    solver._apply_metric_lr_reducer({"coco_eval_bbox": [0.3005]}, epoch=1)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    assert scheduler.base_lrs[0] == pytest.approx(0.05)

    optimizer.step()
    scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


def test_apply_metric_lr_reducer_uses_external_scale_for_self_scheduler():
    solver = _build_solver(
        {
            "metric_lr_reducer": {
                "enabled": True,
                "patience": 1,
                "threshold": 0.001,
                "factor": 0.5,
            }
        }
    )
    solver.self_lr_scheduler = True
    solver.lr_scheduler = _FakeScheduler()
    solver.optimizer = SimpleNamespace(
        param_groups=[
            {"lr": 0.1},
            {"lr": 0.01},
        ]
    )
    solver._build_metric_lr_reducer()

    solver._apply_metric_lr_reducer({"coco_eval_bbox": [0.3000]}, epoch=0)
    result = solver._apply_metric_lr_reducer({"coco_eval_bbox": [0.3005]}, epoch=1)

    assert result["triggered"] is True
    assert solver.lr_scheduler.scales[-1] == pytest.approx(0.5)
    assert solver.optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert solver.optimizer.param_groups[1]["lr"] == pytest.approx(0.01)


def test_apply_metric_lr_reducer_warns_when_self_scheduler_lacks_external_scale(monkeypatch):
    solver = _build_solver(
        {
            "metric_lr_reducer": {
                "enabled": True,
                "patience": 1,
                "threshold": 0.001,
                "factor": 0.5,
            }
        }
    )
    solver.self_lr_scheduler = True
    solver.lr_scheduler = _NoScaleScheduler()
    solver.optimizer = SimpleNamespace(param_groups=[{"lr": 0.1}])
    warnings = []
    monkeypatch.setattr("engine.solver.det_solver.logger.warning", warnings.append)
    solver._build_metric_lr_reducer()

    solver._apply_metric_lr_reducer({"coco_eval_bbox": [0.3000]}, epoch=0)

    assert any("set_external_lr_scale" in message or "external lr scale" in message for message in warnings)


def test_apply_metric_lr_reducer_logs_inactive_status_before_start_epoch(monkeypatch):
    solver = _build_solver(
        {
            "metric_lr_reducer": {
                "enabled": True,
                "threshold": 0.001,
                "patience": 2,
                "factor": 0.5,
                "start_epoch": 10,
            }
        }
    )
    solver._build_metric_lr_reducer()
    messages = []
    monkeypatch.setattr("engine.solver.det_solver.logger.info", messages.append)

    result = solver._apply_metric_lr_reducer({"coco_eval_bbox": [0.3000]}, epoch=5)

    assert result["best_metric"] is None
    assert any("status=inactive" in message for message in messages)
    assert any("waiting_for_start_epoch=10" in message for message in messages)


def test_apply_metric_lr_reducer_bootstraps_best_metric_from_best_stat():
    solver = _build_solver(
        {
            "metric_lr_reducer": {
                "enabled": True,
                "threshold": 0.001,
                "patience": 1,
                "factor": 0.5,
                "start_epoch": 10,
            }
        }
    )
    solver._build_metric_lr_reducer()
    solver.best_stat = {"epoch": 9, "avg_metric": 0.2266}

    result = solver._apply_metric_lr_reducer({"coco_eval_bbox": [0.2264]}, epoch=10)

    assert result["best_metric"] == pytest.approx(0.2266)
    assert solver.metric_lr_reducer.best_metric == pytest.approx(0.2266)
