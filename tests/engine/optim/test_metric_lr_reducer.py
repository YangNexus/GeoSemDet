import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.optim.lr_scheduler import FlatCosineLRScheduler, MetricLrReducer


def test_metric_lr_reducer_uses_historical_best_and_patience():
    reducer = MetricLrReducer(threshold=0.001, patience=3, factor=0.5)

    first = reducer.step(0.3000, epoch=0)
    second = reducer.step(0.3005, epoch=1)
    third = reducer.step(0.3008, epoch=2)
    fourth = reducer.step(0.3009, epoch=3)

    assert first["best_metric"] == pytest.approx(0.3000)
    assert first["bad_epochs"] == 0
    assert second["triggered"] is False
    assert second["bad_epochs"] == 1
    assert third["triggered"] is False
    assert third["bad_epochs"] == 2
    assert fourth["triggered"] is True
    assert fourth["bad_epochs"] == 0
    assert fourth["lr_scale"] == pytest.approx(0.5)
    assert fourth["old_lr_scale"] == pytest.approx(1.0)


def test_metric_lr_reducer_real_improvement_resets_bad_epochs():
    reducer = MetricLrReducer(threshold=0.001, patience=2, factor=0.5)

    reducer.step(0.3000, epoch=0)
    bad = reducer.step(0.3005, epoch=1)
    improved = reducer.step(0.3011, epoch=2)

    assert bad["bad_epochs"] == 1
    assert improved["improvement"] == pytest.approx(0.0011)
    assert improved["best_metric"] == pytest.approx(0.3011)
    assert improved["bad_epochs"] == 0
    assert improved["triggered"] is False


def test_metric_lr_reducer_respects_min_scale():
    reducer = MetricLrReducer(
        threshold=0.001,
        patience=1,
        factor=0.1,
        min_scale=0.05,
    )

    reducer.step(0.3000, epoch=0)
    first_trigger = reducer.step(0.3000, epoch=1)
    second_trigger = reducer.step(0.3000, epoch=2)
    third_trigger = reducer.step(0.3000, epoch=3)

    assert first_trigger["lr_scale"] == pytest.approx(0.1)
    assert second_trigger["lr_scale"] == pytest.approx(0.05)
    assert third_trigger["lr_scale"] == pytest.approx(0.05)
    assert third_trigger["triggered"] is False


def test_metric_lr_reducer_state_dict_round_trip():
    reducer = MetricLrReducer(
        threshold=0.001,
        patience=2,
        factor=0.5,
        cooldown=3,
    )
    reducer.step(0.3000, epoch=0)
    reducer.step(0.3002, epoch=1)
    reducer.step(0.3003, epoch=2)

    state = reducer.state_dict()

    restored = MetricLrReducer(
        threshold=0.123,
        patience=4,
        factor=0.8,
        cooldown=0,
    )
    restored.load_state_dict(state)

    assert restored.best_metric == pytest.approx(reducer.best_metric)
    assert restored.bad_epochs == reducer.bad_epochs
    assert restored.cooldown_counter == reducer.cooldown_counter
    assert restored.lr_scale == pytest.approx(reducer.lr_scale)


def test_metric_lr_reducer_state_resume_matches_original_sequence():
    reducer = MetricLrReducer(
        threshold=0.001,
        patience=1,
        factor=0.5,
        cooldown=2,
    )

    reducer.step(0.3000, epoch=0)
    reducer.step(0.3000, epoch=1)
    state = reducer.state_dict()

    restored = MetricLrReducer(
        threshold=0.001,
        patience=1,
        factor=0.5,
        cooldown=2,
    )
    restored.load_state_dict(state)

    original_result = reducer.step(0.3010, epoch=2)
    restored_result = restored.step(0.3010, epoch=2)

    assert set(state) == {"best_metric", "bad_epochs", "cooldown_counter", "lr_scale"}
    assert original_result == restored_result
    assert restored.state_dict() == reducer.state_dict()


def test_metric_lr_reducer_exact_threshold_boundary_counts_as_improvement():
    reducer = MetricLrReducer(threshold=0.001, patience=2, factor=0.5)

    reducer.step(0.3000, epoch=0)
    result = reducer.step(0.3010, epoch=1)

    assert result["best_metric"] == pytest.approx(0.3010)
    assert result["bad_epochs"] == 0
    assert result["triggered"] is False


def test_metric_lr_reducer_disabled_leaves_state_unchanged():
    reducer = MetricLrReducer(
        threshold=0.001,
        patience=2,
        factor=0.5,
        enabled=False,
    )

    result = reducer.step(0.3000, epoch=0)

    assert result["triggered"] is False
    assert result["best_metric"] is None
    assert result["bad_epochs"] == 0
    assert result["lr_scale"] == pytest.approx(1.0)
    assert reducer.best_metric is None
    assert reducer.bad_epochs == 0
    assert reducer.lr_scale == pytest.approx(1.0)


def test_metric_lr_reducer_waits_until_start_epoch():
    reducer = MetricLrReducer(
        threshold=0.001,
        patience=2,
        factor=0.5,
        start_epoch=3,
    )

    before_start = reducer.step(0.3000, epoch=2)
    at_start = reducer.step(0.3000, epoch=3)

    assert before_start["triggered"] is False
    assert before_start["best_metric"] is None
    assert before_start["bad_epochs"] == 0
    assert before_start["lr_scale"] == pytest.approx(1.0)
    assert at_start["best_metric"] == pytest.approx(0.3000)
    assert at_start["bad_epochs"] == 0
    assert reducer.best_metric == pytest.approx(0.3000)


@pytest.mark.parametrize("threshold", [-0.001, -1.0])
def test_metric_lr_reducer_rejects_negative_threshold(threshold):
    with pytest.raises(ValueError):
        MetricLrReducer(threshold=threshold)


@pytest.mark.parametrize("threshold", [math.inf, -math.inf, math.nan])
def test_metric_lr_reducer_rejects_non_finite_threshold(threshold):
    with pytest.raises(ValueError):
        MetricLrReducer(threshold=threshold)


@pytest.mark.parametrize("metric", [math.inf, -math.inf, math.nan])
def test_metric_lr_reducer_rejects_non_finite_metric(metric):
    reducer = MetricLrReducer()

    with pytest.raises(ValueError):
        reducer.step(metric, epoch=0)


@pytest.mark.parametrize("min_scale", [-0.001, 1.001])
def test_metric_lr_reducer_rejects_min_scale_outside_unit_interval(min_scale):
    with pytest.raises(ValueError):
        MetricLrReducer(min_scale=min_scale)


@pytest.mark.parametrize("value", [-1, -5])
def test_metric_lr_reducer_rejects_negative_cooldown(value):
    with pytest.raises(ValueError):
        MetricLrReducer(cooldown=value)


@pytest.mark.parametrize("value", [-1, -3])
def test_metric_lr_reducer_rejects_negative_start_epoch(value):
    with pytest.raises(ValueError):
        MetricLrReducer(start_epoch=value)


def test_metric_lr_reducer_rejects_invalid_external_lr_scale(tmp_path):
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD(
        [{"params": [param], "lr": 0.1, "initial_lr": 0.1}],
        lr=0.1,
    )
    scheduler = FlatCosineLRScheduler(
        optimizer=optimizer,
        lr_gamma=0.1,
        iter_per_epoch=1,
        total_epochs=10,
        warmup_iter=1,
        flat_epochs=5,
        no_aug_epochs=0,
        lr_scyedule_save_path=tmp_path,
    )

    for scale in (-0.1, math.inf, -math.inf, math.nan):
        with pytest.raises(ValueError):
            scheduler.set_external_lr_scale(scale)


def test_metric_lr_reducer_get_external_lr_scale(tmp_path):
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD(
        [{"params": [param], "lr": 0.1, "initial_lr": 0.1}],
        lr=0.1,
    )
    scheduler = FlatCosineLRScheduler(
        optimizer=optimizer,
        lr_gamma=0.1,
        iter_per_epoch=1,
        total_epochs=10,
        warmup_iter=1,
        flat_epochs=5,
        no_aug_epochs=0,
        lr_scyedule_save_path=tmp_path,
    )

    scheduler.set_external_lr_scale(0.75)

    assert scheduler.get_external_lr_scale() == pytest.approx(0.75)


def test_metric_lr_reducer_cooldown_counts_down_and_allows_improvement():
    reducer = MetricLrReducer(
        threshold=0.001,
        patience=1,
        factor=0.5,
        cooldown=2,
    )

    reducer.step(0.3000, epoch=0)
    trigger = reducer.step(0.3000, epoch=1)
    improving = reducer.step(0.3010, epoch=2)
    cooling_failure = reducer.step(0.3000, epoch=3)
    post_cooldown_failure = reducer.step(0.3000, epoch=4)

    assert trigger["triggered"] is True
    assert trigger["cooldown_counter"] == 2
    assert improving["best_metric"] == pytest.approx(0.3010)
    assert improving["bad_epochs"] == 0
    assert improving["triggered"] is False
    assert cooling_failure["cooldown_counter"] == 0
    assert cooling_failure["bad_epochs"] == 0
    assert cooling_failure["triggered"] is False
    assert post_cooldown_failure["bad_epochs"] == 0
    assert post_cooldown_failure["triggered"] is True


def test_flat_cosine_scheduler_applies_external_lr_scale(tmp_path):
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD(
        [{"params": [param], "lr": 0.1, "initial_lr": 0.1}],
        lr=0.1,
    )
    scheduler = FlatCosineLRScheduler(
        optimizer=optimizer,
        lr_gamma=0.1,
        iter_per_epoch=1,
        total_epochs=10,
        warmup_iter=0,
        flat_epochs=5,
        no_aug_epochs=0,
        lr_scyedule_save_path=tmp_path,
    )

    scheduler.set_external_lr_scale(0.5)
    scheduler.step(2, optimizer)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
