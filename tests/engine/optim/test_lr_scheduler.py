import torch

from engine.optim import lr_scheduler as lr_scheduler_mod


def test_flat_cosine_scheduler_logs_detailed_summary(monkeypatch, tmp_path):
    messages = []

    def _capture_info(msg):
        messages.append(str(msg))

    monkeypatch.setattr(lr_scheduler_mod.logger, "info", _capture_info)
    monkeypatch.setattr(lr_scheduler_mod, "plot_lr_schedule", lambda *args, **kwargs: None)

    p1 = torch.nn.Parameter(torch.ones(1))
    p2 = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD(
        [
            {
                "params": [p1],
                "lr": 1e-3,
                "initial_lr": 1e-3,
                "weight_decay": 0.01,
                "momentum": 0.8,
                "name": "backbone",
            },
            {
                "params": [p2],
                "lr": 2e-3,
                "initial_lr": 2e-3,
                "weight_decay": 0.02,
                "momentum": 0.9,
                "name": "decoder",
            },
        ]
    )

    lr_scheduler_mod.FlatCosineLRScheduler(
        optimizer=optimizer,
        lr_gamma=0.1,
        iter_per_epoch=2,
        total_epochs=6,
        warmup_iter=2,
        flat_epochs=2,
        no_aug_epochs=1,
        lr_scyedule_save_path=tmp_path,
    )

    joined = "\n".join(messages)

    assert "LR Scheduler" in joined
    assert "Scheduler Overview" in joined
    assert "group 0" in joined
    assert "group 1" in joined
    assert "backbone" in joined
    assert "decoder" in joined
    assert "initial_lr" in joined
    assert "1.000000e-03" in joined
    assert "min_lr" in joined
    assert "1.000000e-04" in joined
    assert "warmup_iter" in joined
    assert "no_aug_iter" in joined
    assert "iter 0" in joined
    assert "warmup" in joined
    assert lr_scheduler_mod.GOLD in joined
    assert lr_scheduler_mod.GREEN in joined
    assert lr_scheduler_mod.RESET in joined
