from types import SimpleNamespace

import torch

import engine.solver.det_engine as det_engine


def _build_loader(remap=False):
    dataset = SimpleNamespace(
        remap_mscoco_category=remap,
        category2name={0: "obj"},
        label2category={0: 0},
    )
    return SimpleNamespace(dataset=dataset)


def test_plot_training_modalities_multimodal_uses_rgb_suffix(monkeypatch, tmp_path):
    calls = []

    def _fake_plot_sample(data, category2name, output_dir, coco2category=None):
        calls.append(str(output_dir))

    monkeypatch.setattr(det_engine, "plot_sample", _fake_plot_sample)

    samples = {
        "rgb": torch.rand(2, 3, 16, 16),
        "npy": torch.rand(2, 1, 16, 16),
    }
    targets = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)} for _ in range(2)]

    det_engine._plot_training_modalities(samples, targets, _build_loader(remap=False), tmp_path, epoch=3)

    assert calls == [
        str(tmp_path / "train_batch_3_rgb.png"),
        str(tmp_path / "train_batch_3_npy.png"),
    ]


def test_plot_training_modalities_single_modal_keeps_legacy_name(monkeypatch, tmp_path):
    calls = []

    def _fake_plot_sample(data, category2name, output_dir, coco2category=None):
        calls.append(str(output_dir))

    monkeypatch.setattr(det_engine, "plot_sample", _fake_plot_sample)

    samples = torch.rand(2, 3, 16, 16)
    targets = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)} for _ in range(2)]

    det_engine._plot_training_modalities(samples, targets, _build_loader(remap=False), tmp_path, epoch=4)

    assert calls == [str(tmp_path / "train_batch_4.png")]


def test_plot_training_modalities_normalizes_npy_before_plot(monkeypatch, tmp_path):
    """The plot path runs the prior through normalize_tensor_minmax_per_sample.

    That helper clamps data already in [0, 1] and divides anything larger by 255 (see
    tests/engine/misc/test_modality_utils.py). Real depth priors are stored in [0, 1], so
    they hit the clamp branch; this case feeds out-of-range values to pin the other one.
    """

    captured = {}

    def _fake_plot_sample(data, category2name, output_dir, coco2category=None):
        images, _ = data
        captured[str(output_dir)] = images.clone()

    monkeypatch.setattr(det_engine, "plot_sample", _fake_plot_sample)

    samples = {
        "rgb": torch.rand(2, 3, 16, 16),
        "npy": torch.tensor(
            [
                [[[0.0, 5.0], [10.0, 20.0]]],
                [[[7.0, 7.0], [7.0, 7.0]]],
            ],
            dtype=torch.float32,
        ),
    }
    targets = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)} for _ in range(2)]

    det_engine._plot_training_modalities(samples, targets, _build_loader(remap=False), tmp_path, epoch=5)

    npy_out = captured[str(tmp_path / "train_batch_5_npy.png")]
    # values exceed 1.0, so the helper rescales by 255 rather than per-sample min-max
    assert torch.allclose(npy_out[0], samples["npy"][0] / 255.0)
    assert torch.allclose(npy_out[1], samples["npy"][1] / 255.0)
    assert float(npy_out.min()) == 0.0
    assert 0.0 < float(npy_out.max()) <= 1.0


def test_plot_training_modalities_leaves_unit_range_priors_alone(monkeypatch, tmp_path):
    """A real depth prior is already in [0, 1]: plotting must not rescale it."""
    captured = {}

    def _fake_plot_sample(data, category2name, output_dir, coco2category=None):
        images, _ = data
        captured[str(output_dir)] = images.clone()

    monkeypatch.setattr(det_engine, "plot_sample", _fake_plot_sample)

    prior = torch.tensor(
        [
            [[[0.0, 0.25], [0.5, 1.0]]],
            [[[0.1, 0.1], [0.1, 0.1]]],
        ],
        dtype=torch.float32,
    )
    samples = {"rgb": torch.rand(2, 3, 16, 16), "npy": prior.clone()}
    targets = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)} for _ in range(2)]

    det_engine._plot_training_modalities(samples, targets, _build_loader(remap=False), tmp_path, epoch=5)

    npy_out = captured[str(tmp_path / "train_batch_5_npy.png")]
    assert torch.allclose(npy_out, prior)


def test_target_device_move_supports_text_lists():
    targets = [
        {
            "boxes": torch.zeros(1, 4),
            "labels": torch.zeros(1, dtype=torch.long),
            "texts": ["pedestrian", "car"],
            "text_feats": torch.zeros(2, 8),
        }
    ]

    moved = det_engine.move_samples_to_device(targets, torch.device("cpu"), non_blocking=True)

    assert moved[0]["texts"] == ["pedestrian", "car"]
    assert moved[0]["text_feats"].shape == (2, 8)
