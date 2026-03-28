from types import SimpleNamespace

from click.testing import CliRunner

import biofuse.cli.smoke as smoke_module


def test_smoke_custom_uses_custom_dataset_preset(monkeypatch):
    captured = {}

    def fake_create_experiment_config(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(name="train_custom")

    def fake_run_experiment(exp_config, verbose=False):
        captured["experiment_name"] = exp_config.name
        captured["verbose"] = verbose
        return {}

    monkeypatch.setattr(smoke_module, "create_experiment_config", fake_create_experiment_config)
    monkeypatch.setattr(smoke_module, "run_experiment", fake_run_experiment)

    result = CliRunner().invoke(smoke_module.smoke, ["--preset", "custom", "--models", "CLIP"])

    assert result.exit_code == 0
    assert captured["dataset"] == "custom"
    assert captured["models"] == "CLIP"
    assert captured["experiment_name"] == "smoke_custom"


def test_smoke_medmnist_uses_requested_dataset(monkeypatch):
    captured = {}

    def fake_create_experiment_config(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(name="train_pathmnist")

    def fake_run_experiment(exp_config, verbose=False):
        captured["experiment_name"] = exp_config.name
        return {}

    monkeypatch.setattr(smoke_module, "create_experiment_config", fake_create_experiment_config)
    monkeypatch.setattr(smoke_module, "run_experiment", fake_run_experiment)

    result = CliRunner().invoke(
        smoke_module.smoke,
        ["--preset", "medmnist", "--dataset", "pathmnist", "--models", "CLIP"],
    )

    assert result.exit_code == 0
    assert captured["dataset"] == "pathmnist"
    assert captured["data_root"] == "/data/medmnist"
    assert captured["experiment_name"] == "smoke_pathmnist"


def test_resolve_sample_data_dir_prefers_env_override(tmp_path, monkeypatch):
    sample_dir = tmp_path / "sample-data"
    sample_dir.mkdir()
    for filename in ["xray.jpg", "hist.jpg", "bcc.jpg"]:
        (sample_dir / filename).write_bytes(b"test")

    monkeypatch.setenv("BIOFUSE_SAMPLE_DATA_DIR", str(sample_dir))

    assert smoke_module.resolve_sample_data_dir() == sample_dir


def test_resolve_sample_data_dir_raises_clear_error_when_missing(monkeypatch):
    monkeypatch.delenv("BIOFUSE_SAMPLE_DATA_DIR", raising=False)

    try:
        smoke_module.resolve_sample_data_dir(
            extra_candidates=["/definitely/missing"],
            include_default_candidates=False,
        )
    except FileNotFoundError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("expected FileNotFoundError")

    assert "BIOFUSE_SAMPLE_DATA_DIR" in message
    assert "xray.jpg" in message
