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
