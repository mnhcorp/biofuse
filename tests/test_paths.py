from biofuse.utils.paths import PathManager


def test_get_output_path_creates_timestamped_unique_run_directories(tmp_path):
    path_manager = PathManager(
        data_root=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        model_dir=tmp_path / "models",
        output_dir=tmp_path / "results",
    )

    first = path_manager.get_output_path(
        "train_breastmnist",
        unique=True,
        run_id="20260327-153000",
    )
    second = path_manager.get_output_path(
        "train_breastmnist",
        unique=True,
        run_id="20260327-153000",
    )

    assert first.name == "train_breastmnist_20260327-153000"
    assert second.name == "train_breastmnist_20260327-153000-02"
    assert first.exists()
    assert second.exists()
