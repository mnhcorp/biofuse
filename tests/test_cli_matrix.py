from pathlib import Path

import biofuse.cli.matrix as matrix_module


def test_build_factorized_cases_covers_each_suite_and_marks_missing_inputs(monkeypatch):
    monkeypatch.setattr(matrix_module, "AUTH_TOKEN", None)

    cases = matrix_module.build_factorized_cases(
        selected_models=["CLIP", "UNI"],
        selected_datasets=["custom", "breastmnist", "busi", "imagenet"],
        selected_fusion_methods=["concat", "mean"],
        selected_classifiers=["logistic", "xgboost"],
        custom_root="/tmp/custom",
        medmnist_root="/data/medmnist",
        busi_root=None,
        imagenet_root=None,
        fusion_models="CLIP,rad-dino",
    )

    assert len(cases) == 10
    assert any(case.suite == "encoder" and case.target == "CLIP" for case in cases)
    assert any(case.suite == "encoder" and case.target == "UNI" and case.skip_reason for case in cases)
    assert any(case.suite == "dataset" and case.target == "custom" and case.data_root == "/tmp/custom" for case in cases)
    assert any(case.suite == "dataset" and case.target == "busi" and case.skip_reason == "busi root not provided" for case in cases)
    assert any(case.suite == "fusion" and case.target == "mean" and case.projection_dim == 256 for case in cases)
    assert any(case.suite == "classifier" and case.target == "xgboost" for case in cases)


def test_build_command_contains_expected_cli_arguments():
    case = matrix_module.MatrixCase(
        suite="fusion",
        target="mean",
        dataset="breastmnist",
        models="CLIP,rad-dino",
        fusion_method="mean",
        classifier="logistic",
        projection_dim=256,
        data_root="/data/medmnist",
        download=True,
    )

    command = matrix_module._build_command(
        case,
        output_dir=Path("/workspace/results/matrix"),
        device="cuda",
        batch_size=8,
        img_size=224,
        max_train_samples=32,
        max_val_samples=16,
        max_test_samples=16,
    )

    command_str = " ".join(command)
    assert "biofuse.cli.main" in command_str
    assert "--dataset breastmnist" in command_str
    assert "--models CLIP,rad-dino" in command_str
    assert "--fusion-method mean" in command_str
    assert "--projection-dim 256" in command_str
    assert "--download" in command_str
