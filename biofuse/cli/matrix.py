"""
Compatibility matrix command for BioFuse.

Runs a factorized smoke sweep across encoders, datasets, fusion methods, and
classifiers, and prints a tabular PASS/FAIL/SKIP summary.
"""

from __future__ import annotations

import csv
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import click

from ..config.config import FusionMethod
from ..models.config import AUTH_TOKEN, MODEL_MAP
from .smoke import build_repo_sample_dataset


MEDMNIST_DATASETS = [
    "pathmnist",
    "chestmnist",
    "dermamnist",
    "octmnist",
    "pneumoniamnist",
    "retinamnist",
    "breastmnist",
    "bloodmnist",
    "tissuemnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
]

SPECIAL_DATASETS = ["custom", "busi", "imagenet"]
ALL_DATASETS = MEDMNIST_DATASETS + SPECIAL_DATASETS
ALL_MODELS = list(MODEL_MAP.keys())
ALL_FUSION_METHODS = [method.value for method in FusionMethod]
ALL_CLASSIFIERS = [
    "logistic",
    "xgboost",
    "catboost",
    "nn_mlp",
    "nn_cnn",
    "nn_resnet",
]
GATED_MODELS = {"UNI", "UNI2", "CONCH", "CheXagent"}


@dataclass
class MatrixCase:
    suite: str
    target: str
    dataset: str
    models: str
    fusion_method: str
    classifier: str
    projection_dim: int = 0
    data_root: Optional[str] = None
    download: bool = True
    skip_reason: Optional[str] = None


def _parse_selection(value: str, all_values: Iterable[str]) -> List[str]:
    if value == "all":
        return list(all_values)
    return [item.strip() for item in value.split(",") if item.strip()]


def _safe_case_name(case: MatrixCase) -> str:
    pieces = [
        "matrix",
        case.suite,
        case.target,
        case.dataset,
        case.models.replace(",", "-"),
        case.fusion_method,
        case.classifier,
    ]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", "_".join(pieces))


def _reference_dataset(selected_datasets: List[str]) -> str:
    if "breastmnist" in selected_datasets:
        return "breastmnist"
    if selected_datasets:
        return selected_datasets[0]
    return "breastmnist"


def _build_dataset_cases(
    selected_datasets: List[str],
    reference_model: str,
    custom_root: str,
    medmnist_root: str,
    busi_root: Optional[str],
    imagenet_root: Optional[str],
) -> List[MatrixCase]:
    cases = []
    for dataset in selected_datasets:
        if dataset == "custom":
            cases.append(
                MatrixCase(
                    suite="dataset",
                    target=dataset,
                    dataset="custom",
                    models=reference_model,
                    fusion_method="concat",
                    classifier="logistic",
                    data_root=custom_root,
                    download=False,
                )
            )
            continue

        if dataset == "busi":
            cases.append(
                MatrixCase(
                    suite="dataset",
                    target=dataset,
                    dataset="busi",
                    models=reference_model,
                    fusion_method="concat",
                    classifier="logistic",
                    data_root=busi_root,
                    download=False,
                    skip_reason=None if busi_root else "busi root not provided",
                )
            )
            continue

        if dataset == "imagenet":
            cases.append(
                MatrixCase(
                    suite="dataset",
                    target=dataset,
                    dataset="imagenet",
                    models=reference_model,
                    fusion_method="concat",
                    classifier="logistic",
                    data_root=imagenet_root,
                    download=False,
                    skip_reason=None if imagenet_root else "imagenet root not provided",
                )
            )
            continue

        cases.append(
            MatrixCase(
                suite="dataset",
                target=dataset,
                dataset=dataset,
                models=reference_model,
                fusion_method="concat",
                classifier="logistic",
                data_root=medmnist_root,
                download=True,
            )
        )
    return cases


def build_factorized_cases(
    selected_models: List[str],
    selected_datasets: List[str],
    selected_fusion_methods: List[str],
    selected_classifiers: List[str],
    custom_root: str,
    medmnist_root: str,
    busi_root: Optional[str],
    imagenet_root: Optional[str],
    fusion_models: str,
) -> List[MatrixCase]:
    cases: List[MatrixCase] = []
    reference_dataset = _reference_dataset(selected_datasets)
    reference_model = "CLIP" if "CLIP" in selected_models else selected_models[0]

    for model_name in selected_models:
        skip_reason = None
        if model_name in GATED_MODELS and not AUTH_TOKEN:
            skip_reason = "HF_TOKEN not configured for gated model"

        dataset = "custom" if reference_dataset == "custom" else reference_dataset
        data_root = custom_root if dataset == "custom" else medmnist_root
        download = dataset != "custom"
        cases.append(
            MatrixCase(
                suite="encoder",
                target=model_name,
                dataset=dataset,
                models=model_name,
                fusion_method="concat",
                classifier="logistic",
                data_root=data_root,
                download=download,
                skip_reason=skip_reason,
            )
        )

    cases.extend(
        _build_dataset_cases(
            selected_datasets=selected_datasets,
            reference_model=reference_model,
            custom_root=custom_root,
            medmnist_root=medmnist_root,
            busi_root=busi_root,
            imagenet_root=imagenet_root,
        )
    )

    for fusion_method in selected_fusion_methods:
        cases.append(
            MatrixCase(
                suite="fusion",
                target=fusion_method,
                dataset=reference_dataset if reference_dataset != "custom" else "breastmnist",
                models=fusion_models,
                fusion_method=fusion_method,
                classifier="logistic",
                projection_dim=0 if fusion_method == "concat" else 256,
                data_root=medmnist_root,
                download=True,
            )
        )

    for classifier in selected_classifiers:
        cases.append(
            MatrixCase(
                suite="classifier",
                target=classifier,
                dataset=reference_dataset if reference_dataset != "custom" else "breastmnist",
                models=reference_model,
                fusion_method="concat",
                classifier=classifier,
                data_root=medmnist_root,
                download=True,
            )
        )

    return cases


def _build_command(
    case: MatrixCase,
    output_dir: Path,
    device: Optional[str],
    batch_size: int,
    img_size: int,
    max_train_samples: int,
    max_val_samples: int,
    max_test_samples: int,
) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "biofuse.cli.main",
        "train",
        "--dataset",
        case.dataset,
        "--models",
        case.models,
        "--fusion-method",
        case.fusion_method,
        "--projection-dim",
        str(case.projection_dim),
        "--classifier",
        case.classifier,
        "--batch-size",
        str(batch_size),
        "--img-size",
        str(img_size),
        "--output-dir",
        str(output_dir),
        "--no-cache",
        "--max-train-samples",
        str(max_train_samples),
        "--max-val-samples",
        str(max_val_samples),
        "--max-test-samples",
        str(max_test_samples),
    ]

    if device:
        command.extend(["--device", device])

    if case.data_root:
        command.extend(["--data-root", case.data_root])

    command.append("--download" if case.download else "--no-download")
    return command


def _extract_output_path(stdout: str) -> str:
    match = re.search(r"Config saved to:\s*(.+)", stdout)
    return match.group(1).strip() if match else ""


def _short_note(stdout: str, stderr: str, exc: Optional[BaseException] = None) -> str:
    if exc is not None:
        return f"{type(exc).__name__}: {exc}"

    for stream in (stderr, stdout):
        lines = [line.strip() for line in stream.splitlines() if line.strip()]
        if lines:
            return lines[-1][:200]
    return ""


def _render_table(rows: List[dict]) -> str:
    columns = [
        ("suite", "suite"),
        ("target", "target"),
        ("status", "status"),
        ("duration_s", "sec"),
        ("note", "note"),
    ]
    widths = {}
    for key, header in columns:
        widths[key] = len(header)
        for row in rows:
            widths[key] = min(max(widths[key], len(str(row.get(key, "")))), 80)
        if key == "note":
            widths[key] = min(widths[key], 80)

    def format_value(key: str, value: object) -> str:
        text = str(value)
        if key == "duration_s":
            text = f"{float(value):.1f}" if value not in ("", None) else ""
        if len(text) > widths[key]:
            text = text[: widths[key] - 1] + "…"
        return text.ljust(widths[key])

    header = " | ".join(header.ljust(widths[key]) for key, header in columns)
    separator = "-+-".join("-" * widths[key] for key, _ in columns)
    lines = [header, separator]
    for row in rows:
        lines.append(" | ".join(format_value(key, row.get(key, "")) for key, _ in columns))
    return "\n".join(lines)


def _write_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "suite",
        "target",
        "dataset",
        "models",
        "fusion_method",
        "classifier",
        "status",
        "duration_s",
        "returncode",
        "output_path",
        "note",
        "command",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@click.command("matrix")
@click.option("--device", type=str, help="Device to use (cuda/cpu)")
@click.option("--output-dir", type=str, default="./results/matrix", show_default=True, help="Directory for matrix outputs")
@click.option("--models", type=str, default="all", show_default=True, help="Comma-separated encoder list or 'all'")
@click.option("--datasets", type=str, default="all", show_default=True, help="Comma-separated dataset list or 'all'")
@click.option("--fusion-methods", type=str, default="all", show_default=True, help="Comma-separated fusion method list or 'all'")
@click.option("--classifiers", type=str, default="all", show_default=True, help="Comma-separated classifier list or 'all'")
@click.option("--fusion-models", type=str, default="CLIP,rad-dino", show_default=True, help="Two encoders to use for the fusion sweep")
@click.option("--medmnist-root", type=str, default="/data/medmnist", show_default=True, help="MedMNIST cache/data root")
@click.option("--busi-root", type=str, help="BUSI dataset root")
@click.option("--imagenet-root", type=str, help="ImageNet dataset root")
@click.option("--batch-size", type=int, default=8, show_default=True, help="Batch size for each sweep run")
@click.option("--img-size", type=int, default=224, show_default=True, help="Image size for each sweep run")
@click.option("--max-train-samples", type=int, default=32, show_default=True, help="Train subset size for each run")
@click.option("--max-val-samples", type=int, default=16, show_default=True, help="Validation subset size for each run")
@click.option("--max-test-samples", type=int, default=16, show_default=True, help="Test subset size for each run")
@click.pass_context
def matrix(
    ctx,
    device,
    output_dir,
    models,
    datasets,
    fusion_methods,
    classifiers,
    fusion_models,
    medmnist_root,
    busi_root,
    imagenet_root,
    batch_size,
    img_size,
    max_train_samples,
    max_val_samples,
    max_test_samples,
):
    """
    Run a factorized compatibility sweep and print a PASS/FAIL table.

    The sweep is factorized by suite:
    - one run per encoder
    - one run per dataset
    - one run per fusion method
    - one run per classifier

    This is intentionally smaller and more actionable than the full Cartesian
    product of every possible combination.
    """
    del ctx  # unused

    selected_models = _parse_selection(models, ALL_MODELS)
    selected_datasets = _parse_selection(datasets, ALL_DATASETS)
    selected_fusion_methods = _parse_selection(fusion_methods, ALL_FUSION_METHODS)
    selected_classifiers = _parse_selection(classifiers, ALL_CLASSIFIERS)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[2]
    source_dir = repo_root / "data"

    with tempfile.TemporaryDirectory(prefix="biofuse-matrix-custom-") as tmpdir:
        custom_root = str(Path(tmpdir) / "dataset")
        build_repo_sample_dataset(source_dir, Path(custom_root))

        cases = build_factorized_cases(
            selected_models=selected_models,
            selected_datasets=selected_datasets,
            selected_fusion_methods=selected_fusion_methods,
            selected_classifiers=selected_classifiers,
            custom_root=custom_root,
            medmnist_root=medmnist_root,
            busi_root=busi_root,
            imagenet_root=imagenet_root,
            fusion_models=fusion_models,
        )

        rows = []
        total = len(cases)
        for index, case in enumerate(cases, start=1):
            click.echo(f"[{index}/{total}] {case.suite}:{case.target}")
            command = _build_command(
                case,
                output_dir=output_path,
                device=device,
                batch_size=batch_size,
                img_size=img_size,
                max_train_samples=max_train_samples,
                max_val_samples=max_val_samples,
                max_test_samples=max_test_samples,
            )

            start = time.time()
            if case.skip_reason:
                rows.append(
                    {
                        **asdict(case),
                        "status": "SKIP",
                        "duration_s": 0.0,
                        "returncode": "",
                        "output_path": "",
                        "note": case.skip_reason,
                        "command": shlex.join(command),
                    }
                )
                continue

            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                duration_s = time.time() - start
                status = "PASS" if completed.returncode == 0 else "FAIL"
                rows.append(
                    {
                        **asdict(case),
                        "status": status,
                        "duration_s": duration_s,
                        "returncode": completed.returncode,
                        "output_path": _extract_output_path(completed.stdout),
                        "note": _short_note(completed.stdout, completed.stderr),
                        "command": shlex.join(command),
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                rows.append(
                    {
                        **asdict(case),
                        "status": "FAIL",
                        "duration_s": time.time() - start,
                        "returncode": "",
                        "output_path": "",
                        "note": _short_note("", "", exc=exc),
                        "command": shlex.join(command),
                    }
                )

    csv_path = output_path / "compatibility_matrix.csv"
    _write_csv(rows, csv_path)
    click.echo()
    click.echo(_render_table(rows))
    click.echo()
    click.echo(f"CSV summary: {csv_path}")
    click.echo(
        "Summary: "
        f"{sum(row['status'] == 'PASS' for row in rows)} pass, "
        f"{sum(row['status'] == 'FAIL' for row in rows)} fail, "
        f"{sum(row['status'] == 'SKIP' for row in rows)} skip"
    )
