"""
Smoke-test command for BioFuse CLI.

Provides fast end-to-end checks for custom sample data and MedMNIST subsets.
"""

import shutil
import tempfile
from pathlib import Path

import click

from .train import create_experiment_config, run_experiment


SPLIT_LAYOUT = {
    "train": {
        "class0": ["xray.jpg", "xray.jpg"],
        "class1": ["hist.jpg", "bcc.jpg"],
    },
    "val": {
        "class0": ["xray.jpg"],
        "class1": ["hist.jpg"],
    },
    "test": {
        "class0": ["xray.jpg"],
        "class1": ["bcc.jpg"],
    },
}


def build_repo_sample_dataset(source_dir: Path, dataset_root: Path) -> None:
    """Create a tiny two-class dataset from repository sample images."""
    for split, classes in SPLIT_LAYOUT.items():
        for class_name, filenames in classes.items():
            class_dir = dataset_root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for index, filename in enumerate(filenames):
                shutil.copy2(source_dir / filename, class_dir / f"{index}_{filename}")


@click.command('smoke')
@click.option('--preset', type=click.Choice(['custom', 'medmnist']), default='custom', show_default=True, help='Smoke-test dataset preset')
@click.option('--dataset', default='pathmnist', show_default=True, help='Dataset name for the medmnist preset')
@click.option('--models', '-m', default='CLIP', show_default=True, help='Comma-separated encoder list')
@click.option('--classifier', default='logistic', show_default=True, help='Classifier type')
@click.option('--device', type=str, help='Device to use (cuda/cpu)')
@click.option('--data-root', type=str, help='Dataset root. Required for medmnist if you do not want the default')
@click.option('--output-dir', default='./results', show_default=True, help='Output directory')
@click.option('--img-size', default=224, show_default=True, help='Image size')
@click.option('--batch-size', default=8, show_default=True, help='Batch size')
@click.option('--seed', default=42, show_default=True, help='Random seed')
@click.option('--download/--no-download', default=True, help='Download datasets if missing')
@click.option('--max-train-samples', default=32, show_default=True, help='Train subset size for the smoke run')
@click.option('--max-val-samples', default=16, show_default=True, help='Validation subset size for the smoke run')
@click.option('--max-test-samples', default=16, show_default=True, help='Test subset size for the smoke run')
@click.pass_context
def smoke(
    ctx,
    preset,
    dataset,
    models,
    classifier,
    device,
    data_root,
    output_dir,
    img_size,
    batch_size,
    seed,
    download,
    max_train_samples,
    max_val_samples,
    max_test_samples,
):
    """
    Run a small end-to-end BioFuse check through the normal CLI training path.

    Examples:

    \b
        biofuse smoke

    \b
        biofuse smoke --preset medmnist --dataset pathmnist --models CLIP --device cuda
    """
    ctx.ensure_object(dict)
    verbose = ctx.obj.get('verbose', False)

    if preset == 'custom':
        repo_root = Path(__file__).resolve().parents[2]
        source_dir = repo_root / 'data'

        with tempfile.TemporaryDirectory(prefix='biofuse-smoke-') as tmpdir:
            dataset_root = Path(tmpdir) / 'dataset'
            build_repo_sample_dataset(source_dir, dataset_root)

            exp_config = create_experiment_config(
                dataset='custom',
                models=models,
                classifier=classifier,
                img_size=img_size,
                batch_size=batch_size,
                seed=seed,
                data_root=str(dataset_root),
                output_dir=output_dir,
                no_cache=True,
                device=device,
                download=False,
                max_train_samples=max_train_samples,
                max_val_samples=max_val_samples,
                max_test_samples=max_test_samples,
            )
            exp_config.name = 'smoke_custom'
            run_experiment(exp_config, verbose=verbose)
    else:
        exp_config = create_experiment_config(
            dataset=dataset,
            models=models,
            classifier=classifier,
            img_size=img_size,
            batch_size=batch_size,
            seed=seed,
            data_root=data_root or '/data/medmnist',
            output_dir=output_dir,
            no_cache=True,
            device=device,
            download=download,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            max_test_samples=max_test_samples,
        )
        exp_config.name = f'smoke_{dataset}'
        run_experiment(exp_config, verbose=verbose)
