"""
Training command for BioFuse CLI.

Handles model training, fusion, and evaluation.
"""

import click
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader, Subset
from ..config import ExperimentConfig, merge_configs
from ..utils import set_seed, get_device, ExperimentLogger, PathManager
from ..core import EmbeddingCache
from ..data import (
    create_custom_dataset,
    load_busi,
    load_custom_directory,
    load_imagenet,
    load_medmnist,
)
from ..classifiers import get_classifier
from ..evaluation import Evaluator
from ..models.embedding_extractor import PreTrainedEmbedding


def create_experiment_config(
    config_path=None,
    dataset=None,
    models=None,
    fusion_method='concat',
    projection_dim=0,
    classifier='xgboost',
    img_size=224,
    batch_size=32,
    seed=42,
    data_root=None,
    output_dir='./results',
    no_cache=False,
    device=None,
    download=True,
    max_train_samples=None,
    max_val_samples=None,
    max_test_samples=None,
):
    """Create an experiment config from either a file or CLI-style arguments."""
    if config_path:
        exp_config = ExperimentConfig.load(config_path)

        overrides = {}
        if dataset:
            overrides.setdefault('data', {})['dataset'] = dataset
        if models:
            overrides.setdefault('model', {})['models'] = models.split(',') if isinstance(models, str) else models
        if data_root:
            overrides.setdefault('data', {})['data_root'] = data_root
        if max_train_samples is not None:
            overrides.setdefault('data', {})['max_train_samples'] = max_train_samples
        if max_val_samples is not None:
            overrides.setdefault('data', {})['max_val_samples'] = max_val_samples
        if max_test_samples is not None:
            overrides.setdefault('data', {})['max_test_samples'] = max_test_samples

        if overrides:
            exp_config = merge_configs(exp_config, overrides)
    else:
        if not dataset:
            raise ValueError("--dataset is required when not using --config")

        from ..config import create_default_config

        exp_config = create_default_config(
            experiment_name=f"train_{dataset}",
            dataset=dataset,
            models=models.split(',') if isinstance(models, str) else (models or ['BioMedCLIP']),
        )
        exp_config.data.img_size = img_size
        exp_config.data.batch_size = batch_size
        exp_config.data.data_root = data_root
        exp_config.data.download = download
        exp_config.data.max_train_samples = max_train_samples
        exp_config.data.max_val_samples = max_val_samples
        exp_config.data.max_test_samples = max_test_samples
        exp_config.model.fusion_method = fusion_method
        exp_config.model.projection_dim = projection_dim
        exp_config.classifier.type = classifier
        exp_config.seed = seed
        exp_config.output_dir = output_dir
        exp_config.cache.use_cache = not no_cache
        if device:
            exp_config.device = device

    return exp_config


def maybe_subset_dataset(dataset, max_samples, seed):
    """Return a deterministic subset when max_samples is set."""
    if max_samples is None or max_samples <= 0 or len(dataset) <= max_samples:
        return dataset

    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return Subset(dataset, indices)


def run_experiment(exp_config: ExperimentConfig, verbose: bool = False):
    """Execute a configured BioFuse training run."""
    set_seed(exp_config.seed)
    device = get_device(exp_config.device)
    path_manager = PathManager(
        data_root=exp_config.data.data_root,
        output_dir=exp_config.output_dir,
    )

    exp_logger = ExperimentLogger(
        experiment_name=exp_config.name,
        log_dir=Path(exp_config.output_dir) / 'logs',
        console=verbose,
    )
    exp_logger.log_params(exp_config.to_dict())

    click.echo(f"\n{'='*60}")
    click.echo(f"Experiment: {exp_config.name}")
    click.echo(f"Dataset: {exp_config.data.dataset}")
    click.echo(f"Models: {', '.join(exp_config.model.models)}")
    click.echo(f"Fusion: {exp_config.model.fusion_method}")
    click.echo(f"Classifier: {exp_config.classifier.type}")
    click.echo(f"{'='*60}\n")

    click.echo("Loading data...")
    train_loader, val_loader, test_loader, num_classes = load_data(exp_config)
    click.echo(f"  Train samples: {len(train_loader.dataset)}")
    click.echo(f"  Val samples: {len(val_loader.dataset)}")
    if test_loader:
        click.echo(f"  Test samples: {len(test_loader.dataset)}")
    click.echo(f"  Classes: {num_classes}")

    cache = EmbeddingCache(
        cache_dir=exp_config.cache.cache_dir,
        version=exp_config.cache.cache_version,
    ) if exp_config.cache.use_cache else None

    click.echo("\nExtracting embeddings...")
    train_embeddings, train_labels, val_embeddings, val_labels, test_embeddings, test_labels = \
        extract_embeddings(exp_config, train_loader, val_loader, test_loader, device, cache)

    click.echo(f"  Embedding dimension: {train_embeddings.shape[1]}")

    click.echo(f"\nTraining {exp_config.classifier.type} classifier...")
    start_time = time.time()

    clf = get_classifier(
        exp_config.classifier.type,
        **get_classifier_params(exp_config.classifier)
    )
    clf.fit(train_embeddings, train_labels)

    train_time = time.time() - start_time
    click.echo(f"  Training time: {train_time:.2f}s")

    click.echo("\nEvaluating...")
    evaluator = Evaluator(logger=exp_logger, verbose=True)
    results = evaluator.evaluate_classifier(
        clf,
        val_embeddings,
        val_labels,
        test_embeddings if test_loader else None,
        test_labels if test_loader else None,
        dataset=exp_config.data.dataset,
    )

    click.echo(f"\n{'='*60}")
    click.echo("RESULTS")
    click.echo(f"{'='*60}")

    for split_name, split_results in results.items():
        click.echo(f"\n{split_name.upper()}:")
        for metric, value in split_results.items():
            if isinstance(value, float):
                click.echo(f"  {metric}: {value:.4f}")

    output_path = path_manager.get_output_path(exp_config.name)
    exp_config.save(output_path / 'config.yaml')
    click.echo(f"\nConfig saved to: {output_path / 'config.yaml'}")
    click.echo("\n✓ Training completed successfully!")

    return {
        'results': results,
        'output_path': output_path,
        'num_classes': num_classes,
    }


@click.command('train')
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to config file (YAML/JSON)')
@click.option('--dataset', '-d', type=str, help='Dataset name (e.g., pathmnist, chestmnist, custom)')
@click.option('--models', '-m', type=str, help='Comma-separated list of models (e.g., BioMedCLIP,CONCH)')
@click.option('--fusion-method', type=str, default='concat', help='Fusion method')
@click.option('--projection-dim', type=int, default=0, help='Projection dimension (0 for no projection)')
@click.option('--classifier', type=str, default='xgboost', help='Classifier type')
@click.option('--img-size', type=int, default=224, help='Image size')
@click.option('--batch-size', type=int, default=32, help='Batch size')
@click.option('--seed', type=int, default=42, help='Random seed')
@click.option('--data-root', type=str, help='Root directory for datasets')
@click.option('--output-dir', type=str, default='./results', help='Output directory')
@click.option('--no-cache', is_flag=True, help='Disable embedding cache')
@click.option('--device', type=str, help='Device to use (cuda/cpu)')
@click.option('--download/--no-download', default=True, help='Download datasets if missing')
@click.option('--max-train-samples', type=int, help='Limit the train split for quick smoke runs')
@click.option('--max-val-samples', type=int, help='Limit the validation split for quick smoke runs')
@click.option('--max-test-samples', type=int, help='Limit the test split for quick smoke runs')
@click.pass_context
def train(ctx, config, dataset, models, fusion_method, projection_dim, classifier,
         img_size, batch_size, seed, data_root, output_dir, no_cache, device,
         download, max_train_samples, max_val_samples, max_test_samples):
    """
    Train a BioFuse model on a dataset.

    Examples:

    \b
        # Train with config file
        biofuse train --config experiment.yaml

    \b
        # Train with CLI arguments
        biofuse train --dataset pathmnist --models BioMedCLIP,CONCH

    \b
        # Train with XGBoost classifier
        biofuse train -d chestmnist -m BioMedCLIP --classifier xgboost
    """
    ctx.ensure_object(dict)
    try:
        exp_config = create_experiment_config(
            config_path=config,
            dataset=dataset,
            models=models,
            fusion_method=fusion_method,
            projection_dim=projection_dim,
            classifier=classifier,
            img_size=img_size,
            batch_size=batch_size,
            seed=seed,
            data_root=data_root,
            output_dir=output_dir,
            no_cache=no_cache,
            device=device,
            download=download,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            max_test_samples=max_test_samples,
        )
        run_experiment(exp_config, verbose=ctx.obj.get('verbose', False))

    except Exception as e:
        click.echo(f"\n✗ Error during training: {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        ctx.exit(1)


def load_data(config: ExperimentConfig):
    """Load train/val/test data based on config."""
    dataset_name = config.data.dataset.lower()

    if 'mnist' in dataset_name:
        # MedMNIST
        train_dataset, num_classes = load_medmnist(
            dataset_name,
            split='train',
            img_size=config.data.img_size,
            root=config.data.data_root or '/data/medmnist',
            download=config.data.download
        )
        val_dataset, _ = load_medmnist(
            dataset_name,
            split='val',
            img_size=config.data.img_size,
            root=config.data.data_root or '/data/medmnist'
        )
        test_dataset, _ = load_medmnist(
            dataset_name,
            split='test',
            img_size=config.data.img_size,
            root=config.data.data_root or '/data/medmnist'
        )

        train_dataset = maybe_subset_dataset(train_dataset, config.data.max_train_samples, config.seed)
        val_dataset = maybe_subset_dataset(val_dataset, config.data.max_val_samples, config.seed)
        test_dataset = maybe_subset_dataset(test_dataset, config.data.max_test_samples, config.seed)

        train_loader = DataLoader(train_dataset, batch_size=config.data.batch_size,
                                 shuffle=True, num_workers=config.data.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=config.data.batch_size,
                               shuffle=False, num_workers=config.data.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=config.data.batch_size,
                                shuffle=False, num_workers=config.data.num_workers)

    elif dataset_name == 'imagenet':
        # ImageNet
        train_loader, num_classes = load_imagenet(
            config.data.data_root,
            split='train',
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            subset_size=config.data.subset_size
        )
        val_loader, _ = load_imagenet(
            config.data.data_root,
            split='val',
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers
        )
        test_loader = None  # ImageNet doesn't have public test set

    elif dataset_name == 'busi':
        # BUSI
        train_dataset, num_classes = load_busi(
            config.data.data_root,
            split='train',
            img_size=config.data.img_size
        )
        val_dataset, _ = load_busi(
            config.data.data_root,
            split='val',
            img_size=config.data.img_size
        )
        test_dataset, _ = load_busi(
            config.data.data_root,
            split='test',
            img_size=config.data.img_size
        )

        train_dataset = maybe_subset_dataset(train_dataset, config.data.max_train_samples, config.seed)
        val_dataset = maybe_subset_dataset(val_dataset, config.data.max_val_samples, config.seed)
        test_dataset = maybe_subset_dataset(test_dataset, config.data.max_test_samples, config.seed)

        train_loader = DataLoader(train_dataset, batch_size=config.data.batch_size,
                                 shuffle=True, num_workers=config.data.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=config.data.batch_size,
                               shuffle=False, num_workers=config.data.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=config.data.batch_size,
                                shuffle=False, num_workers=config.data.num_workers)

    elif dataset_name == 'custom':
        if not config.data.data_root:
            raise ValueError("Custom datasets require --data-root pointing at train/val/test folders")

        split_datasets = {}
        num_classes = None
        for split_name in ['train', 'val', 'test']:
            split_dir = Path(config.data.data_root) / split_name
            if not split_dir.exists():
                if split_name == 'test':
                    split_datasets[split_name] = None
                    continue
                raise ValueError(f"Missing required split directory: {split_dir}")

            image_paths, labels, split_num_classes = load_custom_directory(
                split_dir,
                img_size=config.data.img_size,
            )
            split_datasets[split_name] = create_custom_dataset(
                image_paths,
                labels,
                img_size=config.data.img_size,
                from_paths=True,
            )
            max_samples = getattr(config.data, f'max_{split_name}_samples')
            split_datasets[split_name] = maybe_subset_dataset(
                split_datasets[split_name],
                max_samples,
                config.seed,
            )
            num_classes = max(num_classes or 0, split_num_classes)

        train_loader = DataLoader(
            split_datasets['train'],
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        val_loader = DataLoader(
            split_datasets['val'],
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        test_loader = None
        if split_datasets['test'] is not None:
            test_loader = DataLoader(
                split_datasets['test'],
                batch_size=config.data.batch_size,
                shuffle=False,
                num_workers=config.data.num_workers,
            )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_loader, val_loader, test_loader, num_classes


def extract_embeddings(config, train_loader, val_loader, test_loader, device, cache=None):
    """Extract embeddings from dataloaders."""
    import numpy as np
    from tqdm import tqdm

    dataset_name = config.data.dataset
    img_size = config.data.img_size

    all_embeddings = {}

    for split_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        if loader is None:
            all_embeddings[split_name] = (None, None)
            continue

        # Try to load from cache
        split_embeddings_list = []
        split_labels_list = []

        for model_name in config.model.models:
            if cache and cache.exists(dataset_name, model_name, img_size, split_name):
                embeddings, labels = cache.load(dataset_name, model_name, img_size, split_name)
            else:
                # Extract embeddings
                model = PreTrainedEmbedding(model_name, device=device)
                model.eval()

                batch_embeddings = []
                batch_labels = []

                with torch.no_grad():
                    for images, labels in tqdm(loader, desc=f"{model_name} {split_name}"):
                        emb = model(images)
                        batch_embeddings.append(emb.cpu().numpy())
                        batch_labels.append(labels.numpy())

                embeddings = np.vstack(batch_embeddings)
                labels = np.concatenate(batch_labels)

                # Save to cache
                if cache:
                    cache.save(embeddings, labels, dataset_name, model_name, img_size, split_name)

            split_embeddings_list.append(embeddings)
            if len(split_labels_list) == 0:
                split_labels_list = labels

        # Concatenate embeddings from all models
        split_embeddings = np.hstack(split_embeddings_list)
        all_embeddings[split_name] = (split_embeddings, split_labels_list)

    train_emb, train_labels = all_embeddings['train']
    val_emb, val_labels = all_embeddings['val']
    test_emb, test_labels = all_embeddings.get('test', (None, None))

    return train_emb, train_labels, val_emb, val_labels, test_emb, test_labels


def get_classifier_params(clf_config):
    """Extract classifier parameters from config."""
    params = {
        'random_state': clf_config.random_state
    }

    clf_type = clf_config.type.lower()

    if 'xgb' in clf_type or 'cat' in clf_type:
        params.update({
            'n_estimators': clf_config.n_estimators,
            'learning_rate': clf_config.learning_rate,
            'max_depth': clf_config.max_depth,
        })
    elif 'nn_' in clf_type:
        params.update({
            'hidden_dim': clf_config.hidden_dim,
            'dropout': clf_config.dropout,
            'num_epochs': clf_config.num_epochs,
            'learning_rate': clf_config.learning_rate,
        })

    return params
