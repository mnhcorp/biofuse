"""
Training command for BioFuse CLI.

Handles model training, fusion, and evaluation.
"""

import click
from pathlib import Path
import time
import torch
from typing import List

from ..config import ExperimentConfig, merge_configs
from ..utils import set_seed, get_device, ExperimentLogger, PathManager
from ..core import EmbeddingCache
from ..data import load_medmnist, load_imagenet, load_busi
from ..classifiers import get_classifier
from ..evaluation import Evaluator
from ..models.biofuse_model import BioFuseModel
from ..models.embedding_extractor import PreTrainedEmbedding


@click.command('train')
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to config file (YAML/JSON)')
@click.option('--dataset', '-d', type=str, help='Dataset name (e.g., pathmnist, chestmnist)')
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
@click.pass_context
def train(ctx, config, dataset, models, fusion_method, projection_dim, classifier,
         img_size, batch_size, seed, data_root, output_dir, no_cache, device):
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
    logger = ctx.obj.get('logger')

    # Load or create config
    if config:
        click.echo(f"Loading config from: {config}")
        exp_config = ExperimentConfig.load(config)

        # Override with CLI arguments
        overrides = {}
        if dataset:
            overrides['data'] = {'dataset': dataset}
        if models:
            overrides['model'] = {'models': models.split(',')}
        if data_root:
            if 'data' not in overrides:
                overrides['data'] = {}
            overrides['data']['data_root'] = data_root

        if overrides:
            exp_config = merge_configs(exp_config, overrides)
    else:
        # Create config from CLI arguments
        if not dataset:
            click.echo("Error: --dataset is required when not using --config", err=True)
            ctx.exit(1)

        from ..config import create_default_config
        exp_config = create_default_config(
            experiment_name=f"train_{dataset}",
            dataset=dataset,
            models=models.split(',') if models else ['BioMedCLIP']
        )

        # Update with CLI args
        exp_config.data.img_size = img_size
        exp_config.data.batch_size = batch_size
        exp_config.data.data_root = data_root
        exp_config.model.fusion_method = fusion_method
        exp_config.model.projection_dim = projection_dim
        exp_config.classifier.type = classifier
        exp_config.seed = seed
        exp_config.output_dir = output_dir
        exp_config.cache.use_cache = not no_cache
        if device:
            exp_config.device = device

    # Setup
    set_seed(exp_config.seed)
    device = get_device(exp_config.device)
    path_manager = PathManager(
        data_root=exp_config.data.data_root,
        output_dir=exp_config.output_dir
    )

    # Setup experiment logger
    exp_logger = ExperimentLogger(
        experiment_name=exp_config.name,
        log_dir=Path(exp_config.output_dir) / 'logs',
        console=ctx.obj.get('verbose', False)
    )
    exp_logger.log_params(exp_config.to_dict())

    click.echo(f"\n{'='*60}")
    click.echo(f"Experiment: {exp_config.name}")
    click.echo(f"Dataset: {exp_config.data.dataset}")
    click.echo(f"Models: {', '.join(exp_config.model.models)}")
    click.echo(f"Fusion: {exp_config.model.fusion_method}")
    click.echo(f"Classifier: {exp_config.classifier.type}")
    click.echo(f"{'='*60}\n")

    try:
        # Load data
        click.echo("Loading data...")
        train_loader, val_loader, test_loader, num_classes = load_data(exp_config)
        click.echo(f"  Train samples: {len(train_loader.dataset)}")
        click.echo(f"  Val samples: {len(val_loader.dataset)}")
        if test_loader:
            click.echo(f"  Test samples: {len(test_loader.dataset)}")
        click.echo(f"  Classes: {num_classes}")

        # Extract or load embeddings
        cache = EmbeddingCache(
            cache_dir=exp_config.cache.cache_dir,
            version=exp_config.cache.cache_version
        ) if exp_config.cache.use_cache else None

        click.echo("\nExtracting embeddings...")
        train_embeddings, train_labels, val_embeddings, val_labels, test_embeddings, test_labels = \
            extract_embeddings(exp_config, train_loader, val_loader, test_loader, device, cache)

        click.echo(f"  Embedding dimension: {train_embeddings.shape[1]}")

        # Train classifier
        click.echo(f"\nTraining {exp_config.classifier.type} classifier...")
        start_time = time.time()

        clf = get_classifier(
            exp_config.classifier.type,
            **get_classifier_params(exp_config.classifier)
        )
        clf.fit(train_embeddings, train_labels)

        train_time = time.time() - start_time
        click.echo(f"  Training time: {train_time:.2f}s")

        # Evaluate
        click.echo("\nEvaluating...")
        evaluator = Evaluator(logger=exp_logger, verbose=True)
        results = evaluator.evaluate_classifier(
            clf,
            val_embeddings,
            val_labels,
            test_embeddings if test_loader else None,
            test_labels if test_loader else None,
            dataset=exp_config.data.dataset
        )

        # Display results
        click.echo(f"\n{'='*60}")
        click.echo("RESULTS")
        click.echo(f"{'='*60}")

        for split_name, split_results in results.items():
            click.echo(f"\n{split_name.upper()}:")
            for metric, value in split_results.items():
                if isinstance(value, float):
                    click.echo(f"  {metric}: {value:.4f}")

        # Save model and config
        output_path = path_manager.get_output_path(exp_config.name)
        exp_config.save(output_path / 'config.yaml')
        click.echo(f"\nConfig saved to: {output_path / 'config.yaml'}")

        click.echo(f"\n✓ Training completed successfully!")

    except Exception as e:
        click.echo(f"\n✗ Error during training: {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        ctx.exit(1)


def load_data(config: ExperimentConfig):
    """Load train/val/test data based on config."""
    from torch.utils.data import DataLoader

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

        train_loader = DataLoader(train_dataset, batch_size=config.data.batch_size,
                                 shuffle=True, num_workers=config.data.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=config.data.batch_size,
                               shuffle=False, num_workers=config.data.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=config.data.batch_size,
                                shuffle=False, num_workers=config.data.num_workers)

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
                model = PreTrainedEmbedding(model_name).to(device)
                model.eval()

                batch_embeddings = []
                batch_labels = []

                with torch.no_grad():
                    for images, labels in tqdm(loader, desc=f"{model_name} {split_name}"):
                        images = images.to(device)
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
