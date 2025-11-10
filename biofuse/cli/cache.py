"""
Cache management commands for BioFuse CLI.
"""

import click
from ..core import EmbeddingCache


@click.group('cache')
def cache_group():
    """Manage embedding cache."""
    pass


@cache_group.command('list')
@click.option('--cache-dir', type=str, help='Cache directory path')
def list_cache(cache_dir):
    """List all cached embeddings."""
    cache = EmbeddingCache(cache_dir=cache_dir)
    info = cache.get_info()

    click.echo(f"Cache directory: {info['cache_dir']}")
    click.echo(f"Version: {info['version']}")
    click.echo(f"Total entries: {info['num_entries']}")
    click.echo(f"Total size: {info['total_size_mb']:.2f} MB")

    if info['entries']:
        click.echo(f"\nCached embeddings:")
        for entry in sorted(info['entries']):
            click.echo(f"  - {entry}")


@cache_group.command('clear')
@click.option('--cache-dir', type=str, help='Cache directory path')
@click.option('--dataset', type=str, help='Clear only this dataset')
@click.option('--model', type=str, help='Clear only this model')
@click.option('--all', 'clear_all', is_flag=True, help='Clear all cache')
@click.confirmation_option(prompt='Are you sure you want to clear the cache?')
def clear_cache(cache_dir, dataset, model, clear_all):
    """Clear embedding cache."""
    cache = EmbeddingCache(cache_dir=cache_dir)

    if clear_all or (not dataset and not model):
        count = cache.clear()
        click.echo(f"✓ Cleared {count} cache entries")
    else:
        count = cache.clear(dataset=dataset, model=model)
        click.echo(f"✓ Cleared {count} cache entries")


@cache_group.command('info')
@click.option('--cache-dir', type=str, help='Cache directory path')
def cache_info(cache_dir):
    """Display detailed cache information."""
    cache = EmbeddingCache(cache_dir=cache_dir)
    info = cache.get_info()

    click.echo(f"Cache Information")
    click.echo(f"{'='*50}")
    click.echo(f"Directory: {info['cache_dir']}")
    click.echo(f"Version: {info['version']}")
    click.echo(f"Entries: {info['num_entries']}")
    click.echo(f"Size: {info['total_size_mb']:.2f} MB")
