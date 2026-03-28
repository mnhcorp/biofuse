"""
Main CLI entry point for BioFuse.

Provides commands for training, evaluation, extraction, and cache management.
"""

import click
import sys
from pathlib import Path

from ..utils import setup_logger


@click.group()
@click.version_option(version='0.2.0', prog_name='biofuse')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """
    BioFuse - Multi-modal Fusion Framework for Biomedical Foundation Models.

    Train fusion models, extract embeddings, and evaluate performance
    on biomedical imaging tasks.

    Examples:

    \b
        # Train a model
        biofuse train --config config.yaml

    \b
        # Extract embeddings
        biofuse extract --models BioMedCLIP,CONCH --dataset pathmnist

    \b
        # Evaluate model
        biofuse evaluate --model model.pt --dataset pathmnist

    \b
        # Manage cache
        biofuse cache list
    """
    # Initialize context
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose

    # Setup logging
    import logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logger = setup_logger('biofuse', level=log_level, console=True)
    ctx.obj['logger'] = logger


@cli.command()
def info():
    """Display BioFuse installation and system information."""
    import torch
    import numpy as np
    from .. import __version__

    click.echo(f"BioFuse version: {__version__}")
    click.echo(f"Python version: {sys.version.split()[0]}")
    click.echo(f"PyTorch version: {torch.__version__}")
    click.echo(f"NumPy version: {np.__version__}")
    click.echo(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        click.echo(f"CUDA version: {torch.version.cuda}")
        click.echo(f"GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            click.echo(f"  GPU {i}: {torch.cuda.get_device_name(i)}")


def main():
    """Entry point for the CLI."""
    # Import commands
    from .train import train
    from .cache import cache_group
    from .matrix import matrix
    from .smoke import smoke

    # Register commands
    cli.add_command(train)
    cli.add_command(cache_group)
    cli.add_command(matrix)
    cli.add_command(smoke)

    cli(obj={})


if __name__ == '__main__':
    main()
