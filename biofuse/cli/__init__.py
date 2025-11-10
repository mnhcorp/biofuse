"""
Command-line interface for BioFuse.

Provides a modern CLI with commands for training, evaluation,
extraction, and cache management.
"""

from .main import main, cli

__all__ = ['main', 'cli']
