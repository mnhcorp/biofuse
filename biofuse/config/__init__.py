"""
Configuration management for BioFuse.

Provides config classes, validation, and YAML/JSON loading.
"""

from .config import (
    ExperimentConfig,
    DataConfig,
    ModelConfig,
    ClassifierConfig,
    CacheConfig,
    FusionMethod,
    ClassifierType,
    create_default_config,
    merge_configs
)

__all__ = [
    'ExperimentConfig',
    'DataConfig',
    'ModelConfig',
    'ClassifierConfig',
    'CacheConfig',
    'FusionMethod',
    'ClassifierType',
    'create_default_config',
    'merge_configs',
]
