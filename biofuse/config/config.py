"""
Configuration management for BioFuse.

Provides configuration classes and validation for experiments.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import json
import yaml
from enum import Enum


class FusionMethod(str, Enum):
    """Supported fusion methods."""
    CONCAT = 'concat'
    MEAN = 'mean'
    MAX = 'max'
    SUM = 'sum'
    WSUM = 'wsum'
    WMEAN = 'wmean'
    SELF_ATTENTION = 'self_attention'


class ClassifierType(str, Enum):
    """Supported classifier types."""
    LOGISTIC = 'logistic'
    LOGREG = 'logreg'
    XGBOOST = 'xgboost'
    XGB = 'xgb'
    CATBOOST = 'catboost'
    CAT = 'cat'
    NN_MLP = 'nn_mlp'
    NN_CNN = 'nn_cnn'
    NN_RESNET = 'nn_resnet'


@dataclass
class DataConfig:
    """Configuration for data loading."""
    dataset: str
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    data_root: Optional[str] = None
    download: bool = True
    subset_size: float = 1.0
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_test_samples: Optional[int] = None


@dataclass
class ModelConfig:
    """Configuration for model selection and fusion."""
    models: List[str] = field(default_factory=lambda: ['BioMedCLIP'])
    fusion_method: str = 'concat'
    projection_dim: int = 0

    def __post_init__(self):
        """Validate fusion method."""
        if self.fusion_method not in [m.value for m in FusionMethod]:
            raise ValueError(f"Invalid fusion method: {self.fusion_method}")


@dataclass
class ClassifierConfig:
    """Configuration for classifier training."""
    type: str = 'xgboost'

    # XGBoost/CatBoost params
    n_estimators: int = 250
    learning_rate: float = 0.1
    max_depth: int = 6

    # Neural network params
    hidden_dim: int = 256
    dropout: float = 0.3
    num_epochs: int = 100

    # General params
    random_state: int = 42

    def __post_init__(self):
        """Validate classifier type."""
        if self.type not in [c.value for c in ClassifierType]:
            raise ValueError(f"Invalid classifier type: {self.type}")


@dataclass
class CacheConfig:
    """Configuration for embedding cache."""
    cache_dir: Optional[str] = None
    use_cache: bool = True
    cache_version: str = 'v1'


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str
    data: DataConfig
    model: ModelConfig
    classifier: ClassifierConfig
    cache: CacheConfig = field(default_factory=CacheConfig)

    # Training options
    seed: int = 42
    device: Optional[str] = None

    # Paths
    output_dir: str = './results'
    log_dir: Optional[str] = None

    # Evaluation
    test_split: bool = True
    cross_validation: bool = False
    cv_folds: int = 5

    # Robustness
    test_robustness: bool = False
    medmnistc_root: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def save(self, path: Union[str, Path], format: str = 'yaml'):
        """
        Save configuration to file.

        Args:
            path: Path to save config
            format: File format ('yaml' or 'json')
        """
        path = Path(path)
        config_dict = self.to_dict()

        with open(path, 'w') as f:
            if format == 'yaml':
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            elif format == 'json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """
        Create config from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            ExperimentConfig instance
        """
        # Extract nested configs
        data_config = DataConfig(**config_dict.pop('data', {}))
        model_config = ModelConfig(**config_dict.pop('model', {}))
        classifier_config = ClassifierConfig(**config_dict.pop('classifier', {}))
        cache_config = CacheConfig(**config_dict.pop('cache', {}))

        return cls(
            data=data_config,
            model=model_config,
            classifier=classifier_config,
            cache=cache_config,
            **config_dict
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ExperimentConfig':
        """
        Load configuration from file.

        Args:
            path: Path to config file (.yaml or .json)

        Returns:
            ExperimentConfig instance
        """
        path = Path(path)

        with open(path, 'r') as f:
            if path.suffix in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif path.suffix == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

        return cls.from_dict(config_dict)


def create_default_config(
    experiment_name: str,
    dataset: str,
    models: Optional[List[str]] = None
) -> ExperimentConfig:
    """
    Create a default experiment configuration.

    Args:
        experiment_name: Name of the experiment
        dataset: Dataset name (e.g., 'pathmnist')
        models: List of model names (default: ['BioMedCLIP'])

    Returns:
        ExperimentConfig with sensible defaults

    Example:
        >>> config = create_default_config('my_exp', 'pathmnist')
        >>> config.save('config.yaml')
    """
    if models is None:
        models = ['BioMedCLIP']

    return ExperimentConfig(
        name=experiment_name,
        data=DataConfig(dataset=dataset),
        model=ModelConfig(models=models),
        classifier=ClassifierConfig()
    )


def merge_configs(base: ExperimentConfig, overrides: Dict[str, Any]) -> ExperimentConfig:
    """
    Merge configuration with overrides.

    Args:
        base: Base configuration
        overrides: Dictionary of overrides (can be nested)

    Returns:
        New ExperimentConfig with merged values

    Example:
        >>> config = create_default_config('test', 'pathmnist')
        >>> overrides = {'data': {'batch_size': 64}, 'seed': 123}
        >>> new_config = merge_configs(config, overrides)
    """
    config_dict = base.to_dict()

    # Deep merge
    def deep_merge(d1: dict, d2: dict) -> dict:
        result = d1.copy()
        for key, value in d2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    merged_dict = deep_merge(config_dict, overrides)
    return ExperimentConfig.from_dict(merged_dict)
