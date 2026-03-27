"""
Path management utilities for BioFuse.

Provides functions to manage data paths, output paths, and ensure
they are configurable via environment variables.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


class PathManager:
    """
    Manages paths for data, cache, models, and outputs.

    All paths are configurable via environment variables or constructor
    arguments, with sensible defaults.
    """

    def __init__(
        self,
        data_root: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        model_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize path manager.

        Args:
            data_root: Root directory for datasets
            cache_dir: Directory for embedding cache
            model_dir: Directory for saved models
            output_dir: Directory for outputs (results, logs, etc.)
        """
        # Data paths
        self.data_root = self._resolve_path(
            data_root,
            'BIOFUSE_DATA_ROOT',
            '/data'
        )

        # Cache paths
        self.cache_dir = self._resolve_path(
            cache_dir,
            'BIOFUSE_CACHE_DIR',
            '/data/biofuse-embedding-cache'
        )

        # Model paths
        self.model_dir = self._resolve_path(
            model_dir,
            'BIOFUSE_MODEL_DIR',
            './models'
        )

        # Output paths
        self.output_dir = self._resolve_path(
            output_dir,
            'BIOFUSE_OUTPUT_DIR',
            './results'
        )

        # Create directories if they don't exist
        self._ensure_directories()

    def _resolve_path(
        self,
        path: Optional[Union[str, Path]],
        env_var: str,
        default: str
    ) -> Path:
        """
        Resolve a path from argument, environment variable, or default.

        Priority: argument > environment variable > default

        Args:
            path: Path provided as argument
            env_var: Environment variable name
            default: Default path if neither argument nor env var provided

        Returns:
            Resolved Path object
        """
        if path is not None:
            return Path(path)

        env_path = os.environ.get(env_var)
        if env_path:
            return Path(env_path)

        return Path(default)

    def _ensure_directories(self):
        """Create directories if they don't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_dataset_path(self, dataset: str) -> Path:
        """
        Get path for a specific dataset.

        Args:
            dataset: Dataset name (e.g., 'medmnist', 'imagenet')

        Returns:
            Path to dataset directory
        """
        return self.data_root / dataset

    def get_model_path(self, model_name: str) -> Path:
        """
        Get path for a saved model.

        Args:
            model_name: Name of the model

        Returns:
            Path to model file
        """
        return self.model_dir / f"{model_name}.pt"

    def get_output_path(
        self,
        experiment_name: str,
        filename: Optional[str] = None,
        unique: bool = False,
        run_id: Optional[str] = None,
    ) -> Path:
        """
        Get path for experiment outputs.

        Args:
            experiment_name: Name of the experiment
            filename: Optional filename within experiment directory
            unique: Whether to create a unique timestamped run directory
            run_id: Optional explicit run identifier

        Returns:
            Path to output file or directory
        """
        if unique:
            timestamp = run_id or datetime.now().strftime('%Y%m%d-%H%M%S')
            base_dir = self.output_dir / f"{experiment_name}_{timestamp}"
            exp_dir = base_dir
            counter = 2
            while exp_dir.exists():
                exp_dir = self.output_dir / f"{base_dir.name}-{counter:02d}"
                counter += 1
            exp_dir.mkdir(parents=True, exist_ok=False)
        else:
            exp_dir = self.output_dir / experiment_name
            exp_dir.mkdir(parents=True, exist_ok=True)

        if filename:
            return exp_dir / filename
        return exp_dir

    def get_config_path(self) -> Path:
        """
        Get path for configuration files.

        Returns:
            Path to config directory
        """
        config_dir = Path.home() / '.biofuse'
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir

    def __repr__(self) -> str:
        return (
            f"PathManager(\n"
            f"  data_root={self.data_root},\n"
            f"  cache_dir={self.cache_dir},\n"
            f"  model_dir={self.model_dir},\n"
            f"  output_dir={self.output_dir}\n"
            f")"
        )


# Global path manager instance
_default_path_manager = None


def get_path_manager(**kwargs) -> PathManager:
    """
    Get or create the default path manager instance.

    Args:
        **kwargs: Arguments passed to PathManager constructor

    Returns:
        PathManager instance
    """
    global _default_path_manager

    if _default_path_manager is None or kwargs:
        _default_path_manager = PathManager(**kwargs)

    return _default_path_manager
