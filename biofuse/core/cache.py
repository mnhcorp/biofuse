"""
Embedding cache management for BioFuse.

This module provides efficient caching of pre-computed embeddings to speed up
experiments and avoid redundant computation.
"""

import os
import pickle
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from datetime import datetime
import json


class EmbeddingCache:
    """
    Manages caching of embeddings to disk for faster experimentation.

    The cache stores pre-computed embeddings indexed by dataset, model,
    image size, and split. Supports versioning and cache invalidation.

    Attributes:
        cache_dir: Root directory for cache storage
        version: Cache format version for invalidation
    """

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None, version: str = "v1"):
        """
        Initialize the embedding cache.

        Args:
            cache_dir: Directory to store cached embeddings.
                      Defaults to /data/biofuse-embedding-cache or
                      $BIOFUSE_CACHE_DIR environment variable.
            version: Cache version string for invalidation when format changes
        """
        if cache_dir is None:
            cache_dir = os.environ.get(
                'BIOFUSE_CACHE_DIR',
                '/data/biofuse-embedding-cache'
            )

        self.cache_dir = Path(cache_dir)
        self.version = version
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata file to track cache info
        self.metadata_file = self.cache_dir / 'metadata.json'
        if not self.metadata_file.exists():
            self._init_metadata()

    def _init_metadata(self):
        """Initialize cache metadata file."""
        metadata = {
            'version': self.version,
            'created_at': datetime.now().isoformat(),
            'cache_count': 0
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _get_cache_key(
        self,
        dataset: str,
        model: str,
        img_size: int,
        split: str
    ) -> str:
        """
        Generate a unique cache key for the given parameters.

        Args:
            dataset: Dataset name (e.g., 'chestmnist')
            model: Model name (e.g., 'BioMedCLIP')
            img_size: Image size used for preprocessing
            split: Data split ('train', 'val', 'test')

        Returns:
            Cache key string
        """
        # Create a hash to handle long model names and special characters
        key_str = f"{dataset}_{model}_{img_size}_{split}_{self.version}"
        return key_str

    def _get_cache_path(
        self,
        dataset: str,
        model: str,
        img_size: int,
        split: str
    ) -> Path:
        """
        Get the file path for cached embeddings.

        Args:
            dataset: Dataset name
            model: Model name
            img_size: Image size
            split: Data split

        Returns:
            Path to cache file
        """
        cache_key = self._get_cache_key(dataset, model, img_size, split)
        return self.cache_dir / f"{cache_key}.pkl"

    def save(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        dataset: str,
        model: str,
        img_size: int,
        split: str
    ) -> None:
        """
        Save embeddings to cache.

        Args:
            embeddings: Embedding array to cache
            labels: Corresponding labels
            dataset: Dataset name
            model: Model name
            img_size: Image size
            split: Data split
        """
        cache_path = self._get_cache_path(dataset, model, img_size, split)

        # Save with metadata
        cache_data = {
            'embeddings': embeddings,
            'labels': labels,
            'metadata': {
                'dataset': dataset,
                'model': model,
                'img_size': img_size,
                'split': split,
                'version': self.version,
                'cached_at': datetime.now().isoformat(),
                'shape': embeddings.shape,
                'dtype': str(embeddings.dtype)
            }
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Update metadata
        self._update_metadata()

    def load(
        self,
        dataset: str,
        model: str,
        img_size: int,
        split: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load embeddings from cache if available.

        Args:
            dataset: Dataset name
            model: Model name
            img_size: Image size
            split: Data split

        Returns:
            Tuple of (embeddings, labels) if cached, None otherwise
        """
        cache_path = self._get_cache_path(dataset, model, img_size, split)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # Validate cache version
            cached_version = cache_data.get('metadata', {}).get('version')
            if cached_version != self.version:
                print(f"Warning: Cache version mismatch. "
                      f"Expected {self.version}, got {cached_version}. "
                      f"Cache will be regenerated.")
                return None

            return cache_data['embeddings'], cache_data['labels']

        except (pickle.UnpicklingError, EOFError, KeyError) as e:
            print(f"Warning: Failed to load cache from {cache_path}: {e}")
            print("Cache will be regenerated.")
            return None

    def exists(
        self,
        dataset: str,
        model: str,
        img_size: int,
        split: str
    ) -> bool:
        """
        Check if embeddings are cached.

        Args:
            dataset: Dataset name
            model: Model name
            img_size: Image size
            split: Data split

        Returns:
            True if cached, False otherwise
        """
        cache_path = self._get_cache_path(dataset, model, img_size, split)
        return cache_path.exists()

    def clear(
        self,
        dataset: Optional[str] = None,
        model: Optional[str] = None
    ) -> int:
        """
        Clear cache entries.

        Args:
            dataset: If provided, only clear cache for this dataset
            model: If provided, only clear cache for this model

        Returns:
            Number of cache entries deleted
        """
        count = 0

        for cache_file in self.cache_dir.glob('*.pkl'):
            if dataset is None and model is None:
                # Clear all
                cache_file.unlink()
                count += 1
            else:
                # Check if file matches criteria
                parts = cache_file.stem.split('_')
                if len(parts) >= 2:
                    file_dataset = parts[0]
                    file_model = parts[1]

                    should_delete = True
                    if dataset and file_dataset != dataset:
                        should_delete = False
                    if model and file_model != model:
                        should_delete = False

                    if should_delete:
                        cache_file.unlink()
                        count += 1

        self._update_metadata()
        return count

    def get_info(self) -> dict:
        """
        Get cache information and statistics.

        Returns:
            Dictionary with cache statistics
        """
        cache_files = list(self.cache_dir.glob('*.pkl'))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            'cache_dir': str(self.cache_dir),
            'version': self.version,
            'num_entries': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'entries': [f.stem for f in cache_files]
        }

    def _update_metadata(self):
        """Update cache metadata file."""
        cache_files = list(self.cache_dir.glob('*.pkl'))

        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        metadata.update({
            'cache_count': len(cache_files),
            'last_updated': datetime.now().isoformat(),
            'version': self.version
        })

        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)


# Global cache instance (can be configured)
_default_cache = None


def get_cache(cache_dir: Optional[str] = None, version: str = "v1") -> EmbeddingCache:
    """
    Get or create the default cache instance.

    Args:
        cache_dir: Cache directory (uses default if None)
        version: Cache version

    Returns:
        EmbeddingCache instance
    """
    global _default_cache

    if _default_cache is None or cache_dir is not None:
        _default_cache = EmbeddingCache(cache_dir, version)

    return _default_cache
