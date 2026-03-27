from typing import List, Tuple, Union

import numpy as np

from biofuse.data import (
    create_custom_dataset,
    load_busi,
    load_custom_directory,
    load_imagenet,
    load_medmnist,
)


class DataAdapter:
    """Backward-compatible adapter over the v2 data loading utilities."""

    @classmethod
    def from_busi(cls, root: str, split: str, img_size: int):
        return load_busi(root=root, split=split, img_size=img_size)

    @classmethod
    def from_imagenet(
        cls,
        root: str,
        split: str,
        batch_size: int = 32,
        num_workers: int = 1,
        subset_size: float = 1.0,
        img_size: int = 224,
    ):
        return load_imagenet(
            root=root,
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            subset_size=subset_size,
            img_size=img_size,
        )

    @classmethod
    def from_medmnist(
        cls,
        dataset_name: str,
        split: str,
        img_size: int,
        root: str = '/data/medmnist',
    ):
        return load_medmnist(
            dataset_name=dataset_name,
            split=split,
            img_size=img_size,
            root=root,
        )

    @classmethod
    def from_custom(
        cls,
        images: Union[List[str], np.ndarray],
        labels: Union[List[int], np.ndarray],
        dataset_type: str = 'path',
        img_size: int = 224,
    ):
        return create_custom_dataset(
            images=images,
            labels=labels,
            img_size=img_size,
            from_paths=(dataset_type == 'path'),
        )

    @classmethod
    def from_directory(cls, directory_path, img_size=224) -> Tuple[List[str], List[int]]:
        image_paths, labels, _ = load_custom_directory(directory_path, img_size=img_size)
        return image_paths, labels
