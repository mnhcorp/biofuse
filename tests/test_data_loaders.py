import numpy as np
import torch

from biofuse.data import loaders


class DummyMedMNIST:
    def __init__(self, split, download, transform, root, size):
        self.imgs = np.random.randint(0, 255, size=(4, 28, 28), dtype=np.uint8)
        self.labels = np.array([[0], [1], [0], [1]], dtype=np.int64)


def test_load_medmnist_returns_collatable_rgb_tensors(monkeypatch):
    dummy_info = {
        "dummy": {
            "label": {"0": "negative", "1": "positive"},
            "python_class": "DummyMedMNIST",
        },
    }
    dummy_medmnist = type("DummyBackend", (), {"DummyMedMNIST": DummyMedMNIST})
    monkeypatch.setattr(
        loaders,
        "_load_medmnist_backend",
        lambda: (dummy_medmnist, dummy_info),
    )

    dataset, num_classes = loaders.load_medmnist(
        "dummy",
        split="train",
        img_size=32,
        root="/tmp",
        download=False,
    )
    image, label = dataset[0]

    assert num_classes == 2
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 32, 32)
    assert label in [0, 1]
