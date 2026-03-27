from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

from biofuse.cli.train import extract_embeddings


class DummyExtractor:
    instances = 0

    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device
        DummyExtractor.instances += 1

    def eval(self):
        return self

    def __call__(self, images):
        batch_size = images.shape[0]
        return torch.ones(batch_size, 4)


def test_extract_embeddings_reuses_single_model_instance_per_encoder(monkeypatch):
    DummyExtractor.instances = 0
    monkeypatch.setattr("biofuse.cli.train.PreTrainedEmbedding", DummyExtractor)

    dataset = TensorDataset(torch.rand(4, 3, 16, 16), torch.tensor([0, 1, 0, 1]))
    loader = DataLoader(dataset, batch_size=2)
    config = SimpleNamespace(
        data=SimpleNamespace(dataset="dummy", img_size=16),
        model=SimpleNamespace(models=["CLIP"]),
    )

    train_emb, train_labels, val_emb, val_labels, test_emb, test_labels = extract_embeddings(
        config,
        loader,
        loader,
        loader,
        device=torch.device("cpu"),
        cache=None,
    )

    assert DummyExtractor.instances == 1
    assert train_emb.shape == (4, 4)
    assert val_emb.shape == (4, 4)
    assert test_emb.shape == (4, 4)
    assert train_labels.tolist() == [0, 1, 0, 1]
    assert val_labels.tolist() == [0, 1, 0, 1]
    assert test_labels.tolist() == [0, 1, 0, 1]
