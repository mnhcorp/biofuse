import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

import biofuse.biofuse as biofuse_module


class DummyExtractor(nn.Module):
    DIMS = {
        "BioMedCLIP": 512,
        "rad-dino": 768,
    }

    def __init__(self, model_name, device=None):
        super().__init__()
        self.model_name = model_name
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, images):
        batch_size = images.shape[0]
        return torch.ones(batch_size, self.DIMS[self.model_name])


def test_public_api_uses_shared_fusion_path(monkeypatch):
    monkeypatch.setattr(biofuse_module, "PreTrainedEmbedding", DummyExtractor)

    dataset = TensorDataset(
        torch.rand(4, 3, 16, 16),
        torch.tensor([0, 1, 0, 1]),
    )

    biofuse = biofuse_module.BioFuse(
        models=["BioMedCLIP", "rad-dino"],
        fusion_method="concat",
        projection_dim=0,
        device="cpu",
    )

    embeddings, labels, _, _, model = biofuse.generate_embeddings(
        train_data=dataset,
        dataset_type=biofuse_module.BioFuse.CUSTOM,
        batch_size=2,
        num_workers=0,
    )

    assert embeddings.shape == (4, 1280)
    assert sorted(labels.tolist()) == [0, 0, 1, 1]
    assert model.models == ["BioMedCLIP", "rad-dino"]
