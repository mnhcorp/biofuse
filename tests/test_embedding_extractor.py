from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from biofuse.models.embedding_extractor import PreTrainedEmbedding


class DummyVisionModel(nn.Module):
    def __init__(self, embedding_dim=768):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        return SimpleNamespace(
            pooler_output=torch.ones(batch_size, self.embedding_dim),
        )


class DummyProcessor:
    def __call__(self, images, return_tensors="pt"):
        batch_size = len(images)
        return {
            "pixel_values": torch.ones(batch_size, 3, 16, 16),
        }


class DummyClipModel(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.anchor = nn.Parameter(torch.zeros(1))

    def get_image_features(self, pixel_values):
        batch_size = pixel_values.shape[0]
        return SimpleNamespace(
            pooler_output=torch.ones(batch_size, self.embedding_dim),
        )


def test_rad_dino_accepts_raw_tensor_batches_on_cpu():
    with patch.object(PreTrainedEmbedding, "login_to_hf", return_value=None), \
         patch("biofuse.models.embedding_extractor.AutoModel.from_pretrained", return_value=DummyVisionModel()), \
         patch("biofuse.models.embedding_extractor.AutoImageProcessor.from_pretrained", return_value=DummyProcessor()):
        extractor = PreTrainedEmbedding("rad-dino", device="cpu")
        output = extractor(torch.rand(2, 3, 32, 32))

    assert output.shape == (2, 768)
    assert next(extractor.model.parameters()).device.type == "cpu"


def test_clip_structured_output_is_normalized_to_tensor():
    with patch.object(PreTrainedEmbedding, "login_to_hf", return_value=None), \
         patch("biofuse.models.embedding_extractor.CLIPModel.from_pretrained", return_value=DummyClipModel()), \
         patch("biofuse.models.embedding_extractor.CLIPProcessor.from_pretrained", return_value=DummyProcessor()):
        extractor = PreTrainedEmbedding("CLIP", device="cpu")
        output = extractor(torch.rand(2, 3, 32, 32))

    assert output.shape == (2, 512)


def test_login_is_skipped_without_hf_token():
    with patch("biofuse.models.embedding_extractor.AUTH_TOKEN", None), \
         patch("biofuse.models.embedding_extractor.login") as login_mock:
        instance = PreTrainedEmbedding.__new__(PreTrainedEmbedding)
        instance.login_to_hf()

    login_mock.assert_not_called()
