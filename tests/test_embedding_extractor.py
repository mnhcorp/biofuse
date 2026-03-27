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


class DummyTimmModel(nn.Module):
    def __init__(self, embedding_dim=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.anchor = nn.Parameter(torch.zeros(1))
        self.pretrained_cfg = {"input_size": (3, 224, 224)}

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        return torch.ones(batch_size, self.embedding_dim)


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


def test_clip_loader_uses_explicit_cache_dir_and_retries_without_safetensors():
    load_calls = []

    def fake_from_pretrained(model_id, **kwargs):
        load_calls.append((model_id, kwargs))
        if len(load_calls) == 1:
            raise OSError("missing safetensors snapshot")
        return DummyClipModel()

    with patch.object(PreTrainedEmbedding, "login_to_hf", return_value=None), \
         patch("biofuse.models.embedding_extractor.CACHE_DIR", "/tmp/biofuse-hf-cache"), \
         patch("biofuse.models.embedding_extractor.AUTH_TOKEN", None), \
         patch("biofuse.models.embedding_extractor.CLIPModel.from_pretrained", side_effect=fake_from_pretrained), \
         patch("biofuse.models.embedding_extractor.CLIPProcessor.from_pretrained", return_value=DummyProcessor()):
        PreTrainedEmbedding("CLIP", device="cpu")

    assert load_calls[0][0] == "openai/clip-vit-base-patch32"
    assert load_calls[0][1]["cache_dir"] == "/tmp/biofuse-hf-cache"
    assert load_calls[0][1]["use_safetensors"] is True
    assert load_calls[1][1]["cache_dir"] == "/tmp/biofuse-hf-cache"
    assert "use_safetensors" not in load_calls[1][1]


def test_login_is_skipped_without_hf_token():
    with patch("biofuse.models.embedding_extractor.AUTH_TOKEN", None), \
         patch("biofuse.models.embedding_extractor.login") as login_mock:
        instance = PreTrainedEmbedding.__new__(PreTrainedEmbedding)
        instance.login_to_hf()

    login_mock.assert_not_called()


def test_uni_loads_from_hf_hub_via_timm_without_hardcoded_checkpoint():
    create_calls = []

    def fake_create_model(model_name, cache_dir=None, **kwargs):
        create_calls.append((model_name, kwargs, cache_dir))
        return DummyTimmModel()

    with patch.object(PreTrainedEmbedding, "login_to_hf", return_value=None), \
         patch("biofuse.models.embedding_extractor.timm.create_model", new=fake_create_model), \
         patch("biofuse.models.embedding_extractor.resolve_data_config", return_value={}), \
         patch("biofuse.models.embedding_extractor.create_transform", return_value=DummyProcessor()), \
         patch("biofuse.models.embedding_extractor.CACHE_DIR", "/tmp/biofuse-hf-cache"):
        extractor = PreTrainedEmbedding("UNI", device="cpu")

    assert create_calls[0][0] == "hf-hub:MahmoodLab/UNI"
    assert create_calls[0][1]["pretrained"] is True
    assert create_calls[0][2] == "/tmp/biofuse-hf-cache"
    assert extractor.processor is not None


def test_uni2_loads_from_hf_hub_via_timm_without_hardcoded_checkpoint():
    create_calls = []

    def fake_create_model(model_name, cache_dir=None, **kwargs):
        create_calls.append((model_name, kwargs, cache_dir))
        return DummyTimmModel(embedding_dim=1536)

    with patch.object(PreTrainedEmbedding, "login_to_hf", return_value=None), \
         patch("biofuse.models.embedding_extractor.timm.create_model", new=fake_create_model), \
         patch("biofuse.models.embedding_extractor.resolve_data_config", return_value={}), \
         patch("biofuse.models.embedding_extractor.create_transform", return_value=DummyProcessor()), \
         patch("biofuse.models.embedding_extractor.CACHE_DIR", "/tmp/biofuse-hf-cache"):
        extractor = PreTrainedEmbedding("UNI2", device="cpu")

    assert create_calls[0][0] == "hf-hub:MahmoodLab/UNI2-h"
    assert create_calls[0][1]["pretrained"] is True
    assert create_calls[0][2] == "/tmp/biofuse-hf-cache"
    assert extractor.processor is not None
