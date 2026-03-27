import os
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

import biofuse.models.embedding_extractor as embedding_extractor
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


def test_login_is_skipped_without_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with patch("biofuse.models.embedding_extractor.AUTH_TOKEN", None):
        instance = PreTrainedEmbedding.__new__(PreTrainedEmbedding)
        instance.login_to_hf()

    assert "HF_TOKEN" not in os.environ


def test_login_to_hf_only_sets_environment_for_existing_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with patch("biofuse.models.embedding_extractor.AUTH_TOKEN", "hf_test"), \
         patch("biofuse.models.embedding_extractor.CACHE_DIR", "/tmp/biofuse-hf-cache"):
        instance = PreTrainedEmbedding.__new__(PreTrainedEmbedding)
        instance.login_to_hf()

    assert os.environ["HF_TOKEN"] == "hf_test"


def test_load_checkpoint_prefers_weights_only_when_supported():
    captured = {}

    def fake_torch_load(path, map_location=None, weights_only=False):
        captured["path"] = path
        captured["map_location"] = map_location
        captured["weights_only"] = weights_only
        return {}

    with patch("biofuse.models.embedding_extractor.torch.load", new=fake_torch_load):
        embedding_extractor._load_checkpoint("/tmp/checkpoint.bin", map_location="cpu")

    assert captured["path"] == "/tmp/checkpoint.bin"
    assert captured["map_location"] == "cpu"
    assert captured["weights_only"] is True


def test_uni_loads_from_hf_hub_via_timm_without_hardcoded_checkpoint(tmp_path):
    create_calls = []
    download_calls = []

    def fake_create_model(model_name, **kwargs):
        create_calls.append((model_name, kwargs))
        return DummyTimmModel()

    def fake_hf_download(**kwargs):
        download_calls.append(kwargs)
        local_dir = kwargs["local_dir"]
        (local_dir / "pytorch_model.bin").write_bytes(b"checkpoint")
        return str(local_dir / "pytorch_model.bin")

    with patch.object(PreTrainedEmbedding, "login_to_hf", return_value=None), \
         patch("biofuse.models.embedding_extractor.timm.create_model", new=fake_create_model), \
         patch("biofuse.models.embedding_extractor.hf_hub_download", side_effect=fake_hf_download), \
         patch("biofuse.models.embedding_extractor.torch.load", return_value={}) as torch_load_mock, \
         patch.object(DummyTimmModel, "load_state_dict", return_value=None), \
         patch("biofuse.models.embedding_extractor.CACHE_DIR", str(tmp_path)):
        extractor = PreTrainedEmbedding("UNI", device="cpu")

    assert create_calls[0][0] == "vit_large_patch16_224"
    assert create_calls[0][1]["pretrained"] is False
    assert download_calls[0]["repo_id"] == "MahmoodLab/UNI"
    assert str(download_calls[0]["local_dir"]).endswith("ckpts/vit_large_patch16_224.dinov2.uni_mass100k")
    assert extractor.processor is not None


def test_uni2_loads_from_hf_hub_via_timm_without_hardcoded_checkpoint(tmp_path):
    create_calls = []
    download_calls = []

    def fake_create_model(model_name, **kwargs):
        create_calls.append((model_name, kwargs))
        return DummyTimmModel(embedding_dim=1536)

    def fake_hf_download(**kwargs):
        download_calls.append(kwargs)
        local_dir = kwargs["local_dir"]
        (local_dir / "pytorch_model.bin").write_bytes(b"checkpoint")
        return str(local_dir / "pytorch_model.bin")

    with patch.object(PreTrainedEmbedding, "login_to_hf", return_value=None), \
         patch("biofuse.models.embedding_extractor.timm.create_model", new=fake_create_model), \
         patch("biofuse.models.embedding_extractor.hf_hub_download", side_effect=fake_hf_download), \
         patch("biofuse.models.embedding_extractor.torch.load", return_value={}) as torch_load_mock, \
         patch.object(DummyTimmModel, "load_state_dict", return_value=None), \
         patch("biofuse.models.embedding_extractor.CACHE_DIR", str(tmp_path)):
        extractor = PreTrainedEmbedding("UNI2", device="cpu")

    assert create_calls[0][0] == "vit_giant_patch14_224"
    assert create_calls[0][1]["pretrained"] is False
    assert download_calls[0]["repo_id"] == "MahmoodLab/UNI2-h"
    assert str(download_calls[0]["local_dir"]).endswith("ckpts/uni2-h")
    assert extractor.processor is not None


def test_uni_reuses_existing_checkpoint_without_redownloading(tmp_path):
    checkpoint_dir = tmp_path / "ckpts" / "vit_large_patch16_224.dinov2.uni_mass100k"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "pytorch_model.bin"
    checkpoint_path.write_bytes(b"checkpoint")

    with patch.object(PreTrainedEmbedding, "login_to_hf", return_value=None), \
         patch("biofuse.models.embedding_extractor.CACHE_DIR", str(tmp_path)), \
         patch("biofuse.models.embedding_extractor.hf_hub_download") as download_mock, \
         patch("biofuse.models.embedding_extractor.timm.create_model", return_value=DummyTimmModel()), \
         patch("biofuse.models.embedding_extractor.torch.load", return_value={}), \
         patch.object(DummyTimmModel, "load_state_dict", return_value=None):
        PreTrainedEmbedding("UNI", device="cpu")

    download_mock.assert_not_called()
    assert checkpoint_path.exists()
