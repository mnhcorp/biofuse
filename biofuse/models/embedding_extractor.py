import os
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from PIL import Image
import timm
import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
)
from huggingface_hub import hf_hub_download

from biofuse.models.config import AUTH_TOKEN, CACHE_DIR, MODEL_MAP
from biofuse.utils.reproducibility import get_device


def _load_open_clip_backend():
    """Import open_clip lazily so the package can import without the backend."""
    try:
        from open_clip import create_model_from_pretrained, get_tokenizer
    except ImportError as exc:
        raise ImportError(
            "BioMedCLIP support requires the optional `open-clip-torch` package. "
            "Install it with `pip install open-clip-torch`."
        ) from exc

    return create_model_from_pretrained, get_tokenizer


def _load_conch_backend():
    """Import CONCH lazily because it is not part of the base install."""
    try:
        from conch.open_clip_custom import (
            create_model_from_pretrained as create_model_from_pretrained_conch,
        )
    except ImportError as exc:
        raise ImportError(
            "CONCH support requires the optional CONCH package to be installed."
        ) from exc

    return create_model_from_pretrained_conch


def _move_to_device(data: Any, device: torch.device, dtype: Optional[torch.dtype] = None):
    """Move nested model inputs onto the requested device."""
    if isinstance(data, dict):
        return {
            key: _move_to_device(
                value,
                device,
                dtype if key == "pixel_values" else None,
            )
            for key, value in data.items()
        }

    if hasattr(data, "to"):
        if dtype is not None:
            return data.to(device=device, dtype=dtype)
        return data.to(device)

    return data


def _hf_load_kwargs(**extra_kwargs):
    """Build a consistent Hugging Face load config for all model assets."""
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

    load_kwargs: Dict[str, Any] = {"cache_dir": CACHE_DIR}
    if AUTH_TOKEN:
        load_kwargs["token"] = AUTH_TOKEN
    load_kwargs.update(extra_kwargs)
    return load_kwargs


def _load_hf_asset(loader, model_id: str, retry_without_safetensors: bool = False, **kwargs):
    """Load a HF model/processor with explicit cache handling and a small CLIP fallback."""
    load_kwargs = _hf_load_kwargs(**kwargs)

    try:
        return loader(model_id, **load_kwargs)
    except OSError:
        if retry_without_safetensors and load_kwargs.get("use_safetensors"):
            fallback_kwargs = dict(load_kwargs)
            fallback_kwargs.pop("use_safetensors", None)
            return loader(model_id, **fallback_kwargs)
        raise


def _ensure_checkpoint(model_info: Dict[str, Any]) -> Path:
    """Reuse an existing checkpoint or download it to the standard BioFuse layout."""
    checkpoint_dir = Path(CACHE_DIR) / "ckpts" / model_info["checkpoint_subdir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / model_info["checkpoint_file"]

    if checkpoint_path.exists():
        return checkpoint_path

    hf_hub_download(
        repo_id=model_info["hf_model"],
        filename=model_info["checkpoint_file"],
        local_dir=checkpoint_dir,
        token=AUTH_TOKEN,
    )
    return checkpoint_path


def _load_checkpoint(path: Union[str, Path], map_location: Union[str, torch.device]):
    """Load a trusted state dict with the safest torch.load mode this runtime supports."""
    load_kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    return torch.load(path, **load_kwargs)


def _as_tensor_output(output: Any) -> torch.Tensor:
    """Normalize different backend return types into a tensor embedding."""
    if torch.is_tensor(output):
        return output

    if isinstance(output, (list, tuple)) and output:
        first = output[0]
        if torch.is_tensor(first):
            return first

    for attr in ("image_embeds", "pooler_output", "last_hidden_state", "logits"):
        value = getattr(output, attr, None)
        if torch.is_tensor(value):
            if attr == "last_hidden_state" and value.ndim >= 3:
                return value[:, 0, :]
            return value

    if isinstance(output, dict):
        for key in ("image_embeds", "pooler_output", "last_hidden_state", "logits"):
            value = output.get(key)
            if torch.is_tensor(value):
                if key == "last_hidden_state" and value.ndim >= 3:
                    return value[:, 0, :]
                return value

    raise TypeError(f"Unsupported model output type for embedding extraction: {type(output)}")


class PreTrainedEmbedding(nn.Module):
    def __init__(
        self,
        model_name: str,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.transform = None
        self.tokenizer = None
        self.device = (
            device
            if isinstance(device, torch.device)
            else get_device(device)
        )
        self.login_to_hf()
        self._load_model()
        self.model.eval()
        self._freeze_parameters()

    def _freeze_parameters(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def login_to_hf(self):
        """Configure HuggingFace cache and auth for non-interactive model loading."""
        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = CACHE_DIR

        if AUTH_TOKEN:
            os.environ["HF_TOKEN"] = AUTH_TOKEN

    def _load_model(self):
        model_info = MODEL_MAP.get(self.model_name)
        if model_info is None:
            raise ValueError(f"Unsupported model: {self.model_name}")

        if self.model_name == "BioMedCLIP":
            create_model_from_pretrained, get_tokenizer = _load_open_clip_backend()
            self.model, self.processor = create_model_from_pretrained(model_info["model"])
            self.tokenizer = get_tokenizer(model_info["model"])
        elif self.model_name == "CLIP":
            self.model = _load_hf_asset(
                CLIPModel.from_pretrained,
                model_info["model"],
                trust_remote_code=True,
                use_safetensors=True,
                retry_without_safetensors=True,
            )
            self.processor = _load_hf_asset(CLIPProcessor.from_pretrained, model_info["model"])
        elif self.model_name == "BioMistral":
            self.model = _load_hf_asset(AutoModel.from_pretrained, model_info["model"])
            self.processor = _load_hf_asset(AutoTokenizer.from_pretrained, model_info["model"])
        elif self.model_name == "CheXagent":
            self.model = _load_hf_asset(
                AutoModelForCausalLM.from_pretrained,
                model_info["model"],
                trust_remote_code=True,
            )
            self.processor = _load_hf_asset(
                AutoProcessor.from_pretrained,
                model_info["model"],
                trust_remote_code=True,
            )
        elif self.model_name == "CONCH":
            create_model_from_pretrained_conch = _load_conch_backend()
            self.model, self.processor = create_model_from_pretrained_conch(
                model_info["model"],
                model_info["tokenizer"],
            )
        elif self.model_name == "LLama-3-Aloe":
            self.model = _load_hf_asset(AutoModel.from_pretrained, model_info["model"])
            self.processor = _load_hf_asset(AutoTokenizer.from_pretrained, model_info["model"])
        elif self.model_name == "Prov-GigaPath":
            self.model = timm.create_model(
                model_info["model"],
                pretrained=True,
                img_size=224,
            )
            self.processor = model_info["tokenizer"]
        elif self.model_name == "PubMedCLIP":
            self.model = _load_hf_asset(
                CLIPModel.from_pretrained,
                model_info["model"],
                trust_remote_code=True,
                use_safetensors=True,
                retry_without_safetensors=True,
            )
            self.processor = _load_hf_asset(CLIPProcessor.from_pretrained, model_info["model"])
        elif self.model_name == "rad-dino":
            self.model = _load_hf_asset(AutoModel.from_pretrained, model_info["model"])
            self.processor = _load_hf_asset(AutoImageProcessor.from_pretrained, model_info["model"])
        elif self.model_name == "UNI":
            checkpoint_path = _ensure_checkpoint(model_info)
            self.model = timm.create_model(
                model_info["model"],
                pretrained=False,
                patch_size=16,
                init_values=1e-5,
                num_classes=0,
                dynamic_img_size=True,
            )
            self.model.load_state_dict(
                _load_checkpoint(checkpoint_path, map_location="cpu"),
                strict=True,
            )
            self.processor = model_info["tokenizer"]
        elif self.model_name == "UNI2":
            checkpoint_path = _ensure_checkpoint(model_info)
            timm_kwargs = dict(model_info["timm_kwargs"])
            model_name = timm_kwargs.pop("model_name", model_info["model"])
            self.model = timm.create_model(
                model_name,
                pretrained=False,
                **timm_kwargs,
            )
            self.model.load_state_dict(
                _load_checkpoint(checkpoint_path, map_location="cpu"),
                strict=True,
            )
            self.processor = model_info["tokenizer"]
        elif self.model_name == "Hibou-B":
            self.model = _load_hf_asset(
                AutoModel.from_pretrained,
                model_info["model"],
                trust_remote_code=True,
            )
            self.processor = _load_hf_asset(
                AutoImageProcessor.from_pretrained,
                model_info["model"],
                trust_remote_code=True,
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        self.model = self.model.to(self.device)

    def _tensor_batch_to_pil(self, tensor_batch: torch.Tensor) -> List[Image.Image]:
        """Convert raw image tensors into PIL images for model-specific processors."""
        tensor_batch = tensor_batch.detach().cpu()
        if tensor_batch.ndim == 3:
            tensor_batch = tensor_batch.unsqueeze(0)

        if tensor_batch.ndim != 4:
            raise ValueError(
                f"Expected a 3D or 4D image tensor, got shape {tuple(tensor_batch.shape)}"
            )

        pil_images = []
        for image in tensor_batch:
            if image.dtype.is_floating_point:
                image = image.clamp(0.0, 1.0)
            pil_images.append(to_pil_image(image))

        return pil_images

    def _to_pil_images(
        self,
        input_data: Union[Image.Image, torch.Tensor, Sequence[Union[Image.Image, torch.Tensor]]],
    ) -> List[Image.Image]:
        if isinstance(input_data, Image.Image):
            return [input_data]

        if torch.is_tensor(input_data):
            return self._tensor_batch_to_pil(input_data)

        if isinstance(input_data, (list, tuple)):
            pil_images: List[Image.Image] = []
            for item in input_data:
                if isinstance(item, Image.Image):
                    pil_images.append(item)
                elif torch.is_tensor(item):
                    pil_images.extend(self._tensor_batch_to_pil(item))
                else:
                    raise TypeError(f"Unsupported image input type: {type(item)}")
            return pil_images

        raise TypeError(f"Unsupported input type for {self.model_name}: {type(input_data)}")

    def _pixel_dtype(self) -> Optional[torch.dtype]:
        if self.model_name == "CheXagent" and self.device.type == "cuda":
            return torch.float16
        return None

    def prepare_inputs(self, input_data: Any):
        """Prepare raw tensors/PIL images or preprocessed dicts for the current backend."""
        if isinstance(input_data, dict):
            return _move_to_device(input_data, self.device, dtype=self._pixel_dtype())

        if hasattr(input_data, "items") and hasattr(input_data, "to"):
            return _move_to_device(
                dict(input_data.items()),
                self.device,
                dtype=self._pixel_dtype(),
            )

        if self.model_name in ["BioMistral", "LLama-3-Aloe"] and isinstance(input_data, (str, list, tuple)):
            texts = [input_data] if isinstance(input_data, str) else list(input_data)
            inputs = self.processor(texts, return_tensors="pt", padding=True, truncation=True)
            return _move_to_device(inputs, self.device)

        images = [
            image.convert("RGB") if image.mode != "RGB" else image
            for image in self._to_pil_images(input_data)
        ]

        if self.model_name in ["BioMedCLIP", "CONCH", "Prov-GigaPath", "UNI", "UNI2"]:
            processed = [self.processor(image) for image in images]
            return torch.stack(processed).to(self.device)

        if self.model_name in ["PubMedCLIP", "CLIP", "rad-dino", "Hibou-B"]:
            inputs = self.processor(images=images, return_tensors="pt")
            return _move_to_device(inputs, self.device)

        if self.model_name == "CheXagent":
            inputs = self.processor(images=images, return_tensors="pt")
            if "pixel_values" in inputs and inputs["pixel_values"].ndim == 5:
                inputs["pixel_values"] = inputs["pixel_values"].squeeze(1)
            return _move_to_device(inputs, self.device, dtype=self._pixel_dtype())

        return input_data

    def forward(self, input_data):
        prepared_input = self.prepare_inputs(input_data)

        with torch.no_grad():
            if self.model_name == "BioMedCLIP":
                outputs = self.model.encode_image(prepared_input)
            elif self.model_name == "CONCH":
                outputs = self.model.encode_image(
                    prepared_input,
                    proj_contrast=False,
                    normalize=False,
                )
            elif self.model_name in ["Prov-GigaPath", "UNI", "UNI2"]:
                outputs = self.model(prepared_input)
            elif self.model_name in ["PubMedCLIP", "CLIP"]:
                outputs = self.model.get_image_features(**prepared_input)
            elif self.model_name in ["rad-dino", "Hibou-B"]:
                model_output = self.model(**prepared_input)
                if hasattr(model_output, "pooler_output"):
                    outputs = model_output.pooler_output
                else:
                    outputs = model_output.last_hidden_state[:, 0, :]
            elif self.model_name == "CheXagent":
                outputs = self.model.vision_model(**prepared_input).last_hidden_state[:, 0, :]
            else:
                outputs = self.model(**prepared_input).last_hidden_state[:, 0, :]

        outputs = _as_tensor_output(outputs)

        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(0)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return outputs
