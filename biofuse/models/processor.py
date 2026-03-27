from typing import Any, List, Optional

from PIL import Image
import torch
from torchvision.transforms.functional import to_pil_image
from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer, CLIPProcessor

from biofuse.models.config import MODEL_MAP
from biofuse.utils.reproducibility import get_device


def _load_open_clip_backend():
    try:
        from open_clip import create_model_from_pretrained
    except ImportError as exc:
        raise ImportError(
            "BioMedCLIP preprocessing requires the optional `open-clip-torch` package. "
            "Install it with `pip install open-clip-torch`."
        ) from exc

    return create_model_from_pretrained


def _load_conch_backend():
    try:
        from conch.open_clip_custom import (
            create_model_from_pretrained as create_model_from_pretrained_conch,
        )
    except ImportError as exc:
        raise ImportError(
            "CONCH preprocessing requires the optional CONCH package to be installed."
        ) from exc

    return create_model_from_pretrained_conch


def _move_to_device(data: Any, device: torch.device, dtype: Optional[torch.dtype] = None):
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


class ModelPreprocessor:
    def __init__(self, model_name, model_info, device: Optional[str] = None):
        self.model_name = model_name
        self.model_info = model_info
        self.device = get_device(device)
        self.processor = self._setup_preprocessor()

    def _setup_preprocessor(self):
        if self.model_name == "BioMedCLIP":
            create_model_from_pretrained = _load_open_clip_backend()
            _, preprocessor = create_model_from_pretrained(self.model_info["model"])
        elif self.model_name == "CheXagent":
            preprocessor = AutoProcessor.from_pretrained(
                self.model_info["model"],
                trust_remote_code=True,
            )
        elif self.model_name == "CONCH":
            create_model_from_pretrained_conch = _load_conch_backend()
            _, preprocessor = create_model_from_pretrained_conch(
                self.model_info["model"],
                self.model_info["tokenizer"],
            )
        elif self.model_name in ["BioMistral", "LLama-3-Aloe"]:
            preprocessor = AutoTokenizer.from_pretrained(self.model_info["model"])
        elif self.model_name == "Prov-GigaPath":
            preprocessor = self.model_info["tokenizer"]
        elif self.model_name in ["PubMedCLIP", "CLIP"]:
            preprocessor = CLIPProcessor.from_pretrained(self.model_info["model"])
        elif self.model_name == "rad-dino":
            preprocessor = AutoImageProcessor.from_pretrained(self.model_info["model"])
        elif self.model_name in ["UNI", "UNI2"]:
            preprocessor = self.model_info["tokenizer"]
        elif self.model_name == "Hibou-B":
            preprocessor = AutoImageProcessor.from_pretrained(
                self.model_info["model"],
                trust_remote_code=True,
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return preprocessor

    def _pixel_dtype(self) -> Optional[torch.dtype]:
        if self.model_name == "CheXagent" and self.device.type == "cuda":
            return torch.float16
        return None

    def _to_pil_images(self, images) -> List[Image.Image]:
        if isinstance(images, Image.Image):
            return [images]

        if torch.is_tensor(images):
            if images.ndim == 3:
                images = images.unsqueeze(0)
            return [to_pil_image(image.detach().cpu().clamp(0.0, 1.0)) for image in images]

        if isinstance(images, (list, tuple)):
            pil_images = []
            for image in images:
                if isinstance(image, Image.Image):
                    pil_images.append(image)
                elif torch.is_tensor(image):
                    pil_images.extend(self._to_pil_images(image))
                else:
                    raise TypeError(f"Unsupported input type: {type(image)}")
            return pil_images

        raise TypeError(f"Unsupported input type: {type(images)}")

    def preprocess(self, image):
        if isinstance(image, (str, list, tuple)) and self.model_name in ["BioMistral", "LLama-3-Aloe"]:
            texts = [image] if isinstance(image, str) else list(image)
            tokenized = self.processor(texts, return_tensors="pt", padding=True, truncation=True)
            return _move_to_device(tokenized, self.device)

        images = [
            item.convert("RGB") if item.mode != "RGB" else item
            for item in self._to_pil_images(image)
        ]

        if self.model_name in ["BioMedCLIP", "CONCH", "Prov-GigaPath", "UNI", "UNI2"]:
            batch = torch.stack([self.processor(item) for item in images])
            return batch.to(self.device)

        processed = self.processor(images=images, return_tensors="pt")
        if self.model_name == "CheXagent" and "pixel_values" in processed and processed["pixel_values"].ndim == 5:
            processed["pixel_values"] = processed["pixel_values"].squeeze(1)
        return _move_to_device(processed, self.device, dtype=self._pixel_dtype())

    def preprocess_tensor(self, tensor_batch):
        return self.preprocess(tensor_batch)


class MultiModelPreprocessor:
    def __init__(self, model_names, device: Optional[str] = None):
        model_info = MODEL_MAP
        self.preprocessors = [
            ModelPreprocessor(name, model_info[name], device=device)
            for name in model_names
        ]

    def preprocess(self, images):
        return [preprocessor.preprocess(images) for preprocessor in self.preprocessors]

    def preprocess_tensor_batch(self, tensor_batch):
        return [preprocessor.preprocess_tensor(tensor_batch) for preprocessor in self.preprocessors]

    def preprocess_tensor(self, tensor_batch):
        return self.preprocessors[0].preprocess_tensor(tensor_batch)
