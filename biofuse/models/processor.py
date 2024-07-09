from transformers import AutoImageProcessor, CLIPProcessor, AutoProcessor, AutoTokenizer
from open_clip import create_model_from_pretrained, get_tokenizer
from conch.open_clip_custom import create_model_from_pretrained as create_model_from_pretrained_conch
from torchvision import transforms
from biofuse.models.config import MODEL_MAP
import torch

class ModelPreprocessor:
    def __init__(self, model_name, model_info):
        self.model_name = model_name
        self.model_info = model_info
        self.processor = self._setup_preprocessor()

    def _setup_preprocessor(self):
        if self.model_name == "BioMedCLIP":
            _, preprocessor = create_model_from_pretrained(self.model_info["model"])            
        elif self.model_name == "BioMistral":
            preprocessor = AutoTokenizer.from_pretrained(self.model_info["model"])
        elif self.model_name == "CheXagent":
            preprocessor = AutoProcessor.from_pretrained(self.model_info["model"], trust_remote_code=True)
        elif self.model_name == "CONCH":
            _, preprocessor = create_model_from_pretrained_conch(self.model_info["model"], self.model_info["tokenizer"])
        elif self.model_name == "LLama-3-Aloe":
            preprocessor = AutoTokenizer.from_pretrained(self.model_info["model"])
        elif self.model_name == "Prov-GigaPath":
            preprocessor = self.model_info["tokenizer"]
        elif self.model_name == "PubMedCLIP":
            preprocessor = CLIPProcessor.from_pretrained(self.model_info["model"])
        elif self.model_name == "rad-dino":
            preprocessor = AutoImageProcessor.from_pretrained(self.model_info["model"])
        elif self.model_name == "UNI":
            preprocessor = self.model_info["tokenizer"]
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        return preprocessor

    def preprocess(self, image):       
        if self.model_name in ["BioMedCLIP", "CONCH", "Prov-GigaPath", "PubMedCLIP", "rad-dino", "UNI"]:
            if self.model_name in ["BioMedCLIP", "CONCH", "UNI"]:                
                preprocessed_image = self.processor(image.convert('RGB')).unsqueeze(0).to("cuda")
            elif self.model_name == "Prov-GigaPath":
                preprocessed_image = self.processor(image.convert('RGB')).unsqueeze(0).to("cuda")
            elif self.model_name in ["PubMedCLIP", "rad-dino"]:
                preprocessed_image = self.processor(images=image, return_tensors="pt").to("cuda")
            else:
                preprocessed_image = self.processor(image)
        elif self.model_name in ["BioMistral", "CheXagent", "LLama-3-Aloe"]:
            if self.model_name == "CheXagent":
                preprocessed_image = self.processor(images=image, return_tensors="pt").to("cuda", dtype=torch.float16)
                preprocessed_image['pixel_values'] = preprocessed_image['pixel_values'].squeeze(1).to("cuda", dtype=torch.float16)
            else:
                preprocessed_image = self.processor(image, return_tensors='pt').to("cuda")
        else:
            preprocessed_image = image
        
        return preprocessed_image

class MultiModelPreprocessor:
    def __init__(self, model_names):
        model_info = MODEL_MAP
        self.preprocessors = [ModelPreprocessor(name, model_info[name]) for name in model_names]

    def preprocess(self, image):
        preprocessed_images = [preprocessor.preprocess(image) for preprocessor in self.preprocessors]
        return preprocessed_images
