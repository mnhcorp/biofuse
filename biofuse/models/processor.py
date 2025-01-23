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
        elif self.model_name == "CLIP":
            preprocessor = CLIPProcessor.from_pretrained(self.model_info["model"])
        elif self.model_name == "rad-dino":
            preprocessor = AutoImageProcessor.from_pretrained(self.model_info["model"])
        elif self.model_name in ["UNI", "UNI2"]:
            preprocessor = self.model_info["tokenizer"]
        elif self.model_name == "Hibou-B":
            preprocessor = AutoImageProcessor.from_pretrained(self.model_info["model"], trust_remote_code=True)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        return preprocessor

    def preprocess(self, image):       
        if self.model_name in ["BioMedCLIP", "CONCH", "Prov-GigaPath", "PubMedCLIP", "rad-dino", "UNI", "Hibou-B", "CLIP", "UNI2"]:
            if self.model_name in ["BioMedCLIP", "CONCH", "UNI", "UNI2"]:
                if isinstance(image, torch.Tensor):
                    preprocessed_image = image.unsqueeze(0).to("cuda")
                else:                
                    preprocessed_image = self.processor(image.convert('RGB')).unsqueeze(0).to("cuda")
            elif self.model_name == "Prov-GigaPath":
                if isinstance(image, torch.Tensor):
                    preprocessed_image = image.unsqueeze(0).to("cuda")
                else:
                    preprocessed_image = self.processor(image.convert('RGB')).unsqueeze(0).to("cuda")
            elif self.model_name in ["PubMedCLIP", "rad-dino", "Hibou-B", "CLIP"]:               
                if isinstance(image, torch.Tensor):
                    preprocessed_image = {"pixel_values": image.unsqueeze(0).to("cuda")}
                else:
                    if self.model_name == "Hibou-B":
                        image = image.convert('RGB')
                    preprocessed_image = self.processor(images=image, return_tensors="pt").to("cuda")
            else:
                preprocessed_image = image
        elif self.model_name in ["BioMistral", "CheXagent", "LLama-3-Aloe"]:
            if isinstance(image, torch.Tensor):
                if self.model_name == "CheXagent":
                    preprocessed_image = {"pixel_values": image.unsqueeze(0).to("cuda", dtype=torch.float16)}
                else:
                    preprocessed_image = {"pixel_values": image.unsqueeze(0).to("cuda")}
            else:
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
