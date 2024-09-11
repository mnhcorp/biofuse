import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoProcessor, CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModelForCausalLM
from open_clip import create_model_from_pretrained, get_tokenizer
from torchvision import transforms
import timm
import os
from huggingface_hub import login
from biofuse.models.config import AUTH_TOKEN, CACHE_DIR, MODEL_MAP
from conch.open_clip_custom import create_model_from_pretrained as create_model_from_pretrained_conch
import ipdb

class PreTrainedEmbedding(nn.Module):
    def __init__(self, model_name):
        super(PreTrainedEmbedding, self).__init__()
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.transform = None
        self.tokenizer = None
        self.login_to_hf()
        self._load_model()
        self._freeze_parameters()

    def _freeze_parameters(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def login_to_hf(self):
        # set HF_TOKEN environment variable
        os.environ["HF_TOKEN"] = AUTH_TOKEN
        os.environ["HF_HOME"] = "/data/hf-hub/"
        #login()

    def _load_model(self):
        model_info = MODEL_MAP.get(self.model_name)

        if self.model_name == "BioMedCLIP":
            self.model, self.processor = create_model_from_pretrained(model_info["model"])
            self.tokenizer = get_tokenizer(model_info["model"])
        elif self.model_name == "BioMistral":
            self.model = AutoModel.from_pretrained(model_info["model"])
            self.processor = AutoTokenizer.from_pretrained(model_info["model"])
        elif self.model_name == "CheXagent":
            self.model = AutoModelForCausalLM.from_pretrained(model_info["model"], trust_remote_code=True).to("cuda") # torch_dtype=torch.float16, 
            self.processor = AutoProcessor.from_pretrained(model_info["model"], trust_remote_code=True)
        elif self.model_name == "CONCH":
            self.model, _ = create_model_from_pretrained_conch(model_info["model"], model_info["tokenizer"])
        elif self.model_name == "LLama-3-Aloe":
            self.model = AutoModel.from_pretrained(model_info["model"])
            self.processor = AutoTokenizer.from_pretrained(model_info["model"])
        elif self.model_name == "Prov-GigaPath":
            self.model = timm.create_model(model_info["model"], pretrained=True, img_size=224)
            self.processor = model_info["tokenizer"]
        elif self.model_name == "PubMedCLIP":
            self.model = CLIPModel.from_pretrained(model_info["model"])
            self.processor = CLIPProcessor.from_pretrained(model_info["model"])
        elif self.model_name == "rad-dino":
            self.model = AutoModel.from_pretrained(model_info["model"])
            self.processor = AutoImageProcessor.from_pretrained(model_info["model"])
        elif self.model_name == "UNI":
            self.model = timm.create_model(
                model_info["model"], img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
            )
            self.model.load_state_dict(torch.load("/data/hf-hub/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin", map_location="cpu"), strict=True)
            self.processor = model_info["tokenizer"]
        elif self.model_name == "Hibou-B":
            self.model = AutoModel.from_pretrained(model_info["model"], trust_remote_code=True)
            self.processor = AutoImageProcessor.from_pretrained(model_info["model"], trust_remote_code=True)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        # Move model to GPU
        self.model = self.model.to("cuda")

    def forward(self, input_data):
        #ipdb.set_trace()
        #print("In PreTrainedEmbedding forward, calling model forward for input of shape: ", input_data.shape)
        with torch.no_grad():
            if self.model_name == "BioMedCLIP":
                outputs = self.model.encode_image(input_data)
            elif self.model_name == "CONCH":
                model_output = self.model.encode_image(input_data, proj_contrast=False, normalize=False)
                outputs = model_output.clone()
                del model_output
            elif self.model_name in ["Prov-GigaPath", "UNI"]:
                model_output = self.model(input_data)
                outputs = model_output.squeeze().clone()
                del model_output
            elif self.model_name == "PubMedCLIP":
                outputs = self.model.get_image_features(**input_data)
            elif self.model_name in ["rad-dino", "Hibou-B"]:                
                model_output = self.model(**input_data)
                outputs = model_output.pooler_output.clone()
                del model_output             
            elif self.model_name == "CheXagent":
                #input = input_data['pixel_values']
                model_output = self.model.vision_model(**(input_data)).last_hidden_state[:, 0, :]
                outputs = model_output.clone()
                del model_output
            else:
                outputs = self.model(input_data).last_hidden_state[:, 0, :]
        
        torch.cuda.empty_cache()
        return outputs