import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoProcessor, CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModelForCausalLM
from open_clip import create_model_from_pretrained, get_tokenizer
from torchvision import transforms
import timm
import os
from huggingface_hub import login
from biofuse.models.config import AUTH_TOKEN, CACHE_DIR

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

    def login_to_hf(self):
        # set HF_TOKEN environment variable
        os.environ["HF_TOKEN"] = AUTH_TOKEN
        os.environ["HF_HOME"] = "/data/hf-hub/"
        #login()

    def _load_model(self):
        if self.model_name == "BioMedCLIP":
            self.model, self.processor = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        elif self.model_name == "BioMistral":
            self.model = AutoModel.from_pretrained("BioMistral/BioMistral-7B")
            self.processor = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B")
        elif self.model_name == "CheXagent":
            self.model = AutoModelForCausalLM.from_pretrained("StanfordAIMI/CheXagent-8b", torch_dtype=torch.float16, trust_remote_code=True).to("cuda")
            self.processor = AutoProcessor.from_pretrained("StanfordAIMI/CheXagent-8b", trust_remote_code=True)
        elif self.model_name == "CONCH":
            self.model, self.transform = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch")
        elif self.model_name == "LLama-3-Aloe":
            self.model = AutoModel.from_pretrained("HPAI-BSC/Llama3-Aloe-8B-Alpha")
            self.processor = AutoTokenizer.from_pretrained("HPAI-BSC/Llama3-Aloe-8B-Alpha")
        elif self.model_name == "Prov-GigaPath":
            self.model = timm.create_model("vit_giant_patch14_dinov2", pretrained=True, img_size=224)
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        elif self.model_name == "PubMedCLIP":
            self.model = CLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
        elif self.model_name == "rad-dino":
            self.model = AutoModel.from_pretrained("microsoft/rad-dino")
            self.processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
        elif self.model_name == "UNI":
            self.model = timm.create_model(
                "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
            )
            self.model.load_state_dict(torch.load("/data/hf-hub/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin", map_location="cpu"), strict=True)
            self.transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def forward(self, input_data):
        if self.model_name in ["BioMedCLIP", "CONCH", "Prov-GigaPath", "PubMedCLIP", "rad-dino", "UNI"]:
            if self.model_name == "CONCH":
                input_data = self.transform(input_data).unsqueeze(0)
            elif self.model_name == "BioMedCLIP":
                # preprocess
                input_data = self.processor(input_data).unsqueeze(0)
            elif self.model_name == "Prov-GigaPath":
                input_data = self.transform(input_data.convert('RGB')).unsqueeze(0)
            elif self.model_name == "PubMedCLIP":
                input_data = self.processor(images=input_data, return_tensors="pt")
            elif self.model_name == "rad-dino":
                input_data = self.processor(images=input_data, return_tensors="pt")
            elif self.model_name == "UNI":
                input_data = self.transform(input_data).unsqueeze(dim=0)

            with torch.inference_mode():
                if self.model_name == "BioMedCLIP":
                    outputs = self.model.encode_image(input_data)
                elif self.model_name == "CONCH":
                    outputs = self.model.encode_image(input_data, proj_contrast=False, normalize=False)
                elif self.model_name == "Prov-GigaPath":
                    outputs = self.model(input_data).squeeze()
                elif self.model_name == "PubMedCLIP":
                    outputs = self.model.get_image_features(**input_data)
                elif self.model_name == "rad-dino":
                    outputs = self.model(**input_data).pooler_output
                elif self.model_name == "UNI":
                    outputs = self.model(input_data)

        elif self.model_name in ["BioMistral", "CheXagent", "LLama-3-Aloe"]:
            if self.model_name == "CheXagent":
                input_data = self.processor(images=input_data, return_tensors="pt").to("cuda", dtype=torch.float16)
                input_data['pixel_values'] = input_data['pixel_values'].squeeze(1).to("cuda", dtype=torch.float16)
            else:
                input_data = self.processor(input_data, return_tensors='pt')

            with torch.no_grad():
                if self.model_name == "CheXagent":
                    outputs = self.model.vision_model(**input_data).last_hidden_state[:, 0, :]
                    # move to cpu
                    outputs = outputs.detach().cpu().numpy()
                else:
                    outputs = self.model(**input_data).last_hidden_state[:, 0, :]

        return outputs