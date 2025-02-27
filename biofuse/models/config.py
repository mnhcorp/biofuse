# config.py
from torchvision import transforms
import timm
import torch

# HuggingFace authentication token
AUTH_TOKEN = "hf_"

# HuggingFace home directory
CACHE_DIR = "/data/hf-hub/"

MODEL_MAP = {
            "CLIP": {
                "model": "openai/clip-vit-base-patch32",
                "tokenizer": None
            },
            "BioMedCLIP": {
                "model": 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
                "tokenizer": 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            },
            "BioMistral": {
                "model": "BioMistral/BioMistral-7B",
                "tokenizer": None
            },
            "CheXagent": {
                "model": "StanfordAIMI/CheXagent-8b",
                "tokenizer": None
            },
            "CONCH": {
                "model": 'conch_ViT-B-16',
                "tokenizer": "hf_hub:MahmoodLab/conch"
            },
            "LLama-3-Aloe": {
                "model": "HPAI-BSC/Llama3-Aloe-8B-Alpha",
                "tokenizer": None
            },
            "Prov-GigaPath": {
                "model": "vit_giant_patch14_dinov2",
                "tokenizer": transforms.Compose(
                    [
                        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ]
                )
            },
            "PubMedCLIP": {
                "model": "flaviagiammarino/pubmed-clip-vit-base-patch32",
                "tokenizer": None
            },
            "rad-dino": {
                "model": "microsoft/rad-dino",
                "tokenizer": None
            },
            "UNI": {
                "model": "vit_large_patch16_224",
                "tokenizer": transforms.Compose(
                    [
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ]
                )
            },
            "UNI2": {
                "model": "uni2-h",
                "tokenizer": transforms.Compose(
                    [
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ]
                ),
                "timm_kwargs": {
                            'model_name': 'vit_giant_patch14_224',
                            'img_size': 224, 
                            'patch_size': 14, 
                            'depth': 24,
                            'num_heads': 24,
                            'init_values': 1e-5, 
                            'embed_dim': 1536,
                            'mlp_ratio': 2.66667*2,
                            'num_classes': 0, 
                            'no_embed_class': True,
                            'mlp_layer': timm.layers.SwiGLUPacked, 
                            'act_layer': torch.nn.SiLU, 
                            'reg_tokens': 8, 
                            'dynamic_img_size': True
                            }
            },
            "Hibou-B": {
                "model": "histai/hibou-b",
                "tokenizer": None
            }
        }
