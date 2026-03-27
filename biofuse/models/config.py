# config.py
from torchvision import transforms
import timm
import torch
import os
from pathlib import Path

def get_hf_token():
    """
    Get HuggingFace token from multiple sources in order of priority:
    1. Environment variable HF_TOKEN
    2. Environment variable HUGGINGFACE_TOKEN
    3. ~/.huggingface/token file (default HF CLI location)
    4. .env file in project root

    Returns:
        str: HuggingFace token or None if not found
    """
    # Try environment variables first
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    if token:
        return token

    # Try default HuggingFace CLI location
    hf_token_path = Path.home() / '.huggingface' / 'token'
    if hf_token_path.exists():
        try:
            return hf_token_path.read_text().strip()
        except Exception:
            pass

    # Try .env file in project root
    try:
        env_file = Path(__file__).parent.parent.parent / '.env'
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith('HF_TOKEN=') or line.startswith('HUGGINGFACE_TOKEN='):
                    return line.split('=', 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass

    return None

def get_cache_dir():
    """
    Get HuggingFace cache directory from environment or use default.

    Returns:
        str: Cache directory path
    """
    # Try environment variable
    cache_dir = os.environ.get('HF_HOME') or os.environ.get('HUGGINGFACE_HUB_CACHE')
    if cache_dir:
        return cache_dir

    # Check if /data exists and is writable (common in clusters)
    data_dir = Path('/data/hf-hub')
    if data_dir.parent.exists() and os.access(data_dir.parent, os.W_OK):
        return str(data_dir)

    # Default to user's home directory
    return str(Path.home() / '.cache' / 'huggingface')

# HuggingFace authentication token
AUTH_TOKEN = get_hf_token()

# HuggingFace cache directory
CACHE_DIR = get_cache_dir()

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
