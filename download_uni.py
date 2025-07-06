import os
from huggingface_hub import login, hf_hub_download

# === 1. Read token from hf.token ===
token_path = "hf.token"
if not os.path.exists(token_path):
    raise FileNotFoundError("No hf.token file found in current directory.")

with open(token_path, "r") as f:
    hf_token = f.read().strip()

login(token=hf_token)

# === 2. Define models and target dirs ===
models = {
    "vit_large_patch16_224.dinov2.uni_mass100k":    "MahmoodLab/UNI",
    "uni2-h": "MahmoodLab/UNI2-h"
}

base_dir = "/data/hf-hub/ckpts"

# === 3. Download if not already present ===
for name, repo in models.items():
    local_dir = os.path.join(base_dir, name)
    ckpt_path = os.path.join(local_dir, "pytorch_model.bin")

    if os.path.exists(ckpt_path):
        print(f"✅ {name} already exists at {ckpt_path}, skipping download.")
        continue

    os.makedirs(local_dir, exist_ok=True)
    print(f"⬇️ Downloading {name} → {local_dir}")

    hf_hub_download(
        repo_id=repo,
        filename="pytorch_model.bin",
        local_dir=local_dir,
        force_download=True
    )

print("🎉 Done.")