# 1A.  Install Kaggle CLI if you haven’t
pip install -q kaggle

cp /data/kaggle.json ~/.kaggle/kaggle.json

# 1B.  Put your API token in ~/.kaggle/kaggle.json  (Kaggle ▸ Settings ▸ Create Token)
chmod 600 ~/.kaggle/kaggle.json

# 1C.  Download & unzip
mkdir -p /data/busi && cd /data/busi
kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset
unzip breast-ultrasound-images-dataset.zip