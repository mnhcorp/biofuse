#!/usr/bin/env python3
# run_medmnistc_xgb.py
#
# Example:
#   python run_medmnistc_xgb.py --dataset dermamnist \
#       --medmnist_root /data/medmnist \
#       --out dermamnist_xgb.csv
#
# Requirements:
#   pip install medmnist medmnistc xgboost scikit-learn pandas numpy pillow

import sys, types
sys.modules['wand'] = types.ModuleType('wand')          # fake top-level package
sys.modules['wand.image'] = types.ModuleType('wand.image')
# provide the attribute medmnistc expects
setattr(sys.modules['wand.image'], 'Image', object)     # dummy stand-in

import argparse, pathlib, time, json
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from medmnist import INFO
import medmnist
from medmnistc.dataset_manager import DatasetManager
from medmnistc.dataset import CorruptionDataset
from medmnistc.corruptions.registry import CORRUPTIONS_DS
from medmnistc.assets.baseline import alexnet_be, alexnet_be_clean  # supplied by medmnistc



##########################################################################
# helpers
##########################################################################
def load_split(name, split, root):
    info = INFO[name]
    DS = getattr(medmnist, info['python_class'])
    ds = DS(split=split, root=root, download=True, transform=None)
    imgs = ds.imgs.reshape(len(ds), -1).astype(np.float32) / 255.0   # flatten to 1-D
    labels = ds.labels.squeeze().astype(np.int64)
    return imgs, labels, info

def train_xgb(X_train, y_train, n_classes, gpu=True):
    params = dict(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        objective='multi:softprob' if n_classes > 2 else 'binary:logistic',
        num_class=n_classes if n_classes > 2 else None,
        tree_method='gpu_hist' if gpu else 'hist',
        predictor='gpu_predictor' if gpu else 'auto',
        eval_metric='mlogloss' if n_classes > 2 else 'logloss',
        random_state=42
    )
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model

def predict_xgb(model, X, n_classes):
    prob = model.predict_proba(X)
    if n_classes == 2:       # binary returns shape (N,) in some xgboost versions
        prob = np.column_stack([1-prob, prob])
    return prob.argmax(1)

##########################################################################
# main
##########################################################################
def main(args):
    gpu_ok = False
    try:
        import cupy  # noqa: F401
        gpu_ok = True
    except Exception:
        pass

    # ------------------------------ load data
    X_tr, y_tr, info = load_split(args.dataset, 'train', args.medmnist_root)
    X_te, y_te, _    = load_split(args.dataset, 'test',  args.medmnist_root)
    n_classes = info['n_classes']

    # ------------------------------ train
    print(f"Training XGBoost on {len(X_tr)} images...")
    t0 = time.time()
    model = train_xgb(X_tr, y_tr, n_classes, gpu=gpu_ok)
    print(f"Done in {(time.time()-t0)/60:.1f} min")

    # ------------------------------ clean accuracy / BE
    y_pred = predict_xgb(model, X_te, n_classes)
    bacc_clean = balanced_accuracy_score(y_te, y_pred)
    be_clean   = 1.0 - bacc_clean

    # ------------------------------ create corrupted dataset once
    c_root = pathlib.Path(args.medmnist_root) / f"{args.dataset}-C"
    if not c_root.exists():
        print("Creating corrupted dataset (one-off, may take a few minutes)...")
        DatasetManager(args.medmnist_root, c_root).create_dataset(dataset_name=args.dataset)

    # ------------------------------ evaluate corruptions
    per_corr = {}                         # corruption → list[severity1-5] BE
    for corr in CORRUPTIONS_DS[args.dataset]:
        be_levels = []
        for sev in range(1, 6):
            ds_c = CorruptionDataset(args.dataset, split='test',
                                     corruption=corr, severity=sev,
                                     root=c_root, transform=None)
            X_c = ds_c.imgs.reshape(len(ds_c), -1).astype(np.float32) / 255.0
            y_c = ds_c.labels.squeeze()
            y_hat = predict_xgb(model, X_c, n_classes)
            be_levels.append(1.0 - balanced_accuracy_score(y_c, y_hat))
        per_corr[corr] = np.array(be_levels)

    df = pd.DataFrame(per_corr, index=[1,2,3,4,5])   # rows are severities
    df.to_csv(args.out, index_label='severity')

    # ------------------------------ BE & rBE normalised
    be_sum   = df.values.sum()
    alex_sum = sum(alexnet_be[args.dataset][c] for c in df.columns)
    BE_norm  = be_sum / alex_sum * 100                       # Eq.(1)

    rbe_sum  = (df.values - be_clean).sum()
    alex_r   = (np.array([alexnet_be[args.dataset][c] for c in df.columns]) -
                alexnet_be_clean[args.dataset]).sum()
    rBE_norm = rbe_sum / alex_r * 100                       # Eq.(2)

    # ------------------------------ print compact row
    print("\n=== MedMNIST-C summary ===")
    print(f"| {args.modelname:15s} | {bacc_clean*100:5.1f} | "
          f"{BE_norm:6.1f} | {rBE_norm:6.1f} |")

    print(f"\nPer-corruption results written to {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="one of the 12 MedMNIST+ names, e.g. dermamnist")
    ap.add_argument("--medmnist_root", default="/data/medmnist",
                    help="root directory holding the MedMNIST files")
    ap.add_argument("--out", default="medmnistc_xgb.csv",
                    help="CSV file for per-corruption BE values")
    ap.add_argument("--modelname", default="XGBoost-GPU")
    args = ap.parse_args()
    main(args)