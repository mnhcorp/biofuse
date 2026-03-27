import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml


SPLIT_LAYOUT = {
    "train": {
        "class0": ["xray.jpg", "xray.jpg"],
        "class1": ["hist.jpg", "bcc.jpg"],
    },
    "val": {
        "class0": ["xray.jpg"],
        "class1": ["hist.jpg"],
    },
    "test": {
        "class0": ["xray.jpg"],
        "class1": ["bcc.jpg"],
    },
}


def build_tiny_dataset(source_dir: Path, dataset_root: Path) -> None:
    for split, classes in SPLIT_LAYOUT.items():
        for class_name, filenames in classes.items():
            class_dir = dataset_root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for index, filename in enumerate(filenames):
                src = source_dir / filename
                dst = class_dir / f"{index}_{filename}"
                shutil.copy2(src, dst)


def write_smoke_config(config_path: Path, dataset_root: Path, output_dir: Path, model: str, device: str) -> None:
    config = {
        "name": "docker_smoke",
        "data": {
            "dataset": "custom",
            "data_root": str(dataset_root),
            "img_size": 224,
            "batch_size": 2,
            "num_workers": 0,
        },
        "model": {
            "models": [model],
            "fusion_method": "concat",
            "projection_dim": 0,
        },
        "classifier": {
            "type": "logistic",
        },
        "cache": {
            "use_cache": False,
        },
        "device": device,
        "output_dir": str(output_dir),
    }

    config_path.write_text(yaml.safe_dump(config, sort_keys=False))


def main():
    parser = argparse.ArgumentParser(description="Run a tiny end-to-end BioFuse smoke test.")
    parser.add_argument("--model", default="CLIP", help="Public model to exercise in the smoke test.")
    parser.add_argument("--device", default="cuda", help="Device passed through to the CLI.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    source_dir = repo_root / "data"

    with tempfile.TemporaryDirectory(prefix="biofuse-smoke-") as tmpdir:
        tmp_root = Path(tmpdir)
        dataset_root = tmp_root / "dataset"
        output_dir = tmp_root / "results"
        config_path = tmp_root / "smoke.yaml"

        build_tiny_dataset(source_dir, dataset_root)
        write_smoke_config(config_path, dataset_root, output_dir, args.model, args.device)

        command = ["biofuse", "train", "--config", str(config_path)]
        print("Running smoke command:", " ".join(command))
        subprocess.run(command, check=True, cwd=repo_root)

        saved_config = output_dir / "docker_smoke" / "config.yaml"
        if not saved_config.exists():
            raise SystemExit(f"Smoke test finished without producing {saved_config}")

        print(f"Smoke test completed successfully. Output config: {saved_config}")


if __name__ == "__main__":
    main()
