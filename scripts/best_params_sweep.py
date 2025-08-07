import wandb
import sys
import json
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python best_params.py <sweep_id>")
        print("Example: python best_params.py your-entity/your-project/sweeps/abcd1234")
        sys.exit(1)

    sweep_path = sys.argv[1]

    api = wandb.Api()
    sweep = api.sweep(sweep_path)

    best_run = max(
        sweep.runs,
        key=lambda r: r.summary.get("val_accuracy", float('-inf'))
    )

    params = best_run.config
    dataset_name = params.get("dataset", "best_params")

    filename = f"{dataset_name}_wandb_params.json"
    with open(filename, "w") as f:
        json.dump(params, f, indent=2)

    print(f"\n🏆 Best Run ID: {best_run.id}")
    print(f"✅ Best val_accuracy: {best_run.summary.get('val_accuracy')}")
    print(f"📦 Saved best params to: {filename}")

if __name__ == "__main__":
    main()