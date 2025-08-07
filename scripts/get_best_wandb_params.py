import wandb
import sys
import json
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python get_best_wandb_params.py <entity/project>")
        print("Example: python get_best_wandb_params.py mnh3/2025-with-uni2-static-250")
        sys.exit(1)

    project_path = sys.argv[1]
    try:
        entity, project = project_path.split("/")
    except ValueError:
        print("❌ Error: Please provide project path in the format 'entity/project'")
        sys.exit(1)

    output_dir = f"{project}-best-params"
    os.makedirs(output_dir, exist_ok=True)

    api = wandb.Api()
    # 🔥 Grab all sweeps manually by filtering sweep IDs from project runs
    all_runs = api.runs(f"{entity}/{project}")
    sweep_map = {}

    for run in all_runs:
        if run.sweep is not None:
            sweep_id = run.sweep.id
            if sweep_id not in sweep_map:
                sweep_map[sweep_id] = []
            sweep_map[sweep_id].append(run)

    if not sweep_map:
        print(f"⚠️ No sweeps found in project: {project_path}")
        return

    print(f"🔍 Found {len(sweep_map)} sweeps in {project_path}")

    for sweep_id, runs in sweep_map.items():
        if not runs:
            continue

        best_run = max(
            runs,
            key=lambda r: r.summary.get("val_accuracy", float('-inf'))
        )

        params = best_run.config
        dataset_name = params.get("dataset", f"sweep_{sweep_id}")
        filename = f"{dataset_name}_{sweep_id}_wandb_params.json".replace("/", "_")
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w") as f:
            json.dump(params, f, indent=2)

        print(f"✅ Sweep {sweep_id} → {filename} (val_accuracy={best_run.summary.get('val_accuracy')})")

    print(f"\n📂 All best params saved to: {output_dir}/")

if __name__ == "__main__":
    main()