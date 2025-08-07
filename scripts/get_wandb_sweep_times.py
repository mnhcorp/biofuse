import wandb
import sys
import csv
import os
from datetime import datetime

def main():
    if len(sys.argv) != 2:
        print("Usage: python get_wandb_sweep_times.py <entity/project>")
        print("Example: python get_wandb_sweep_times.py mnh3/2025-with-uni2-static-250")
        sys.exit(1)

    project_path = sys.argv[1]
    try:
        entity, project = project_path.split("/")
    except ValueError:
        print("❌ Error: Please provide project path in the format 'entity/project'")
        sys.exit(1)

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

    # Prepare CSV data
    csv_data = []
    
    for sweep_id, runs in sweep_map.items():
        if not runs:
            continue

        # Get dataset name from any run in the sweep
        dataset_name = runs[0].config.get("dataset", f"sweep_{sweep_id}")
        
        # Calculate total time for this sweep
        total_time_seconds = 0
        run_count = 0
        
        for run in runs:
            run_time = None
            
            try:
                # Method 1: Check for duration in _attrs
                if hasattr(run, '_attrs') and 'duration' in run._attrs:
                    run_time = run._attrs['duration']
                # Method 2: Check for _runtime in summary
                elif run.summary.get('_runtime'):
                    run_time = run.summary.get('_runtime')
                # Method 3: Try to calculate from timestamps (with proper error handling)
                elif hasattr(run, 'created_at') and hasattr(run, 'finished_at'):
                    if run.created_at and run.finished_at:
                        created = datetime.fromisoformat(run.created_at.replace('Z', '+00:00'))
                        finished = datetime.fromisoformat(run.finished_at.replace('Z', '+00:00'))
                        run_time = (finished - created).total_seconds()
                # Method 4: Try heartbeat timestamps as fallback
                elif hasattr(run, 'heartbeat_at') and hasattr(run, 'created_at'):
                    if run.created_at and run.heartbeat_at:
                        created = datetime.fromisoformat(run.created_at.replace('Z', '+00:00'))
                        heartbeat = datetime.fromisoformat(run.heartbeat_at.replace('Z', '+00:00'))
                        run_time = (heartbeat - created).total_seconds()
            except (AttributeError, ValueError, TypeError) as e:
                # Skip this run if we can't get timing data
                print(f"⚠️ Skipping run {run.id} (crashed/incomplete): {str(e)}")
                continue
            
            if run_time is not None and run_time > 0:
                total_time_seconds += run_time
                run_count += 1
        
        if run_count > 0:
            csv_data.append({
                'sweep_name': dataset_name,
                'total_time_seconds': round(total_time_seconds, 2)
            })
            print(f"✅ Sweep {sweep_id} ({dataset_name}): {run_count} runs, {total_time_seconds:.2f} seconds total")
        else:
            print(f"⚠️ Sweep {sweep_id} ({dataset_name}): No timing data available")

    # Write to CSV
    csv_filename = "sweep_times.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['sweep_name', 'total_time_seconds']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)

    print(f"\n📊 Sweep times saved to: {csv_filename}")
    print(f"📈 Total sweeps processed: {len(csv_data)}")
    
    # Print summary statistics
    if csv_data:
        total_time = sum(row['total_time_seconds'] for row in csv_data)
        avg_time = total_time / len(csv_data)
        max_time = max(row['total_time_seconds'] for row in csv_data)
        min_time = min(row['total_time_seconds'] for row in csv_data)
        
        print(f"⏱️  Summary:")
        print(f"   Total time across all sweeps: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        print(f"   Average time per sweep: {avg_time:.2f} seconds ({avg_time/3600:.2f} hours)")
        print(f"   Longest sweep: {max_time:.2f} seconds ({max_time/3600:.2f} hours)")
        print(f"   Shortest sweep: {min_time:.2f} seconds ({min_time/3600:.2f} hours)")

if __name__ == "__main__":
    main()