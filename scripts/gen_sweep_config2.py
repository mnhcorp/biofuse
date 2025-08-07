#!/usr/bin/env python3
# filepath: /sharedscratch/mnh3/biofuse/biofuse.git2/find_best_configs.py

import os
import pandas as pd
import yaml
import re
import argparse
import glob
from pathlib import Path
from shutil import copyfile

# Global configuration
SWEEP_CONFIG_DIR = "./sweep-configs/self-attention"  # Output directory for sweep configs
BASE_CONFIG_FILE = "sweep_config.yaml"  # Base config file to use as template
EXPERIMENT_FOLDERS = [
    "./experiments-2.0/autofuse-self-attention-256",
    "./experiments-2.0/autofuse-self-attention-512",
    "./experiments-2.0/autofuse-self-attention-768"
]

class PreserveArrayIndentDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(PreserveArrayIndentDumper, self).increase_indent(flow, False)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Find best configurations across experiment folders')
    args = parser.parse_args()
    
    # Ensure the output directory exists
    os.makedirs(SWEEP_CONFIG_DIR, exist_ok=True)
    
    # Define the reverse mapping for model abbreviations
    model_legend = {
        'BC': 'BioMedCLIP',
        'PC': 'PubMedCLIP',
        'CO': 'CONCH',
        'RD': 'rad-dino',
        'UN': 'UNI',
        'PG': 'Prov-GigaPath',
        'HB': 'Hibou-B',
        'CA': 'CheXagent',
        'UN2': 'UNI2',
    }
    
    reverse_model_legend = {v: k for k, v in model_legend.items()}
    
    # Check if base config file exists
    if not os.path.exists(BASE_CONFIG_FILE):
        print(f"Error: Base config file {BASE_CONFIG_FILE} not found.")
        return
    
    # Find all unique dataset names across folders
    all_datasets = set()
    for folder in EXPERIMENT_FOLDERS:
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} does not exist, skipping.")
            continue
            
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        for csv_file in csv_files:
            match = re.search(r'results_(.+)_224\.csv', os.path.basename(csv_file))
            if match:
                all_datasets.add(match.group(1))
    
    print(f"Found {len(all_datasets)} unique datasets: {', '.join(sorted(all_datasets))}")
    
    # Process each dataset
    for dataset in sorted(all_datasets):
        print(f"\n{'='*80}\nProcessing dataset: {dataset}")
        
        best_val_accuracy = -1
        best_csv_file = None
        best_row = None
        
        # Find the best configuration across all experiment folders
        for folder in EXPERIMENT_FOLDERS:
            csv_path = os.path.join(folder, f"results_{dataset}_224.csv")
            if not os.path.exists(csv_path):
                continue
                
            try:
                df = pd.read_csv(csv_path)
                
                if 'Val Accuracy' not in df.columns:
                    print(f"Warning: 'Val Accuracy' column not found in {csv_path}, skipping.")
                    continue
                    
                # Find the row with the highest validation accuracy in this file
                best_idx = df['Val Accuracy'].idxmax()
                curr_best_val_accuracy = df.loc[best_idx]['Val Accuracy']
                
                # Keep track of the best configuration across all files
                if curr_best_val_accuracy > best_val_accuracy:
                    best_val_accuracy = curr_best_val_accuracy
                    best_csv_file = csv_path
                    best_row = df.loc[best_idx].copy()
                    
                print(f"- {csv_path}: Best Val Accuracy = {curr_best_val_accuracy:.4f}")
                
            except Exception as e:
                print(f"Error processing {csv_path}: {e}")
        
        if best_row is None:
            print(f"No valid data found for dataset {dataset}, skipping.")
            continue
        
        # Extract the best model combination
        if 'Models' not in best_row:
            print(f"Error: 'Models' column not found in best row for dataset {dataset}")
            continue
            
        models_full = best_row['Models'].split(',')
        
        # Get the projection dimension
        if 'Projection Dim' not in best_row:
            print(f"Warning: 'Projection Dim' column not found for dataset {dataset}, using default 0")
            projection_dim = 0
        else:
            projection_dim = int(best_row['Projection Dim'])
        
        # Convert full model names to abbreviations
        best_models = []
        for model_name in models_full:
            model_name = model_name.strip()
            if model_name in reverse_model_legend:
                best_models.append(reverse_model_legend[model_name])
            else:
                print(f"Warning: Model '{model_name}' not found in legend, skipping.")
                
        if not best_models:
            print(f"No valid models found in the best row for dataset {dataset}")
            continue
            
        # Create comma-separated string of model abbreviations
        models_string = ','.join(best_models)
        
        # Print the best configuration details
        print(f"\nBest configuration for {dataset}:")
        print(f"- File: {best_csv_file}")
        print(f"- Projection Dim: {projection_dim}")
        print(f"- Val Accuracy: {best_val_accuracy:.4f}")
        print(f"- Model combination: {models_string}")
        print(f"- Full row: {best_row.to_dict()}")
        
        # Create the sweep config file
        output_config = os.path.join(SWEEP_CONFIG_DIR, f"sweep_config_{dataset}.yaml")
        
        # Read the original file to preserve formatting
        with open(BASE_CONFIG_FILE, 'r') as f:
            config_content = f.read()
            config = yaml.safe_load(config_content)
        
        # Set the display name for the sweep to the dataset name
        config['name'] = f"{dataset}"
        
        # Update the models, dataset, and projection dimension values
        config['parameters']['models']['values'] = [models_string]
        config['parameters']['dataset']['values'] = [dataset]
        
        # Set fusion_methods to self_attention
        config['parameters']['fusion_methods']['values'] = ['self_attention']
        
        # Add or update the projections parameter
        if 'projections' not in config['parameters']:
            config['parameters']['projections'] = {'values': [projection_dim]}
        else:
            config['parameters']['projections']['values'] = [projection_dim]
        
        # Write the updated config back with preserved formatting
        with open(output_config, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, Dumper=PreserveArrayIndentDumper)
        
        print(f"Created sweep config: {output_config}")
    
    print(f"\nDone! Created {len(all_datasets)} sweep configuration files in {SWEEP_CONFIG_DIR}")

if __name__ == "__main__":
    main()