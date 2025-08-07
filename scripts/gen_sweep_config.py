#!/usr/bin/env python3
# filepath: /sharedscratch/mnh3/biofuse/biofuse.git2/create_sweep_config.py

import os
import pandas as pd
import yaml
import re
import argparse
from pathlib import Path
from shutil import copyfile

# Global configuration
SWEEP_CONFIG_DIR = "./sweep-configs/oracle-single"  # You can modify this to change the output directory

class PreserveArrayIndentDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(PreserveArrayIndentDumper, self).increase_indent(flow, False)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Create sweep config based on best models from results CSV')
    parser.add_argument('csv_file', help='Path to results CSV file (e.g., results_<dataset>_224.csv)')
    parser.add_argument('--no-config', action='store_true', help='Only output the best model combination without generating config')
    args = parser.parse_args()
    
    result_file = args.csv_file
    
    # Ensure the output directory exists
    if not args.no_config:
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
    
    # Verify the file exists
    if not os.path.exists(result_file):
        print(f"Error: File {result_file} does not exist.")
        return
    
    # Extract dataset name from filename
    match = re.search(r'results_(.+)_224\.csv', os.path.basename(result_file))
    if not match:
        print(f"Could not extract dataset name from {result_file}")
        return
    
    dataset = match.group(1)
    print(f"Processing dataset: {dataset}")
    
    # Read the CSV file
    try:
        df = pd.read_csv(result_file)
        
        # Find the row with the highest validation accuracy
        if 'Val Accuracy' not in df.columns:
            print(f"Column 'Val Accuracy' not found in {result_file}")
            return
        
        best_row = df.loc[df['Val Accuracy'].idxmax()]
        print(f"Best validation accuracy: {best_row['Val Accuracy']:.4f}")
        
        # Get the models directly from the Models column
        if 'Models' not in df.columns:
            print(f"Column 'Models' not found in {result_file}")
            return
        
        # Extract model names from the Models column
        models_full = best_row['Models'].split(',')
        
        # Convert full model names to abbreviations
        best_models = []
        for model_name in models_full:
            model_name = model_name.strip()
            if model_name in reverse_model_legend:
                best_models.append(reverse_model_legend[model_name])
            else:
                print(f"Warning: Model '{model_name}' not found in legend")
        
        if not best_models:
            print(f"No valid models found in the best row for {result_file}")
            return
        
        # Create comma-separated string of model abbreviations
        models_string = ','.join(best_models)
        print(f"Best models: {models_string}")
        
        # If no-config flag is set, just return here
        if args.no_config:
            return
        
        # Copy the sweep config and update it
        output_config = os.path.join(SWEEP_CONFIG_DIR, f"sweep_config_{dataset}.yaml")
        base_config = "sweep_config.yaml"
        
        if not os.path.exists(base_config):
            print(f"Error: Base config file {base_config} not found.")
            return
            
        # Instead of copying, let's read the original file to preserve formatting
        with open(base_config, 'r') as f:
            config_content = f.read()
            config = yaml.safe_load(config_content)
            
        # Set the display name for the sweep to the dataset name
        config['name'] = f"{dataset}"
        
        # Update the models and dataset values
        config['parameters']['models']['values'] = [models_string]
        config['parameters']['dataset']['values'] = [dataset]
        
        # Write the updated config back with preserved formatting
        with open(output_config, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, Dumper=PreserveArrayIndentDumper)
        
        print(f"Created {output_config}")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error processing {result_file}: {e}")

if __name__ == "__main__":
    main()