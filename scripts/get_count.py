#!/usr/bin/env python3
# filepath: /sharedscratch/mnh3/biofuse/biofuse.git2/count_model_combinations.py

import pandas as pd
import argparse
from collections import Counter

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Count unique model combinations in results CSV')
    parser.add_argument('csv_file', help='Path to results CSV file (e.g., results_<dataset>_224.csv)')
    args = parser.parse_args()
    
    result_file = args.csv_file
    
    try:
        # Read the CSV file
        df = pd.read_csv(result_file)
        
        # Check if 'Models' column exists
        if 'Models' not in df.columns:
            print(f"Error: 'Models' column not found in {result_file}")
            return
        
        # Extract and standardize model combinations
        model_combinations = []
        for model_str in df['Models'].values:
            # Split and sort each model combination for consistent comparison
            models = sorted([m.strip() for m in model_str.split(',')])
            model_combinations.append(','.join(models))
        
        # Count unique combinations
        unique_combinations = set(model_combinations)
        combination_counts = Counter(model_combinations)
        
        # Print results
        print(f"\nAnalysis of file: {result_file}")
        print(f"Total rows: {len(df)}")
        print(f"Unique model combinations: {len(unique_combinations)}")
        print("\nTop 10 most common combinations:")
        
        for combo, count in combination_counts.most_common(10):
            print(f"  {combo}: {count} occurrences")
            
        # Count by number of models in combination
        models_per_combo = {}
        for combo in unique_combinations:
            num_models = len(combo.split(','))
            if num_models not in models_per_combo:
                models_per_combo[num_models] = 0
            models_per_combo[num_models] += 1
            
        print("\nDistribution by number of models:")
        for num_models in sorted(models_per_combo.keys()):
            print(f"  {num_models} model(s): {models_per_combo[num_models]} unique combinations")
        
        # Model frequency analysis
        individual_models = {}
        for combo in model_combinations:
            for model in combo.split(','):
                if model not in individual_models:
                    individual_models[model] = 0
                individual_models[model] += 1
                
        print("\nIndividual model frequency:")
        for model, count in sorted(individual_models.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(model_combinations)) * 100
            print(f"  {model}: {count} occurrences ({percentage:.1f}%)")
            
    except Exception as e:
        print(f"Error processing {result_file}: {e}")

if __name__ == "__main__":
    main()