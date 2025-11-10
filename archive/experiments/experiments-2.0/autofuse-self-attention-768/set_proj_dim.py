#!/usr/bin/env python3
# filepath: /sharedscratch/mnh3/biofuse/biofuse.git2/set_projection_dim.py

import os
import sys
import pandas as pd
import glob

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: ./set_projection_dim.py <projection_dim>")
        print("Example: ./set_projection_dim.py 256")
        return
    
    try:
        # Parse the projection dimension value
        projection_dim = int(sys.argv[1])
        
        # Find all CSV files in the current directory
        csv_files = glob.glob("*.csv")
        
        if not csv_files:
            print("No CSV files found in the current directory.")
            return
        
        # Process each CSV file
        for csv_file in csv_files:
            print(f"Processing {csv_file}...")
            
            try:
                # Read the CSV file
                df = pd.read_csv(csv_file)
                
                # Check if 'Projection Dim' column exists
                if 'Projection Dim' not in df.columns:
                    print(f"Warning: 'Projection Dim' column not found in {csv_file}, skipping.")
                    continue
                
                # Update the 'Projection Dim' column
                original_value = df['Projection Dim'].iloc[0]
                df['Projection Dim'] = projection_dim
                
                # Save the modified CSV
                df.to_csv(csv_file, index=False)
                print(f"Updated {csv_file}: Projection Dim changed from {original_value} to {projection_dim}")
                
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
        
        print(f"Completed! All CSV files have been updated with Projection Dim = {projection_dim}")
        
    except ValueError:
        print("Error: Projection dimension must be an integer value.")
        print("Usage: ./set_projection_dim.py <projection_dim>")

if __name__ == "__main__":
    main()