#!/usr/bin/env python3
# filepath: extract_medmnist_labels.py

from medmnist import INFO

def print_medmnist_labels():
    """Print all label names from each MedMNIST dataset"""
    
    for data_flag in INFO.keys():
        info = INFO[data_flag]
        task = info['task']
        labels = info['label']
        
        print(f"\n=== {data_flag} ===")
        print(f"Task: {task}")
        
        # Handle different label formats based on task type
        if task == 'multi-label, binary-class':
            # For multi-label datasets (like ChestMNIST), labels is a list
            print("Labels:")
            for i, label_name in enumerate(labels):
                print(f"  {i}: {label_name}")
        else:
            # For classification datasets, labels is a dictionary
            print("Labels:")
            for class_id, class_name in labels.items():
                print(f"  {class_id}: {class_name}")

if __name__ == "__main__":
    print_medmnist_labels()