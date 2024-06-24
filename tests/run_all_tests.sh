#!/bin/bash

# Define the table of configurations
datasets=("breastmnist")
image_sizes=(224)
models=("BioMedCLIP") # "BioMedCLIP,rad-dino" "BioMedCLIP,rad-dino,PubMedCLIP")
fusion_methods=("mean" "concat" "max" "sum" "mul" "wsum" "wmean" "ifusion")
projection_dims=(0 128 256 512 768 1024)
epochs=100

# Function to run the experiment
run_experiment() {
    dataset=$1
    img_size=$2
    models=$3
    fusion_method=$4
    projection_dim=$5

    echo "Running experiment with dataset=$dataset, img_size=$img_size, models=$models, fusion_method=$fusion_method, projection_dim=$projection_dim, epochs=$epochs"
    python tests/test_linear_probe_trainable.py --dataset $dataset --img_size $img_size --models $models --fusion_method $fusion_method --projection_dim $projection_dim --num_epochs $epochs >> results.csv
}

# Write the CSV header
echo "Dataset,Image Size,Pre-trained Models,Fusion Method,Projection Layer Dim,Epochs,Val Accuracy" > results.csv

# Iterate through all combinations
for dataset in "${datasets[@]}"; do
    for img_size in "${image_sizes[@]}"; do
        for model in "${models[@]}"; do
            for fusion_method in "${fusion_methods[@]}"; do
                for projection_dim in "${projection_dims[@]}"; do
                    if [ "$fusion_method" == "-" ]; then
                        projection_dim="-"
                    fi
                    run_experiment $dataset $img_size $model $fusion_method $projection_dim
                done
            done
        done
    done
done
