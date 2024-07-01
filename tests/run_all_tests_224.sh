#!/bin/bash

# Define the table of configurations
datasets=("octmnist")
image_sizes=(224)
models=("BioMedCLIP,rad-dino") # "BioMedCLIP,rad-dino,PubMedCLIP")
#models=("BioMedCLIP,rad-dino,PubMedCLIP")

fusion_methods=("mean" "concat" "max" "sum" "mul" "wsum" "wmean" "ifusion" "self_attention")
projection_dims=(512 1024)
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
#echo "Dataset,Image Size,Pre-trained Models,Fusion Method,Projection Layer Dim,Epochs,Val Accuracy" > results.csv

# Iterate through all combinations
for dataset in "${datasets[@]}"; do
    for img_size in "${image_sizes[@]}"; do
        for model in "${models[@]}"; do
            for fusion_method in "${fusion_methods[@]}"; do
                for projection_dim in "${projection_dims[@]}"; do
                # only concat works if projection_dim is 0
                    if [ "$projection_dim" == "0" ] && [ "$fusion_method" != "concat" ]; then
                        continue
                    fi
                    run_experiment $dataset $img_size $model $fusion_method $projection_dim
                done
            done
        done
    done
done
