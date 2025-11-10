#!/bin/bash

#dataset=$1
size=224
models="UNI,Hibou-B,CheXagent"
#models="CONCH,rad-dino,UNI,Prov-GigaPath,CheXagent"
#models="BioMedCLIP,CONCH,PubMedCLIP,rad-dino,UNI,Hibou-B,Prov-GigaPath,CheXagent"
#models="BioMedCLIP,PubMedCLIP,rad-dino,UNI,CheXagent,Hibou-B,Prov-GigaPath"
fusion_methods="concat"
epochs=1

#datasets=("dermamnist" "octmnist" "pathmnist" "pneumoniamnist" "retinamnist" "breastmnist" "organamnist" "organsmnist" "organcmnist")
datasets=("bloodmnist")


for dataset in ${datasets[@]}; do
    echo "python tests/test_linear_probe_trainable2.py --dataset $dataset --img_size $size --models $models --fusion_methods $fusion_methods --num_epochs=$epochs"
    python tests/test_linear_probe_trainable2.py --dataset $dataset --img_size $size --models $models --fusion_methods $fusion_methods --num_epochs=$epochs --single #--projections 512,1024,2048
done