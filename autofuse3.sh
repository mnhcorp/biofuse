#!/bin/bash

#dataset=$1
size=224
models=$1
fusion_methods="concat"
epochs=1

# Model Legend: BC: BioMedCLIP, PC: PubMedCLIP, CO: CONCH, RD: rad-dino, UN: UNI, PG: Prov-GigaPath, HB: Hibou-B, CA: CheXagent, CL: CLIP
# replace the code with model names from 'models' variable
models=$(echo $models | sed 's/BC/BioMedCLIP/g' | sed 's/PC/PubMedCLIP/g' | sed 's/CO/CONCH/g' | sed 's/RD/rad-dino/g' | sed 's/UN/UNI/g' | sed 's/PG/Prov-GigaPath/g' | sed 's/HB/Hibou-B/g' | sed 's/CA/CheXagent/g' | sed 's/CL/CLIP/g')

datasets=("breastmnist" "dermamnist" "octmnist" "pathmnist" "pneumoniamnist" "retinamnist" "organamnist" "organsmnist" "organcmnist" "bloodmnist" "chestmnist" "tissuemnist")

for dataset in ${datasets[@]}; do
    echo "python tests/test_linear_probe_trainable2.py --dataset $dataset --img_size $size --models $models --fusion_methods $fusion_methods --num_epochs=$epochs"
    python tests/test_linear_probe_trainable2.py --dataset $dataset --img_size $size --models $models --fusion_methods $fusion_methods --num_epochs=$epochs --single #--projections 512,1024,2048
done

# echo "python tests/test_linear_probe_trainable2.py --dataset $dataset --img_size $size --models $models --fusion_methods $fusion_methods --num_epochs=$epochs --single"
# python tests/test_linear_probe_trainable2.py --dataset $dataset --img_size $size --models $models --fusion_methods $fusion_methods --num_epochs=$epochs --single #--projections 512,1024,2048

