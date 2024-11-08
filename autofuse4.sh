#!/bin/bash

dataset=$1
size=224
models=$2
fusion_methods="concat"
epochs=1

# Model Legend: BC: BioMedCLIP, PC: PubMedCLIP, CO: CONCH, RD: rad-dino, UN: UNI, PG: Prov-GigaPath, HB: Hibou-B, CA: CheXagent
# replace the code with model names from 'models' variable
models=$(echo $models | sed 's/BC/BioMedCLIP/g' | sed 's/PC/PubMedCLIP/g' | sed 's/CO/CONCH/g' | sed 's/RD/rad-dino/g' | sed 's/UN/UNI/g' | sed 's/PG/Prov-GigaPath/g' | sed 's/HB/Hibou-B/g' | sed 's/CA/CheXagent/g')

#datasets=("dermamnist" "octmnist" "pathmnist" "pneumoniamnist" "retinamnist" "breastmnist" "organamnist" "organsmnist" "organcmnist")
#datasets=("bloodmnist")


echo "python tests/test_linear_probe_trainable_hp.py --dataset $dataset --img_size $size --models $models --fusion_methods $fusion_methods --num_epochs=$epochs"
python tests/test_linear_probe_trainable_hp.py --dataset $dataset --img_size $size --models $models --fusion_methods $fusion_methods --num_epochs=$epochs #--projections 512,1024,2048