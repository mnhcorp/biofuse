#!/bin/bash

dataset=$1
size=$2
# all these models  model_dims = {
        #     "BioMedCLIP": 512,
        #     "BioMistral": 4096,
        #     "CheXagent": 1408,
        #     "CONCH": 512,
        #     "LLama-3-Aloe": 4096,
        #     "Prov-GigaPath": 1536,
        #     "PubMedCLIP": 512,
        #     "rad-dino": 768,
        #     "UNI": 1024,
        #     "Prov-GigaPath": 1536,
        #     "Hibou-B": 768,
        # }
#models="BioMedCLIP,CONCH" #,PubMedCLIP,rad-dino,UNI,Hibou-B,Prov-GigaPath"
models="BioMedCLIP,CONCH,Hibou-B"
fusion_methods="concat"
epochs=100

echo "python tests/test_linear_probe_trainable2.py --dataset $dataset --img_size $size --models $models --fusion_methods $fusion_methods --num_epochs=$epochs"
python tests/test_linear_probe_trainable2.py --dataset $dataset --img_size $size --models $models --fusion_methods $fusion_methods --num_epochs=$epochs --projections 512,1024,2048