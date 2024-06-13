import torch
import torch.nn as nn
from biofuse.models.embedding_extractor import PreTrainedEmbedding
from biofuse.models.processor import MultiModelPreprocessor

class BioFuseModel(nn.Module):
    def __init__(self, models, fusion_method='concat', projection_dim=512):
        super(BioFuseModel, self).__init__()
        self.models = models
        self.fusion_method = fusion_method
        self.projection_dim = projection_dim
        self.preprocessor = MultiModelPreprocessor(models)
        
        self.embedding_extractors = nn.ModuleList()
        self.projection_layers = nn.ModuleList()

        for model in models:
            self.embedding_extractors.append(PreTrainedEmbedding(model))
            # Assume each model has its own dimensionality, which we project to a common dimension
            self.projection_layers.append(nn.Linear(self.get_model_dim(model), projection_dim))            

    def get_model_dim(self, model_name):
        model_dims = {
            "BioMedCLIP": 512,
            "BioMistral": 4096,
            "CheXagent": 1408,
            "CONCH": 512,
            "LLama-3-Aloe": 4096,
            "Prov-GigaPath": 1536,
            "PubMedCLIP": 512,
            "rad-dino": 768,
            "UNI": 1024
        }

        return model_dims.get(model_name, 512)  # default to 512    

    def forward(self, input):
        # print("Input: ", input)
        # print("Type: ", type(input))
        # For every input image/text, this will return num_models tensors
        processed_images = self.preprocessor.preprocess(input[0])
        embeddings = []       
        
        if self.fusion_method == 'concat':                        
            # No projection layer for concatenation
            for img, extractor in zip(processed_images, self.embedding_extractors):
                embedding = extractor(img)
                embeddings.append(embedding)
            fused_embedding = torch.cat(embeddings, dim=-1)
            
        elif self.fusion_method == 'mean':
            # For mean, we project to a common dim
            for img, extractor, projection in zip(processed_images, self.embedding_extractors, self.projection_layers):
                embedding = extractor(img)
                embedding = projection(embedding)
                embeddings.append(embedding)
            fused_embedding = torch.mean(torch.stack(embeddings), dim=0)
        else:
            raise ValueError(f'Fusion method {self.fusion_method} not supported')
        
        return fused_embedding