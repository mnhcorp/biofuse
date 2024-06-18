import torch
import torch.nn as nn
from biofuse.models.embedding_extractor import PreTrainedEmbedding
from biofuse.models.processor import MultiModelPreprocessor

class BioFuseModel(nn.Module):
    def __init__(self, models, fusion_method='concat', projection_dim=512, hidden_dim=1024):
        super(BioFuseModel, self).__init__()
        self.models = models
        self.fusion_method = fusion_method
        self.projection_dim = projection_dim
        self.preprocessor = MultiModelPreprocessor(models)
        self.embedding_extractors = nn.ModuleList()
        self.projection_layers = nn.ModuleList()

        for model in models:
            embedding_extractor = PreTrainedEmbedding(model)
            self.embedding_extractors.append(PreTrainedEmbedding(model))

            #projection_layer = nn.Linear(self.get_model_dim(model), projection_dim)

            # Create an MLP for each projection layer
            projection_layer = nn.Sequential(
                nn.Linear(self.get_model_dim(model), projection_dim),
                # nn.Linear(self.get_model_dim(model), hidden_dim),
                # nn.ReLU(),  # Non-linear activation
                # nn.Linear(hidden_dim, projection_dim)
                nn.LayerNorm(projection_dim)
            )
            self.projection_layers.append(projection_layer)
            
        self.cached_raw_embeddings = []

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
        return model_dims.get(model_name, 512)

    def forward(self, input, cache_raw_embeddings=False, projection=True, index=None):
        if self.cached_raw_embeddings is not None and not cache_raw_embeddings and index is not None:            
            raw_embeddings = self.cached_raw_embeddings[index]
        else:
            processed_images = self.preprocessor.preprocess(input[0])
            raw_embeddings = []
            for img, extractor in zip(processed_images, self.embedding_extractors):
                embedding = extractor(img)
                raw_embeddings.append(embedding)

            if cache_raw_embeddings:
                # append the raw embeddings to the cache
                self.cached_raw_embeddings.append(raw_embeddings)

        embeddings = []
        if self.fusion_method == 'concat':
            for raw_embedding in raw_embeddings:
                embeddings.append(raw_embedding)
            fused_embedding = torch.cat(embeddings, dim=-1)
        elif self.fusion_method == 'mean':
            for raw_embedding, projection in zip(raw_embeddings, self.projection_layers):                
                # if len(self.models) > 1:                    
                #     embedding = projection(raw_embedding)                
                # else:
                #     
                embedding = projection(raw_embedding)
                embeddings.append(embedding)
            fused_embedding = torch.mean(torch.stack(embeddings), dim=0)
        else:
            raise ValueError(f'Fusion method {self.fusion_method} not supported')

        return fused_embedding

    def clear_cached_embeddings(self):
        self.cached_raw_embeddings = None