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

            if projection_dim > 0:
                projection_layer = nn.Sequential(
                    nn.Linear(self.get_model_dim(model), projection_dim),
                    #nn.ReLU(),
                    #nn.LayerNorm(projection_dim)
                )
            else:
                print("No projection layer")
                projection_layer = nn.Identity()
                for param in projection_layer.parameters():
                    param.requires_grad = False
            
            self.projection_layers.append(projection_layer)
            
        self.cached_train_embeddings = {model: {} for model in models}
        self.cached_val_embeddings = {model: {} for model in models}

        # Initialize learnable weights for 'wsum' and 'wmean' fusion methods
        if self.fusion_method in ['wsum', 'wmean']:
            self.fusion_weights = nn.Parameter(torch.ones(len(models)) / len(models))

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

    def forward(self, input, cache_raw_embeddings=False, index=None, is_training=True):
        cache = self.cached_train_embeddings if is_training else self.cached_val_embeddings

        if index is not None and all(index in cache[model] for model in self.models):
            raw_embeddings = [cache[model][index] for model in self.models]
        else:
            processed_images = self.preprocessor.preprocess(input[0])
            raw_embeddings = []
            for img, extractor in zip(processed_images, self.embedding_extractors):
                embedding = extractor(img)
                raw_embeddings.append(embedding)

            if cache_raw_embeddings and index is not None:
                for model, embedding in zip(self.models, raw_embeddings):
                    cache[model][index] = embedding

        # Print size of the cache to check if it is growing
        #print("Size of cache: ", sum([len(cache[model]) for model in self.models]))

        embeddings = [projection(raw_embedding) for raw_embedding, projection in zip(raw_embeddings, self.projection_layers)]
        
        if self.fusion_method == 'concat':
            fused_embedding = torch.cat(embeddings, dim=-1)
        elif self.fusion_method == 'mean':
            fused_embedding = torch.mean(torch.stack(embeddings), dim=0)
        elif self.fusion_method == 'max':
            fused_embedding = torch.max(torch.stack(embeddings), dim=0)[0]
        elif self.fusion_method == 'sum':
            fused_embedding = torch.sum(torch.stack(embeddings), dim=0)
        elif self.fusion_method == 'mul':
            fused_embedding = torch.prod(torch.stack(embeddings), dim=0)
        elif self.fusion_method == 'wsum':
            stacked_embeddings = torch.stack(embeddings)
            fused_embedding = torch.sum(stacked_embeddings * self.fusion_weights.unsqueeze(1).unsqueeze(2), dim=0)
        elif self.fusion_method == 'wmean':
            stacked_embeddings = torch.stack(embeddings)
            weighted_sum = torch.sum(stacked_embeddings * self.fusion_weights.unsqueeze(1).unsqueeze(2), dim=0)
            fused_embedding = weighted_sum / torch.sum(self.fusion_weights)
        else:
            raise ValueError(f'Fusion method {self.fusion_method} not supported')

        return fused_embedding

    def forward_test(self, input):
        processed_images = self.preprocessor.preprocess(input[0])
        raw_embeddings = []
        for img, extractor in zip(processed_images, self.embedding_extractors):
            embedding = extractor(img)
            raw_embeddings.append(embedding)

        embeddings = [projection(raw_embedding) for raw_embedding, projection in zip(raw_embeddings, self.projection_layers)]
        
        if self.fusion_method == 'concat':
            fused_embedding = torch.cat(embeddings, dim=-1)
        elif self.fusion_method == 'mean':
            fused_embedding = torch.mean(torch.stack(embeddings), dim=0)
        elif self.fusion_method == 'max':
            fused_embedding = torch.max(torch.stack(embeddings), dim=0)[0]
        elif self.fusion_method == 'sum':
            fused_embedding = torch.sum(torch.stack(embeddings), dim=0)
        elif self.fusion_method == 'mul':
            fused_embedding = torch.prod(torch.stack(embeddings), dim=0)
        elif self.fusion_method == 'wsum':
            stacked_embeddings = torch.stack(embeddings)
            fused_embedding = torch.sum(stacked_embeddings * self.fusion_weights.unsqueeze(1).unsqueeze(2), dim=0)
        elif self.fusion_method == 'wmean':
            stacked_embeddings = torch.stack(embeddings)
            weighted_sum = torch.sum(stacked_embeddings * self.fusion_weights.unsqueeze(1).unsqueeze(2), dim=0)
            fused_embedding = weighted_sum / torch.sum(self.fusion_weights)
        else:
            raise ValueError(f'Fusion method {self.fusion_method} not supported')

        return fused_embedding

    def clear_cached_embeddings(self):
        self.cached_train_embeddings = {model: {} for model in self.models}
        self.cached_val_embeddings = {model: {} for model in self.models}

    def half(self):
        for projection in self.projection_layers:
            projection.half()
        return self
