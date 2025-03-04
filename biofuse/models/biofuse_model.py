import torch
import torch.nn as nn
import torch.nn.functional as F
from biofuse.models.embedding_extractor import PreTrainedEmbedding
from biofuse.models.processor import MultiModelPreprocessor
import ipdb

class BioFuseModel(nn.Module):
    def __init__(self, models, fusion_method='concat', projection_dim=512):
        super(BioFuseModel, self).__init__()
        self.models = models
        self.fusion_method = fusion_method
        self.projection_dim = projection_dim
        self.projection_layers = nn.ModuleList()

        for model in models:
            if projection_dim > 0:
                projection_layer = nn.Sequential(
                    nn.Linear(self.get_model_dim(model), projection_dim),
                )
            else:
                print("No projection layer")
                projection_layer = nn.Identity()
            
            self.projection_layers.append(projection_layer)

        # Initialize learnable weights for 'wsum', 'wmean', and 'ifusion' fusion methods
        if self.fusion_method in ['wsum', 'wmean']:
            self.fusion_weights = nn.Parameter(torch.ones(len(models)) / len(models))
        elif self.fusion_method == 'ifusion':
            self.chunk_size = nn.Parameter(torch.tensor(64.0))  # Initialize chunk size to 64
            self.num_models = len(models)
        elif self.fusion_method == 'self_attention':
            self.attention = nn.MultiheadAttention(embed_dim=projection_dim, num_heads=8, batch_first=True)
            self.layer_norm = nn.LayerNorm(projection_dim)

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
            "UNI": 1024,
            "Prov-GigaPath": 1536,
            "Hibou-B": 768,
            "CLIP": 512,
            "UNI2": 1536
        }
        return model_dims.get(model_name, 512)

    def _ifusion(self, embeddings):
        chunk_size = int(self.chunk_size.item())
        
        # Ensure all embeddings are of the same size
        if not all(embedding.size() == embeddings[0].size() for embedding in embeddings):
            raise ValueError("All embeddings must be of the same size for ifusion")
        
        # Split all embeddings into chunks
        all_chunks = [torch.split(embedding, chunk_size, dim=-1) for embedding in embeddings]
        
        # Interleave chunks from all embeddings
        interleaved_chunks = []
        for chunks in zip(*all_chunks):
            interleaved_chunks.extend(chunks)
        
        # Concatenate interleaved chunks
        return torch.cat(interleaved_chunks, dim=-1)

    def forward(self, embeddings):
        embeddings = [projection(embedding) for embedding, projection in zip(embeddings, self.projection_layers)]
        
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
        elif self.fusion_method == 'ifusion':
            fused_embedding = self._ifusion(embeddings)
        elif self.fusion_method == 'self_attention':
            # Add the batch dimension to the embeddings
            embeddings = [embedding.unsqueeze(0) for embedding in embeddings]

            # Stack the embeddings along the model dimension, to get [batch_size, num_models, embedding_dim]
            stacked_embeddings = torch.stack(embeddings, dim=1)
            
            # Self-attention, Q, K, V are all the same in the shape [batch_size, num_models, embedding_dim]
            # See: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
            attn_output, _ = self.attention(stacked_embeddings, stacked_embeddings, stacked_embeddings)

            # Average over the models
            fused_embedding = self.layer_norm(attn_output.mean(dim=1))
        else:
            raise ValueError(f'Fusion method {self.fusion_method} not supported')

        return fused_embedding

    def forward_test(self, embeddings):
        with torch.no_grad():
            projected_embeddings = [projection(embedding) for embedding, projection in zip(embeddings, self.projection_layers)]
            
            if self.fusion_method == 'concat':
                fused_embedding = torch.cat(projected_embeddings, dim=-1)
            elif self.fusion_method == 'mean':
                fused_embedding = torch.mean(torch.stack(projected_embeddings), dim=0)
            elif self.fusion_method == 'max':
                fused_embedding = torch.max(torch.stack(projected_embeddings), dim=0)[0]
            elif self.fusion_method == 'sum':
                fused_embedding = torch.sum(torch.stack(projected_embeddings), dim=0)
            elif self.fusion_method == 'mul':
                fused_embedding = torch.prod(torch.stack(projected_embeddings), dim=0)
            elif self.fusion_method == 'wsum':
                stacked_embeddings = torch.stack(projected_embeddings)
                fused_embedding = torch.sum(stacked_embeddings * self.fusion_weights.unsqueeze(1).unsqueeze(2), dim=0)
            elif self.fusion_method == 'wmean':
                stacked_embeddings = torch.stack(projected_embeddings)
                weighted_sum = torch.sum(stacked_embeddings * self.fusion_weights.unsqueeze(1).unsqueeze(2), dim=0)
                fused_embedding = weighted_sum / torch.sum(self.fusion_weights)
            elif self.fusion_method == 'ifusion':
                fused_embedding = self._ifusion(projected_embeddings)
            elif self.fusion_method == 'self_attention':
                projected_embeddings = [embedding.unsqueeze(0) for embedding in projected_embeddings]
                stacked_embeddings = torch.stack(projected_embeddings, dim=1)  # [batch_size, num_models, embedding_dim]
                attn_output, _ = self.attention(stacked_embeddings, stacked_embeddings, stacked_embeddings)
                fused_embedding = self.layer_norm(attn_output.mean(dim=1))  # Average over models
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
