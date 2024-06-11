import torch
import torch.nn as nn
from embedding_extractor import PreTrainedEmbedding

class BioFuseModel(nn.Module):
    def __init__(self, models, fusion_method='concat'):
        super(BioFuseModel, self).__init__()
        self.models = models
        self.fusion_method = fusion_method
        
        self.embedding_extractors = nn.ModuleList()
        for model in models:
            self.embedding_extractors.append(PreTrainedEmbedding(model))

    def forward(self, inputs):
        embeddings = []
        for extractor in self.embedding_extractors:
            embedding = extractor(inputs)
            embeddings.append(embedding)
        
        if self.fusion_method == 'concat':
            fused_embedding = torch.cat(embeddings, dim=-1)
        elif self.fusion_method == 'mean':
            fused_embedding = torch.mean(torch.stack(embeddings), dim=0)
        else:
            raise ValueError(f'Fusion method {self.fusion_method} not supported')
        
        return fused_embedding