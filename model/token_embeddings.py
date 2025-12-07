import torch.nn as nn
from math import sqrt

class TokenEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.dim_model)
        self.scale = sqrt(config.dim_model)

    def forward(self, input_ids):
        return self.embedding(input_ids) * self.scale
