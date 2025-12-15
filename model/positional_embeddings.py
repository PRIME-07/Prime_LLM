import torch
import torch.nn as nn
from config import Config

class PositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Learnable positional embeddings 
        self.embedding = nn.Embedding(config.max_seq_len, config.dim_model)
        
        # Random weight initialization
        nn.init.normal_(self.embedding.weight, mean = 0.0, std=0.02)

    def forward(self, input_ids):
        """
        input ids are tensors of shape [batch, seq_len]
        """
        batch_size, seq_len = input_ids.shape

        # Create position indices from indices 0 to seq_len-1 (in this case seq_len is 512)
        position_ids = torch.arange(seq_len, device=input_ids.device)
        # Repeat position indices for each batch, so after 512, starts from index 0 again
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        # Return positional embeddings
        return self.embedding(position_ids)
