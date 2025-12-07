    import torch.nn as nn
    from config import Config
    from token_embeddings import TokenEmbedding
    from positional_embeddings import PositionalEmbedding


    class EmbeddingBlock(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.token_embedding = TokenEmbedding(config)
            self.positional_embedding = PositionalEmbedding(config)
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, input_ids):
            # Get the token embeddings
            token_emb = self.token_embedding(input_ids)
            # Get the positional embeddings
            pos_emb = self.positional_embedding(input_ids)
            # Add the token and positional embeddings and apply dropout
            combined_emb = self.dropout(token_emb + pos_emb)
            
            return combined_emb
