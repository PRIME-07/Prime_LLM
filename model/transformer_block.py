import torch
import torch.nn as nn
from attention_block import Attention
from feed_forward_block import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Normalization layers
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)

        # Submodules
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        # Attention block
        attention_input = self.norm1(x)                     # normalize token representation
        attention_out = self.attention(attention_input)     # compute self-attention

        # Feedforward block
        feed_forward_input = self.norm2(x + attention_out)
        feed_forward_out = self.feed_forward(feed_forward_input)
        x = x + feed_forward_out        # residual connection

        return x


