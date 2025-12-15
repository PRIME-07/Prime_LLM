import torch
import torch.nn as nn
import math
from config import Config

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Hyperparameters
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.dim_model = config.dim_model
        self.use_flash = config.use_flash_attention

        # Q, K, V projections
        self.q_proj = nn.Linear(config.dim_model, config.dim_model)
        self.k_proj = nn.Linear(config.dim_model, config.dim_model)
        self.v_proj = nn.Linear(config.dim_model, config.dim_model)
        
        # Output projection
        self.out_proj = nn.Linear(config.dim_model, config.dim_model)

        # Scaling factor for dot product
        self.scale = 1 / math.sqrt(self.head_dim)

        self._warned = False

    def forward(self, x):
        # Get batch size (B), sequence length (S), and embedding dimension (D)
        B, S, D = x.shape

        # Compute Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape Q, K, V for multi-head attention
        Q = Q.view(B, S, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(B, S, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(B, S, self.num_heads, self.head_dim).transpose(1,2)
        # Now: [B, H, S, head_dim]
        

        # Flash attention path
        if self.use_flash:
            # PyTorch built-in flash attention
            out = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, 
                attn_mask = None,
                dropout_p = 0.0,
                is_causal = True
                )
            # out shape = [B, H, S, head_dim]

        else: # Use manual attention computation
            if not self._warned:
                print("WARNING: Using manual attention computation (flash not enabled).")
                self._warned = True

            # Attention Scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            # scores shape = [B, H, S, S]
            
            # Causal mask to prevent looking ahead
            mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))

            # softmax over last dimension
            weights = torch.softmax(scores, dim=-1)

            # weighted sum of values
            out = torch.matmul(weights, V)

        # Merge results
        out = out.transpose(1, 2).contiguous().view(B, S, D)

        return self.out_proj(out)
                