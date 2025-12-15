# Model hyperparameters finetuned for the hardware I have
import torch

class Config:
    def __init__(self):
        # Vocab and sequence length
        self.vocab_size = 8000
        self.max_seq_len = 512
        
        # Transformer structure
        self.num_layers = 16
        self.num_heads = 12
        self.dim_model = 768
        self.dim_feedforward = 3072
        self.head_dim = 64
        self.dropout = 0.1

        # Optimization
        self.learning_rate = 6e-4
        self.weight_decay = 0.01
        self.betas = (0.9, 0.999)
        self.warmup_steps = 1000
        self.max_steps = 200000

        # Misc.
        self.tie_weights = True
        self.use_flash_attention = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"    
