import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.act = nn.GELU()  
        self.fc2 = nn.Linear(config.dim_feedforward, config.dim_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x shape: [B, S, D]
        x = self.fc1(x)      # [B, S, FF]
        x = self.act(x)      # [B, S, FF]
        x = self.fc2(x)      # [B, S, D]
        x = self.dropout(x)  # dropout only applied on output

        return x
