import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding_block import EmbeddingBlock
from transformer_block import TransformerBlock
from config import Config


class PrimeGPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.config = config

        # 1) Embeddings (token + positional)
        self.embedding_block = EmbeddingBlock(config)

        # 2) Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )

        # 3) Final LN
        self.final_ln = nn.LayerNorm(config.dim_model)

        # 4) LM head
        self.lm_head = nn.Linear(config.dim_model, config.vocab_size, bias=False)

        # 5) Weight tying
        if config.tie_weights:
            self.lm_head.weight = self.embedding_block.token_embedding.embedding.weight

        # 6) Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialization used by GPT-series models."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids):
        """
        input_ids: [B, S] integers
        returns:   [B, S, V] logits
        """
        if input_ids.size(1) > self.config.max_seq_len:
            input_ids = input_ids[:, -self.config.max_seq_len:]

        x = self.embedding_block(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=200,
        temperature=1.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.0,
        eos_token_ids=None,
        chorus_token_id=None,
        banned_token_ids=None,
        device=None,
    ):
        self.eval()

        if device is None:
            device = next(self.parameters()).device

        input_ids = input_ids.to(device)

        seen_chorus = False

        for _ in range(max_new_tokens):
            if input_ids.size(1) > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]

            logits = self.forward(input_ids)
            logits = logits[:, -1, :]

            # Hard-ban control tokens
            if banned_token_ids is not None:
                for tid in banned_token_ids:
                    if tid >= 0:
                        logits[:, tid] = -1e9

            # Temperature decay after chorus
            current_temp = temperature * (0.7 if seen_chorus else 1.0)
            logits = logits / max(current_temp, 1e-8)

            # Repetition penalty (last 64 tokens)
            if repetition_penalty != 1.0:
                window = min(64, input_ids.size(1))
                recent_tokens = input_ids[:, -window:]
                for b in range(logits.size(0)):
                    for token_id in recent_tokens[b].tolist():
                        logits[b, token_id] /= repetition_penalty

            # Top-k
            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                min_values = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_values, -1e9, logits)

            # Top-p
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = probs.cumsum(dim=-1)

                remove_mask = cumulative_probs > top_p
                remove_mask[..., 1:] = remove_mask[..., :-1].clone()
                remove_mask[..., 0] = False

                for b in range(logits.size(0)):
                    logits[b, sorted_indices[b, remove_mask[b]]] = -1e9

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Track chorus appearance
            if chorus_token_id is not None and next_token.item() == chorus_token_id:
                seen_chorus = True

            # Early stopping on </song>
            if eos_token_ids is not None and next_token.item() in eos_token_ids:
                break

        return input_ids
