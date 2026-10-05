import torch
import torch.nn as nn

from ..config import GPT_CONFIG_124M
from .block import DeepSeekBlock


class MultiTokenPredictionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = nn.ModuleList(
            nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
            for _ in range(cfg["mtp_depth"]))
## simple MTP class no Merge and project layer
## Might work poorly
    def forward(self, x):
        return [head(x) for head in self.heads]


class DeepSeekV3(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = dict(GPT_CONFIG_124M if cfg is None else cfg)
        self.token_emb = nn.Embedding(self.cfg["vocab_size"], self.cfg["emb_dim"])
        self.pos_emb = nn.Embedding(self.cfg["context_length"], self.cfg["emb_dim"])
        self.drop_out = nn.Dropout(self.cfg["drop_rate"])
        self.blocks = nn.ModuleList(
            DeepSeekBlock(self.cfg) for _ in range(self.cfg["n_layers"])
        )
        self.norm = nn.LayerNorm(self.cfg["emb_dim"])
        self.output_head = nn.Linear(self.cfg["emb_dim"], self.cfg["vocab_size"], bias=False)
        self.mtp_head = MultiTokenPredictionHead(self.cfg)
        self.cache_len = 0

    def forward(self, input_ids, use_cache=False, return_mtp=False):
        batch_size, seq_length = input_ids.shape
        start_pos = self.cache_len if use_cache else 0
        positions = torch.arange(
            start_pos,
            start_pos + seq_length,
            device=input_ids.device
        )

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.drop_out(x)

        for block in self.blocks:
            x = block(x, use_cache=use_cache)

        x = self.norm(x)
        logits = self.output_head(x)

        if use_cache:
            self.cache_len += seq_length

        if return_mtp:
            return logits, self.mtp_head(x)
        return logits

    def reset_cache(self):
        self.cache_len = 0
        for block in self.blocks:
            block.reset_cache()
