import torch.nn as nn

from ..modules.attention import MHLA
from ..modules.moe import SparseMOE


class DeepSeekBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MHLA(
            d_model=cfg["emb_dim"],
            d_cache=cfg["d_cache"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            n_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
            rope_theta=cfg["rope_theta"],
        )
        self.moe = SparseMOE(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, use_cache=use_cache)
        x = shortcut + self.drop_shortcut(x)

        shortcut = x
        x = self.norm2(x)
        x = self.moe(x)
        x = shortcut + self.drop_shortcut(x)
        return x

    def reset_cache(self):
        self.attn.reset_cache()
