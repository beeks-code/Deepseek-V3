import torch


def build_rope_cache(seq_len, head_dim, theta=10000.0, device=None):
    assert head_dim % 2 == 0, "RoPE requires an even head_dim"
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: [batch, heads, tokens, head_dim]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out
