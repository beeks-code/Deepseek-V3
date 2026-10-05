import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepseek.config import GPT_CONFIG_124M
from deepseek.models import DeepSeekV3
from deepseek.modules import MHLA, SparseMOE, quantize_tensor, dequantize_tensor


def tiny_cfg():
    cfg = dict(GPT_CONFIG_124M)
    cfg.update({
        "vocab_size": 128,
        "context_length": 16,
        "emb_dim": 48,
        "n_heads": 6,
        "n_layers": 2,
        "n_experts": 2,
        "top_k": 1,
        "d_cache": 24,
        "mtp_depth": 2,
    })
    return cfg


def test_mhla_shape():
    cfg = tiny_cfg()
    module = MHLA(
        cfg["emb_dim"],
        cfg["d_cache"],
        cfg["context_length"],
        cfg["drop_rate"],
        cfg["n_heads"],
        cfg["qkv_bias"],
        cfg["rope_theta"],
    )
    x = torch.randn(2, 8, cfg["emb_dim"])
    assert module(x).shape == x.shape


def test_moe_shape():
    cfg = tiny_cfg()
    module = SparseMOE(cfg)
    x = torch.randn(2, 8, cfg["emb_dim"])
    assert module(x).shape == x.shape


def test_model_and_mtp_shapes():
    cfg = tiny_cfg()
    model = DeepSeekV3(cfg)
    input_ids = torch.randint(0, cfg["vocab_size"], (2, 8))
    logits, mtp_logits = model(input_ids, return_mtp=True)
    assert logits.shape == (2, 8, cfg["vocab_size"])
    assert len(mtp_logits) == cfg["mtp_depth"]
    assert mtp_logits[0].shape == logits.shape


def test_quantization_round_trip_shape():
    x = torch.randn(4, 4)
    q, scale = quantize_tensor(x)
    y = dequantize_tensor(q, scale)
    assert q.dtype == torch.int8
    assert y.shape == x.shape
