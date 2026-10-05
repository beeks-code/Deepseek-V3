
import sys
from pathlib import Path

import torch

# Make the src package available when running tests directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepsheek.config import GPT_CONFIG_124M
from deepsheek.models import DeepSeekV3
from deepsheek.modules import (
    MHLA,
    SparseMOE,
    dequantize_tensor,
    quantize_tensor,
)


def make_test_config():
    """Create a small configuration so tests run quickly."""
    config = dict(GPT_CONFIG_124M)

    config.update(
        {
            "vocab_size": 128,
            "context_length": 16,
            "emb_dim": 48,
            "n_heads": 6,
            "n_layers": 2,
            "n_experts": 2,
            "top_k": 1,
            "d_cache": 24,
            "mtp_depth": 2,
        }
    )

    return config


def test_mhla_preserves_shape():
    config = make_test_config()

    attention = MHLA(
        config["emb_dim"],
        config["d_cache"],
        config["context_length"],
        config["drop_rate"],
        config["n_heads"],
        config["qkv_bias"],
        config["rope_theta"],
    )

    x = torch.randn(2, 8, config["emb_dim"])
    output = attention(x)

    assert output.shape == x.shape


def test_moe_preserves_shape():
    config = make_test_config()
    moe = SparseMOE(config)

    x = torch.randn(2, 8, config["emb_dim"])
    output = moe(x)

    assert output.shape == x.shape


def test_model_output_shape():
    config = make_test_config()
    model = DeepSeekV3(config)

    input_ids = torch.randint(
        0,
        config["vocab_size"],
        (2, 8),
    )

    logits, mtp_logits = model(input_ids, return_mtp=True)

    expected_shape = (
        2,
        8,
        config["vocab_size"],
    )

    assert logits.shape == expected_shape
    assert len(mtp_logits) == config["mtp_depth"]

    for mtp_output in mtp_logits:
        assert mtp_output.shape == expected_shape


def test_quantization_round_trip():
    x = torch.randn(4, 4)

    quantized, scale = quantize_tensor(x)
    reconstructed = dequantize_tensor(
        quantized,
        scale,
    )

    assert quantized.dtype == torch.int8
    assert reconstructed.shape == x.shape

