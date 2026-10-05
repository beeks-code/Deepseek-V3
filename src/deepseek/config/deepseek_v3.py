GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False,     # Query-key-value bias

    # Simple DeepSeek-style extras
    "n_experts": 2,        # Keep this low for a small local model
    "top_k": 2,            # Number of experts selected per token
    "d_cache": 192,        # Compressed latent KV size for MLHA
    "rope_theta": 10000.0, # RoPE base
    "mtp_depth": 2,        # Predict next 1 and next 2 tokens
    "quant_bits": 8        # Basic per-tensor quantization helpers
}
