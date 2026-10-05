# Deepsheek

A small GPT-style DeepSeek-inspired model using the simple `GPT_CONFIG_124M`
shape: MLHA with RoPE, low-expert MoE, multi-token prediction heads, and basic
quantization helpers.

## Structure

- `src/deepsheek/config`: model configuration values
- `src/deepsheek/modules`: reusable modules such as activation, MoE, and latent attention
- `src/deepsheek/models`: transformer block and full model
- `src/deepsheek/data`: dataset and dataloader code
- `src/deepsheek/training`: losses and training loop
- `src/deepsheek/inference`: generation helpers
- `scripts`: command entry points
- `tests`: module tests

`deepseek_v3.py` is kept as a compatibility re-export.
