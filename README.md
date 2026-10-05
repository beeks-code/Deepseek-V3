# Deepsheek-V3

A small GPT-style DeepSeek-inspired model using the simple `GPT_CONFIG_124M` shape, inspired by my [GPT-From-Scratch](https://github.com/beeks-code/GPT-From-scratch) implementation.

The model experiments with **MLA with RoPE, low-expert(2) MoE, simple multi-token prediction heads, and basic quantization**.

## Structure

* `src/deepsheek/config`: model configuration values
* `src/deepsheek/modules`: reusable modules such as activation, MoE, and latent attention
* `src/deepsheek/models`: transformer block and full model
* `src/deepsheek/data`: dataset and dataloader code
* `src/deepsheek/training`: losses and training loop
* `src/deepsheek/inference`: generation helpers
* `scripts`: command entry points
* `tests`: module tests

`deepseek_v3.py` is kept as a compatibility re-export.
