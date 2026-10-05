import torch
import torch.nn as nn


def quantize_tensor(x, bits=8):
    qmax = (2 ** (bits - 1)) - 1
    scale = x.abs().max().clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax).to(torch.int8)
    return q, scale


def dequantize_tensor(q, scale):
    return q.float() * scale


def quantize_linear_weights(module, bits=8):
    for child in module.modules():
        if isinstance(child, nn.Linear):
            q_weight, scale = quantize_tensor(child.weight.data, bits)
            child.weight.data.copy_(dequantize_tensor(q_weight, scale).to(child.weight.dtype))
    return module
