import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import GELU


class Expert(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),
            GELU(),
            nn.Linear(4*cfg["emb_dim"],cfg["emb_dim"]),
            nn.Dropout(cfg['drop_rate'])
        )
    def forward(self,x):
        return self.net(x)

class Router(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.top_k=cfg["top_k"]
        self.router=nn.Linear(cfg["emb_dim"],cfg["n_experts"])
        self.noise=nn.Linear(cfg["emb_dim"],cfg["n_experts"]) ## Gaussian Noise
    def forward(self,x):
        selector=self.router(x)
        if self.training:
            noise_logits=self.noise(x)
            noisy_std=F.softplus(noise_logits)
            noise=torch.randn_like(selector)* noisy_std
            selector = selector + noise

        topk_val,indices=torch.topk(selector,k=self.top_k,dim=-1)
        mask=torch.full_like(selector,float('-inf'))
        expert_selector=mask.scatter(-1,indices,topk_val)
        gating_output=torch.softmax(expert_selector,dim=-1)
        return gating_output,indices

class SparseMOE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.router = Router(cfg)
        self.experts = nn.ModuleList(Expert(cfg) for _ in range(cfg["n_experts"]))
        self.top_k = cfg["top_k"]

    def forward(self, x):
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        for i, expert in enumerate(self.experts):

            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(-1)
                weighted_output = expert_output * gating_scores

                final_output[expert_mask] += weighted_output

        return final_output
