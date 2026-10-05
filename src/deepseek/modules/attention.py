import torch
import torch.nn as nn

from .rope import apply_rope, build_rope_cache


class MHLA(nn.Module):
    def __init__(self, d_model, d_cache, context_length, dropout, n_heads, qkv_bias=False, rope_theta=10000.0):
        super().__init__()
        assert d_model % n_heads ==0, "Wrong n_head size"
        self.d_model=d_model
        self.d_cache=d_cache
        self.n_heads = n_heads
        self.context_length = context_length
        self.head_dim = d_model // n_heads
        self.rope_theta = rope_theta
        self.W_dkv=nn.Linear(d_model,d_cache,bias=qkv_bias)
        self.W_q=nn.Linear(d_model,d_model,bias=qkv_bias)
        self.W_uk=nn.Linear(d_cache,d_model,bias=qkv_bias)
        self.W_uv=nn.Linear(d_cache,d_model,bias=qkv_bias)
        self.register_buffer("mask", torch.triu(
            torch.ones(context_length, context_length), diagonal=1))
        self.out_proj = nn.Linear(d_model,d_model)
        self.register_buffer("cache_kv",None)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,x,use_cache=False):
        b,n_tokens,dim=x.shape
        q=self.W_q(x) # [b,n_tokens,d_model]
        C_kv=self.W_dkv(x)  # [B, n_token, d_cache]
        past_len= 0 if not use_cache or self.cache_kv is None else self.cache_kv.shape[1]
        if use_cache:
            if self.cache_kv is None:
                self.cache_kv=C_kv
            else:
                self.cache_kv=torch.cat([self.cache_kv,C_kv],dim=1)
                C_kv=self.cache_kv

        kv_len=C_kv.shape[1]
        k=   self.W_uk (C_kv)
        v=    self.W_uv(C_kv)
        k=k.view(b,-1,self.n_heads,self.head_dim).transpose(1,2)
        v=v.view(b,-1,self.n_heads,self.head_dim).transpose(1,2) ## -1 means figure out cureent dim as it varies depending on cache_kv size
        q=q.view(b,n_tokens,self.n_heads,self.head_dim).transpose(1,2)

        cos, sin = build_rope_cache(kv_len, self.head_dim, self.rope_theta, x.device)
        q_cos = cos[past_len:past_len + n_tokens]
        q_sin = sin[past_len:past_len + n_tokens]
        q = apply_rope(q, q_cos, q_sin)
        k = apply_rope(k, cos, sin)

        attention_score= q @ k.transpose(2,3)
        mask_bool=self.mask.bool()[past_len:past_len + n_tokens, :kv_len]

        attention_score=attention_score.masked_fill(mask_bool,-torch.inf)

        scale = self.head_dim ** 0.5
        attention_weights=torch.softmax(attention_score / scale,dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector= (attention_weights @ v).transpose(1,2)
        context_vector = context_vector.contiguous().view(b, n_tokens, self.d_model)
        context_vector=self.out_proj(context_vector)
        return context_vector
    def reset_cache(self):
        self.cache_kv=None


MLHA = MHLA
