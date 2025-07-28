import math
from inspect import isfunction
from functools import partial
#from einops import rearrange, reduce, einsum
#from einops.layers.torch import Rearrange

#import torch
#from torch import nn, einsum
#import torch.nn.functional as F
import jittor as jt
from jittor import nn, Module
import jittor.nn as F

from utils.basic_functions import *

class Residual(Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def execute(self, x, *args, **kwargs):
        return x + self.fn(x, *args, **kwargs)
    
def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )

def Downsample(dim, dim_out=None):
    return nn.Sequential(
        #Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        #lambda x: rearrange(x, "b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        # 替换 rearrange 操作为 Jittor 原生操作
        lambda x: x.reshape(
            x.shape[0],           # batch (b)
            x.shape[1],           # channel (c)
            x.shape[2] // 2, 2,   # height (h p1)
            x.shape[3] // 2, 2    # width (w p2)
        ).transpose(2, 3)         # 重排维度
        .reshape(
            x.shape[0], 
            x.shape[1] * 4,       # c*p1*p2 (2*2=4)
            x.shape[2] // 2, 
            x.shape[3] // 2
        ),
        nn.Conv2d(dim*4, default(dim_out, dim), 1),
    )

class SinusoidalPositionEmbeddings(Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def execute(self, time):
        half_dim = self.dim //2
        embeddings = math.log(10000) / (half_dim-1)
        embeddings = jt.exp(jt.arange(half_dim) * -embeddings)
        embeddings = time[:,None] * embeddings[None,:]
        embeddings = jt.concat([embeddings.sin(), embeddings.cos()],dim=-1)
        return embeddings

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def execute(self, x):
        eps = 1e-5 if x.dtype == 'float32' else 1e-3

        weight = self.weight
        #mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        #var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        
        mean = weight.mean(dims=[1, 2, 3], keepdims=True)  # [O, 1, 1, 1]
        #var = weight.var(dims=[1, 2, 3], keepdims=True, unbiased=False)  # [O, 1, 1, 1]
        variance = ((weight - mean)**2).mean(dims=[1, 2, 3], keepdims=True)

        #normalized_weight = (weight - mean) * (var + eps).rsqrt()# rsqrt = 1/sqrt(x)
        normalized_weight = (weight - mean) / jt.sqrt(variance + eps)
        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
    
class SiLU(Module):
    def execute(self, x):
        return x * x.sigmoid()
    
class Block(Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups,dim_out)
        self.act = SiLU()

    def execute(self, x, scale_shift=None):
        x=self.proj(x)
        x=self.norm(x)

        if exists(scale_shift):
            scale,shift = scale_shift
            x = x*(scale+1)+shift
        
        x=self.act(x)
        return x
    
class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(SiLU(), nn.Linear(time_emb_dim, dim_out*2)) if exists(time_emb_dim) else None
        )
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
    
    def execute(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            #time_emb = rearrange(time_emb, "b c -> b c 1 1")
            time_emb = time_emb.reshape(time_emb.shape[0], time_emb.shape[1], 1, 1)
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class Attention(Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = heads * dim_head
        self.to_qkv = nn.Conv2d(dim, hidden_dim*3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def execute(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        """ q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        ) """
        q, k, v = [
            t.reshape(b, self.heads, -1, h*w).transpose(2, 3)
            for t in qkv
        ]
        q = q * self.scale

        #sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = q @ k.transpose(-2, -1)
        sim -= sim.max(dim=-1, keepdims=True).detach()
        attn = sim.softmax(dim=-1)

        #out = einsum("b h i j, b h d j -> b h i d", attn, v)
        #out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        out = attn @ v
        out = out.transpose(2, 3).reshape(b, -1, h, w)
        return self.to_out(out)

class LinearAttention(Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))
        
    def execute(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        """ q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        ) """
        q, k, v = [
            t.reshape(b, self.heads, -1, h*w).transpose(2, 3)
            for t in qkv
        ]

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        #context = einsum("b h d n, b h e n -> b h d e", k, v)
        context = jt.matmul(k, v.transpose(-2, -1))

        #out = einsum("b h d e, b h d n -> b h e n", context, q)
        out = jt.matmul(context, q)
        #out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        out = out.reshape(b, -1, h, w) 
        return self.to_out(out)
    
class PreNorm(Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)
    
    def execute(self, x):
        x = self.norm(x)
        return self.fn(x)