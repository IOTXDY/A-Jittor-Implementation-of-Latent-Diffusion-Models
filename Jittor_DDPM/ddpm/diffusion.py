import math
import jittor as jt
import jittor.nn as F
from utils.basic_functions import *

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = jt.linspace(0, timesteps, steps)
    alphas_cumprod = jt.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jt.clamp(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return jt.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return jt.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = jt.linspace(-6, 6, timesteps)
    return jt.sigmoid(betas) * (beta_end - beta_start) + beta_start

def get_shedule(schedule_func=linear_beta_schedule, timesteps=300):
    betas = schedule_func(timesteps=timesteps)

    alphas = 1. - betas
    alphas_cumprod = jt.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = jt.sqrt(1.0 / alphas)

    # q(x_t | x_{t-1})
    sqrt_alphas_cumprod = jt.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = jt.sqrt(1. - alphas_cumprod)

    # q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    return betas, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,posterior_variance

def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,noise=None):
    noise = default(noise, jt.randn_like(x_start))

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

