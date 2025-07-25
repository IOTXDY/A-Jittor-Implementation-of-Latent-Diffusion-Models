from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from utils.basic_functions import *
from ddpm.diffusion import q_sample

def p_losses(denoise_model, x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,noise=None, loss_type="l1"):
    noise = default(noise, torch.randn_like(x_start))

    x_noisy = q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == "l1":
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == "l2":
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()
    
    return loss

@torch.no_grad()
def p_sample(model, x, t, t_index, betas, sqrt_one_minus_alphas_cumprod,sqrt_recip_alphas,posterior_variance):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas,t,x.shape)

    model_mean = sqrt_recip_alphas_t * (x - betas_t*model(x, t)/sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)

    return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, shape,timesteps, betas, sqrt_one_minus_alphas_cumprod,sqrt_recip_alphas,posterior_variance):
    device = next(model.parameters()).device
    
    b = shape[0]
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, betas, sqrt_one_minus_alphas_cumprod,sqrt_recip_alphas,posterior_variance)
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(model, image_size, timesteps, betas, sqrt_one_minus_alphas_cumprod,sqrt_recip_alphas,posterior_variance,batch_size=16, channels=3):
    return p_sample_loop(model, (batch_size, channels, image_size, image_size),timesteps, betas, sqrt_one_minus_alphas_cumprod,sqrt_recip_alphas,posterior_variance)