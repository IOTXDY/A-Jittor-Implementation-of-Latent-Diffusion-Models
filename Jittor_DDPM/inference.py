import argparse
#import torch
import jittor as jt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from pathlib import Path
import numpy as np

from noise_predict_model.UNet import Unet
from ddpm.denoising import sample
from ddpm.diffusion import *

def save_sampling_gif_grid(samples, filename, image_size, channels, fps=10, n_frames=100,is_gray=True):
    frame_indices = np.linspace(0, len(samples)-1, min(n_frames, len(samples)), dtype=int)
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    plt.subplots_adjust(wspace=0, hspace=0)
    
    def update(frame_idx):
        for i, ax in enumerate(axes.flat):
            ax.clear()
            if is_gray:
                ax.imshow(samples[frame_idx][i].reshape(image_size, image_size, channels), cmap='gray')
            else:
                img = samples[frame_idx][i]
                img = np.clip(img, -1, 1)                 
                img = img * 0.5 + 0.5                    
                img = np.transpose(img, (1, 2, 0))        
                ax.imshow(img)
            ax.axis('off')
        fig.suptitle(f'Step {frame_idx}/{len(samples)-1}', y=0.92)
    
    anim = FuncAnimation(fig, update, frames=frame_indices, interval=1000/fps)
    anim.save(filename, writer='pillow', fps=fps, dpi=100)
    plt.close()

def sample_fmnist(args):
    image_size = 28
    channels = 1
    timesteps = 1000

    samples_folder = Path("./samples")
    samples_folder.mkdir(exist_ok=True)
    gifs_folder = Path("./sampling_gifs")
    gifs_folder.mkdir(exist_ok=True)

    model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4))
    model.load('model_ckpts/20250803_125737/best_model_epoch32_loss0.0172.pkl')
    model.eval()

    betas, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,posterior_variance = get_shedule(schedule_func=linear_beta_schedule, timesteps=timesteps)

    samples = sample(model, image_size, timesteps, betas, sqrt_one_minus_alphas_cumprod,sqrt_recip_alphas,posterior_variance,batch_size=16, channels=channels)
    
    # 4x4 grid
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    plt.subplots_adjust(wspace=0, hspace=0.02) 
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[-1][i].reshape(image_size, image_size, channels), cmap="gray")
        ax.axis('off')
    existing_files = list(samples_folder.glob("generated_sample_*.png"))
    next_num = len(existing_files) + 1
    plt.savefig(str(samples_folder /f"generated_sample_{next_num}.png"), bbox_inches='tight', pad_inches=0) 

    # gif
    if args.save_gif:
        existing_gifs = list(gifs_folder.glob("sampling_process_*.gif"))
        next_gif_num = len(existing_gifs) + 1
        save_sampling_gif_grid(samples, str(gifs_folder / f"sampling_process_{next_gif_num}.gif"), image_size, channels)

def sample_cifar10(args):
    image_size = 32
    channels = 3
    timesteps = 1000
    samples_folder = Path("./cifar10_samples")
    samples_folder.mkdir(exist_ok=True)
    gifs_folder = Path("./cifar10_sampling_gifs")
    gifs_folder.mkdir(exist_ok=True)

    model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4))
    model.load('model_ckpts/20250803_114506/best_model_epoch39_loss0.0156.pkl')
    model.eval()

    betas, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,posterior_variance = get_shedule(schedule_func=linear_beta_schedule, timesteps=timesteps)

    samples = sample(model, image_size, timesteps, betas, sqrt_one_minus_alphas_cumprod,sqrt_recip_alphas,posterior_variance,batch_size=16, channels=channels)
    # 4x4 grid
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    plt.subplots_adjust(wspace=0, hspace=0.02) 
    for i, ax in enumerate(axes.flat):
        img = samples[-1][i]
        img = np.clip(img, -1, 1)              
        img = img * 0.5 + 0.5                     
        img = np.transpose(img, (1, 2, 0))       
        
        ax.imshow(img)
        ax.axis('off')
    
    existing_files = list(samples_folder.glob("generated_sample_*.png"))
    next_num = len(existing_files) + 1
    plt.savefig(str(samples_folder /f"generated_sample_{next_num}.png"), bbox_inches='tight', pad_inches=0) 

    # gif
    if args.save_gif:
        existing_gifs = list(gifs_folder.glob("sampling_process_*.gif"))
        next_gif_num = len(existing_gifs) + 1
        save_sampling_gif_grid(samples, str(gifs_folder / f"sampling_process_{next_gif_num}.gif"), image_size, channels,is_gray=False)
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="fmnist")
    parser.add_argument('--save_gif', default=True)
    args = parser.parse_args()
    
    jt.flags.use_cuda = 1

    if args.dataset == "fmnist":
        sample_fmnist(args)
    elif args.dataset == "cifar10":
        sample_cifar10(args)
    else:
        raise NotImplementedError()