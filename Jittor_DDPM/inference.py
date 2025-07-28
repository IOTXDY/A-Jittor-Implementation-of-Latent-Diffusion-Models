import argparse
#import torch
import jittor as jt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

from noise_predict_model.UNet import Unet
from ddpm.denoising import sample
from ddpm.diffusion import *

def sample_fmnist(args):
    # settings for fashion mnist
    image_size = 28
    channels = 1
    timesteps = 300

    # output path
    samples_folder = Path("./samples")
    samples_folder.mkdir(exist_ok=True)
    gifs_folder = Path("./sampling_gifs")
    gifs_folder.mkdir(exist_ok=True)

    # load denoising model
    model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4))
    model.load('model_ckpts/fmnist_e6.pkl')
    model.eval()

    # get shedule varibles for sampling
    betas, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,posterior_variance = get_shedule(schedule_func=linear_beta_schedule, timesteps=timesteps)

    # final result
    samples = sample(model, image_size, timesteps, betas, sqrt_one_minus_alphas_cumprod,sqrt_recip_alphas,posterior_variance,batch_size=16, channels=channels)
    random_index = 5
    plt.imshow(samples[-1][random_index].reshape(image_size, image_size, channels), cmap="gray")
    existing_files = list(samples_folder.glob("generated_sample_*.png"))
    next_num = len(existing_files) + 1
    plt.savefig(str(samples_folder /f"generated_sample_{next_num}.png"), bbox_inches='tight', pad_inches=0) 

    # gif
    if args.save_gif:
        fig = plt.figure()
        ims = []
        for i in range(timesteps):
            im = plt.imshow(samples[i][random_index].reshape(image_size, image_size, channels), cmap="gray", animated=True)
            ims.append([im])
        animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
        existing_files = list(gifs_folder.glob("diffusion_*.gif"))
        next_num = len(existing_files) + 1
        animate.save(str(gifs_folder /f"diffusion_{next_num}.gif"), savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0})
        #plt.show()

def sample_cifar10(args):
    image_size = 32
    channels = 3
    timesteps = 900
    # output path
    samples_folder = Path("./cifar10_samples")
    samples_folder.mkdir(exist_ok=True)
    gifs_folder = Path("./cifar10_sampling_gifs")
    gifs_folder.mkdir(exist_ok=True)

    # load denoising model
    model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4))
    model.load('model_ckpts/c_e10_t900.pkl')
    model.eval()

    # get shedule varibles for sampling
    betas, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,posterior_variance = get_shedule(schedule_func=linear_beta_schedule, timesteps=timesteps)

    # final result
    samples = sample(model, image_size, timesteps, betas, sqrt_one_minus_alphas_cumprod,sqrt_recip_alphas,posterior_variance,batch_size=16, channels=channels)
    random_index = 5
    plt.imshow((samples[-1][random_index].clip(-1, 1).reshape(image_size, image_size, channels)*0.5 + 0.5))
    #plt.imshow(samples[-1][random_index].reshape(image_size, image_size, channels), cmap="gray")
    existing_files = list(samples_folder.glob("generated_sample_*.png"))
    next_num = len(existing_files) + 1
    plt.savefig(str(samples_folder /f"generated_sample_{next_num}.png"), bbox_inches='tight', pad_inches=0) 

    # gif
    if args.save_gif:
        fig = plt.figure()
        ims = []
        for i in range(timesteps):
            img = samples[i][random_index].clip(-1, 1).reshape(image_size, image_size, channels)*0.5 + 0.5
            #im = plt.imshow(samples[i][random_index].reshape(image_size, image_size, channels), cmap="gray", animated=True)
            im = plt.imshow(img, animated=True) 
            ims.append([im])
        animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
        existing_files = list(gifs_folder.glob("diffusion_*.gif"))
        next_num = len(existing_files) + 1
        animate.save(str(gifs_folder /f"diffusion_{next_num}.gif"), savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0})

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="fmnist")
    parser.add_argument('--save_gif', default=True)
    args = parser.parse_args()
    
    jt.flags.use_cuda = 1
    #torch.manual_seed(0)

    if args.dataset == "fmnist":
        sample_fmnist(args)
    elif args.dataset == "cifar10":
        sample_cifar10(args)
    else:
        raise NotImplementedError()