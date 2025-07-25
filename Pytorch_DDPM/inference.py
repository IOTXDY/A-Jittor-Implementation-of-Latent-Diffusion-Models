import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

from noise_predict_model.UNet import Unet
from ddpm.denoising import sample
from ddpm.diffusion import *

if __name__ == '__main__':
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
    model.load_state_dict(torch.load('model_ckpts/final_model_e6.pth'))
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