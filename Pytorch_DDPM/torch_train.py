import argparse
from pathlib import Path
from torch.optim import Adam
from torchvision.utils import save_image

from noise_predict_model.UNet import Unet
from data_processing.get_data import get_dataloader
from utils.basic_functions import *
from ddpm.denoising import *
from ddpm.diffusion import *

def train(args):
    device = args.device
    save_and_sample_every = args.save_and_sample_every
    timesteps = args.timesteps
    loss_type = args.loss_type
    torch.manual_seed(0)
    betas, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,posterior_variance = get_shedule(schedule_func=linear_beta_schedule, timesteps=timesteps)

    dataloader, image_size, channels, batch_size = get_dataloader()
    model = Unet(dim=image_size, channels=channels, dim_mults=(1,2,4),)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    results_folder = Path("./results")
    results_folder.mkdir(exist_ok=True)
    ckpt_folder = Path("./model_ckpts")
    ckpt_folder.mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            batch_size = batch["pixel_values"].shape[0]
            batch = batch["pixel_values"].to(device)

            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            loss = p_losses(model, batch, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type=loss_type)

            if step % 100 == 0:
                print("Loss:", loss.item())

            loss.backward()
            optimizer.step()

            if step != 0 and step % save_and_sample_every == 0:
                milestone = step // save_and_sample_every
                batches = num_to_groups(4, batch_size)
                all_images_list = list(map(lambda n: sample(model, batch_size=n, channels=channels), batches))
                all_images = torch.cat(all_images_list, dim=0)
                all_images = (all_images + 1) * 0.5
                save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = 6)

    torch.save(model.state_dict(), str(ckpt_folder / "final_model.pth"))
    print("Training complete! Model saved to 'results/final_model.pth'.")

#python torch_train.py --device cuda:0 --epochs 6 --timesteps 300
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save_and_sample_every', type=int, required=False,default=1000)
    #parser.add_argument('--results_folder', required=False, default="./results")
    parser.add_argument('--device', required=True, default="cuda:0")
    parser.add_argument('--epochs', type=int, required=True, default=6)
    parser.add_argument('--timesteps', type=int, required=True, default=300)
    parser.add_argument('--loss_type',  required=False, default="huber")
    parser.add_argument('--schedule_func',  required=False, default="linear")
    
    args = parser.parse_args()

    #save_and_sample_every = 1000
    train(args)
