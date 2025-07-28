import argparse
from pathlib import Path
#import torch
#from torch.optim import Adam
#from torchvision.utils import save_image

from noise_predict_model.UNet import Unet
from data_processing.get_data import get_fmnist_dataloader, get_cifar10_dataloader
from utils.basic_functions import *
#from utils.eval_functions import *
from ddpm.denoising import *
from ddpm.diffusion import *

def train(args):
    #torch.manual_seed(0)
    
    betas, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,posterior_variance = get_shedule(schedule_func=linear_beta_schedule, timesteps=args.timesteps)
    
    if args.dataset == "fmnist":
        dataloader, image_size, channels, batch_size = get_fmnist_dataloader()
    elif args.dataset == "cifar10":
        dataloader, image_size, channels, batch_size = get_cifar10_dataloader()
    else:
        print("不支持的数据集")
        return
    
    model = Unet(dim=image_size, channels=channels, dim_mults=(1,2,4),)
    #model.to(args.device)
    optimizer = jt.optim.Adam(model.parameters(), lr=1e-3)

    results_folder = Path("./results")
    results_folder.mkdir(exist_ok=True)
    ckpt_folder = Path("./model_ckpts")
    ckpt_folder.mkdir(exist_ok=True)
    
    for epoch in range(args.epochs):
        print("Epoch:", epoch)
        for step, batch in enumerate(dataloader):
            #print(step)
            optimizer.zero_grad()

            batch_size = batch["pixel_values"].shape[0]
            batch = batch["pixel_values"]

            t = jt.randint(0, args.timesteps, (batch_size,)).int32()
            loss = p_losses(model, batch, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type=args.loss_type)

            if step % 100 == 0:
                print("Loss:", loss.item())

            #loss.backward()
            optimizer.backward(loss)
            optimizer.step()

            """ if step != 0 and step % save_and_sample_every == 0:
                milestone = step // save_and_sample_every
                batches = num_to_groups(4, batch_size)
                all_images_list = list(map(lambda n: sample(model, batch_size=n, channels=channels), batches))
                all_images = torch.cat(all_images_list, dim=0)
                all_images = (all_images + 1) * 0.5
                save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = 6) """

    #torch.save(model.state_dict(), str(ckpt_folder / "cifar_e6.pth"))
    model.save(str(ckpt_folder / "c_e10_t900.pkl"))
    print("Training complete! Model saved to 'model_ckpts/'.")

def eval_fid(args):
    from datasets import load_dataset
    from torchvision import transforms
    import os
    from PIL import Image

    dataset = load_dataset("fashion_mnist")
    real_images = torch.stack([
        transforms.ToTensor()(dataset["test"][i]["image"].convert("L")) * 2 - 1 
        for i in range(2)  # 只取10张
    ])

    fake_images = []
    for i in range(1, 3):
        img_path = os.path.join("samples", f"generated_sample_{i}.png")
        img = Image.open(img_path).convert("L")
        img_tensor = transforms.ToTensor()(img) * 2 - 1
        fake_images.append(img_tensor)
        
    fake_images = torch.stack(fake_images)  # [10, 1, H, W]

    fid_score = calculate_fid(real_images.to(args.device), fake_images.to(args.device))
    print(f"FID Score: {fid_score:.2f}")

#python torch_train.py --device cuda:0 --epochs 6 --timesteps 300
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--batch_size', type=int, default=64)
    #parser.add_argument('--save_and_sample_every', type=int, required=False,default=1000)
    #parser.add_argument('--results_folder', required=False, default="./results")
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--timesteps', type=int, default=900)
    parser.add_argument('--loss_type', default="huber")
    parser.add_argument('--schedule_func', default="linear")
    parser.add_argument('--mode', default="train")
    parser.add_argument('--dataset', default="fmnist")
    
    args = parser.parse_args()

    #torch.manual_seed(0)
    jt.flags.use_cuda = 1
    jt.set_global_seed(0)

    if args.mode == "train":
        train(args)
    elif args.mode == "fid":
        eval_fid(args)
    else:
        raise NotImplementedError()
