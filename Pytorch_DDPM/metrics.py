from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from tqdm import tqdm
import time
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datasets import load_dataset
from torchvision import transforms

from noise_predict_model.UNet import Unet
from ddpm.denoising import sample
from ddpm.diffusion import *

def eval_fid_dynamic(args):
    num_fake = 10000  # 需要生成的图片数量
    batch_size = 100
    device = args.device
    timesteps = 1000

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # 加载真实图片的统计量
    print("Processing real images...")
    if args.dataset == "cifar10":
        image_size = 32
        channels = 3
        
        dataset = load_dataset("cifar10")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [0,1] -> [-1,1]
        ])

        # 分批处理真实图片
        for i in tqdm(range(0, len(dataset["test"]), batch_size)):
            batch = dataset["test"][i:i+batch_size]["img"]
            real_batch = torch.stack([transform(img.convert("RGB")) for img in batch])
            fid.update(real_batch.to(device), real=True)
            
    elif args.dataset == "fmnist":
        image_size = 28
        channels = 1
        
        dataset = load_dataset("fashion_mnist")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) 
            #transforms.Lambda(lambda t: (t * 2) - 1),
        ])
        # 分批处理真实图片
        for i in tqdm(range(0, len(dataset["test"]), batch_size)):
            batch = dataset["test"][i:i+batch_size]["image"]
            real_batch = torch.stack([transform(img.convert("L")) for img in batch])
            if real_batch.size(1) == 1:
                real_batch = real_batch.repeat(1, 3, 1, 1)
            fid.update(real_batch.to(device), real=True)

    print("Generating fake images on-the-fly...")
    model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4))
    model.load_state_dict(torch.load('model_ckpts/20250803_170605/best_model_epoch32_loss0.0161.pth'))
    model.to(args.device)
    model.eval()

    betas, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,posterior_variance = get_shedule(schedule_func=linear_beta_schedule, timesteps=timesteps)
    
    
    # 生成循环
    total_time = 0
    generated_count = 0
    while generated_count < num_fake:
        with torch.no_grad():
            batch_start_time = time.time()
            samples = sample(
                model, image_size, timesteps, betas, 
                sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas,
                posterior_variance, batch_size=batch_size, 
                channels=channels
            )
            batch_time = time.time() - batch_start_time
            print(batch_time)
            total_time += batch_time
            # samples[-1] 是最终生成的图片（形状 [batch, C, H, W]）
            fake_batch = torch.from_numpy(samples[-1]).float().to(device)
            fake_batch = torch.clamp(fake_batch, -1, 1)
            if fake_batch.size(1) == 1:
                fake_batch = fake_batch.repeat(1, 3, 1, 1)
            fid.update(fake_batch, real=False)
            
            generated_count += batch_size

    # 计算最终FID
    fid_score = fid.compute()
    print(f"FID Score: {fid_score:.2f}")
    print(f"Time:{total_time:.3f}s")
    return fid_score

def eval_is_dynamic(args):
    num_fake = 10000  # 需要生成的图片数量
    batch_size = 100
    device = args.device

    if args.dataset == "cifar10":
        image_size = 32
        channels = 3
    elif args.dataset == "fmnist":
        image_size = 28
        channels = 1
    timesteps = 1000

    inception = InceptionScore(normalize=True).to(device)

    print("Generating fake images on-the-fly for IS calculation...")
    model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4))
    model.load_state_dict(torch.load(args.model_path))
    #model.load_state_dict(torch.load('model_ckpts/20250803_162759/best_model_epoch32_loss0.0152.pth'))
    model.to(args.device)
    model.eval()

    betas, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance = get_shedule(schedule_func=linear_beta_schedule, timesteps=timesteps)
    
    # 生成循环
    total_time = 0
    generated_count = 0
    while generated_count < num_fake:
        with torch.no_grad():
            batch_start_time = time.time()
            samples = sample(
                model, image_size, timesteps, betas, 
                sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas,
                posterior_variance, batch_size=batch_size, 
                channels=channels
            )
            batch_time = time.time() - batch_start_time
            print(batch_time)
            total_time += batch_time
            # samples[-1] 是最终生成的图片（形状 [batch, C, H, W]）
            fake_batch = torch.from_numpy(samples[-1]).float().to(device)
            
            fake_batch = torch.clamp(fake_batch, -1, 1)
            
            if fake_batch.size(1) == 1:
                fake_batch = fake_batch.repeat(1, 3, 1, 1)
            
            inception.update(fake_batch)
            
            generated_count += batch_size

    # 计算最终IS
    is_mean, is_std = inception.compute()
    print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")
    print(f"Time:{total_time:.3f}s")
    return is_mean

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--batch_size', type=int, default=64)
    #parser.add_argument('--save_and_sample_every', type=int, required=False,default=1000)
    #parser.add_argument('--results_folder', required=False, default="./results")
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--dataset', default="fmnist")
    parser.add_argument('--model_path', default="model_ckpts/20250803_170605/best_model_epoch32_loss0.0161.pth")
    parser.add_argument('--cate', default="fid")
    args = parser.parse_args()
    if args.cate == "fid":
        eval_fid_dynamic(args)
    elif args.cate == "is":
        eval_is_dynamic(args)
    else:
        raise NotImplementedError()