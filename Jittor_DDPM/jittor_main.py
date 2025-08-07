import argparse
import datetime
import time
from datetime import datetime, timedelta
import sys
import logging
from pathlib import Path
import numpy as np

from noise_predict_model.UNet import Unet
from data_processing.get_data import get_fmnist_dataloader, get_cifar10_dataloader
from utils.basic_functions import *
from ddpm.denoising import *
from ddpm.diffusion import *

def train(args):
    # 实验文件夹
    experiment_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_folder = Path("./model_ckpts") / experiment_time
    ckpt_folder.mkdir(parents=True, exist_ok=True)

    # 设置日志
    log_file = ckpt_folder / "training_log.txt"

    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    logger.info(f"Experiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Checkpoints will be saved to: {ckpt_folder}")

    betas, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,posterior_variance = get_shedule(schedule_func=linear_beta_schedule, timesteps=args.timesteps)
    
    if args.dataset == "fmnist":
        dataloader,val_dataloader,image_size, channels, batch_size = get_fmnist_dataloader()
    elif args.dataset == "cifar10":
        dataloader, val_dataloader,image_size, channels, batch_size = get_cifar10_dataloader()
    else:
        print("不支持的数据集")
        return
    
    model = Unet(dim=image_size, channels=channels, dim_mults=(1,2,4),)
    #model.to(args.device)
    optimizer = jt.optim.Adam(model.parameters(), lr=1e-3)

    # 仅用于可视化
    data_for_vis = {"time_per_epoch":[],"loss_per_epoch":[],"loss_per_step":[],"val_loss_per_epoch":[],}
    
    best_val_loss = float('inf')
    total_train_time = 0
    total_eval_time = 0
    
    for epoch in range(args.epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"{'='*50}")
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0
        train_start_time = time.time()
        for step, batch in enumerate(dataloader):
            batch_start_time = time.time()
            
            optimizer.zero_grad()

            batch_size = batch["pixel_values"].shape[0]
            batch = batch["pixel_values"]

            t = jt.randint(0, args.timesteps, (batch_size,)).int32()
            loss = p_losses(model, batch, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type=args.loss_type)

            optimizer.backward(loss)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            num_train_batches += 1
            
            data_for_vis["loss_per_step"].append(loss.item())
            
            if (step % 100 == 0 and step!=0):
                batch_time = time.time() - batch_start_time
                logger.info(
                    f"Train | Epoch: {epoch+1}/{args.epochs} | "
                    f"Batch: {step}/{len(dataloader)/batch_size} | "
                    f"Loss: {loss.item():.6f} | "
                    f"Time: {batch_time:.3f}s"
                )
                
        train_time = time.time() - train_start_time
        data_for_vis["time_per_epoch"].append(train_time)
        total_train_time += train_time
        
        avg_train_loss = epoch_train_loss / num_train_batches
        data_for_vis["loss_per_epoch"].append(avg_train_loss)
        logger.info(
            f"\nTrain Summary | Epoch: {epoch+1}/{args.epochs} | "
            f"Avg Loss: {avg_train_loss:.6f} | "
            f"Time: {str(timedelta(seconds=train_time))} | "
            f"Total Train Time: {str(timedelta(seconds=total_train_time))}"
        )

        # 评估阶段
        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0
        eval_start_time = time.time()
        
        with jt.no_grad():
            for val_step, val_batch in enumerate(val_dataloader):
                batch_start_time = time.time()

                batch_size = val_batch["pixel_values"].shape[0]
                val_batch = val_batch["pixel_values"]
                
                t = jt.randint(0, args.timesteps, (batch_size,)).int32()
                val_loss = p_losses(model, val_batch, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type=args.loss_type)

                total_val_loss += val_loss.item()
                num_val_batches += 1
                
                if val_step % 35 == 0 and val_step!=0:
                    batch_time = time.time() - batch_start_time
                    logger.info(
                        f"Eval  | Epoch: {epoch+1}/{args.epochs} | "
                        f"Batch: {val_step}/{len(val_dataloader)/batch_size} | "
                        f"Loss: {val_loss.item():.6f} | "
                        f"Time: {batch_time:.3f}s"
                    )
        eval_time = time.time() - eval_start_time
        total_eval_time += eval_time
        avg_val_loss = total_val_loss / num_val_batches
        data_for_vis["val_loss_per_epoch"].append(avg_val_loss)
        logger.info(
            f"\nEval Summary  | Epoch: {epoch+1}/{args.epochs} | "
            f"Avg Loss: {avg_val_loss:.6f} | "
            f"Time: {str(timedelta(seconds=eval_time))} | "
            f"Total Eval Time: {str(timedelta(seconds=total_eval_time))}"
        )
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = ckpt_folder / f"best_model_epoch{epoch}_loss{avg_val_loss:.4f}.pkl"
            jt.save(model.state_dict(), model_path)
            logger.info(f"New best model saved to: {model_path}")

    np.savez("metrics.npz",
         val_loss_per_epoch=data_for_vis["val_loss_per_epoch"],
         time_per_epoch=data_for_vis["time_per_epoch"],
         loss_per_step=data_for_vis["loss_per_step"],
         loss_per_epoch=data_for_vis["loss_per_epoch"],)
    
    # 总结
    logger.info("\n" + "="*50)
    logger.info("Training Completed!")
    logger.info(f"Total Train Time: {str(timedelta(seconds=total_train_time))}")
    logger.info(f"Total Eval Time: {str(timedelta(seconds=total_eval_time))}")
    logger.info(f"Best Val Loss: {best_val_loss:.6f}")
    logger.info("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--loss_type', default="huber")
    #parser.add_argument('--schedule_func', default="linear")
    parser.add_argument('--mode', default="train")
    parser.add_argument('--dataset', default="fmnist")
    
    args = parser.parse_args()

    jt.flags.use_cuda = 1
    _seed_ = 0
    jt.set_global_seed(_seed_)
    #random.seed(_seed_)
    np.random.seed(_seed_)

    train(args)

