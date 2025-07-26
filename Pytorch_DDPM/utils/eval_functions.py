import torch
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import numpy as np

def calculate_fid(real_images, fake_images, device="cuda"):
    inception = inception_v3(pretrained=True, aux_logits=True).to(device)
    inception.AuxLogits = None
    inception.eval()
    
    # 预处理函数（适配Inception输入）
    def preprocess(x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # [N,1,H,W] -> [N,3,H,W]
        x = torch.nn.functional.interpolate(x, size=(299, 299), mode="bilinear")
        #x = (x - 0.5) * 2
        return x
    
    def get_features(x):
        with torch.no_grad():
            x = preprocess(x)
            x = inception(x)  # [N, 2048]（pool3特征）
        return x.cpu().numpy()
    
    real_features = get_features(real_images)
    fake_features = get_features(fake_images)
    
    # 计算均值和协方差
    mu_r, sigma_r = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_g, sigma_g = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
    
    # 计算FID
    diff = mu_r - mu_g
    covmean = sqrtm(sigma_r.dot(sigma_g))
    fid = diff.dot(diff) + np.trace(sigma_r + sigma_g - 2 * covmean.real)
    return fid