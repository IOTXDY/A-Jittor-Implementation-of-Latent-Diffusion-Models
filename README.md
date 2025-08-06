# Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models



## 1.实验环境

AutoDL镜像: Jittor  1.3.1 Python  3.8(ubuntu18.04) CUDA  11.3

GPU: RTX 3090(24GB) * 1

CPU: 14 vCPU Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz

安装其他依赖：

````
pip install numpy matplotlib datasets scipy einops tqdm torch torchvision
# 如果需要评估（计算 fid 和 is）
pip install torchmetrics torch-fidelity
````

## 2.数据集准备

**FashionMNIST:**来自 10 种类别的共 7 万个不同商品的正面图片。60000/10000 的训练测试数据划分，28x28 的灰度图片。

**Cifar-10:**包含 10 个类别的共 6 万张 RGB 彩色图片。50000/10000 的训练测试数据划分，图片的尺寸为 32×32。

P.S. 数据集的获取和处理包含在训练及评估流程中





## 3.训练


````
# pytorch
cd Pytorch_DDPM
python torch_main.py
# jittor
cd Jittor_DDPM
python jittor_main.py
````


## 4.推理

````
# pytorch
cd Pytorch_DDPM
python inference.py
# jittor
cd Jittor_DDPM
python inference.py
````

## 5.评估

````
# pytorch
cd Pytorch_DDPM
python metrics.py --dataset cifar10 --cate fid
python metrics.py --dataset fmnist --cate fid
python metrics.py --dataset cifar10 --cate is
python metrics.py --dataset fmnist --cate is
# jittor
cd Jittor_DDPM
python metrics.py --dataset cifar10 --cate fid
python metrics.py --dataset fmnist --cate fid
python metrics.py --dataset cifar10 --cate is
python metrics.py --dataset fmnist --cate is
````

## 6.实验结果

### 6.1日志

（1000步采样，训练40epoch）

pytorch: [on Fashion-MNIST](https://cg.cs.tsinghua.edu.cn/jittor/)、[on CIFAR10](https://cg.cs.tsinghua.edu.cn/jittor/)

jittor: [on Fashion-MNIST](https://cg.cs.tsinghua.edu.cn/jittor/)、[on CIFAR10](https://cg.cs.tsinghua.edu.cn/jittor/)

### 6.2训练损失曲线


![GitHub Logo](https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models/blob/main/Assets/cifar_train.png)


![GitHub Logo](https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models/blob/main/Assets/fmnist_train.png)

### 6.3验证损失曲线

![GitHub Logo](https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models/blob/main/Assets/cifar_val.png)


![GitHub Logo](https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models/blob/main/Assets/fmnist_val.png)

### 6.4训练用时

![GitHub Logo](https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models/blob/main/Assets/cifar_time.png)


![GitHub Logo](https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models/blob/main/Assets/fmnist_time.png)

### 6.5生成效果

#### 6.5.1灰度图



| 框架   | 采样1w张用时（秒） |
|--------|------|
| pytorch   | fmnist   |
| jittor   | cifar10   |

#### 6.5.2彩色图

| 框架   | 采样1w张用时（秒） |
|--------|------|
| pytorch   | 2189.36   |
| jittor   | 4941.18   |

![GitHub Logo](https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models/blob/main/Assets/j_cifar_1.png)

### 6.6评估结果

**FID（Frechet Inception Distance）:**衡量生成图片分布与真实图片分布的距离，数值越小，生成质量越高。

| 框架   | 数据集 | 得分     |
|--------|------|----------|
| pytorch   | fmnist   | 0   |
| pytorch   | cifar10   | 19.05   |
| jittor   | fmnist   | 0   |
| jittor   | cifar10   | 38.96   |


**IS（Inception Score）:**衡量生成图片的质量和多样性，数值越高，生成效果越好。

| 框架   | 训练数据集 | 得分     |
|--------|------|----------|
| pytorch   | fmnist   | 0   |
| pytorch   | cifar10   | 3.95 ± 0.08   |
| jittor   | fmnist   | 0   |
| jittor   | cifar10   | 0   |

## 7.参考

[The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)

