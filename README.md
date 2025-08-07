# Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models



## 1.实验环境

AutoDL镜像: Jittor  1.3.1 Python  3.8(ubuntu18.04) CUDA  11.3

GPU: RTX 3090(24GB) * 1

CPU: 14 vCPU Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz

安装其他依赖：

````
pip install numpy matplotlib datasets scipy einops tqdm torch torchvision cupy
# 如果需要评估（计算 fid 和 is）
pip install torchmetrics torch-fidelity
````

## 2.数据准备

Fashion-MNIST:来自 10 种类别共 7 万个不同商品的正面图片。60000/10000 的训练测试数据划分，28x28 灰度图片。

Cifar-10:包含 10 个类别的共 6 万张 RGB 彩色图片。50000/10000 的训练测试数据划分，图片的尺寸为 32×32。

P.S. 数据集的获取和处理包含在训练及评估流程中


## 3.目录结构

```
Pytorch-DDPM/
├── data_processing/
│   └── get_data.py # 获取数据，预处理
├── ddpm/
│   └── __init__.py
│   └── denoising.py # 去噪（采样）过程
│   └── diffusion.py # 扩散过程
├── noise_predict_model/
│   └── UNet.py # 噪声预测网络
├── utils/
│   └── __init__.py
│   └── basic_functions.py # 一些辅助函数
│   └── network_helpers.py # 网络基本模块
├── inference.py # 采样
├── torch_main.py # 训练
└── metrics.py # 评估

```
(Jittor版本的结构完全对应)

## 4.训练



| 参数名       | 类型      | 默认值     | 描述                     |
|-------------|----------|-----------|--------------------------|
| device     | str | 'cuda:0'         | 设备     |
| epochs       | int    | 40       | 训练轮数            |
| timesteps   | int   | 1000   | 采样时间步         |
| loss_type   | str   | 'huber'   | 损失函数         |
| dataset   | str   | 'fmnist'   | 训练数据集         |

````
# pytorch
cd Pytorch_DDPM
python torch_main.py
# jittor
cd Jittor_DDPM 
python jittor_main.py
````


## 5.推理

| 参数名       | 类型      | 默认值     | 描述                     |
|-------------|----------|-----------|--------------------------|
| device     | str | 'cuda:0'         | 设备     |
| save_gif   | bool   | True  | 是否保存gif         |
| dataset   | str   | 'fmnist'   | 对应训练数据集         |


````
# pytorch
cd Pytorch_DDPM
python inference.py
# jittor
cd Jittor_DDPM
python inference.py
````

## 6.评估

| 参数名       | 类型      | 默认值     | 描述                     |
|-------------|----------|-----------|--------------------------|
| device     | str | 'cuda:0'         | 设备     |
| model_path   | str   | -  | 模型ckpt路径         |
| dataset   | str   | 'fmnist'   | 对应训练数据集         |
| cate   | str   | 'fid'   | 评估指标         |

````
# pytorch
cd Pytorch_DDPM
python metrics.py --model_path "your_model.pth"
# jittor
cd Jittor_DDPM
python metrics.py --model_path "your_model.pkl"
````

## 7.实验结果

（1000步采样，训练40epochs）

### 7.1日志

**pytorch:** [on Fashion-MNIST](https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models/blob/main/Assets/170605.txt)
、[on CIFAR10](https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models/blob/main/Assets/162759.txt)

**jittor:** [on Fashion-MNIST](https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models/blob/main/Assets/125737.txt)
、[on CIFAR10](https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models/blob/main/Assets/114506.txt)

### 7.2训练损失曲线


![GitHub Logo](https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models/blob/main/Assets/cifar_train.png)


![GitHub Logo](https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models/blob/main/Assets/fmnist_train.png)

### 7.3验证损失曲线

![GitHub Logo](https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models/blob/main/Assets/cifar_val.png)


![GitHub Logo](https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models/blob/main/Assets/fmnist_val.png)

### 7.4训练用时

![GitHub Logo](https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models/blob/main/Assets/cifar_time.png)


![GitHub Logo](https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models/blob/main/Assets/fmnist_time.png)

### 7.5生成效果

#### 7.5.1灰度图

| 框架   | 生成1万张用时（秒） |
|--------|------|
| pytorch   | 2015.97   |
| jittor   | 3641.01   |

Jittor生成效果：

![GitHub Logo](https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models/blob/main/Assets/j_fmni_1.png)

Jittor采样过程：

![GitHub Logo](https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models/blob/main/Assets/j_fmni_1.gif)

Pytorch生成效果：

![GitHub Logo](https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models/blob/main/Assets/t_fmni_1.png)

Pytorch采样过程：

![GitHub Logo](https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models/blob/main/Assets/t_fmni_1.gif)

#### 7.5.2彩色图

| 框架   | 生成1万张用时（秒） |
|--------|------|
| pytorch   | 2189.36   |
| jittor   | 4941.18   |

Jittor生成效果：

![GitHub Logo](https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models/blob/main/Assets/j_cifar_1.png)

Jittor采样过程：

![GitHub Logo](https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models/blob/main/Assets/j_cifar_1.gif)

Pytorch生成效果：

![GitHub Logo](https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models/blob/main/Assets/t_cifar_1.png)

Pytorch采样过程：

![GitHub Logo](https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models/blob/main/Assets/t_cifar_1.gif)



### 7.6评估结果

FID（Frechet Inception Distance）:衡量生成图片分布与真实图片分布的距离，数值越小，生成质量越高。

| 框架   | 数据集 | FID     |
|--------|------|----------|
| pytorch   | fmnist   | 16.14   |
| pytorch   | cifar10   | 19.05   |
| jittor   | fmnist   | 14.94   |
| jittor   | cifar10   | 38.96   |


IS（Inception Score）:衡量生成图片的质量和多样性，数值越高，生成效果越好。

| 框架   | 训练数据集 | IS     |
|--------|------|----------|
| pytorch   | fmnist   | 4.24 ± 0.12   |
| pytorch   | cifar10   | 3.95 ± 0.08   |
| jittor   | fmnist   | 4.42 ± 0.08   |
| jittor   | cifar10   | 3.29 ± 0.11   |

## 8.参考

[Denoising Diffusion Probabilistic Models](https://proceedings.neurips.cc/paper_files/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)

[The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)

