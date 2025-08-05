# Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models

参考：[The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)

## 配置环境

````
git clone https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models.git
cd Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models
# pytorch
conda env create -f environment.yml
conda activate ddpm
# jittor
python -m pip install jittor
python -m jittor.test.test_core
python -m jittor.test.test_example
python -m jittor.test.test_cudnn_op
````

## 数据集准备

**FashionMNIST:**来自 10 种类别的共 7 万个不同商品的正面图片。60000/10000 的训练测试数据划分，28x28 的灰度图片。

**Cifar-10:**包含 10 个类别的共 6 万张 RGB 彩色图片。50000/10000 的训练测试数据划分，图片的尺寸为 32×32。

P.S. 数据集的获取和处理包含在训练及评估流程中

## 训练


````
# pytorch
cd Pytorch_DDPM
python torch_main.py
# jittor
cd Jittor_DDPM
python jittor_main.py
````


## 推理

````
# pytorch
cd Pytorch_DDPM
python inference.py
# jittor
cd Jittor_DDPM
python inference.py
````

## 评估

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

## 实验结果

### 训练日志

（1000步采样，训练40epoch）

pytorch: [fmnist](https://cg.cs.tsinghua.edu.cn/jittor/)、[cifar10](https://cg.cs.tsinghua.edu.cn/jittor/)

jittor: [fmnist](https://cg.cs.tsinghua.edu.cn/jittor/)、[cifar10](https://cg.cs.tsinghua.edu.cn/jittor/)

### Loss曲线

pytorch:

![GitHub Logo](https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png "GitHub")

jittor:

![GitHub Logo](https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png "GitHub")

### 生成效果

**灰度图：**

| 框架   | 采样1w张用时（秒） |
|--------|------|
| pytorch   | fmnist   |
| jittor   | cifar10   |

**彩色图：**

| 框架   | 采样1w张用时（秒） |
|--------|------|
| pytorch   | 2189.36   |
| jittor   | 4941.18   |

### 评估结果

**FID（Frechet Inception Distance）:**衡量生成图片分布与真实图片分布的距离，数值越小，生成质量越高。

| 框架   | 数据集 | 得分     |
|--------|------|----------|
| pytorch   | fmnist   | 0   |
| pytorch   | cifar10   | 19.05   |
| jittor   | fmnist   | 0   |
| jittor   | cifar10   | 38.96   |


**IS（Inception Score）:**衡量生成图片的质量和多样性，数值越高，生成效果越好。

| 框架   | 数据集 | 得分     |
|--------|------|----------|
| pytorch   | fmnist   | 0   |
| pytorch   | cifar10   | 3.95 ± 0.08   |
| jittor   | fmnist   | 0   |
| jittor   | cifar10   | 0   |

