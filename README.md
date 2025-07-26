# Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models

参考：[The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)

## 1.配置环境

````
git clone https://github.com/IOTXDY/Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models.git
cd Pytorch-and-Jittor-Implementations-of-Denoising-Diffusion-Probabilistic-Models
conda env create -f environment.yml
conda activate ddpm
````

## 2.训练

### pytorch:

````
cd Pytorch_DDPM
python torch_main.py
````

### Jittor:

````
cd Jittor_DDPM
python jittor_main.py
````

## 3.推理

### pytorch:

````
cd Pytorch_DDPM
python inference.py
````

### Jittor:

````
cd Jittor_DDPM
python inference.py
````


# 4.评估

### pytorch:

````
cd Pytorch_DDPM
python torch_main.py --mode fid
````

### Jittor:

````
cd Jittor_DDPM
python jittor_main.py --mode fid
````
