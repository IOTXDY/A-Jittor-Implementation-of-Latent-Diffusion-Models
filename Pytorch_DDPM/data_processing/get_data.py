from datasets import load_dataset
from torchvision import transforms
#from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torch.utils.data import DataLoader
import os

def fmnist_transforms(examples):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ])
    examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
    del examples["image"]

    return examples

def get_fmnist_dataloader():
    dataset = load_dataset("fashion_mnist")

    # fashion_mnist的参数
    image_size = 28
    channels = 1
    batch_size = 128

    transformed_dataset = dataset.with_transform(fmnist_transforms).remove_columns("label")
    dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)
    testdataloader = DataLoader(transformed_dataset["test"], batch_size=batch_size, shuffle=True)
    return dataloader, testdataloader, image_size, channels, batch_size

def cifar10_transforms(examples):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [0,1] -> [-1,1]
    ])
    examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["img"]]  # CIFAR-10的键是"img"
    del examples["img"]
    return examples

def get_cifar10_dataloader():
    #os.environ['http_proxy'] = 'http://127.0.0.1:7890'
    #os.environ['https_proxy'] = 'http://127.0.0.1:7890'
    dataset = load_dataset("cifar10")

    # CIFAR-10的参数
    image_size = 32
    channels = 3
    batch_size = 128

    transformed_dataset = dataset.with_transform(cifar10_transforms).remove_columns("label")
    
    dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)
    testdataloader = DataLoader(transformed_dataset["test"], batch_size=batch_size, shuffle=True)
    return dataloader, testdataloader, image_size, channels, batch_size



