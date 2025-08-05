from datasets import load_dataset
#from torchvision import transforms
#from torch.utils.data import DataLoader
import os
import jittor as jt
from jittor.dataset import Dataset
from PIL import Image
import numpy as np

def fmnist_transforms(examples):
    """ transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ])
    examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]] """
    transformed_images = []
    for image in examples["image"]:
        img = image.convert("L")
        if np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
        #(H,W) -> (1,H,W)
        arr = np.expand_dims(arr, axis=0)
        transformed_images.append(arr)

    examples["pixel_values"] = transformed_images
    del examples["image"]
    return examples

class FashionMNISTDataset(Dataset):
    def __init__(self, hf_dataset):
        super().__init__()
        self.dataset = hf_dataset
        self.total_len = len(hf_dataset)
        self.set_attrs(batch_size=128, shuffle=True)
    
    def __getitem__(self, index):
        item = self.dataset[index]
        x = jt.array(item["pixel_values"])
        return {"pixel_values": x}
        #return x, 0  # 返回0作为伪标签，保持接口统一
    
    def __len__(self):
        return self.total_len

def get_fmnist_dataloader():
    dataset = load_dataset("fashion_mnist")

    # fashion_mnist的参数
    image_size = 28
    channels = 1
    batch_size = 128

    transformed_dataset = dataset.with_transform(fmnist_transforms).remove_columns("label")
    
    dataloader = FashionMNISTDataset(transformed_dataset["train"])
    
    testdataloader = FashionMNISTDataset(transformed_dataset["test"])
    #dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)
    return dataloader,testdataloader, image_size, channels, batch_size

def cifar10_transforms(examples):
    #transform = transforms.Compose([
    #    transforms.RandomHorizontalFlip(),
    #    transforms.ToTensor(),
    #    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [0,1] -> [-1,1]
    #])
    #examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["img"]]  # CIFAR-10的键是"img"
    transformed_images = []
    for img in examples["img"]:
        img = img.convert("RGB")
        
        if np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        arr = np.array(img, dtype=np.float32) / 255.0
        
        arr = (arr - 0.5) / 0.5  # mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]
        
        # HWC -> CHW
        arr = arr.transpose(2, 0, 1)
        
        transformed_images.append(arr)
    
    examples["pixel_values"] = transformed_images
    del examples["img"]
    return examples

def get_cifar10_dataloader():
    dataset = load_dataset("cifar10")

    # CIFAR-10的参数
    image_size = 32
    channels = 3
    batch_size = 128

    transformed_dataset = dataset.with_transform(cifar10_transforms).remove_columns("label")
    dataloader = FashionMNISTDataset(transformed_dataset["train"])
    testdataloader = FashionMNISTDataset(transformed_dataset["test"])
    #dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)
    return dataloader,testdataloader, image_size, channels, batch_size



