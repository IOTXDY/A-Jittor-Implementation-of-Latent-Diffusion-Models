from datasets import load_dataset
from torchvision import transforms
#from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torch.utils.data import DataLoader
import os
#os.environ['http_proxy'] = 'http://127.0.0.1:7890'
#os.environ['https_proxy'] = 'http://127.0.0.1:7890'

def transformss(examples):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ])
    examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
    del examples["image"]

    return examples

def get_dataloader():
    dataset = load_dataset("fashion_mnist")

    image_size = 28
    channels = 1
    batch_size = 128

    transformed_dataset = dataset.with_transform(transformss).remove_columns("label")
    dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)
    return dataloader, image_size, channels, batch_size