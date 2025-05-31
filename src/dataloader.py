import os
import numpy as np
import torch
from torchvision import datasets, transforms
def get_imagenet_dataloader(data_dir: str, batch_size: int = 32, num_workers: int = 4, augment: bool = True):
    """
    Get the ImageNet dataloader.

    Args:
        data_dir (str): Path to the ImageNet dataset directory with specified subset eg. dataset/train.
        batch_size (int): Batch size for the dataloader.
        num_workers (int): Number of workers for data loading.
        transform (transforms.Compose): Transformations to apply to the images.
        augment (bool): Whether to apply data augmentation.

    Returns:
        torch.utils.data.DataLoader: Dataloader for the ImageNet dataset.
    """
    if augment:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader
