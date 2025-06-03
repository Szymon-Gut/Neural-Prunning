import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import numpy as np


class CIFARDatasetFromArrays(Dataset):
    def __init__(self, images, labels, transform=None , encoder=None):
        self.images = images
        self.transform = transform
        self.labels = labels
        self.encoder = encoder
        if self.encoder is not None:
            self.labels = self.encoder.transform(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.labels[idx])

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
    
def get_cifar_dataloader(pkl_path: str, batch_size: int = 32, num_workers: int = 4, augment: bool = True):
    """
    Get the ImageNet dataloader.

    Args:
        pkl_path (str): Path to the Cifar dataset directory.
        batch_size (int): Batch size for the dataloader.
        num_workers (int): Number of workers for data loading.
        augment (bool): Whether to apply data augmentation.

    Returns:
        torch.utils.data.DataLoader: Dataloader for the ImageNet dataset.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    test_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.99, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    le = LabelEncoder()
    le.fit(y_train)
    train_dataset = CIFARDatasetFromArrays(X_train, y_train, transform=train_transform, encoder = le)
    test_dataset = CIFARDatasetFromArrays(X_test, y_test, transform=test_transform, encoder = le)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
