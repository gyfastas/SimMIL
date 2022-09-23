import os, sys
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from PIL import ImageFilter
import random
from PIL import Image
from PIL import ImageFile
from torch.utils.data import DataLoader, Dataset
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

class SelectedCamelyon16(Dataset):
    def __init__(self, images, labels, augmentation):
        self.images = images
        self.labels = labels
        self.augmentation = augmentation

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        if self.augmentation is not None:
            image = self.augmentation(image)

        return {"data":image, "label": self.labels[idx]}


