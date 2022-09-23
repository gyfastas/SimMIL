import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision
from torchvision.datasets import ImageFolder
import os, sys
from .base_bag_dataset import BaseBagDataset

class StdBagDataset(BaseBagDataset):
    """
    A standard bag dataset that wraps a image-folder like dataset

    Args:
        pos_target: (int) target in dataset that is the positive class for bag
    """
    def __init__(self, dataset, label_file, pos_target=8):
        super().__init__(dataset, label_file)
        self.dataset = dataset
        self.label_file = label_file
        self.pos_target = pos_target
    
    def bag_label_assign(self, bag_sample):
        return int((bag_sample[:, 1].astype(np.int32) == self.pos_target).max())
