import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision
from torchvision.datasets import ImageFolder
import os, sys
from .base_bag_dataset import BaseBagDataset

class CountBagDataset(BaseBagDataset):
    """
    A CountMIL bag dataset that wraps a image-folder like dataset

    Args:
        pos_target: (int) target in dataset that is the positive class for bag
        pos_ratio: (float) only when pos target num in a bag > pos_ratio * bag_length, the
        bag is seen as positive bag
    """
    def __init__(self, dataset, label_file, pos_target=8, pos_ratio=0.2, max_pos_ratio=1.0):
        super().__init__(dataset, label_file)
        self.dataset = dataset
        self.label_file = label_file
        self.pos_target = pos_target
        self.pos_ratio = pos_ratio
        self.max_pos_ratio = max_pos_ratio
    
    def bag_label_assign(self, bag_sample):
        condition_min = (bag_sample[:, 1].astype(np.int32) == self.pos_target).sum() > self.pos_ratio * len(bag_sample)
        condition_max = (bag_sample[:, 1].astype(np.int32) == self.pos_target).sum() <= self.max_pos_ratio * len(bag_sample)
        return int(condition_max and condition_min)

