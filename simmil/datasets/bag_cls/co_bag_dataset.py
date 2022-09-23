import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision
from torchvision.datasets import ImageFolder
import os, sys
from .base_bag_dataset import BaseBagDataset

class CoBagDataset(BaseBagDataset):
    """
    A Co-occurance bag dataset that wraps a image-folder like dataset

    Args:
        pos_target: [list(int)] target in dataset that are the positive class for bag. The bag is positive if
        these target appear together
        pos_ratio: (float) only when pos target num in a bag > pos_ratio * bag_length, the
        bag is seen as positive bag
    """
    def __init__(self, dataset, label_file, pos_target=8):
        super().__init__(dataset, label_file)
        self.dataset = dataset
        self.label_file = label_file
        self.pos_target = pos_target
    
    def bag_label_assign(self, bag_sample):
        ins_labels = bag_sample[:, 1].astype(np.int32)
        bag_label = np.array([(ins_labels==x).max() for x in self.pos_target])
        return int(bag_label.min())

