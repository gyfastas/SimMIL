import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision
from torchvision.datasets import ImageFolder

class BaseBagDataset(Dataset):
    def __init__(self, dataset, label_file):
        self.dataset = dataset
        self.label_file = label_file
        self._scan()

    def gen_ins_samples(self):
        with open(self.label_file, 'r') as f:
            samples = [x.strip().split(" ") for x in f.readlines()]
        
        samples = list(map(lambda x:(x[0], x[1], x[2]), samples))
        return samples

    def _scan(self):
        # ins-oriented 
        self.ins_samples = np.array(self.gen_ins_samples())
        # bag-oriented
        self.bag_samples = [self.ins_samples[np.where(self.ins_samples[:,-1]==k)] for k in np.unique(self.ins_samples[:,-1])]
    
    def bag_label_assign(self, bag_sample):
        raise NotImplementedError

    def __len__(self):
        return len(self.bag_samples)

    def __getitem__(self, idx):
        samples = self.bag_samples[idx]
        img_dirs, label = samples[:, 0], self.bag_label_assign(samples)
        imgs = [self.dataset.loader(img_dir) for img_dir in img_dirs]
        if self.dataset.transform is not None:
            imgs = [self.dataset.transform(sample) for sample in imgs]
        
        return {"data": torch.stack(imgs, 0), "label": label}
