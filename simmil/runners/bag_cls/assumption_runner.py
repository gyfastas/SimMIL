import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from simmil.datasets import StdBagDataset, CoBagDataset, CountBagDataset, ImageFolderDict
from .base_runner import BaseRunner

class AssumptionRunner(BaseRunner):
    def build_data_loaders(self):
        self.logger.info("Building datasets")
        if self.args.assumption=="std":
            train_dataset = StdBagDataset(dataset=ImageFolderDict(os.path.join(self.args.data, self.args.train_subdir),
                                    transform=self.train_augmentation),
                                        label_file=self.args.train_label_file,
                                        pos_target=self.args.pos_targets)
            
            val_dataset =  StdBagDataset(dataset=ImageFolderDict(os.path.join(self.args.data, self.args.eval_subdir),
                                        transform=self.val_augmentation),
                                        label_file=self.args.val_label_file,
                                        pos_target=self.args.pos_targets)
        elif self.args.assumption=="count":
            train_dataset = CountBagDataset(dataset=ImageFolderDict(os.path.join(self.args.data, self.args.train_subdir),
                                    transform=self.train_augmentation),
                                        label_file=self.args.train_label_file,
                                        pos_target=self.args.pos_targets,
                                        pos_ratio=self.args.pos_ratio,
                                        max_pos_ratio=self.args.max_pos_ratio)
            
            val_dataset =  CountBagDataset(dataset=ImageFolderDict(os.path.join(self.args.data, self.args.eval_subdir),
                                        transform=self.val_augmentation),
                                        label_file=self.args.val_label_file,
                                        pos_target=self.args.pos_targets,
                                        pos_ratio=self.args.pos_ratio,
                                        max_pos_ratio=self.args.max_pos_ratio)
        elif self.args.assumption=="co":
            train_dataset = CoBagDataset(dataset=ImageFolderDict(os.path.join(self.args.data, self.args.train_subdir),
                                    transform=self.train_augmentation),
                                        label_file=self.args.train_label_file,
                                        pos_target=self.args.pos_targets)
            
            val_dataset =  CoBagDataset(dataset=ImageFolderDict(os.path.join(self.args.data, self.args.eval_subdir),
                                        transform=self.val_augmentation),
                                        label_file=self.args.val_label_file,
                                        pos_target=self.args.pos_targets)
        else:
            train_dataset = StdBagDataset(dataset=ImageFolderDict(os.path.join(self.args.data, self.args.train_subdir),
                                    transform=self.train_augmentation),
                                        label_file=self.args.train_label_file,
                                        pos_target=self.args.pos_targets)
            
            val_dataset =  StdBagDataset(dataset=ImageFolderDict(os.path.join(self.args.data, self.args.eval_subdir),
                                        transform=self.val_augmentation),
                                        label_file=self.args.val_label_file,
                                        pos_target=self.args.pos_targets)


        bag_labels = [train_dataset.bag_label_assign(sample) for sample in train_dataset.bag_samples]
        self.logger.info("number of pos bags in train dataset: {}".format(sum(bag_labels)))
        self.logger.info("number of neg bagas in train dataset: {}".format(len(bag_labels) - sum(bag_labels)))
        bag_labels = [train_dataset.bag_label_assign(sample) for sample in val_dataset.bag_samples]
        self.logger.info("number of pos bags in val dataset: {}".format(sum(bag_labels)))
        self.logger.info("number of neg bagas in val dataset: {}".format(len(bag_labels) - sum(bag_labels)))
        
        train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=self.args.batch_size,
        shuffle=True,
        num_workers=self.args.workers,
        drop_last=True,
        pin_memory=True,
    )

        val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=self.args.batch_size,
        shuffle=False,
        num_workers=self.args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)
        return train_loader, val_loader