"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from openslide.deepzoom import DeepZoomGenerator
import openslide
from PIL import ImageFilter
import random
from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
shuffle_index = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
    [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
    [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]]
class HISMIL(ImageFolder):
    def __init__(self, root, patch_size, augmentation, floder=0, ratio=0.8, train=False):
        super(HISMIL, self).__init__(root)
        self.patch_size = patch_size
        self.classes = 4
        self.classes_len = 100
        self.transform = transforms.Compose(augmentation)
        # self._save_patches()
        self.folder = self._shuffle_CV(floder, ratio, train)

    # def _save_patches(self):
    #     import os
    #     for path, target in self.samples:
    #         slide = openslide.open_slide(path)
    #         dz = DeepZoomGenerator(slide, self.patch_size, 0, True)
    #         x_idx, y_dx = dz.level_tiles[dz.level_count - 1]
    #         if self.train:
    #             patch_path = os.path.join('/home/tclin/Downloads/ICIAR2018_BACH_Challenge/datasets/patch/train', path.split('/')[7])
    #         else:
    #             patch_path = os.path.join('/home/tclin/Downloads/ICIAR2018_BACH_Challenge/datasets/patch/test',
    #                                       path.split('/')[7])
    #         if not os.path.exists(patch_path):
    #             os.makedirs(patch_path)
    #         # bag_list = torch.tensor([])
    #         for i in range(x_idx - 1):
    #             for j in range(y_dx - 1):
    #                 sample = dz.get_tile(dz.level_count - 1, (i, j))
    #
    #                 patch_name = path.split('/')[-1].split('.')[0]+ '_' + str(i) + '_' + str(j)+'.png'
    #                 # patch_path = os.path.join(path.split('.')[0], str(i) + '_' + str(j)+'.png')
    #                 sample.save(os.path.join(patch_path,patch_name))
    def _shuffle_CV(self, floder, ratio, train):
        shuffle_idx = shuffle_index[floder]
        chosen_list=[]
        if train:
            chosen_idx = shuffle_idx[:int(ratio*len(shuffle_idx))]
        else:
            chosen_idx = shuffle_idx[int(ratio * len(shuffle_idx)):]
        for i in range(self.classes):
            tmp_class_sample = self.samples[i*self.classes_len:(i+1)*self.classes_len]
            chosen_list += [tmp_class_sample[k] for k in chosen_idx]
        return chosen_list

    def collate_fn(self, batch):
        idx_lists = [x[0] for x in batch]
        bag_lists = [x[1] for x in batch]
        targets  = [x[2] for x in batch]
        batches = []
        for i in range(len(batch)):
            batches.extend([i]* len(idx_lists[i]))
        
        batches = torch.tensor(batches).long()
        return torch.cat(idx_lists), torch.cat(bag_lists), torch.cat(targets), batches



    def __getitem__(self, index):
        path, target = self.folder[index]
        # if target >1:
        #     target = torch.tensor([1]).long()
        # else:
        #     target = torch.tensor([0]).long()
        # sample = self.loader(path)
        slide = openslide.open_slide(path)
        dz = DeepZoomGenerator(slide, self.patch_size, 0, True)
        x_idx, y_idx = dz.level_tiles[dz.level_count-1]
        bag_list = torch.tensor([])
        idx_list = torch.tensor([]).long()
        cnt = 0
        idx_init = index*x_idx*y_idx
        for i in range(x_idx):
            for j in range(y_idx):
                sample = dz.get_tile(dz.level_count-1, (i, j))
                idx = idx_init + cnt
                cnt += 1
                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                bag_list = torch.cat((bag_list, sample.unsqueeze(0)))
                idx_list = torch.cat((idx_list, torch.tensor([idx])))
        # batch_idx = torch.ones_like(idx_list)*index
        return idx_list, bag_list, torch.tensor([target]).long()
    def __len__(self):
        return len(self.folder)


class HISMIL_DoubleAug(ImageFolder):
    def __init__(self, root, patch_size, transform, floder=0, ratio=0.7, train=False):
        super(HISMIL_DoubleAug, self).__init__(root)
        self.patch_size = patch_size
        self.classes = 4
        self.classes_len = 100
        self.transform = transform
        # self._save_patches()
        self.folder = self._shuffle_CV(floder, ratio, train)

    # def _save_patches(self):
    #     import os
    #     for path, target in self.samples:
    #         slide = openslide.open_slide(path)
    #         dz = DeepZoomGenerator(slide, self.patch_size, 0, True)
    #         x_idx, y_dx = dz.level_tiles[dz.level_count - 1]
    #         if self.train:
    #             patch_path = os.path.join('/home/tclin/Downloads/ICIAR2018_BACH_Challenge/datasets/patch/train', path.split('/')[7])
    #         else:
    #             patch_path = os.path.join('/home/tclin/Downloads/ICIAR2018_BACH_Challenge/datasets/patch/test',
    #                                       path.split('/')[7])
    #         if not os.path.exists(patch_path):
    #             os.makedirs(patch_path)
    #         # bag_list = torch.tensor([])
    #         for i in range(x_idx - 1):
    #             for j in range(y_dx - 1):
    #                 sample = dz.get_tile(dz.level_count - 1, (i, j))
    #
    #                 patch_name = path.split('/')[-1].split('.')[0]+ '_' + str(i) + '_' + str(j)+'.png'
    #                 # patch_path = os.path.join(path.split('.')[0], str(i) + '_' + str(j)+'.png')
    #                 sample.save(os.path.join(patch_path,patch_name))
    def _shuffle_CV(self, floder, ratio, train):
        shuffle_idx = shuffle_index[floder]
        chosen_list = []
        if train:
            chosen_idx = shuffle_idx[:int(ratio * len(shuffle_idx))]
        else:
            chosen_idx = shuffle_idx[int(ratio * len(shuffle_idx)):]
        for i in range(self.classes):
            tmp_class_sample = self.samples[i * self.classes_len:(i + 1) * self.classes_len]
            chosen_list += [tmp_class_sample[k] for k in chosen_idx]
        return chosen_list

    def collate_fn(self, batch):
        idx_lists = [x[0] for x in batch]
        bag_lists1 = [x[1][0] for x in batch]
        bag_lists2 = [x[1][1] for x in batch]
        targets = [x[2] for x in batch]
        batches = []
        for i in range(len(batch)):
            batches.extend([i] * len(idx_lists[i]))

        batches = torch.tensor(batches).long()
        return torch.cat(idx_lists), (torch.cat(bag_lists1), torch.cat(bag_lists2)), torch.cat(targets), batches

    def __getitem__(self, index):
        path, target = self.folder[index]
        # if target >1:
        #     target = torch.tensor([1]).long()
        # else:
        #     target = torch.tensor([0]).long()
        # sample = self.loader(path)
        slide = openslide.open_slide(path)
        dz = DeepZoomGenerator(slide, self.patch_size, 0, True)
        x_idx, y_idx = dz.level_tiles[dz.level_count - 1]
        bag_list1 = torch.tensor([])
        bag_list2 = torch.tensor([])
        idx_list = torch.tensor([]).long()
        cnt = 0
        idx_init = index * x_idx * y_idx
        for i in range(x_idx):
            for j in range(y_idx):
                sample = dz.get_tile(dz.level_count - 1, (i, j))
                idx = idx_init + cnt
                cnt += 1
                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                bag_list1 = torch.cat((bag_list1, sample[0].unsqueeze(0)))
                bag_list2 = torch.cat((bag_list2, sample[1].unsqueeze(0)))
                idx_list = torch.cat((idx_list, torch.tensor([idx])))
        # batch_idx = torch.ones_like(idx_list)*index
        return idx_list, (bag_list1, bag_list2), torch.tensor([target]).long()

    def __len__(self):
        return len(self.folder)

class SimpleMIL(ImageFolder):
    def __init__(self, root, train=True):
        super(SimpleMIL, self).__init__(root)
        self.train=train
        if train:
            augmentation = [
                # transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                # transforms.RandomApply([
                #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                # ], p=0.8),
                # transforms.RandomGrayscale(p=0.2),
                # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                # transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]
        else:
            augmentation = [
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        self.transform = transforms.Compose(augmentation)
    def __getitem__(self, index):
        image_data = list(self.dataset.__getitem__(index))
        # important to return the index!
        data = [index] + image_data
        return tuple(data)

class GaussianBlur(object):
        """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

        def __init__(self, sigma=[.1, 2.]):
            self.sigma = sigma

        def __call__(self, x):
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
            return x

class MnistBags(data_utils.Dataset):
    def __init__(self, target_number=9, mean_bag_length=10, var_bag_length=2, num_bag=250, seed=1, train=False):
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train

        self.r = np.random.RandomState(seed)

        self.num_in_train = 60000
        self.num_in_test = 10000

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        if self.train:
            loader = data_utils.DataLoader(datasets.MNIST('../datasets',
                                                          train=True,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_train,
                                           shuffle=False)
        else:
            loader = data_utils.DataLoader(datasets.MNIST('../datasets',
                                                          train=False,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_test,
                                           shuffle=False)

        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        bags_list = []
        labels_list = []

        for i in range(self.num_bag):
            bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            if bag_length < 1:
                bag_length = 1

            if self.train:
                indices = torch.LongTensor(self.r.randint(0, self.num_in_train, bag_length))
            else:
                indices = torch.LongTensor(self.r.randint(0, self.num_in_test, bag_length))

            labels_in_bag = all_labels[indices]
            labels_in_bag = labels_in_bag == self.target_number

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)

        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = [max(self.train_labels_list[index]), self.train_labels_list[index]]
        else:
            bag = self.test_bags_list[index]
            label = [max(self.test_labels_list[index]), self.test_labels_list[index]]

        return bag, label

if __name__ == "__main__":
    train_path = '/home/tclin/Documents/ICIAR/WSI/train'
    aug_train = [
        # aug_plus
        # transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        # transforms.RandomApply([
        #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        # ], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        # transforms.RandomHorizontalFlip(),
        # aug
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        # lincls
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])]
    aug_test = [transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    HISMIL(train_path, 256, aug_train)