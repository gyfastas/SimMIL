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
from PIL import Image
import os
from scipy.io import loadmat

# ImageFile.LOAD_TRUNCATED_IMAGES = True

shuffle_index = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
    [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
    [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
    ]

random.seed(1)

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
    def __init__(self, root, patch_size, transform, floder=0, ratio=0.8, train=False):
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
        bag_idx_lists = [x[0][0] for x in batch]
        ins_idx_lists = [x[0][1] for x in batch]
        bag_lists1 = [x[1][0] for x in batch]
        bag_lists2 = [x[1][1] for x in batch]
        targets = [x[2] for x in batch]
        batches = []
        for i in range(len(batch)):
            batches.extend([i] * len(ins_idx_lists[i]))

        batches = torch.tensor(batches).long()
        return torch.cat(bag_idx_lists), torch.cat(ins_idx_lists), (torch.cat(bag_lists1), torch.cat(bag_lists2)), torch.cat(targets), batches

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
        ins_idx_list = torch.tensor([]).long()
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
                ins_idx_list = torch.cat((ins_idx_list, torch.tensor([idx])))
        # batch_idx = torch.ones_like(idx_list)*index
        return (torch.tensor([index]), ins_idx_list), (bag_list1, bag_list2), torch.tensor([target]).long()

    def __len__(self):
        return len(self.folder)

class MIL_CRC(ImageFolder):
    def __init__(self, root, transform, floder=0, ratio=0.8, train=False):
        super(MIL_CRC, self).__init__(root)
        self.transform = transform
        self.train = train
        self.bags_list, self.labels_list, self.labels_list_real,\
            self.labels_list_frombag = self._form_bags()
        self._shuffle_CV(floder, ratio, train)

    def _form_bags(self):
        bags_list, ins_list, labels_list, labels_list_real, labels_list_frombag = [], [], [], [], []
        bag_name = []
        ins_list_label_real = []
        index_for_bag = []
        cnt = 0

        for i in self.imgs:
            ins_path = i[0]
            bag_path, ins_name = os.path.split(ins_path)
            if 'epithelial' in ins_name:
                ins_label_real = 0
            elif 'fibroblast' in ins_name:
                ins_label_real = 1
            elif 'inflammatory' in ins_name:
                ins_label_real = 2
            else:
                ins_label_real = 3
            if not(bag_path in bag_name):
                bag_name.append(bag_path)
                if len(ins_list) != 0:
                    bags_list.append(ins_list)
                    labels_list.append(ins_label_frombag)
                    if ins_label_frombag == 0:
                        labels_list_frombag.append([0]*(len(ins_list)))
                    else:
                        labels_list_frombag.append([1]*(len(ins_list)))
                    labels_list_real.append(ins_list_label_real)
                    ins_list = []
                    ins_list_label_real = []
                ins_list.append(ins_path)
                ins_list_label_real.append(ins_label_real)
                ins_label_frombag = i[1]
            else:
                ins_list.append(ins_path)
                ins_list_label_real.append(ins_label_real)
        bags_list.append(ins_list)
        labels_list.append(ins_label_frombag)
        if ins_label_frombag == 0:
            labels_list_frombag.append([0]*(len(ins_list)))
        else:
            labels_list_frombag.append([1]*(len(ins_list)))
        labels_list_real.append(ins_list_label_real)

        return bags_list, labels_list, labels_list_real, labels_list_frombag

    def _shuffle_CV(self, folder, ratio, train):
        random.seed(1)
        index = [i for i in range(100)]
        random.shuffle(index)
        bags_list = np.array(self.bags_list)[index]
        labels_list = np.array(self.labels_list)[index]
        labels_list_real = np.array(self.labels_list_real)[index]
        labels_list_frombag = np.array(self.labels_list_frombag)[index]
        
        # random.shuffle(self.bags_list)
        shuffle_idx = shuffle_index[folder]
        chosen_list = []
        if train:
            chosen_idx = shuffle_idx[:int(ratio * len(shuffle_idx))]
        else:
            chosen_idx = shuffle_idx[int(ratio * len(shuffle_idx)):]            
        # for i in range(self.classes):
            # tmp_class_sample = self.bags_list[i * self.classes_len:(i + 1) * self.classes_len]
            # chosen_list += [tmp_class_sample[k] for k in chosen_idx]
        self.folder_img = [bags_list[k] for k in chosen_idx]
        self.folder_label = [labels_list[k] for k in chosen_idx]
        self.folder_inslabel_real = [labels_list_real[k] for k in chosen_idx]
        self.folder_inslabel_frombag = [labels_list_frombag[k] for k in chosen_idx]

    def collate_fn(self, batch):
        bag_idx_lists = [x[0].long() for x in batch]
        # ins_idx_lists = [x[0][1].long() for x in batch]
        bag_lists1 = [x[1][0] for x in batch]
        bag_lists2 = [x[1][1] for x in batch]
        bag_label = [x[2][0] for x in batch]
        ins_label_real = [x[2][1] for x in batch]
        ins_label_frombag = [x[2][2] for x in batch]
        batches = []
        for i in range(len(batch)):
            batches.extend([i] * len(ins_label_real[i]))
        batches = torch.tensor(batches).long()
        bag_label = torch.torch.tensor(bag_label).long()
        ins_label_real = torch.tensor(np.concatenate(np.array(ins_label_real))).long()
        ins_label_frombag = torch.tensor(np.concatenate(np.array(ins_label_frombag))).long()
        return torch.cat(bag_idx_lists), (torch.cat(bag_lists1), torch.cat(bag_lists2)),\
             (bag_label, ins_label_real, ins_label_frombag), batches

    def __getitem__(self, index):

        path = self.folder_img[index]
        target = [self.folder_label[index], self.folder_inslabel_real[index], self.folder_inslabel_frombag[index]]

        # img = np.array(Image.open(path))

        bag_list1 = torch.tensor([])
        bag_list2 = torch.tensor([])
        ins_idx_list = torch.tensor([]).long()
        # cnt = 0

        for i in path:
            # cnt+=1
            ins_sample = Image.open(i)
            if self.transform is not None:
                ins_sample = self.transform(ins_sample)
            bag_list1 = torch.cat((bag_list1, ins_sample[0].unsqueeze(0)))
            bag_list2 = torch.cat((bag_list2, ins_sample[1].unsqueeze(0)))
            # ins_idx_list = torch.cat((ins_idx_list, torch.tensor([cnt])))
        return torch.tensor([index]), (bag_list1, bag_list2), target

    def __len__(self):
        return len(self.folder_img)

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

class CifarBags(data_utils.Dataset):
    def __init__(self, target_number=1, mean_bag_length=10, var_bag_length=2, num_bag=250, seed=1, train=False, transform=None):
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train

        self.r = np.random.RandomState(seed)

        self.num_in_train = 50000
        self.num_in_test = 10000
        self.transform = transform
        if self.train:
            self.train_bags_list, \
            self.train_labels_list, self.train_labels_list_real, \
            self.train_labels_frombag, self.train_bag_index_mb = self._form_bags()
        else:
            self.test_bags_list, \
            self.test_labels_list, self.test_labels_list_real,\
            self.test_labels_frombag, self.test_bag_index_mb = self._form_bags()

    @staticmethod
    def _test(target_numbers, labels_in_bag):
        ins_labels = []
        if type(target_numbers) is int:
            target_numbers = [target_numbers]
        for i in range(len(labels_in_bag)):
            if labels_in_bag[i] in target_numbers:
                ins_labels.append(torch.tensor([1]))
            else:
                ins_labels.append(torch.tensor([0]))
        return torch.cat(ins_labels)

    def _form_bags(self):
        if self.train:
            bags_list = []
            labels_list = []
            labels_list_real = []
            labels_list_frombag = []
            index_for_bag = []
            valid_bags_counter = 0
            label_of_last_bag = 0
            cnt = 0
            dataset = datasets.CIFAR10('../datasets',
                                  train=True,
                                  download=True,)
            numbers = dataset.data
            labels = torch.tensor(dataset.targets)
            while valid_bags_counter < self.num_bag:
                bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
                if bag_length <= 1:
                    bag_length = 2
                indices = torch.LongTensor(self.r.randint(0, self.num_in_train, bag_length))
                labels_in_bag = labels[indices]
                ins_labels = self._test(self.target_number, labels_in_bag)
                if (ins_labels.sum().item()>0) and (label_of_last_bag == 0):
                    # real label, the same label as bag, index for mb
                    labels_list_real.append(labels_in_bag)
                    labels_list_frombag.append(torch.ones_like((labels_in_bag)))
                    index_for_bag.append(torch.arange(cnt, cnt+len((labels_in_bag))))
                    cnt += len(labels_in_bag)

                    labels_in_bag = ins_labels
                    labels_list.append(labels_in_bag)
                    bags_list.append(numbers[indices])
                    label_of_last_bag = 1
                    valid_bags_counter += 1
                elif label_of_last_bag == 1:
                    index_list = []
                    bag_length_counter = 0
                    while bag_length_counter < bag_length:
                        index = torch.LongTensor(self.r.randint(0, self.num_in_train, 1))
                        label_temp = labels[index]
                        # if label_temp.numpy()[0] != self.target_number:
                        if label_temp not in self.target_number:
                            index_list.append(index)
                            bag_length_counter += 1

                    index_list = np.array(index_list)
                    labels_in_bag = labels[index_list]
                    labels_list_real.append(labels_in_bag)
                    labels_list_frombag.append(torch.zeros_like(labels_in_bag))
                    index_for_bag.append(torch.arange(cnt, cnt + len((labels_in_bag))))
                    cnt += len(labels_in_bag)
                    labels_in_bag = self._test(self.target_number, labels_in_bag)
                    labels_list.append(labels_in_bag)
                    bags_list.append(numbers[index_list])
                    label_of_last_bag = 0
                    valid_bags_counter += 1
                else:
                    pass

        else:
            bags_list = []
            labels_list = []
            labels_list_real = []
            labels_list_frombag = []
            index_for_bag = []
            valid_bags_counter = 0
            label_of_last_bag = 0
            cnt = 0
            dataset = datasets.CIFAR10('../datasets',
                                   train=False,
                                   download=True,)
            numbers = dataset.data
            labels = torch.tensor(dataset.targets)
            while valid_bags_counter < self.num_bag:
                bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
                if bag_length <= 1:
                    bag_length = 2
                indices = torch.LongTensor(self.r.randint(0, self.num_in_test, bag_length))
                labels_in_bag = labels[indices]
                ins_labels = self._test(self.target_number, labels_in_bag)
                if (ins_labels.sum().item()>0) and (label_of_last_bag == 0):
                    labels_list_real.append(labels_in_bag)
                    labels_list_frombag.append(torch.ones_like(labels_in_bag))
                    index_for_bag.append(torch.arange(cnt, cnt + len((labels_in_bag))))
                    cnt += len(labels_in_bag)
                    labels_in_bag = ins_labels
                    labels_list.append(labels_in_bag)
                    bags_list.append(numbers[indices])
                    label_of_last_bag = 1
                    valid_bags_counter += 1
                elif label_of_last_bag == 1:
                    index_list = []
                    bag_length_counter = 0
                    while bag_length_counter < bag_length:
                        index = torch.LongTensor(self.r.randint(0, self.num_in_test, 1))
                        label_temp = labels[index]
                        # if label_temp.numpy()[0] != self.target_number:
                        if label_temp not in self.target_number:
                            index_list.append(index)
                            bag_length_counter += 1
                    index_list = np.array(index_list)
                    labels_in_bag = labels[index_list]
                    labels_list_real.append(labels_in_bag)
                    labels_list_frombag.append(torch.zeros_like(labels_in_bag))
                    index_for_bag.append(torch.arange(cnt, cnt + len((labels_in_bag))))
                    cnt += len(labels_in_bag)

                    labels_in_bag = self._test(self.target_number, labels_in_bag)
                    labels_list.append(labels_in_bag)
                    bags_list.append(numbers[index_list])
                    label_of_last_bag = 0
                    valid_bags_counter += 1
                else:
                    pass

        return bags_list, labels_list, labels_list_real, labels_list_frombag, index_for_bag

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        bag_list = torch.tensor([])
        bag_list_aug0 = torch.tensor([])
        bag_list_aug1 = torch.tensor([])
        bag_list_aug2 = torch.tensor([])
        if self.train:
            bag = self.train_bags_list[index]
            label = [max(self.train_labels_list[index]), self.train_labels_list[index], self.train_labels_frombag[index]]
            index_list = self.train_bag_index_mb[index]
        else:
            bag = self.test_bags_list[index]
            label = [max(self.test_labels_list[index]), self.test_labels_list[index], self.test_labels_frombag[index]]
            index_list = self.test_bag_index_mb[index]
        for i in range(len(bag)):
            img = Image.fromarray(bag[i])
            if self.transform:
                img = self.transform(img)
                if self.train:
                    bag_list_aug0 = torch.cat((bag_list_aug0, img[0].unsqueeze(0)))
                    bag_list_aug1 = torch.cat((bag_list_aug1, img[1].unsqueeze(0)))
                    bag_list_aug2 = torch.cat((bag_list_aug2, img[2].unsqueeze(0)))
                else:
                    bag_list = torch.cat((bag_list, img.unsqueeze(0)))
        if self.train:
            bag_list = (bag_list_aug0, bag_list_aug1, bag_list_aug2)
        return bag_list, label, index_list

    # rotated_imgs = [
    #     self.transform(img0),
    #     self.transform(rotate_img(img0, 90)),
    #     self.transform(rotate_img(img0, 180)),
    #     self.transform(rotate_img(img0, 270))
    # ]
    # rotation_labels = torch.LongTensor([0, 1, 2, 3])
    # return torch.stack(rotated_imgs, dim=0), rotation_labels

    def collate_fn(self, batch):
        bag_label_lists = [x[1][0].long() for x in batch]
        ins_label_lists = [x[1][1].long() for x in batch]
        ins_fb_label_lists = [x[1][2].long() for x in batch]
        index_list = [x[2] for x in batch]
        batches = []
        for i in range(len(batch)):
            batches.extend([i] * len(ins_label_lists[i]))

        batches = torch.tensor(batches).long()
        if self.train:
            bag_lists_aug0 = [x[0][0] for x in batch]
            bag_lists_aug1 = [x[0][1] for x in batch]
            bag_lists_aug2 = [x[0][2] for x in batch]
            return (torch.cat(bag_lists_aug0), torch.cat(bag_lists_aug1), torch.cat(bag_lists_aug2)),\
                   torch.stack(bag_label_lists), torch.cat(ins_label_lists), torch.cat(ins_fb_label_lists), \
                   batches, torch.cat(index_list)
        else:
            bag_lists = [x[0] for x in batch]
            return torch.cat(bag_lists), torch.stack(bag_label_lists), torch.cat(ins_label_lists), batches, torch.cat(index_list)

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