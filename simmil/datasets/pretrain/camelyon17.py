import os, sys
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from openslide.deepzoom import DeepZoomGenerator
import openslide
from PIL import ImageFilter
import random
from PIL import Image
from PIL import ImageFile
from torch.utils.data import DataLoader, Dataset
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

class CameLyon17(Dataset):
    """
    Folder organization:
        root/
            class1/
                bag1/
                    p1.png
                    p2.png
                    ...
                bag2/
                    ...
                ...
            class2/
                ...
    3. changing `cls_label_dict` might change the order of the bag!!! Be careful!
        If you find `bag_lengths` and the bag lens stored in memory bank are not the same,
        check if you changed it.
        
    Args:
        root (str): the root directory of the data
        ins_transform (torchvision.transforms): transforms on instance
        label_transform (torchvision.transforms): transforms on label
        cls_label_dict (dict): key-value pair of class name and its encoded label number.
                        (list of dict): you can also pass a list of dict, which enable multi-label.
        use_indexs: (bool) set True to return bag index and inner index for each instance.
        
        partial_class (list) a list of class name that should be balanced with others.
            For example, partial_class = ['neg'], cls_label_dict = {'pos1':1, 'pos2':1, 'neg':0}
            then sample number of 'neg' would be randomly eliminated
        
        getter: (callable) [optional] a post getter to handle the output of __getitem__()
    """
    def __init__(self, root, ins_transform=None, label_transform=None,
                 cls_label_dict=None, use_indexs=True, bag_name_list=None,
                 partial_class=None, getter=None):
        self.root = root
        self.cls_label_dict = cls_label_dict
        self.class_path_list = self.init_cls_path(cls_label_dict)
        self.ins_transform = ins_transform
        self.label_transform = label_transform
        self.use_indexs = use_indexs
        self.bag_name_list = bag_name_list
        self.partial_class = partial_class
        self.getter = getter
        self.bag_names = []
        self.bag_paths = []
        self.bag_labels = []
        self.bag_lengths = []
        self.instance_labels = []
        ##instance_infos: [bag index, inner index, nodule ratios]
        self.instance_infos = []
        self.instance_paths = []
        self._scan()

    def _scan(self):
        if self.partial_class is None:
            class_path_list = self.class_path_list
            partial_cls_path_list = None
        else:
            class_path_list = [os.path.join(self.root, x) for x in self.cls_label_dict.keys() if x not in self.partial_class]
            partial_cls_path_list = [os.path.join(self.root, x) for x in self.partial_class]
        self._scan_classes(class_path_list)
        self._scan_partial(partial_cls_path_list)

    def _scan_classes(self, class_path_list):
        bag_idx = 0
        for class_path in class_path_list:
            class_folder = class_path.rsplit("/", 1)[-1]
            for bag_dir in os.listdir(class_path):
                ## if bag name list is pre-defined, use it.
                if self.bag_name_list is not None:
                    if bag_dir not in self.bag_name_list:
                        continue
                self.bag_names.append(bag_dir)
                self.bag_paths.append(os.path.join(class_path, bag_dir))
                label = self.assign_bag_label(class_folder)
                self.bag_labels.append(label)
                inner_idx = 0
                for instance_file in os.listdir(os.path.join(class_path, bag_dir)):
                    if is_image_file(os.path.join(class_path, bag_dir, instance_file)):
                        self.instance_infos.append([bag_idx, inner_idx])
                        self.instance_paths.append(os.path.join(
                            class_path, bag_dir, instance_file))
                        self.instance_labels.append(label)
                        inner_idx += 1
                self.bag_lengths.append(inner_idx)
                bag_idx += 1

    def _scan_partial(self, partial_cls_path_list):
        bag_idx = len(self.bag_labels)
        ## partial class
        if partial_cls_path_list:
            cur_bag_nums = len(self.bag_labels)
            cur_ins_nums = len(self.instance_labels)
            avg_ins_nums = cur_ins_nums / cur_bag_nums
            for class_path in partial_cls_path_list:
                class_folder = class_path.rsplit("/", 1)[-1]
                bag_names = []
                bag_paths = []
                bag_labels = []
                bag_lengths = []
                instance_infos = []
                instance_labels = []
                instance_paths = []
                ## get instance
                for bag_dir in os.listdir(class_path):
                    bag_names.append(bag_dir)
                    bag_paths.append(os.path.join(class_path, bag_dir))
                    label = self.assign_bag_label(class_folder)
                    bag_labels.append(label)
                    inner_idx = 0
                    for instance_file in os.listdir(os.path.join(class_path, bag_dir)):
                        if is_image_file(os.path.join(class_path, bag_dir, instance_file)):
                            instance_infos.append([bag_idx, inner_idx])
                            instance_paths.append(os.path.join(
                                class_path, bag_dir, instance_file))
                            instance_labels.append(label)
                            inner_idx += 1
                            if (inner_idx+1) > avg_ins_nums: 
                                break
                    bag_lengths.append(inner_idx)
                    if len(instance_paths) > cur_ins_nums:
                        break
                    bag_idx += 1
                ## merge list
                self.bag_names.extend(bag_names)
                self.bag_paths.extend(bag_paths)
                self.bag_labels.extend(bag_labels)
                self.bag_lengths.extend(bag_lengths)
                self.instance_infos.extend(instance_infos)
                self.instance_paths.extend(instance_paths)
                self.instance_labels.extend(instance_labels)

    def assign_bag_label(self, class_folder):
        """
        Get the bag lebel from self.cls_label_dict if given.
        If not, we use the default setting (easy to understand).
        """
        ##single-label
        if isinstance(self.cls_label_dict, dict):
            return self.cls_label_dict[class_folder]
        ##multi-label
        elif isinstance(self.cls_label_dict, list):
            return [x[class_folder] for x in self.cls_label_dict]
        else:
            raise Exception("The class folder is incorrect!")
    
    def init_cls_path(self, cls_label_dict):
        """
        Class paths are sub-folders in the root. Folder name is
        the class name.
        If multi-label enabled, use the order of first class-label pair.
        """
        if isinstance(cls_label_dict, dict):
            return_list = []
            for key, value in cls_label_dict.items():
                return_list.append(os.path.join(self.root, key))
            return return_list
        elif isinstance(cls_label_dict, list):
            return_list = []
            for key, value in cls_label_dict[0].items():
                return_list.append(os.path.join(self.root, key))
            return return_list
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        """
        Return:
            img: (?) an instance
            label: (int) bag label
            bag_idx: (int) the bag index
            inner_idx: (int) inner index of current instance
            nodule_ratio: (float) the nodule ratio of current instance.
        """
        img_dir = self.instance_paths[idx]
        bag_idx, inner_idx = self.instance_infos[idx][0], self.instance_infos[idx][1]
        img = Image.open(img_dir).convert('RGB')
        label = self.instance_labels[idx]
        
        if callable(self.ins_transform):
            img = self.ins_transform(img)
        
        if callable(self.label_transform):
            label = self.label_transform(label)
        
        if self.use_indexs:
            return self.post_get({"data":img, "label":label, "bag_idx": bag_idx, "inner_idx": inner_idx})
        else:
            return self.post_get({"data": img, "label": label})
    

    def post_get(self, data_dict):
        if self.getter is not None:
            return self.getter(data_dict)
        else:
            return data_dict

    def __len__(self):
        return len(self.instance_paths)
    
    @property
    def bag_num(self):
        return len(self.bag_names)

    @property
    def max_ins_num(self):
        return max(self.bag_lengths)

    def __str__(self):
        print("bag_idx-name-class-instance:\n")
        for idx, bag_name in enumerate(self.bag_names):
            print("{}, {}, {}, {}\n".format(idx, bag_name, self.bag_labels[idx], 
                                    self.bag_lengths[idx]))

        return "print done!"
