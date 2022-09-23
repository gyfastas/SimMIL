"""
Script for generating bag dataset from ImageFolder class.
"""
import argparse
import os, sys
import random
import shutil
import numpy as np
from datasets.imagefolder_dict import ImageFolderDict


class BaseBagGenerator(object):
    """
    Bag Dataset Generator for MyImageFolder based dataset.
    """
    def __init__(self, dataset, pos_targets, target_root, *args, **kwargs):
        self.dataset = dataset
        self.pos_targets = pos_targets
        if isinstance(self.pos_targets, int):
            self.pos_targets = [self.pos_targets]
        self.target_root = target_root
    
    def get_data(self):
        return list(map(lambda x:x[0], self.dataset.samples)), list(map(lambda x:x[1], self.dataset.samples))
    
    @staticmethod
    def split_pos_neg(all_ins, all_labels, target_label):
        pos_instances = list(map(lambda x:x[0], filter(lambda x:x[1] in target_label, zip(all_ins, all_labels))))
        neg_instances = list(map(lambda x:x[0], filter(lambda x:x[1] not in target_label, zip(all_ins, all_labels))))
        return pos_instances, neg_instances

    def run(self):
        raise NotImplementedError

    def save_label_file(self, label_file, pos_bags, neg_bags, length_check=None):
        print("=> Saving label file to {}".format(label_file))

        bag_idx = 0
        with open(label_file, 'w+') as f:
            for (bags, cls_id) in [(pos_bags, 1), (neg_bags, 0)]:
                for bag in bags:
                    if length_check is not None:
                        if len(bag)!=length_check:
                            print("bag {} is not len as {}, droped".format(bag_idx, length_check))
                            continue
                        
                    for ins in bag:
                        ## ins path, target, bag idx
                        f.write('{} {} {} \n'.format(ins, cls_id, bag_idx))
                    bag_idx+=1

    def save(self, pos_bags, neg_bags):
        """
        save files by copying
        """
        print("=> Saving generated dataset to {}".format(self.target_root))
        pos_dir = os.path.join(self.target_root, "Pos")
        neg_dir = os.path.join(self.target_root, "Neg")
        if not os.path.exists(pos_dir):
            os.makedirs(pos_dir)
        if not os.path.exists(neg_dir):
            os.makedirs(neg_dir)
        
        bag_idx = 0
        for (bags, cls_dir) in [(pos_bags, pos_dir), (neg_bags, neg_dir)]:
            for bag in bags:
                bag_dir = os.path.join(cls_dir, str(bag_idx))
                if not os.path.exists(bag_dir):
                    os.makedirs(bag_dir)

                for ins_path in bag:
                    print("copying {} to {}".format(ins_path, os.path.join(bag_dir, ins_path.rsplit("/")[-1])))
                    shutil.copy(ins_path, os.path.join(bag_dir, ins_path.rsplit("/")[-1]))
                bag_idx += 1

class BalanceBagGenerator(BaseBagGenerator):

    def __init__(self, bag_length, ratio_interval, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bag_length = bag_length
        self.ratio_interval = ratio_interval

    def run(self, copy_and_save=False, label_file='./label.txt'):
        # calculate parameters for ratio group generation
        total_ins_num = len(self.dataset)
        print("=> total instance num in dataset: {}".format(total_ins_num))
        pos_ins_num = len(list(filter(lambda x:x[1] in self.pos_targets, self.dataset.samples)))
        print("=> positive instance num in dataset: {}".format(pos_ins_num))
        mean_ratio = 2 * pos_ins_num / total_ins_num
        print("=> average bag ratio: {}".format(mean_ratio))
        pos_bag_nums = total_ins_num / (2 * self.bag_length)
        # generate ratio group
        ratio_group = self.generate_ratio_group(pos_bag_nums, mean_ratio)
        # get pos and neg bags
        pos_bags, neg_bags = self.fixed_length_bag_assign(self.bag_length, ratio_group)
        # saving
        if copy_and_save:
            self.save(pos_bags, neg_bags)
        else:
            self.save_label_file(label_file, pos_bags, neg_bags, self.bag_length)

    def generate_ratio_group(self, pos_bag_num, mean_ratio):
        """
        Generation rule:
            number of bags of ratio m　= total_ratio / (m * number of available ratios)
        """
        print("=> Generating ratio groups")
        available_ratio_list = np.arange(self.ratio_interval, 1.0+self.ratio_interval, self.ratio_interval)        
        nbags_of_ratio =  (pos_bag_num * mean_ratio) / (available_ratio_list * len(available_ratio_list))

        ratio_group_generator = ([available_ratio_list[i]] * int(np.round(n)) for (i, n) in enumerate(nbags_of_ratio))
        ratio_group = []
        for ratios in ratio_group_generator:
            ratio_group.extend(ratios)
        
        print("=> Ratio list: {}".format(available_ratio_list))
        print("=> Number of bags of each ratio: {}".format(nbags_of_ratio))
        print("=> Expected sum of ratio groups: {}".format(pos_bag_num * mean_ratio))
        print("=> Sum of ratio groups: {}".format(sum(ratio_group)))

        return ratio_group

    def fixed_length_bag_assign(self, bag_length, ratio_group):
        """
        Args:
            bag_length: number of instance contained in each bag
            ratio_group: {r1,r2...rK} a group of ratio we want (0<ri<=1)
            K: number of positive bags we want

        Returns:
            bos bags: list(list)
            neg bags: list(list)
        
        Notes:
            ratio means how many pos instance in a pos bag
            Notice that: sum(ratio_group) * bag_length == len(pos_instances)
        """
        all_instance, all_labels = self.get_data()
        pos_instances, neg_instances =  self.split_pos_neg(all_instance, all_labels, self.pos_targets)
        # pos_instances: {x1,x2...xNp}, Np: number of pos instances
        # neg_instances: {x1,x2...xNn}, Nn: number of neg instances
        pos_bags = []
        neg_bags = []
        random.shuffle(pos_instances)
        random.shuffle(neg_instances)
        for ratio in ratio_group:
            try:
                p_ins = [pos_instances.pop() for _ in range(0, int(ratio*bag_length))]
                n_ins = [neg_instances.pop() for _ in range(0, bag_length - int(ratio*bag_length))]
                p_ins.extend(n_ins)
                pos_bags.append(p_ins)
            except:
                print("failed to generate bag with ratio {}".format(ratio))
                continue
        
        if len(pos_instances):
            pos_instances.extend([neg_instances.pop() for _ in range(len(pos_instances))])
            pos_bags.append(pos_instances)
        
        while len(neg_instances):
            if len(neg_instances) < bag_length:
                break
            neg_bags.append([neg_instances.pop() for _ in range(0, bag_length)])
        
        return pos_bags, neg_bags


def re_assgin_labels(origin_label_file, target_label_file, dataset):
    f1 = open(origin_label_file, "r")
    f2 = open(target_label_file, "w+")
    ins_samples = np.array(dataset.samples)
    print(ins_samples.shape)
    samples = [x.strip().split(" ") for x in f1.readlines()]
    for sample in samples:
        file_name, ins_label, bag_index = sample[0], sample[1], sample[2]
        re_assigned_label = ins_samples[ins_samples[:, 0]==file_name][0, 1]
        print("writing {} {} {}".format(file_name, re_assigned_label, bag_index))
        f2.write("{} {} {}\n".format(file_name, re_assigned_label, bag_index))

    f1.close()
    f2.close()

def parse_args():
    parser = argparse.ArgumentParser("Bag generation tool for NCTCRC dataset.")
    parser.add_argument("--data_root", type=str, default="./data/NCTCRC/", 
                        help="data root of NCTCRC (instance level data)")
    parser.add_argument("--target_root", type=str, default="./data/NCTCRC-BAGS/", 
                        help="target data root of NCTCRC bag data.")
    parser.add_argument("--bag_length", type=int, default=50, 
                        help="length of each bag.")
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()

    bag_length = args.bag_length
    for pos_target in range(9):
        for split in ['train', 'val', 'test']:

            if split == "test":
                data_root = os.path.join(args.data_root, "VAL_TEST/test/")
            elif split == "val":
                data_root = os.path.join(args.data_root, "VAL_TEST/val/") 
            elif split == "train":
                data_root = os.path.join(args.data_root, "NCT-CRC-HE-100K/")    
            save_dir = "./data/samples/nctcrc_bags/BL{}/target{}/".format(bag_length, pos_target)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            label_file = os.path.join(save_dir, '{}.txt'.format(split))
            dataset = ImageFolderDict(data_root, None)
            print("Dataset classes: {}".format(dataset.class_to_idx))
            data_generator = BalanceBagGenerator(bag_length, 0.05, dataset, pos_target, args.target_root)
            data_generator.run(False, label_file)
            ins_label_file = os.path.join(save_dir, '{}_ins.txt'.format(split))
            re_assgin_labels(label_file, ins_label_file, dataset)