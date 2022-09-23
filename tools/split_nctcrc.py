import argparse
import os
import shutil

def split_by_file(data_root, split_file, target_root):
    with open(split_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split(",")
        file_name, folder_name, split = line[0], line[1], line[2]
        origin_file = os.path.join(data_root, folder_name, file_name)
        target_file = os.path.join(target_root, folder_name, file_name)
        target_folder = os.path.basename(target_file)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
            shutil.copy(origin_file, target_file)

parser = argparse.ArgumentParser("Split data according to given ratio")
parser.add_argument('--data_root', type=str, default="./data/NCTCRC/CRC-VAL-HE-7K/")
parser.add_argument('--split_file', type=str, default="./")
parser.add_argument('--train_ratios', type=float, default=0.7)
parser.add_argument('--target_root', type=str, default="")

if __name__=="__main__":
    args = parser.parse_args()
    root = args.root
    class_names = args.class_names
    train_ratios = args.train_ratios
    target_root = args.target_root
    split_by_ratio(root, class_names, train_ratios, target_root)