# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import torchvision.transforms as transforms

from . import data
from .utils import export, GaussianBlur, TwoCropsTransform


#Added support for nctcrc: data augmentation we use the same as imgnet pretrained
@export
def nctcrc(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_augmentation = [
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        normalize
    ]
    train_augmentation = TwoCropsTransform(transforms.Compose(train_augmentation))
    val_augmentation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return {
        'train_transformation': train_augmentation,
        'eval_transformation': val_augmentation,
    }
