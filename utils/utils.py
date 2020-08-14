from PIL import ImageFilter
import random
import math
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
"""
Local Aggregation Objective as defined in 
https://arxiv.org/abs/1903.12355

Code is based on Tensorflow implementation: 
https://github.com/neuroailab/LocalAggregation
"""

import faiss
import torch

import numpy as np
import time
from termcolor import colored


DEFAULT_KMEANS_SEED = 1234

def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)
def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def adjust_learning_rate(optimizer, epoch, args, logger):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        logger.log_string('lr:{}'.format(lr))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
        logger.log_string('lr:{}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def adjust_weight(weight, epoch, args):
    if epoch >= (args.epochs/2):
        weight = weight
    else:
        weight *= 0.5 * (1. + math.cos(math.pi * (args.epochs/2-epoch) / (args.epochs/2)))
    print('weight{}'.format(weight))
    return weight
def print_args(args, logger):
    logger.log_string("==========================================")
    logger.log_string("==========       CONFIG      =============")
    logger.log_string("==========================================")
    for arg, content in args.__dict__.items():
        logger.log_string("{}:{}".format(arg, content))
    logger.log_string("\n")

def cal_metrics(preds, gt, args, logger, epoch,mode):
    preds = np.array(preds)
    gt = np.array(gt)
    target_names = ['normal', 'ITC', 'micro', 'macro']
    cls_rep=classification_report(gt, preds, target_names=target_names,output_dict=False)
    con_ma=confusion_matrix(gt, preds)
    logger.log_string(cls_rep)
    logger.log_string(con_ma)
    cls_rep_dic = classification_report(gt, preds, target_names=target_names, output_dict=True)
    logger.log_scalar(mode, 'acc', cls_rep_dic['accuracy'], epoch)
# def cal_metrics_train(preds, gt, args, logger, epoch):
#     if args.mil == 1:
#         preds = np.concatenate(preds)
#         gt = np.concatenate(gt)
#
#     else:
#         preds = np.array(preds)
#         gt = np.array(gt)
#     # fpr, tpr, thresholds = metrics.roc_curve(gt, preds, pos_label=1)
#     # AUC = metrics.auc(fpr, tpr)
#     # y_pred = (preds>0.5).astype(int)
#     # print('AUC:', AUC)
#     target_names = ['normal', 'ITC', 'micro', 'macro']
#     cls_rep_str = classification_report(gt, preds, target_names=target_names, output_dict=False)
#     con_ma = confusion_matrix(gt, preds)
#     logger.log_string(cls_rep_str)
#     logger.log_string(con_ma)
#     cls_rep_dic = classification_report(gt, preds, target_names=target_names, output_dict=True)
#     logger.log_scalar_train('acc', cls_rep_dic['accuracy'], epoch)


