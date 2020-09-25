from PIL import ImageFilter
import random
import math

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np



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

class TwoSameTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        return [q, q]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    print('lr:{}'.format(lr))

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length

        return float(np.exp(-5.0 * phase * phase))
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

# def cal_metrics(preds, gt, args, logger, epoch,mode):
#     preds = np.array(preds)
#     gt = np.array(gt)
#     target_names = ['normal', 'ITC', 'micro', 'macro']
#     cls_rep=classification_report(gt, preds, target_names=target_names,output_dict=False)
#     con_ma=confusion_matrix(gt, preds)
#     logger.log_string(cls_rep)
#     logger.log_string(con_ma)
#     cls_rep_dic = classification_report(gt, preds, target_names=target_names, output_dict=True)
#     logger.log_scalar(mode, 'acc', cls_rep_dic['accuracy'], epoch)

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
def cal_metrics(preds, gt, cls_num, mode, tag, epoch, logger=None):
    preds = np.array(preds)
    gt = np.array(gt)
    target_names = [str(i) for i in range(cls_num)]
    cls_rep = classification_report(gt, preds, target_names=target_names, output_dict=False)
    con_ma = confusion_matrix(gt, preds)
    fpr, tpr, _ = roc_curve(gt, preds, pos_label=1)
    AUC = auc(fpr, tpr)
    if logger is None:
        print(cls_rep)
        print(con_ma)
        print('AUC:{}'.format(AUC))
    else:
        logger.log_string(cls_rep)
        logger.log_string(con_ma)
        logger.log_string('AUC:{}'.format(AUC))
        logger.log_scalar(mode, tag, AUC, epoch)
def set_param_groups(net, lr_mult_dict):
    params = []
    modules = net._modules
    for name in modules:
        module = modules[name]
        if name in lr_mult_dict:
            params += [{'params': module.parameters(), 'lr_mult': lr_mult_dict[name]}]
        else:
            params += [{'params': module.parameters(), 'lr_mult': 1.0}]

    return params


def sample_ins_labels(dataset, ratio, seed):
    bag_num = len(dataset.bags_list)
    random.seed(seed)
    choosen_bags = random.sample(range(0, int(bag_num/2)), int(ratio * bag_num/2))
    pos_bag_mask = [i*2 for i in choosen_bags]
    neg_bag_mask = [i*2+1 for i in choosen_bags]
    mask = np.concatenate((pos_bag_mask, neg_bag_mask))
    data = []
    label = []
    for idx in mask:
        data.append(dataset.bags_list[idx])
        label.append(dataset.labels_list[idx])
    data = np.concatenate(data)
    label = np.concatenate(label)
    return data, label

import torch
def set_reweight(labels, K , reweight_pow=0.5):
    hist = np.bincount(labels, minlength=K).astype(np.float32)
    inv_hist = (1. / (hist + 1e-5)) ** reweight_pow
    weight = inv_hist / inv_hist.sum()
    return hist, torch.from_numpy(weight).cuda()


def l_byol(x,y):
    x = F.normalize(x, dim=1, p=2)
    y = F.normalize(y, dim=1, p=2)
    return (2 - 2*(x*y).sum(dim=-1)).mean()
