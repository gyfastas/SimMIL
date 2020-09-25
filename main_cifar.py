from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

from dataloader import CifarBags
# from Builder.e2e import Attention, GatedAttention, Res18, Res18_SAFS
from Builder import FeaAgg
# from Builder.FeaAgg import ResBackbone, AggNet
# from Builder.GraphCon import GraphCon
from Builder.cifar_res import cifar_shakeshake26
from model import MT_ResAMIL
from tqdm import tqdm
from torchvision import models, transforms
import os
from logger import Logger
import datetime
from MemoryBank.memorybank import MemoryBank, InstanceDiscriminationLossModule, LocalAggregationLossModule, \
    Kmeans, MocoLossModule
from MemoryBank.mb_utils import _init_cluster_labels
from utils.utils import TwoCropsTransform, GaussianBlur
from utils.setup import process_config
from utils.utils import fix_bn, GaussianBlur, adjust_learning_rate, print_args, cal_metrics, RandomTranslateWithReflect
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import find_class_by_name, set_param_groups

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CifarBags')
# optimizer
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=0, metavar='R',
                    help='weight decay')
parser.add_argument('--target_number', type=int, nargs='+', default=[0, 1, 8, 9], metavar='T')
parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML',
                    help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL',
                    help='variance of bag length')
parser.add_argument('--bs', type=int, default=4,
                    help='Batch size')
parser.add_argument('--num_bags_train', type=int, default=50, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=1000, metavar='NTest',
                    help='number of bags in test set')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--cos', action='store_true', default=True,
                    help='use cosine lr schedule')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
#############################################################
# parser.add_argument('--pretrained', action='store_true', default=True,
#                     help='imagenet_pretrained and fix_bn to aviod shortcut')
# parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
#                     choices=model_names,
#                     help='model architecture: ' +
#                         ' | '.join(model_names) +
#                         ' (default: resnet50)')
parser.add_argument('--attention', type=str, default='Agg_Attention',
                    help='Agg_GAttention/Agg_SAttention/Agg_Attention')
parser.add_argument('--graph', type=str, default='GraphMILNet',
                    help='Choose graph')
parser.add_argument('--n_topk', type=int, default=4,
                    help='K of KNN')
parser.add_argument('--agg_mode', type=str, default='attention',
                    help='attention/graph/residual(concat)')
# parser.add_argument('--momentum', type=float, default=0.999,
#                     help='momentum used in MOCO')                    
parser.add_argument('--ema', type=float, default=0.5, metavar='EMA',
                    help='exponential moving average')
#memory bank
# parser.add_argument('--mb_dim', default=128, type=int,
                    # help='feature dimension (default: 128)')
# parser.add_argument('--mb-t', default=0.07, type=float,
                    # help='softmax temperature (default: 0.07)')
# loss weight
parser.add_argument('--weight_self', type=float, default=1)
parser.add_argument('--weight_mse', type=float, default=0)
parser.add_argument('--weight_bag', type=float, default=0)
# Data and logging
# parser.add_argument('--ratio', type=float, default=0.8,
                    # help='the ratio of train dataset')
# parser.add_argument('--folder', type=int, default=0,
                    # help='CV folder')
# parser.add_argument('--data_dir', type=str, default='/media/walkingwind/Transcend',
                    # help='local: /media/walkingwind/Transcend'
                        #  'remote: /remote-home/my/datasets/BASH')
parser.add_argument('--log_dir', default='./experiments', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--config', type=str, default='./config/imagenet_la.json',
                        help='path to config file')

args = parser.parse_args()
config = process_config(args.config)

torch.manual_seed(args.seed)
cudnn.benchmark = True
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

#setting logger
log_path = os.path.join(args.log_dir, str(datetime.datetime.now()))
logger = Logger(logdir=log_path)

# logger.backup_files(
#         [x for x in os.listdir('./') if x.endswith('.py')] + [args.config])

# print('Load Train and Test Set')
loader_kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

# ins_path = os.path.join(args.data_dir, 'patch/train')
# bag_path = os.path.join(args.data_dir, 'BASH')

channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                     std=[0.2470,  0.2435,  0.2616])

aug_cls = [transforms.Resize((28, 28)),
           RandomTranslateWithReflect(4),
           transforms.RandomHorizontalFlip(),
           transforms.ToTensor(),
           transforms.Normalize(**channel_stats)]

aug_train = [
            #aug Moco v2's aug
            transforms.RandomResizedCrop(28, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)]

aug_test = [transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)]

train_dataset = CifarBags(target_number=args.target_number, 
                            mean_bag_length=args.mean_bag_length, 
                            var_bag_length=args.var_bag_length,
                            num_bag=args.num_bags_train,
                            seed=args.seed,
                            train=True,
                            transform = TwoCropsTransform(transforms.Compose(aug_cls) ,transforms.Compose(aug_train)))

test_dataset = CifarBags(target_number=args.target_number, 
                            mean_bag_length=args.mean_bag_length, 
                            var_bag_length=args.var_bag_length,
                            num_bag=args.num_bags_test,
                            seed=args.seed,
                            train=False,
                            transform = transforms.Compose(aug_test))

train_loader = data_utils.DataLoader(train_dataset,
                                     batch_size=args.bs,
                                     collate_fn=train_dataset.collate_fn,
                                     shuffle=True,
                                     **loader_kwargs)
test_loader = data_utils.DataLoader(test_dataset,
                                    batch_size=args.bs,
                                    collate_fn=test_dataset.collate_fn,
                                    shuffle=False,
                                    **loader_kwargs)

init_ins_labels = torch.cat(train_dataset.train_labels_frombag)

print('Init Model')

# memory bank
if args.weight_self != 0:
    data_len = len(torch.cat(train_dataset.train_bag_index_mb))
    memory_bank = MemoryBank(data_len, config.model_params.out_dim, config.gpu_device)
    # init self-module
    if config.loss_params.loss == 'LocalAggregationLossModule':
        # init cluster
        cluster_labels = _init_cluster_labels(config, data_len)
        Ins_Module = LocalAggregationLossModule(memory_bank.bank_broadcast,
                         cluster_labels,
                         k=config.loss_params.k,
                         t=config.loss_params.t,
                         m=config.loss_params.m)
        if config.loss_params.kmeans_freq is None:
           config.loss_params.kmeans_freq = (len(train_loader))
    # elif config.loss_params.loss == 'MocoLossModule':
        # Ins_Module = MocoLossModule(memory_bank.bank_broadcast, 
                        #  k=config.loss_params.k,
                        #  t=config.loss_params.t)
    else:
        Ins_Module = InstanceDiscriminationLossModule(memory_bank.bank_broadcast,
                         cluster_labels_broadcast=None,
                         k=config.loss_params.k,
                         t=config.loss_params.t,
                         m=config.loss_params.m)

# choose agg model
attention = find_class_by_name(args.attention, [FeaAgg])
# graph = find_class_by_name(args.graph, [FeaAgg])
# consistent = (args.weight_mse != 0) | (args.weight_bag != 0)
# bag_label = torch.stack(([torch.tensor(i[1]) for i in train_dataset.folder])).cuda()
model = MT_ResAMIL(num_classes=2, attention=attention, L=384, m=args.ema, init_ins_labels=init_ins_labels)
if args.cuda:
    model.cuda()
# print(model)

# parameters = set_param_groups(model, dict({'fc2': 0}))
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

criterion_ce_sum = nn.CrossEntropyLoss(size_average=False, ignore_index=-1)
criterion_ce = nn.CrossEntropyLoss()
criterion_mse_mean = nn.MSELoss()
# criterion_l1 = nn.L1Loss()

def train(epoch):
    model.train()
    train_loss = 0.
    train_error = 0.
    CE_loss = 0.
    INS_loss = 0.
    MSE_loss = 0.
    # BagInfo_loss = 0.
    preds_list = []
    gt_list = []
    ins_preds_list = []
    ins_gt_list = []
    mse_weight = args.weight_mse
    cnt = 0
    total_len = 0
    nce_loss = 0 
    ins_loss = 0
    # ramp_weight = mse_weight*sigmoid_rampup(epoch, 100)
    # print('mse_weight{}'.format(ramp_weight))
    # adjusted_weight = adjust_weight(args.weight_mse, epoch, args)
    for batch_idx, (data, bag_label, ins_label, ins_fb_label, batch, index) in enumerate(tqdm(train_loader)):
        data_aug0 = data[0]
        data_aug1 = data[1]
        data_aug2 = data[2]
        if args.cuda:
            data_aug0, data_aug1, data_aug2, bag_label, ins_label, ins_fb_label, batch, index= \
                data_aug0.cuda(), data_aug1.cuda(), data_aug2.cuda(), bag_label.cuda(), ins_label.cuda(), ins_fb_label.cuda(), batch.cuda(), index.cuda()
        # reset gradients
        optimizer.zero_grad()
        # forward
        # (bag_preds_q, bag_preds_k), (logits, labels) = model(data_aug0, data_aug1, data_aug2, batch)
        bag_preds_q, ins_q, ins_k = model(data_aug0, data_aug1, data_aug2, batch)
        # calculate loss and metrics
        ce_loss = criterion_ce(bag_preds_q, bag_label)
        # nce_loss =criterion_ce(logits, labels)

        if args.weight_self != 0:
            ins_loss, new_data_memory = Ins_Module(index, ins_q, ins_k, torch.arange(len(config.gpu_device)))
            # loss = ce_loss + args.weight_self * ins_loss

        total_loss = ce_loss + mse_weight * nce_loss + args.weight_self * ins_loss
        CE_loss += ce_loss.item()
        # MSE_loss += nce_loss.item()
        # INS_loss += ins_loss.item()

        # if args.weight_bag != 0:
        #     baginfo_loss = criterion_ce(logits, labels)
        #     BagInfo_loss += baginfo_loss.item()
        #     total_loss = total_loss + args.weight_bag * baginfo_loss
        # if args.weight_mse != 0:
        #     mse_loss = criterion_mse(Affinity_q, Affinity_k)/data[0].shape[0]
        #     MSE_loss += mse_loss.item()
        #     total_loss = total_loss + args.weight_mse * mse_loss
        # if args.weight_self != 0:
        #     ins_loss, new_data_memory = Ins_Module(
        #         ins_idx, self_fea_q, torch.arange(len(config.gpu_device)))
        #     INS_loss += ins_loss.item()
        #     total_loss = total_loss + args.weight_self * ins_loss

        train_loss += total_loss.item()
        preds_list.extend([torch.argmax(bag_preds_q, dim=1)[i].item() for i in range(bag_label.shape[0])])
        gt_list.extend([bag_label[i].item() for i in range(bag_label.shape[0])])
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        # update memo and cluster
        if args.weight_self != 0:
            with torch.no_grad():
                memory_bank.update(index, new_data_memory)
                Ins_Module.memory_bank_broadcast = memory_bank.bank_broadcast  # update loss_fn
                if (config.loss_params.loss == 'LocalAggregationLossModule' and
                        (Ins_Module.first_iteration_kmeans or batch_idx == (config.loss_params.kmeans_freq-1))):
                    if Ins_Module.first_iteration_kmeans:
                        Ins_Module.first_iteration_kmeans = False
                        # get kmeans clustering (update our saved clustering)
                    k = [config.loss_params.kmeans_k for _ in
                         range(config.loss_params.n_kmeans)]
                    # NOTE: we use a different gpu for FAISS otherwise cannot fit onto memory
                    km = Kmeans(k, memory_bank, gpu_device=config.gpu_device)
                    cluster_labels = km.compute_clusters()

                    # broadcast the cluster_labels after modification
                    for i in range(len(config.gpu_device)):
                        device = Ins_Module.cluster_label_broadcast[i].device
                        Ins_Module.cluster_label_broadcast[i] = cluster_labels.to(
                            device)

    # calculate loss and error for epoch
    CE_loss /= len(train_loader)
    INS_loss /= len(train_loader)
    MSE_loss /= len(train_loader)
    # BagInfo_loss /= len(train_loader)
    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    logger.log_string('====================Train')
    logger.log_string('Epoch: {}, ceLoss: {:.4f}, insLoss: {:.4f}, MseLoss: {:.4f}, Train loss: {:.4f}'
                      .format(epoch, CE_loss, INS_loss, MSE_loss, train_loss))
    cal_metrics(preds_list, gt_list, args, logger, epoch, 'train')

def test(epoch):
    model.eval()
    test_loss = 0.
    test_error = 0.
    preds_list = []
    gt_list = []
    with torch.no_grad():
        for batch_idx, (data, bag_label, instance_labels, batch, index) in enumerate(tqdm(test_loader)):
            # bag_label = label
            # img_k = data[0]
            # img_q = data[1]
            if args.cuda:
                # img_k, img_q, bag_label, batch = img_k.cuda(), img_q.cuda(), bag_label.cuda(), batch.cuda()
                data, bag_label, batch = data.cuda(), bag_label.cuda(), batch.cuda()
            # preds = model(img_k, None, batch=batch,  bag_idx=None, label=None)
            preds = model(data, data, None, batch)
            preds_list.extend([torch.argmax(preds, dim=1)[i].item() for i in range(preds.shape[0])])
            gt_list.extend([bag_label[i].item() for i in range(bag_label.shape[0])])

        test_error /= len(test_loader)
        test_loss /= len(test_loader)
        logger.log_string('Test====================')
        logger.log_string(
            '\nEpoch: {}, Test Set, Loss: {:.4f}, Test error: {:.4f}'.format(epoch, test_loss, test_error))
        cal_metrics(preds_list, gt_list, args, logger, epoch, 'test')

if __name__ == "__main__":
    logger.log_string('Start Training')
    print_args(args, logger)
    for epoch in range(1, args.epochs + 1):
        # test(epoch)
        adjust_learning_rate(optimizer, epoch, args, logger)
        # train(epoch)
        train(epoch)
        # test(epoch)

        if epoch >= 195:
            logger.log_string('Start Testing')
            test(epoch)
