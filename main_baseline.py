from __future__ import print_function

import os
import tqdm
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from dataloader import MIL_CRC
from model import MT_ResAMIL_BagIns, MT_ResAMIL_ODC
from utils.utils import cal_metrics, find_class_by_name, set_param_groups, GaussianBlur, TwoCropsTransform, adjust_learning_rate
from Builder import FeaAgg, clustering
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import math
import numpy as np
# import data
from torchvision import transforms

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CRC')
# optimizer
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=0, metavar='R',
                    help='weight decay')
parser.add_argument('--bs', type=int, default=4,
                    help='Batch size')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--cos', action='store_true', default=True,
                    help='use cosine lr schedule')
# parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    # help='learning rate schedule (when to drop lr by 10x)')
#############################################################
# parser.add_argument('--pretrained', action='store_true', default=False,
                    # help='imagenet_pretrained and fix_bn to aviod shortcut')
parser.add_argument('--attention', type=str, default='Agg_GAttention',
                    help='Agg_GAttention/Agg_SAttention/Agg_Attention')
# parser.add_argument('--momentum', type=float, default=0.999,
                    # help='momentum used in MOCO')                    
parser.add_argument('--ema', type=float, default=0.99, metavar='EMA',
                    help='exponential moving average')
#memory bank
parser.add_argument('--mb_dim', default=256, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--mb_t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--mb_ema', default=0.5, type=float,
                    help='Momentum for memory bank (default: 0.5)')
parser.add_argument('--mb_K', default=4, type=int,
                    help='K ins classes (default: 4)')
# loss weight
parser.add_argument('--weight_self', type=float, default=0)
parser.add_argument('--weight_mse', type=float, default=0)
# parser.add_argument('--weight_bag', type=float, default=0)
# Data and logging
parser.add_argument('--ratio', type=float, default=0.8,
                    help='the ratio of train dataset')
parser.add_argument('--folder', type=int, default=0,
                    help='CV folder')
parser.add_argument('--data_dir', type=str, default='/remote-home/source/DATA',
                    help='local: /media/walkingwind/Transcend'
                         'remote: /remote-home/source/DATA')
# parser.add_argument('--log_dir', default='./experiments', type=str,
                    # help='path to moco pretrained checkpoint')
# parser.add_argument('--config', type=str, default='./config/imagenet_byol.json',
                        # help='path to config file')

args = parser.parse_args()
# config = process_config(args.config)

torch.manual_seed(args.seed)
# cudnn.benchmark = True
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load Train and Test Set')
bag_path = os.path.join(args.data_dir, 'CRCHistoPhenotypes')

for arg, content in args.__dict__.items():
    print("{}:{}".format(arg, content))

channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                     std=[0.2470,  0.2435,  0.2616])

loader_kwargs = {'num_workers': 16, 'pin_memory': True} if args.cuda else {}

aug_train = [
            #aug Moco v2's aug
            # transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)]

aug_test = [
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)]

train_dataset = MIL_CRC(bag_path, TwoCropsTransform(transforms.Compose(aug_train)),
                                 floder=args.folder, ratio=args.ratio, train=True)

test_dataset  = MIL_CRC(bag_path, transforms.Compose(aug_test),
                                floder=args.folder, ratio=args.ratio, train=False)

train_loader = data_utils.DataLoader(train_dataset,
                                     batch_size=args.bs,
                                     shuffle=True,
                                     collate_fn=train_dataset.collate_fn,
                                     **loader_kwargs)

test_loader = data_utils.DataLoader(test_dataset,
                                     batch_size=args.bs,
                                     shuffle=False,
                                     collate_fn=test_dataset.collate_fn,
                                     **loader_kwargs)

print('Init Model')

# # memory bank
# BYOL_flag=False
# if args.weight_self != 0 and config.loss_params.loss != 'BYOLModule':
#     data_len = len(train_dataset)*48 #WSIs * patches
#     memory_bank = MemoryBank(data_len, args.mb_dim, config.gpu_device) #128D, gpu-id=0
#     # init self-module
#     if config.loss_params.loss == 'LocalAggregationLossModule':
#         # init cluster
#         cluster_labels = _init_cluster_labels(config, data_len)
#         Ins_Module = LocalAggregationLossModule(memory_bank.bank_broadcast,
#                          cluster_labels,
#                          k=config.loss_params.k,
#                          t=config.loss_params.t,
#                          m=config.loss_params.m)
#         if config.loss_params.kmeans_freq is None:
#            config.loss_params.kmeans_freq = (
#                     len(train_loader) )
#     elif config.loss_params.loss == 'MocoLossModule':
#         Ins_Module = MocoLossModule(memory_bank.bank_broadcast, 
#                          k=config.loss_params.k,
#                          t=config.loss_params.t)
#     else:
#         Ins_Module = InstanceDiscriminationLossModule(memory_bank.bank_broadcast,
#                          cluster_labels_broadcast=None,
#                          k=config.loss_params.k,
#                          t=config.loss_params.t,
#                          m=config.loss_params.m)
# elif args.weight_self != 0 and config.loss_params.loss == 'BYOLModule':
#     BYOL_flag=True

attention = find_class_by_name(args.attention, [FeaAgg])
# graph = find_class_by_name(args.graph, [FeaAgg])
# consistent = (args.weight_mse != 0) | (args.weight_bag != 0)
# bag_label = torch.stack([torch.tensor(i) for i in train_dataset.folder_label]).cuda()
# model = GraphCon(args.agg_mode,
                #  ResBackbone, args.arch, args.pretrained,
                #  AggNet, attention, graph, args.n_topk,
                #  m=args.ema, T=args.mb_t, dim=args.mb_dim, K=len(train_dataset), bag_label=bag_label, 
                #  BYOL_flag=BYOL_flag, BYOL_size=config.projection_size, BYOL_hidden_size=config.projection_hidden_size)
tr_length = len(np.concatenate(np.array(train_dataset.labels_list_frombag)))
model = MT_ResAMIL_BagIns(num_classes=2, attention=attention, L=384) #basline model
if args.cuda:
    model.cuda()
print(model)

parameters = [{'params': model.parameters()}]
optimizer = optim.Adam(parameters, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

criterion_ce = nn.CrossEntropyLoss()
criterion_mse = nn.MSELoss()
# criterion_l1 = nn.L1Loss()

# def init_memory():
#     model.eval()
#     # 1.get features
#     features = torch.tensor([])
#     indexs = torch.tensor([]).long()
#     for batch_idx, (data, bag_label, ins_real_label, ins_label, ins_fb_label, batch, index) in enumerate(train_loader):
#         data = data[0]
#         if args.cuda:
#             data, bag_label, ins_real_label, ins_label, ins_fb_label, batch = \
#                 data.cuda(), bag_label.cuda(), ins_real_label.cuda(), ins_label.cuda(), ins_fb_label.cuda(), batch.cuda()
#         features = torch.cat((features, model.extract(data, batch)))
#         indexs = torch.cat((indexs, index))

#     # 2.get labels
#     clustering_algo = clustering.Kmeans(k=args.K, pca_dim=-1)
#     # Features are normalized during clustering
#     clustering_algo.cluster(features.numpy(), verbose=True)
#     assert isinstance(clustering_algo.labels, np.ndarray)
#     new_labels = clustering_algo.labels.astype(np.int64)
#     evaluate(new_labels)

#     #3.assin new labels
#     new_labels_tensor = torch.from_numpy(new_labels)
#     # argsort
#     features = features[torch.argsort(indexs)]
#     new_labels_tensor = new_labels_tensor[torch.argsort(indexs)]
#     features = features.numpy()
#     new_labels = new_labels_tensor.numpy()
#     new_labels_list = list(new_labels)
#     train_dataset.assign_labels(new_labels_list)

#     # 4.init mb

#     model.memory_bank.init_memory(features, new_labels)

#     # reweight
#     ce_reweight=model.set_reweight(new_labels, 0.5)
#     return ce_reweight.cuda()

def train(epoch):
    model.train()
    train_loss = 0.
    train_error = 0.
    CE_loss = 0.
    INS_loss = 0.
    MSE_loss = 0.
    # BagInfo_loss = 0.
    preds_list_1 = []
    preds_list_2 = []
    gt_list = []

    ins_preds_list = []
    ins_gt_list = []

    # ce_reweight = init_weight 
    # ce_w = 1.
    # ce_m = 1.

    # adjusted_weight = adjust_weight(args.weight_mse, epoch, args)
    for batch_idx, (bag_idx, data, label, batch) in enumerate(train_loader):
        bag_label, ins_real_label, ins_fb_label = label
        img_q, img_k = data
        if args.cuda:
            # img_k, img_q, bag_label = img_k.cuda(non_blocking=True), img_q.cuda(non_blocking=True), bag_label.cuda(non_blocking=True)
            # bag_idx, batch = bag_idx.cuda(non_blocking=True), batch.cuda(non_blocking=True)
            img_k, img_q, bag_label, ins_real_label, ins_fb_label, bag_idx, batch = \
                img_k.cuda(), img_q.cuda(), bag_label.cuda(), ins_real_label.cuda(), ins_fb_label.cuda(), bag_idx.cuda(), batch.cuda()
        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        bag_preds_q, bag_preds_k = model(data_aug1, batch)
        ce_loss_bag = criterion_ce(bag_preds_q, bag_label)
        ce_loss_ins = criterion_ce(bag_preds_k, bag_label)
        loss = args.ce1_weight*ce_loss_bag + args.mse_weight*ce_loss_ins
        CE_loss += ce_loss_bag.item()
        MSE_loss += ce_loss_ins.item()


        

        # if args.weight_mse != 0:
            # mse_loss = criterion_mse(Y_prob_q_sf, Y_prob_k_sf)/data[0].shape[0]
            # mse_loss = criterion_mse(Y_prob_q_sf, Y_prob_k_sf)
            # MSE_loss += mse_loss.item()
            # total_loss = total_loss + args.weight_mse * mse_loss
        # if args.weight_self != 0 and BYOL_flag == False:
            # ins_loss, new_data_memory = Ins_Module(
                # ins_idx, self_fea_q, torch.arange(len(config.gpu_device)))
            # BYOL module
            # ins_loss = Ins_Module(self_fea_q, self_fea_k)
            # INS_loss += ins_loss.item()
            # total_loss = total_loss + args.weight_self * ins_loss
        # if args.weight_self != 0 and BYOL_flag == True:
            # INS_loss += ins_loss.item()
            # total_loss = total_loss + args.weight_self * ins_loss

        train_loss += loss.item()
        preds_list_1.extend([torch.argmax(bag_preds_q, dim=1)[i].item() for i in range(bag_preds_q.shape[0])])
        preds_list_2.extend([torch.argmax(bag_preds_k, dim=1)[i].item() for i in range(bag_preds_k.shape[0])])
        gt_list.extend([bag_label[i].item() for i in range(bag_label.shape[0])])
        # backward pass
        loss.backward()
        # step
        optimizer.step()

    # calculate loss and error for epoch
    CE_loss /= len(train_loader)
    INS_loss /= len(train_loader)
    MSE_loss /= len(train_loader)
    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    cal_metrics(preds_list_1, gt_list)
    cal_metrics(preds_list_2, gt_list)
    print('Epoch: {}, Loss: {:.4f}, {:.4f}, {:.4f}'.format(epoch, train_loss, CE_loss, MSE_loss))

def test():
    model.eval()
    test_loss = 0.
    test_error = 0.
    preds_list = []
    gt_list = []
    with torch.no_grad():
        for batch_idx, (_, _, data, label, batch) in enumerate(tqdm(test_loader)):
            bag_label = label
            img_k = data[0]
            img_q = data[1]
            if args.cuda:
                img_k, img_q, bag_label, batch = img_k.cuda(), img_q.cuda(), bag_label.cuda(), batch.cuda()
            preds = model(img_k, None, batch=batch,  bag_idx=None, label=None)
            preds_list.extend([torch.argmax(preds, dim=1)[i].item() for i in range(preds.shape[0])])
            gt_list.extend([bag_label[i].item() for i in range(bag_label.shape[0])])

        test_error /= len(test_loader)
        test_loss /= len(test_loader)
        logger.log_string('Test====================')
        logger.log_string(
            '\nEpoch: {}, Test Set, Loss: {:.4f}, Test error: {:.4f}'.format(epoch, test_loss, test_error))
        cal_metrics(preds_list, gt_list, args, logger, epoch, 'test')

if __name__ == "__main__":
    print('Start Training')
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # w = init_memory()
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch, args)
        train(epoch)

        if epoch % 5 == 0:
            test()
