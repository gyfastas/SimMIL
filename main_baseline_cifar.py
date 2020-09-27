from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from tqdm import tqdm
from logger import Logger
import datetime
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from dataloader import CifarBags
from model import MT_ResAMIL_BagIns
from utils.utils import find_class_by_name, set_param_groups, GaussianBlur, TwoCropsTransform, adjust_learning_rate
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
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=0, metavar='R',
                    help='weight decay')
parser.add_argument('--bs', type=int, default=8,
                    help='Batch size')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--cos', action='store_true', default=True,
                    help='use cosine lr schedule')
parser.add_argument('--attention', type=str, default='Agg_GAttention',
                    help='Agg_GAttention/Agg_SAttention/Agg_Attention')                 
parser.add_argument('--ema', type=float, default=0.99, metavar='EMA',
                    help='exponential moving average')
# parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    # help='learning rate schedule (when to drop lr by 10x)')
# parser.add_argument('--pretrained', action='store_true', default=False,
                    # help='imagenet_pretrained and fix_bn to aviod shortcut')
# cifar bag
parser.add_argument('--target_number', type=int, nargs='+', default=[9], metavar='T',
                    help='single_train_target')
parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML',
                    help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL',
                    help='variance of bag length')
parser.add_argument('--num_bags_train', type=int, default=500, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=1000, metavar='NTest',
                    help='number of bags in test set')
# #memory bank
# parser.add_argument('--mb_dim', default=256, type=int,
#                     help='feature dimension (default: 128)')
# parser.add_argument('--mb_t', default=0.07, type=float,
#                     help='softmax temperature (default: 0.07)')
# parser.add_argument('--mb_ema', default=0.5, type=float,
#                     help='Momentum for memory bank (default: 0.5)')
# parser.add_argument('--mb_K', default=4, type=int,
#                     help='K ins classes (default: 4)')
# train paradigm
parser.add_argument('--pool_model', type=str, default='mean')
parser.add_argument('--paradigm', type=str, default='instance')
# loss weight
parser.add_argument('--weight_self', type=float, default=1)
parser.add_argument('--weight_mse', type=float, default=1)
# parser.add_argument('--weight_bag', type=float, default=0)
# Data and logging
# parser.add_argument('--ratio', type=float, default=0.8,
#                     help='the ratio of train dataset')
# parser.add_argument('--folder', type=int, default=0,
#                     help='CV folder')
# parser.add_argument('--data_dir', type=str, default='/remote-home/source/DATA',
#                     help='local: /media/walkingwind/Transcend'
#                          'remote: /remote-home/source/DATA')
parser.add_argument('--checkpoint_dir', default='./experiments_data', type=str,
                    help='path to checkpoint')
parser.add_argument('--log_dir', default='./experiments', type=str)                    
# parser.add_argument('--config', type=str, default='./config/imagenet_byol.json',
                        # help='path to config file')

args = parser.parse_args()
# config = process_config(args.config)

torch.manual_seed(args.seed)
# cudnn.benchmark = True
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

#setting logger
log_path = os.path.join(args.log_dir, str(datetime.datetime.now()))
logger = Logger(logdir=log_path)
print('Load Train and Test Set')
# bag_path = os.path.join(args.data_dir, 'CRCHistoPhenotypes')

for arg, content in args.__dict__.items():
    print("{}:{}".format(arg, content))

channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                     std=[0.2470,  0.2435,  0.2616])

loader_kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

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

train_dataset = CifarBags(target_number=args.target_number,
                          mean_bag_length=args.mean_bag_length,
                          var_bag_length=args.var_bag_length,
                          num_bag=args.num_bags_train,
                          seed=args.seed,
                          train=True,
                          transform=transforms.Compose(aug_train))
test_dataset = CifarBags(target_number=args.target_number,
                         mean_bag_length=args.mean_bag_length,
                         var_bag_length=args.var_bag_length,
                         num_bag=args.num_bags_test,
                         seed=args.seed,
                         train=False,
                         transform=transforms.Compose(aug_test))

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
# tr_length = len(np.concatenate(np.array(train_dataset.labels_list_frombag)))
model = MT_ResAMIL_BagIns(num_classes=2, attention=attention, L=128) #basline model
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

def cal_metrics(preds, gt, logger, epoch, train):
    preds = np.array(preds)
    gt = np.array(gt)
    target_names = ['Negative', 'Positive']
    cls_rep = classification_report(gt, preds, target_names=target_names, output_dict=False)
    con_ma = confusion_matrix(gt, preds)
    fpr, tpr, _ = roc_curve(gt, preds, pos_label=1)
    AUC = auc(fpr, tpr)
    logger.log_string(cls_rep)
    logger.log_string(con_ma)
    logger.log_string('AUC: {}'.format(AUC))
    cls_rep_dic = classification_report(gt, preds, target_names=target_names, output_dict=True)
    if train:
        logger.log_scalar_train('acc', cls_rep_dic['accuracy'], epoch)
    else:
        logger.log_scalar_eval('acc', cls_rep_dic['accuracy'], epoch)

def save_snapshot(state, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))

def save_checkpoint(filename="checkpoint.pth.tar"):
    out_dict = {
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    save_snapshot(out_dict, folder=args.checkpoint_dir, filename=args.paradigm+'_'+str(epoch)+'_'+filename)

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

    for _, (data, label, batch) in enumerate(tqdm(train_loader)):
        bag_label, ins_real_label, ins_fb_label = label
        img = data
        if args.cuda:
            img, bag_label, ins_real_label, ins_fb_label, batch = \
                img.cuda(), bag_label.cuda(), ins_real_label.cuda(), ins_fb_label.cuda(), batch.cuda()
        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        preds_img = model(img, batch, args.pool_model, args.paradigm)
        ce_loss_bag = criterion_ce(preds_img, bag_label)
        # ce_loss_ins = criterion_ce(ins_preds_q, ins_fb_label)
        # loss = args.weight_self*ce_loss_bag + args.weight_mse*ce_loss_ins
        loss = args.weight_self*ce_loss_bag
        CE_loss += ce_loss_bag.item()
        # MSE_loss += ce_loss_ins.item()

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
        preds_list_1.extend([torch.argmax(preds_img, dim=1)[i].item() for i in range(preds_img.shape[0])])
        # preds_list_2.extend([torch.argmax(bag_preds_k, dim=1)[i].item() for i in range(bag_preds_k.shape[0])])
        gt_list.extend([bag_label[i].item() for i in range(bag_label.shape[0])])
        # backward pass
        loss.backward()
        # step
        optimizer.step()

    # calculate loss and error for epoch
    CE_loss /= len(train_loader)
    # INS_loss /= len(train_loader)
    # MSE_loss /= len(train_loader)
    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    logger.log_string('====================Train')
    logger.log_string('Epoch: {}, Loss: {:.4f}'.format(epoch, train_loss))
    cal_metrics(preds_list_1, gt_list, logger, epoch, train=True)
    # cal_metrics(preds_list_2, gt_list)
    
def test(epoch):
    model.eval()
    test_loss = 0.
    test_error = 0.
    preds_list = []
    gt_list = []
    with torch.no_grad():
        for batch_idx, (data, label, batch) in enumerate(tqdm(test_loader)):
            bag_label = label[0]
            img = data
            # img_q = data[1]
            if args.cuda:
                img, bag_label, batch = img.cuda(), bag_label.cuda(), batch.cuda()
            preds_bag = model(img, batch, args.pool_model, args.paradigm)
            preds_list.extend([torch.argmax(preds_bag, dim=1)[i].item() for i in range(preds_bag.shape[0])])
            gt_list.extend([bag_label[i].item() for i in range(bag_label.shape[0])])

        test_error /= len(test_loader)
        test_loss /= len(test_loader)
        logger.log_string('Test====================')
        logger.log_string('Epoch: {}, Test Set, Loss: {:.4f}'.format(epoch, test_loss))
        cal_metrics(preds_list, gt_list, logger, epoch, train=False)

if __name__ == "__main__":
    logger.log_string('Start Training')
    # torch.cuda.empty_cache()
    # w = init_memory()
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch, args)
        train(epoch)

        if (epoch % 5 == 0) and (epoch >= 150):
            logger.log_string('Start Testing')
            test(epoch)
        
        if (epoch>=195):
            save_checkpoint()