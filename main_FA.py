from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

from dataloader import MnistBags, HISMIL, SimpleMIL
from model import Attention, GatedAttention, Res18, Res18_SAFS
from model import ResBackbone
from model import Agg_GAttention, Agg_SAttention, Agg_Attention, Agg_Residual
from tqdm import tqdm
import numpy as np
import math
from torchvision import models, transforms
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import os
from PIL import ImageFilter
import random
from logger import Logger
import datetime
from utils.utils import MemoryBank, InstanceDiscriminationLossModule, LocalAggregationLossModule, Kmeans
from utils.setup import process_config
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
from utils.utils import fix_bn, GaussianBlur, adjust_learning_rate, print_args, cal_metrics, cal_metrics_train,\
    _init_cluster_labels
from models.resnet import resnet18

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='gated_attention',
                    help='Choose b/w attention and gated_attention')
parser.add_argument('--mil', type=int, default=0,
                    help='0=Attention_MIL, 1=SimpleMIL')
#############################################################
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='imagenet_pretrained and fix_bn to aviod shortcut')
# parser.add_argument('--fixbn', action='store_true', default=False,
#                     help='fix bn')
parser.add_argument('--insnorm', action='store_true', default=False,
                    help='Ins norm')
parser.add_argument('--ratio', type=float, default=0.8,
                    help='the ratio of train dataset')
parser.add_argument('--folder', type=int, default=0,
                    help='CV folder')


parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--weight', type=float, default=1)
parser.add_argument('--data_dir', type=str, default='..',
                    help='local: ../'
                         'remote: /remote-home/my/datasets/BASH')
parser.add_argument('--log_dir', default='./experiments', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--config', type=str, default='./config/imagenet_ir.json',
                        help='path to config file')

args = parser.parse_args()
config = process_config(args.config)

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

#setting logger
log_path = os.path.join(args.log_dir, str(datetime.datetime.now()))
logger = Logger(logdir=log_path)
logger.backup_files(
        [x for x in os.listdir('./') if x.endswith('.py')] + [args.config])
# print('Load Train and Test Set')
loader_kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

ins_path = os.path.join(args.data_dir, 'patch/train')
# ins_test_path = os.path.join(args.data_dir, 'patch/test')
bag_path = os.path.join(args.data_dir, 'WSI/train')
# bag_test_path = os.path.join(args.data_dir, 'WSI/test')

aug_train = [
            #aug_plus
            # transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            # ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            # transforms.RandomHorizontalFlip(),
            #aug
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            #lincls
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]
aug_test = [transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

train_dataset = HISMIL(bag_path, 256, aug_train, floder=args.folder, ratio=args.ratio, train=True)
test_dataset = HISMIL(bag_path, 256, aug_test, floder=args.folder, ratio=args.ratio, train=False)
train_loader = data_utils.DataLoader(train_dataset,
                                     batch_size=1,
                                     shuffle=True,
                                     **loader_kwargs)
test_loader = data_utils.DataLoader(test_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    **loader_kwargs)
print('Init Model')




# memory bank
data_len = len(train_loader)*48 #WSIs * patches
memory_bank = MemoryBank(data_len,  config.model_params.out_dim, config.gpu_device) #128D, gpu-id=0
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
       config.loss_params.kmeans_freq = (
                len(train_loader) )
else:
    Ins_Module = InstanceDiscriminationLossModule(memory_bank.bank_broadcast,
                     cluster_labels_broadcast=None,
                     k=config.loss_params.k,
                     t=config.loss_params.t,
                     m=config.loss_params.m)

# feature extractor
if args.insnorm:
    model1 = resnet18(num_classes=128)
else:
    model1 = ResBackbone(args.arch, args.pretrained)
# aggregation
if args.model == 'attention':
    model2 = Agg_Attention()
if args.model == 'gated_attention':
    model2 = Agg_GAttention()
elif args.model == 'self':
    model2 = Agg_SAttention()
elif args.model == 'residual':
    model2 = Agg_Residual()

if args.cuda:
    model1.cuda()
    model2.cuda()


# if args.pretrained:
#     if os.path.isfile(args.pretrained):
#         print("=> loading checkpoint '{}'".format(args.pretrained))
#         checkpoint = torch.load(args.pretrained, map_location="cpu")
#
#         # rename moco pre-trained keys
#         state_dict = checkpoint['state_dict']
#         # for k in list(state_dict.keys()):
#         #     # retain only encoder_q up to before the embedding layer
#         #     if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
#         #         # remove prefix
#         #         state_dict[k[len("module.encoder_q."):]] = state_dict[k]
#         #     # delete renamed or unused k
#         #     del state_dict[k]
#         for k in list(state_dict.keys()):
#             if k.startswith('attention') or k.startswith('classifier'):
#                 del state_dict[k]
#
#
#         args.start_epoch = 0
#         msg = model1.load_state_dict(state_dict, strict=False)
#         if msg.missing_keys:
#             print(msg.missing_keys)
#         print("=> loaded pre-trained model '{}'".format(args.pretrained))
#     else:
#         print("=> no checkpoint found at '{}'".format(args.pretrained))
#
# parameters = list(filter(lambda p: p.requires_grad, model2.parameters()))
print(model1)
print(model2)
parameters = [{'params': model1.parameters()}, {'params': model2.parameters()}]
optimizer = optim.Adam(parameters, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)


def train(epoch):
    model1.train()
    if not args.insnorm:
        model1.apply(fix_bn)
    model2.train()
    train_loss = 0.
    CE_loss = 0.
    INS_loss = 0.
    train_error = 0.
    preds_list = []
    gt_list = []
    for batch_idx, (idx, data, label) in enumerate(tqdm(train_loader)):
        bag_label = label
        idx = idx.squeeze(0)
        # print(idx)
        if args.cuda:
            data, bag_label, idx = data.cuda(), bag_label.cuda(), idx.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics

        feature, self_feat = model1(data)
        ce_loss, error, _, preds, gt = model2.calculate_objective(feature, bag_label)


        # print(ce_loss, iwns_loss)
        # loss
        if args.weight != 0:
            ins_loss, new_data_memory = Ins_Module(idx, self_feat, torch.arange(len(config.gpu_device)))
            loss = ce_loss + args.weight * ins_loss
        else:
            loss = ce_loss

        # print info
        CE_loss += ce_loss.item()
        # INS_loss += ins_loss.item()
        train_loss += loss.item()
        train_error += error
        preds_list.append(preds)
        gt_list.append(gt)

        # backward pass
        loss.backward()
        # step
        optimizer.step()
        #update memo and cluster
        if args.weight != 0:
            with torch.no_grad():
                memory_bank.update(idx, new_data_memory)
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
                        Ins_Module.cluster_label_broadcast[i] = cluster_labels.to(device)

    # calculate loss and error for epoch
    CE_loss /= len(train_loader)
    INS_loss /= len(train_loader)
    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    logger.log_string('====================Train')
    # logger.log_string('Epoch: {}, ceLoss: {:.4f}, Train error: {:.4f}'
    #                   .format(epoch, ce_loss.item(), train_error))
    logger.log_string('Epoch: {}, ceLoss: {:.4f}, insLoss: {:.4f}, Train loss: {:.4f}'
                      .format(epoch, CE_loss, INS_loss, train_loss))
    cal_metrics_train(preds_list, gt_list, args, logger, epoch)
    # if epoch == args.epochs-1:
    #     torch.save({'state_dict': model1.state_dict()}, 'checkpoint_{:04d}.pth.tar'.format(epoch))


def test(epoch):
    model1.eval()
    model2.eval()

    # test_loss = 0.
    # test_error = 0.
    # preds_list = []
    # gt_list = []
    # with torch.no_grad():
    #     for batch_idx, (_, data, label) in enumerate(tqdm(train_loader)):
    #         bag_label = label
    #         # instance_labels = label[1]
    #         if args.cuda:
    #             data, bag_label = data.cuda(), bag_label.cuda()
    #         data, bag_label = Variable(data), Variable(bag_label)
    #         feature, _ = model1(data)
    #         preds, gt = model2.calculate_classification_error(feature, bag_label)
    #         preds_list.append(preds)
    #         gt_list.append(gt)
    #
    #     test_error /= len(test_loader)
    #     test_loss /= len(test_loader)
    #     print('========Test============')
    #     cal_metrics(preds_list, gt_list, args, logger, epoch)
    #     print('\nTest Set, Test error: {:.4f}'.format(test_error))

    test_loss = 0.
    test_error = 0.
    preds_list = []
    gt_list = []
    with torch.no_grad():
        for batch_idx, (_, data, label) in enumerate(tqdm(test_loader)):
            bag_label = label
            # instance_labels = label[1]
            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
            data, bag_label = Variable(data), Variable(bag_label)
            feature, _ = model1(data)
            preds, gt = model2.calculate_classification_error(feature, bag_label)
            # test_loss += loss.item()
            # # error, predicted_label = model.calculate_classification_error(data, bag_label)
            # test_error += error
            preds_list.append(preds)
            gt_list.append(gt)
            # if batch_idx < 5:  # plot bag labels and instance labels for first 5 bags
            #     bag_level = (bag_label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))
            #     instance_level = list(zip(instance_labels.numpy()[0].tolist(),
            #                          np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))
            #
            #     print('\nTrue Bag Label, Predicted Bag Label: {}\n'
            #           'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))

        test_error /= len(test_loader)
        test_loss /= len(test_loader)
        logger.log_string('Test====================')
        logger.log_string(
            '\nEpoch: {}, Test Set, Loss: {:.4f}, Test error: {:.4f}'.format(epoch, test_loss, test_error))
        cal_metrics(preds_list, gt_list, args, logger, epoch)

if __name__ == "__main__":
    logger.log_string('Start Training')
    print_args(args, logger)
    for epoch in range(1, args.epochs + 1):
        # test()
        adjust_learning_rate(optimizer, epoch, args, logger)
        train(epoch)
        # test()
        if epoch % 5 == 0:
            logger.log_string('Start Testing')
            test(epoch)
