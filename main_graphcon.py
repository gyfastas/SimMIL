from __future__ import print_function
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data_utils
import torch.optim as optim
from dataloader import HISMIL, HISMIL_DoubleAug
from Builder.e2e import Attention, GatedAttention, Res18, Res18_SAFS
from Builder import FeaAgg
from Builder.FeaAgg import ResBackbone, AggNet
from Builder.GraphCon import GraphCon
from tqdm import tqdm
from torchvision import models, transforms
import os
from logger import Logger
import datetime
from MemoryBank.memorybank import MemoryBank, InstanceDiscriminationLossModule, LocalAggregationLossModule, \
    Kmeans, _init_cluster_labels
from utils.utils import TwoCropsTransform
from utils.setup import process_config
from utils.utils import fix_bn, GaussianBlur, adjust_learning_rate, print_args, cal_metrics
import torch.nn as nn
from utils.utils import find_class_by_name, adjust_weight
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# Training settings
parser = argparse.ArgumentParser(description='PyTorch HISMIL')
# optimizer
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
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
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
#############################################################
parser.add_argument('--pretrained', action='store_true', default=True,
                    help='imagenet_pretrained and fix_bn to aviod shortcut')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--attention', type=str, default='Agg_Attention',
                    help='Agg_GAttention/Agg_SAttention/Agg_Attention')
parser.add_argument('--graph', type=str, default='GraphMILNet',
                    help='Choose graph')
parser.add_argument('--n_topk', type=int, default=4,
                    help='K of KNN')
parser.add_argument('--agg_mode', type=str, default='attention',
                    help='attention/graph/residual(concat)')
parser.add_argument('--ema', type=float, default=0.99, metavar='EMA',
                    help='exponential moving average')
#memory bank
parser.add_argument('--mb_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--mb-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
# loss weight
parser.add_argument('--weight_self', type=float, default=0)
parser.add_argument('--weight_mse', type=float, default=0)
parser.add_argument('--weight_bag', type=float, default=0)
# Data and logging
parser.add_argument('--ratio', type=float, default=0.8,
                    help='the ratio of train dataset')
parser.add_argument('--folder', type=int, default=0,
                    help='CV folder')
parser.add_argument('--data_dir', type=str, default='..',
                    help='local: ../'
                         'remote: /remote-home/my/datasets/BASH')
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
logger.backup_files(
        [x for x in os.listdir('./') if x.endswith('.py')] + [args.config])
# print('Load Train and Test Set')
loader_kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

ins_path = os.path.join(args.data_dir, 'patch/train')
bag_path = os.path.join(args.data_dir, 'WSI/train')

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

train_dataset = HISMIL_DoubleAug(bag_path, 256, TwoCropsTransform(transforms.Compose(aug_train)),
                                 floder=args.folder, ratio=args.ratio, train=True)
test_dataset = HISMIL_DoubleAug(bag_path, 256, TwoCropsTransform(transforms.Compose(aug_test)),
                                floder=args.folder, ratio=args.ratio, train=False)

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
###Original single bag loader
# train_loader = data_utils.DataLoader(train_dataset,
#                                      batch_size=1,
#                                      shuffle=True,
#                                      **loader_kwargs)
# test_loader = data_utils.DataLoader(test_dataset,
#                                     batch_size=1,
#                                     shuffle=False,
#                                     **loader_kwargs)
print('Init Model')

# memory bank
if args.weight_self != 0:
    data_len = len(train_dataset)*48 #WSIs * patches
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

# choose agg model
attention = find_class_by_name(args.attention, [FeaAgg])
graph = find_class_by_name(args.graph, [FeaAgg])
# consistent = (args.weight_mse != 0) | (args.weight_bag != 0)
bag_label = torch.stack(([torch.tensor(i[1]) for i in train_dataset.folder])).cuda()
model = GraphCon(args.agg_mode,
                 ResBackbone, args.arch, args.pretrained,
                 AggNet, attention, graph, args.n_topk,
                 m=args.ema, T=args.mb_t, dim=args.mb_dim, K=len(train_dataset),
                 bag_label=bag_label)
if args.cuda:
    model.cuda()
print(model)

parameters = [{'params': model.parameters()}]
optimizer = optim.Adam(parameters, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

criterion_ce = nn.CrossEntropyLoss()
criterion_mse = nn.MSELoss(reduction='sum')
criterion_l1 = nn.L1Loss()

def train_multi_batch(epoch):
    model.train()
    train_loss = 0.
    CE_loss = 0.
    INS_loss = 0.
    MSE_loss = 0.
    BagInfo_loss = 0.
    preds_list = []
    gt_list = []
    # adjusted_weight = adjust_weight(args.weight_mse, epoch, args)
    for batch_idx, (bag_idx, ins_idx, data, label, batch) in enumerate(tqdm(train_loader)):
        bag_label = label
        img_q = data[0]
        img_k = data[1]
        if args.cuda:
            img_k, img_q, bag_label = img_k.cuda(), img_q.cuda(), bag_label.cuda()
            bag_idx, ins_idx, batch = bag_idx.cuda(), ins_idx.cuda(), batch.cuda()
        # reset gradients
        optimizer.zero_grad()
        # forward
        Y_prob_q, (logits, labels), (self_fea_q, self_fea_k), (Attention_q, Attention_k), (Affinity_q, Affinity_k)\
            = model(img_q, img_k, batch, bag_idx, bag_label) ##preds: [N] , gt： [N]
        # calculate loss and metrics
        ce_loss = criterion_ce(Y_prob_q, bag_label)
        total_loss = ce_loss
        CE_loss += ce_loss.item()
        if args.weight_bag != 0:
            baginfo_loss = criterion_ce(logits, labels)
            BagInfo_loss += baginfo_loss.item()
            total_loss = total_loss + baginfo_loss
        if args.weight_mse != 0:
            mse_loss = criterion_mse(Affinity_q, Affinity_k)/data[0].shape[0]
            MSE_loss += mse_loss.item()
            total_loss = total_loss + mse_loss
        if args.weight_self != 0:
            ins_loss, new_data_memory = Ins_Module(
                ins_idx, self_fea_q, torch.arange(len(config.gpu_device)))
            INS_loss += ins_loss.item()
            total_loss = total_loss + ins_loss

        train_loss += total_loss.item()
        preds_list.extend([torch.argmax(Y_prob_q, dim=1)[i].item() for i in range(Y_prob_q.shape[0])])
        gt_list.extend([bag_label[i].item() for i in range(bag_label.shape[0])])
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        #update memo and cluster
        if args.weight_self != 0:
            with torch.no_grad():
                memory_bank.update(ins_idx, new_data_memory)
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
    BagInfo_loss /= len(train_loader)
    train_loss /= len(train_loader)
    logger.log_string('====================Train')
    logger.log_string('Epoch: {}, ceLoss: {:.4f}, insLoss: {:.4f}, MseLoss: {:.4f}, BagLoss: {:.4f},Train loss: {:.4f}'
                      .format(epoch, CE_loss, INS_loss, MSE_loss, BagInfo_loss, train_loss))
    cal_metrics(preds_list, gt_list, args, logger, epoch, 'train')
    # if epoch == args.epochs-1:
    #     torch.save({'state_dict': model1.state_dict()}, 'checkpoint_{:04d}.pth.tar'.format(epoch))

def test(epoch):
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
    logger.log_string('Start Training')
    print_args(args, logger)
    for epoch in range(1, args.epochs + 1):
        # test(epoch)
        adjust_learning_rate(optimizer, epoch, args, logger)
        # train(epoch)
        train_multi_batch(epoch)
        # test(epoch)

        if epoch % 5 == 0:
            logger.log_string('Start Testing')
            test(epoch)
