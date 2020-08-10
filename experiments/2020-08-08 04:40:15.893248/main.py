from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

from dataloader import HISMIL, SimpleMIL
from model import Attention, GatedAttention, Res18, Res18_SAFS
from tqdm import tqdm


from torchvision import models, transforms
from logger import Logger

import os
import datetime
from utils.utils import fix_bn, GaussianBlur, adjust_learning_rate, print_args,cal_metrics, cal_metrics_train

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--model', type=str, default='attention',
                    help='Choose b/w attention and gated_attention')
parser.add_argument('--mil', type=int, default=0,
                    help='0=Attention_MIL, 1=SimpleMIL')
parser.add_argument('--pretrained', default='./checkpoint_0199.pth.tar', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--log_dir', default='./experiments', type=str,
                    help='path to moco pretrained checkpoint')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

#setting logger
log_path = os.path.join(args.log_dir, str(datetime.datetime.now()))
logger = Logger(logdir=log_path)
logger.backup_files(
        [x for x in os.listdir('./') if x.endswith('.py')])

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
ins_train_path = '../dataset_patch/train'
ins_test_path = '../dataset_patch/test'
bag_train_path = '../dataset_WSI/train'
bag_test_path = '../dataset_WSI/test'

aug_train = [
            # aug_plus
            # transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            # ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            # transforms.RandomHorizontalFlip(),
            # aug
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            # transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            # lincls
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]
aug_test = [transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])]
if args.mil == 0:
    # attention MIL
    train_path = bag_train_path
    test_path = bag_test_path
    train_loader = data_utils.DataLoader(HISMIL(train_path, 256, aug_train),
                                                        batch_size=1,
                                                        shuffle=True,
                                                        **loader_kwargs)
    test_loader = data_utils.DataLoader(HISMIL(test_path, 256, aug_test),
                                                        batch_size=1,
                                                        shuffle=False,
                                                        **loader_kwargs)
    print('Init Model')
    if args.model == 'attention':
        model = Attention()
    elif args.model == 'gated_attention':
        model = GatedAttention()
    elif args.model == 'self':
        model = Res18_SAFS()
    if args.cuda:
        model.cuda()

elif args.mil == 1:
    # simple MIL
    train_path = ins_train_path
    test_path = bag_test_path
    train_loader = data_utils.DataLoader(SimpleMIL(train_path, train=True),
                                         batch_size=64,
                                         shuffle=True,
                                         **loader_kwargs)
    test_loader = data_utils.DataLoader(HISMIL(test_path, 256, train=False),
                                                        batch_size=1,
                                                        shuffle=False,
                                                        **loader_kwargs)
    model = Res18()
    if args.cuda:
        model.cuda()
# elif args.mil == 2:
#     # feature extractor + aggregation
#     train_path = bag_train_path
#     test_path = bag_test_path
#     train_loader = data_utils.DataLoader(HISMIL(train_path, 256, train=True),
#                                                         batch_size=1,
#                                                         shuffle=True,
#                                                         **loader_kwargs)
#     test_loader = data_utils.DataLoader(HISMIL(test_path, 256, train=False),
#                                                         batch_size=1,
#                                                         shuffle=False,
#                                                         **loader_kwargs)
#     print('Init Model')
#     if args.model == 'attention':
#         model = Attention()
#     elif args.model == 'gated_attention':
#         model = GatedAttention()
#     elif args.model == 'self':
#         model = Res18_SAFS()
#     if args.cuda:
#         model.cuda()
#
#
#     for name, param in model.named_parameters():
#         if name not in ['attention.0.weight','attention.0.bias','attention.2.weight','attention.2.bias',
#         'attention_V.0.weight','attention_V.0.bias','attention_U.0.weight','attention_U.0.bias',
#         'attention_weights.weight','attention_weights.bias','classifier.weight','classifier.bias']:
#             param.requires_grad = False
#     model.attention[0].weight.data.normal_(mean=0.0, std=0.01)
#     model.attention[0].bias.data.zero_()
#     model.attention[2].weight.data.normal_(mean=0.0, std=0.01)
#     model.attention[2].bias.data.zero_()
#     model.attention_V[0].weight.data.normal_(mean=0.0, std=0.01)
#     model.attention_V[0].bias.data.zero_()
#     model.attention_U[0].weight.data.normal_(mean=0.0, std=0.01)
#     model.attention_U[0].bias.data.zero_()
#     model.attention_weights.weight.data.normal_(mean=0.0, std=0.01)
#     model.attention_weights.bias.data.zero_()
#     model.classifier.weight.data.normal_(mean=0.0, std=0.01)
#     model.classifier.bias.data.zero_()
#     if args.pretrained:
#         if os.path.isfile(args.pretrained):
#             print("=> loading checkpoint '{}'".format(args.pretrained))
#             checkpoint = torch.load(args.pretrained, map_location="cpu")
#
#             # rename moco pre-trained keys
#             state_dict = checkpoint['state_dict']
#             for k in list(state_dict.keys()):
#                 # retain only encoder_q up to before the embedding layer
#                 if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
#                     # remove prefix
#                     state_dict[k[len("module.encoder_q."):]] = state_dict[k]
#                 # delete renamed or unused k
#                 del state_dict[k]
#
#             args.start_epoch = 0
#             msg = model.load_state_dict(state_dict, strict=False)
#             print()
#             assert set(msg.missing_keys) == {'attention.0.weight','attention.0.bias','attention.2.weight','attention.2.bias',
#         'attention_V.0.weight','attention_V.0.bias','attention_U.0.weight','attention_U.0.bias',
#         'attention_weights.weight','attention_weights.bias','classifier.weight','classifier.bias'}
#
#             print("=> loaded pre-trained model '{}'".format(args.pretrained))
#         else:
#             print("=> no checkpoint found at '{}'".format(args.pretrained))
#     if args.cuda:
#         model.cuda()
#     parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
#     assert len(parameters) == 12  # fc.weight, fc.bias

print(model)
print('Path of dataset:{},{}'.format(train_path, test_path))

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)



def train(epoch):

    model.train()
    if args.mil == 0:
        model.apply(fix_bn)
    train_loss = 0.
    train_error = 0.
    preds_list = []
    gt_list = []
    for batch_idx, (data, label) in enumerate(tqdm(train_loader)):
        bag_label = label
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        loss, error, _, preds, gt = model.calculate_objective(data, bag_label)
        train_loss += loss.item()
        preds_list.append(preds)
        gt_list.append(gt)
        train_error += error
        # backward pass
        loss.backward()
        # step
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    logger.log_string('====================Train')
    logger.log_string('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss, train_error))
    cal_metrics_train(preds_list, gt_list, args, logger, epoch)
    if epoch == args.epochs-1:
        torch.save({'state_dict': model.state_dict()}, 'checkpoint_{:04d}.pth.tar'.format(epoch))
def test(epoch):
    model.eval()
    # test_loss = 0.
    # test_error = 0.
    # preds_list = []
    # gt_list = []
    # with torch.no_grad():
    #     for batch_idx, (data, label) in enumerate(tqdm(train_loader)):
    #         bag_label = label[0]
    #         # instance_labels = label[1]
    #         if args.cuda:
    #             data, bag_label = data.cuda(), bag_label.cuda()
    #         data, bag_label = Variable(data), Variable(bag_label)
    #         loss, error, attention_weights, preds, gt = model.calculate_objective(data, bag_label)
    #         test_loss += loss.item()
    #         # error, predicted_label = model.calculate_classification_error(data, bag_label)
    #         test_error += error
    #         preds_list.append(preds)
    #         gt_list.append(gt)
    #         # if batch_idx == 1:
    #         #     print(preds, gt)
    #         # if batch_idx < 5:  # plot bag labels and instance labels for first 5 bags
    #         #     bag_level = (bag_label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))
    #         #     instance_level = list(zip(instance_labels.numpy()[0].tolist(),
    #         #                          np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))
    #         #
    #         #     print('\nTrue Bag Label, Predicted Bag Label: {}\n'
    #         #           'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))
    #
    #     test_error /= len(test_loader)
    #     test_loss /= len(test_loader)
    #     print('========Test============')
    #     cal_metrics(preds_list, gt_list)
    #     print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss, test_error))
    test_loss = 0.
    test_error = 0.
    preds_list = []
    gt_list = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(tqdm(test_loader)):
            bag_label = label
            # instance_labels = label[1]
            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
            data, bag_label = Variable(data), Variable(bag_label)
            preds, gt = model.calculate_classification_error(data, bag_label)
            preds_list.append(preds)
            gt_list.append(gt)
        test_error /= len(test_loader)
        test_loss /= len(test_loader)
        logger.log_string('Test====================')

        logger.log_string('\nEpoch: {}, Test Set, Loss: {:.4f}, Test error: {:.4f}'.format(epoch,test_loss, test_error))
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
