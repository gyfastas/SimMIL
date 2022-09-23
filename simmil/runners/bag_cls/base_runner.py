import argparse
import builtins
import os
import random
import shutil
import time
import warnings
import os, sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from simmil.utils.logger import Logger
import json
from simmil.utils.core import AverageMeter, ProgressMeter, accuracy
import simmil.core.bag_cls.aggregators as aggregators
from simmil.core.bag_cls.bag_mil_model import BagMILModel
from simmil.datasets.bag_cls.bag_dataset import BagDataset
from simmil.datasets import ImageFolderDict
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, accuracy_score

class BaseRunner(object):
    def __init__(self, args):
        self.args = args
        self.logger = Logger(self.args)

    def adjust_learning_rate(self, optimizer, epoch):
        """Decay the learning rate based on schedule"""
        lr = self.args.lr
        for milestone in self.args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def build_losses(self):
        return nn.CrossEntropyLoss().cuda(self.args.gpu)

    def build_data_loaders(self):
        self.logger.info("Building datasets")

        train_dataset = BagDataset(dataset=ImageFolderDict(os.path.join(self.args.data, self.args.train_subdir),
                                transform=self.train_augmentation),
                                    label_file=self.args.train_label_file)
        
        val_dataset =  BagDataset(dataset=ImageFolderDict(os.path.join(self.args.data, self.args.eval_subdir),
                                    transform=self.val_augmentation),
                                    label_file=self.args.val_label_file)

        train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=self.args.batch_size,
        shuffle=True,
        num_workers=self.args.workers,
        drop_last=True,
        pin_memory=True,
    )

        val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=self.args.batch_size,
        shuffle=False,
        num_workers=self.args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)
        return train_loader, val_loader

    @property
    def train_augmentation(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        train_augmentation = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        normalize
        ])
        return train_augmentation

    @property
    def val_augmentation(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        val_augmentation = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        return val_augmentation

    def build_optimizer(self, model):
        # optimize only params that require grads
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        # assert len(parameters) == 2  # fc.weight, fc.bias
        optimizer = torch.optim.SGD(parameters, self.args.lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        return optimizer

    def build_model(self):
        # create model
        self.logger.info("=> creating model '{}'".format(self.args.arch))
        backbone_factory = models.__dict__[self.args.arch]
        aggregator_factory = aggregators.__dict__[self.args.aggregator]
        backbone_params = dict(pretrained=self.args.imgnet_pretrained)
        print("=> backbone param:", backbone_params)
        model = BagMILModel(backbone_factory=backbone_factory, aggregator_factory=aggregator_factory, class_num=self.args.class_num,
                           backbone_params=backbone_params)
        # freeze all layers but the last fc
        for name, param in model.backbone.named_parameters():
            param.requires_grad = False
        return model

    def set_device(self, model):
        if self.args.gpu is not None:
            model = model.cuda()
        return model

    def load_pretrained(self, model):
        try:
            self.logger.info("=> loading checkpoint '{}'".format(self.args.load))
            checkpoint = torch.load(self.args.load, map_location="cpu")
            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            if self.args.pretrained_type in ['moco']:
                # rename moco pre-trained keys
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                        # remove prefix
                        state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
                msg = model.backbone.load_state_dict(state_dict, strict=False)
            else:
                def load_model(model, state_dict, strict=False):
                    key_list = list(filter(lambda x: x.split(".")[-2]=="fc", state_dict.keys()))

                    for k in key_list:
                        state_dict.pop(k)
                        self.logger.info("Not loading {}".format(k))
                    
                    # Check module as prefix (temp)
                    key_list = list(filter(lambda x: x.startswith('module.'), state_dict.keys()))
                    for k in key_list:
                        self.logger.info("Transforming {} to {}".format(k, k.split(".",1)[1]))
                        state_dict["{}".format(k.split(".",1)[1])] = state_dict.pop("{}".format(k))
                    msg = model.backbone.load_state_dict(state_dict, strict=strict)
                    return model, msg
                model, msg = load_model(model, state_dict, strict=False)
            
            self.logger.info(msg)
            # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
            self.logger.info("=> loaded pre-trained model '{}'".format(self.args.load))
        except:
            self.logger.info("=> no checkpoint found at '{}'".format(self.args.load))

        return model
        
    def resume(self, model, optimizer):
        try:
            checkpoint_filepath = self.args.resume
            if os.path.isfile(checkpoint_filepath):
                self.logger.info("=> loading checkpoint '{}'".format(checkpoint_filepath))
                if self.args.gpu is None:
                    checkpoint = torch.load(checkpoint_filepath)
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = 'cuda:{}'.format(self.args.gpu)
                    checkpoint = torch.load(checkpoint_filepath, map_location=loc)

                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                self.args.start_epoch = checkpoint['epoch']
                self.best_acc1 = checkpoint['best_acc1']
                if self.args.gpu is not None:
                    # best_acc1 may be from a checkpoint from a different GPU
                    self.best_acc1 = self.best_acc1.to(self.args.gpu)
                self.logger.info("=> (resume) loaded checkpoint '{}' (epoch {})"
                    .format(self.args.resume, checkpoint['epoch']))
        except:
            self.logger.info("=> resume failed! Nothing resumed")

        return model, optimizer

    def run(self):
        model = self.build_model()
        self.args.start_epoch = 0
        if self.args.load:
            model = self.load_pretrained(model)
        model = self.set_device(model)
        optimizer = self.build_optimizer(model)
        if self.args.resume:
            model, optimizer = self.resume(model, optimizer)
        criterion = self.build_losses()
        train_loader, val_loader = self.build_data_loaders()

        # Check for grad requirement
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.logger.info("{} requires grad".format(name))
        cudnn.benchmark = True

        if self.args.evaluate:
            self.validate(val_loader, model, criterion)
            return

        if self.args.evaluate_before_train:
            self.validate(val_loader, model, criterion)

        self.best_acc1 = 0.0
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.adjust_learning_rate(optimizer, epoch)
            # train for one epoch
            self.train(train_loader, model, criterion, optimizer, epoch)

            # evaluate on validation set
            if epoch % self.args.validate_interval == 0:
                self.logger.info("Evaluating...")
                self.validate(val_loader, model, criterion)
            # remember best acc@1 and save checkpoint
            is_best = self.acc1 > self.best_acc1
            self.best_acc1 = max(self.acc1, self.best_acc1)

            self.save_checkpoint({
                'epoch': epoch + 1,
                'arch': self.args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': self.best_acc1,
                'optimizer' : optimizer.state_dict(),
                }, is_best, filename=os.path.join(self.args.log_dir, 'checkpoint_{:04d}.pth'.format(epoch)))

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(filename.rsplit('/',1)[0],'model_best.pth'))

    def get_meters(self):
        return dict(batch_time = AverageMeter('Time', ':6.3f'),
        data_time = AverageMeter('Data', ':6.3f'),
        losses = AverageMeter('Loss', ':.4e'),
        BACC = AverageMeter('BACC', ':6.2f'),
        BAUC = AverageMeter('BAUC', ':6.2f'),
        )

    def train(self, train_loader, model, criterion, optimizer, epoch):
        meter_set = self.get_meters()
        progress = ProgressMeter(
            len(train_loader),
            meter_set,
            prefix="Epoch: [{}]".format(epoch),
            logger=self.logger)
        
        model.train()
        model.backbone.eval()

        end = time.time()
        for i,  data  in enumerate(train_loader):
            images = data['data']
            target = data['label']
            # measure data loading time
            progress.update('data_time', time.time() - end)

            if self.args.gpu is not None:
                images = images.cuda(self.args.gpu)
                target = target.cuda(self.args.gpu)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc, _ = accuracy(output, target, topk=(1, 2))
            progress.update('losses', loss.item(), images.size(0))
            progress.update('BACC', acc[0], images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            progress.update('batch_time', time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                progress.display(i)


    def validate(self, val_loader, model, criterion):
        meter_set = self.get_meters()
        progress = ProgressMeter(
            len(val_loader),
            meter_set,
            prefix='Test: ',
            logger=self.logger)
        model.eval()
        outputs = []
        targets = []
        with torch.no_grad():
            end = time.time()
            for i, data in enumerate(val_loader):
                images = data['data']
                target = data['label'] 
                if self.args.gpu is not None:
                    images = images.cuda(self.args.gpu, non_blocking=True)
                    target = target.cuda(self.args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                outputs.append(output.cpu())
                targets.append(target.cpu())
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc, _ = accuracy(output, target, topk=(1, 2))
                progress.update('losses', loss.item(), images.size(0))
                progress.update('BACC', acc[0], images.size(0))

                # measure elapsed time
                progress.update('batch_time', time.time() - end)
                end = time.time()

                if i % self.args.print_freq == 0:
                    progress.display(i)

        # Compute AUC
        outputs = torch.cat(outputs, 0).softmax(-1).numpy()
        targets = torch.cat(targets, 0).numpy()
        cls_report = classification_report(targets, outputs.argmax(-1),
                        labels=[0,1],
                        target_names=["neg", "pos"])

        confusion_mat = confusion_matrix(targets, outputs.argmax(-1))
        self.logger.info("\n classification report: {}".format(cls_report))
        self.logger.info("\n confusion matrix: \n {}".format(confusion_mat))
        auc = roc_auc_score(targets, outputs[:, 1])

        for k, v in progress.meters.items():
            self.logger.info(' * {} {:.5f}'.format(k, v.avg))
    
        self.logger.info(' * BAUC {:.5f}'
            .format(auc))
        self.acc1 = meter_set['BACC'].avg
