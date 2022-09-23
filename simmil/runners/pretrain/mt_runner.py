
import re
import argparse
import os
import shutil
import time
import math
import logging
import typing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets
from simmil.utils.core import AverageMeter, ProgressMeter, accuracy
from simmil.core.mean_teacher import architectures, augmentations, data, losses, ramps, cli
from simmil.core.mean_teacher.model import MeanTeacher
from simmil.core.mean_teacher.data import NO_LABEL, TwoStreamDataLoader
from simmil.datasets import ImageFolderDict
from .cls_runner import ClsRunner

class MeanTeacherRunner(ClsRunner):
    def __init__(self, gpu, ngpus_per_node, args):
        super().__init__(gpu, ngpus_per_node, args)
        self.global_step = 0
        assert self.args.dataset in ['NCTCRC']

    def adjust_learning_rate(self, optimizer, epoch, step_in_epoch, total_steps_in_epoch):
        lr = self.args.lr
        epoch = epoch + step_in_epoch / total_steps_in_epoch

        # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
        lr = ramps.linear_rampup(epoch, self.args.lr_rampup) * (self.args.lr - self.args.initial_lr) +self.args.initial_lr

        # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
        if self.args.lr_rampdown_epochs:
            lr *= ramps.cosine_rampdown(epoch, self.args.lr_rampdown_epochs)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def set_ddp_batch_size(self):
        self.args.batch_size = int(self.args.batch_size / self.ngpus_per_node)
        self.args.labeled_batch_size = int(self.args.labeled_batch_size / self.ngpus_per_node)

    def build_model(self):
        model_factory = architectures.__dict__[self.args.arch]
        model_params = dict(pretrained=self.args.imgnet_pretrained, num_classes=self.args.class_num)
        model = MeanTeacher(self.args.ema_decay, model_factory, **model_params)

        return model

    def resume(self, model, optimizer):
        if self.args.resume:
            try:
                self.logger.info("=> loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.args.start_epoch = checkpoint['epoch']
                self.global_step = checkpoint['global_step']
                self.best_prec1 = checkpoint['best_prec1']
                model.module.model.load_state_dict(checkpoint['state_dict'])
                model.module.ema_model.load_state_dict(checkpoint['ema_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                self.logger.info("=> loaded checkpoint '{}' (epoch {})".format(self.args.resume, checkpoint['epoch']))
            except:
                self.logger.info("=> resume failed")

        return model, optimizer

    def run(self):
        model = self.build_model()
        model = self.set_device(model)
        criterion = self.build_losses()
        optimizer = self.build_optimizer(model)
        model, optimizer = self.resume(model, optimizer)
        train_loader, eval_loader = self.build_data_loaders()
        cudnn.benchmark = True

        if self.args.evaluate:
            self.logger.info("Evaluating the primary model:")
            model.module.use_ema = False
            self.validate(eval_loader, model, criterion)
            self.logger.info("Evaluating the EMA model:")
            model.module.use_ema = True
            self.validate(eval_loader, model, criterion)
            return

        best_prec1 = 0
        for epoch in range(self.args.start_epoch, self.args.epochs):
            start_time = time.time()
            # train for one epoch
            self.train(train_loader, model, optimizer, epoch)
            self.logger.info("--- training epoch in %s seconds ---" % (time.time() - start_time))

            if self.args.evaluation_epochs and (epoch + 1) % self.args.evaluation_epochs == 0:
                start_time = time.time()
                self.logger.info("Evaluating the primary model:")
                model.module.use_ema = False
                prec1 = self.validate(eval_loader, model, criterion)
                self.logger.info("Evaluating the EMA model:")
                model.module.use_ema = True
                ema_prec1 = self.validate(eval_loader, model, criterion)
                self.logger.info("--- validation in %s seconds ---" % (time.time() - start_time))
                is_best = ema_prec1 > best_prec1
                best_prec1 = max(ema_prec1, best_prec1)
            else:
                is_best = False

            if not self.args.multiprocessing_distributed or (self.args.multiprocessing_distributed
                                and self.args.rank % self.ngpus_per_node == 0):
                if self.args.checkpoint_epochs and (epoch + 1) % self.args.checkpoint_epochs == 0:
                    self.save_checkpoint({
                        'epoch': epoch + 1,
                        'global_step': self.global_step,
                        'state_dict': model.module.model.state_dict(),
                        'ema_state_dict': model.module.ema_model.state_dict(),
                        'best_prec1': best_prec1,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best, filename=os.path.join(self.args.log_dir, 'checkpoint_{:04d}.pth'.format(epoch)))

    def build_data_loaders(self):
        transforms = augmentations.nctcrc(self.args)
        train_transformation = transforms['train_transformation']
        eval_transformation = transforms['eval_transformation']
        traindir = os.path.join(self.args.data, self.args.train_subdir)
        evaldir = os.path.join(self.args.data, self.args.eval_subdir)
        train_dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)

        with open(self.args.labels) as f:
            labels = dict(line.split(' ') for line in f.read().splitlines())
        labeled_subset, unlabeled_subset = data.relabel_dataset(train_dataset, labels)
        
        if self.args.distributed:
            labeled_sampler = torch.utils.data.distributed.DistributedSampler(labeled_subset)
            unlabeled_sampler = torch.utils.data.distributed.DistributedSampler(unlabeled_subset)
        else:
            labeled_sampler = None
            unlabeled_sampler = None

        labeled_train_loader = torch.utils.data.DataLoader(
            labeled_subset,
            batch_size=self.args.labeled_batch_size,
            shuffle=(labeled_sampler is None),
            sampler=labeled_sampler,
            num_workers=self.args.workers,
            drop_last=True,
            pin_memory=True,
        )

        unlabeled_train_loader =  torch.utils.data.DataLoader(
            unlabeled_subset,
            batch_size=self.args.batch_size - self.args.labeled_batch_size,
            shuffle=(unlabeled_sampler is None),
            sampler=unlabeled_sampler,
            num_workers=self.args.workers,
            drop_last=True,
            pin_memory=True,
        )

        train_loader = TwoStreamDataLoader(unlabeled_train_loader, labeled_train_loader)

        eval_loader = torch.utils.data.DataLoader(
            ImageFolderDict(evaldir, eval_transformation),
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,  # Needs images twice as fast
            pin_memory=True,
            drop_last=False)

        return train_loader, eval_loader

    def get_meters(self):
        return dict(batch_time = AverageMeter('Time', ':6.3f'),
        data_time = AverageMeter('Data', ':6.3f'),
        losses = AverageMeter('Loss', ':.4e'),
        top1 = AverageMeter('Acc@1', ':6.2f'),
        top5 = AverageMeter('Acc@5', ':6.2f'),
        ema_top1 = AverageMeter('EMA Acc@1', ':6.2f'),
        ema_top5 = AverageMeter('EMA Acc@5', ':6.2f'),
        lr = AverageMeter('lr', ''),
        res_loss = AverageMeter('ResLoss', ':.4e'),
        class_loss = AverageMeter('ClassLoss', ':.4e'),
        ema_class_loss = AverageMeter('EMAClassLoss', ':.4e'),
        cons_weight = AverageMeter('ConsWeight', ':.4e'),
        cons_loss = AverageMeter('ConsLoss', ':.4e'),
        )

    def train(self, train_loader, model, optimizer, epoch):
        class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
        if self.args.consistency_type == 'mse':
            consistency_criterion = losses.softmax_mse_loss
        elif self.args.consistency_type == 'kl':
            consistency_criterion = losses.softmax_kl_loss
        else:
            assert False, self.args.consistency_type
        
        residual_logit_criterion = losses.symmetric_mse_loss

        meter_set = self.get_meters()
        progress = ProgressMeter(
            len(train_loader),
            meter_set,
            prefix="Epoch: [{}]".format(epoch),
            logger=self.logger)

        # switch to train mode
        model.train()
        model.module.train()
        model.module.single_head = False
        res_loss = 0
        consistency_loss = 0
        end = time.time()

        with model.join():
            for i, ((img, ema_img), target) in enumerate(train_loader):
                # measure data loading time
                self.adjust_learning_rate(optimizer, epoch, i, len(train_loader))
                if self.args.gpu is not None:
                    img = img.cuda(self.args.gpu, non_blocking=True)
                    ema_img = ema_img.cuda(self.args.gpu, non_blocking=True)
                    target = target.cuda(self.args.gpu, non_blocking=True)
                progress.update('data_time', time.time() - end)
                progress.update('lr', optimizer.param_groups[0]['lr'])
                minibatch_size = self.args.batch_size
                labeled_minibatch_size = self.args.labeled_batch_size
                
                ema_logit = model(ema_img, True)
                logit = model(img, False)
                if isinstance(logit, typing.Sequence):
                    class_logit, cons_logit = logit
                    ema_logit, _ = ema_logit
                
                ema_logit = ema_logit.detach()

                if self.args.logit_distance_cost >= 0:
                    res_loss = self.args.logit_distance_cost * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
                    progress.update('res_loss', res_loss.item())

                class_loss = class_criterion(class_logit, target) / minibatch_size
                progress.update('class_loss', class_loss.item())
                ema_class_loss = class_criterion(ema_logit, target) / minibatch_size
                progress.update('ema_class_loss', ema_class_loss.item())

                if self.args.consistency:
                    consistency_weight = self.get_current_consistency_weight(epoch)
                    progress.update('cons_weight', consistency_weight)
                    consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit) / minibatch_size
                    progress.update('cons_loss', consistency_loss.item())

                loss = class_loss + consistency_loss + res_loss
                progress.update('losses', loss.item())
                # target: LongTensor([BS]), where[:LBS] are -1
                prec1, prec5 = accuracy(class_logit, target, topk=(1, 5))
                progress.update('top1', prec1[0], labeled_minibatch_size)
                progress.update('top5', prec5[0], labeled_minibatch_size)

                ema_prec1, ema_prec5 = accuracy(ema_logit, target, topk=(1, 5))
                progress.update('ema_top1', ema_prec1[0], labeled_minibatch_size)
                progress.update('ema_top5', ema_prec5[0], labeled_minibatch_size)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.global_step += 1
                if self.args.distributed:
                    model.module.momentum_update_ema_model(self.global_step)
                else:
                    model.momentum_update_ema_model(self.global_step)
                # measure elapsed time
                progress.update('batch_time', time.time() - end)
                end = time.time()

                if i % self.args.print_freq == 0:
                    progress.display(i)

    def get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.args.consistency * ramps.sigmoid_rampup(epoch, self.args.consistency_rampup)
