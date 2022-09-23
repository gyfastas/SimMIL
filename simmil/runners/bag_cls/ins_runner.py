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
from utils.logger import Logger
import json
from utils.core import AverageMeter, ProgressMeter, accuracy
import aggregators
from core.ins_mil_model import InsMILModel
from datasets.debug_dataset import DebugDataset
from datasets.bag_dataset import BagDataset
from datasets.my_imagefolder import MyImageFolder
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, accuracy_score
from .assumption_runner import AssumptionRunner

class InsRunner(AssumptionRunner):
    def __init__(self, args):
        self.args = args
        self.logger = Logger(self.args)

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

    def build_model(self):
        # create model
        self.logger.info("=> creating model '{}'".format(self.args.arch))
        backbone_factory = models.__dict__[self.args.arch]
        # TODO: aggregator params
        model = InsMILModel(backbone_factory=backbone_factory, ins_aggregation=self.args.ins_aggregation, class_num=self.args.class_num,
                            pos_class=self.args.pos_class)
        # freeze all layers but the last fc
        for name, param in model.backbone.named_parameters():
            param.requires_grad = False
        return model

    def set_device(self, model):
        if self.args.gpu is not None:
            model = model.cuda()
        return model
        
    def resume(self, model):

        checkpoint_filepath = self.args.resume
        if os.path.isfile(checkpoint_filepath):
            self.logger.info("=> loading checkpoint '{}'".format(checkpoint_filepath))
            if self.args.gpu is None:
                checkpoint = torch.load(checkpoint_filepath)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(self.args.gpu)
                checkpoint = torch.load(checkpoint_filepath, map_location=loc)

            def load_model(model, state_dict, strict=False):                    
                # Check module as prefix (temp)
                key_list = list(filter(lambda x: x.startswith('module.'), state_dict.keys()))
                for k in key_list:
                    self.logger.info("Transforming {} to {}".format(k, k.split(".",1)[1]))
                    state_dict["{}".format(k.split(".",1)[1])] = state_dict.pop("{}".format(k))
                
                if self.args.pretrained_type == "mt":
                    state_dict["fc.weight"] = state_dict["fc1.weight"]
                    state_dict["fc.bias"] = state_dict["fc1.bias"]
                msg = model.load_state_dict(state_dict, strict=strict)
                return model, msg
            
            state_dict = checkpoint['state_dict']
            model.backbone, msg = load_model(model.backbone, state_dict, strict=False)
            self.logger.info(msg)
            self.logger.info("=> (resume) loaded checkpoint '{}' (epoch {})"
                .format(self.args.resume, checkpoint['epoch']))

        return model

    def run(self):
        model = self.build_model()
        self.args.start_epoch = 0
        model = self.set_device(model)
        if self.args.resume:
            model = self.resume(model)
        _, val_loader = self.build_data_loaders()

        # Check for grad requirement
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.logger.info("{} requires grad".format(name))
        cudnn.benchmark = True
        self.validate(val_loader, model)

    def get_meters(self):
        return dict(batch_time = AverageMeter('Time', ':6.3f'),
        data_time = AverageMeter('Data', ':6.3f'),
        losses = AverageMeter('Loss', ':.4e'),
        BACC = AverageMeter('BACC', ':6.2f')
        )

    def validate(self, val_loader, model):
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
                output = model(images) # [N, C]
                outputs.append(output.cpu())
                targets.append(target.cpu())
                # measure accuracy and record loss
                acc, _ = accuracy(output, target, topk=(1, 2))
                progress.update('BACC', acc[0], images.size(0))

                # measure elapsed time
                progress.update('batch_time', time.time() - end)
                end = time.time()
            self.logger.info(' * BACC {BACC.avg:.3f}'
                .format(BACC=meter_set['BACC']))

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
            self.logger.info(' * {} {:.3f}'.format(k, v.avg))
    
        self.logger.info(' * BAUC {:.3f}'
            .format(auc))

        self.acc1 = meter_set['BACC'].avg