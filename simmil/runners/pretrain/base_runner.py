
import json
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from simmil.utils.logger import Logger

class BaseRunner(object):
    def __init__(self, gpu, ngpus_per_node, args):
        args.gpu = gpu
        self.gpu = gpu
        self.ngpus_per_node = ngpus_per_node
        self.args = args
        self.logger = Logger(self.args)
        self.init_ddp()
    
    def run(self):
        raise NotImplementedError

    def init_ddp(self):
        # suppress printing and auto backup if not master
        if self.args.multiprocessing_distributed and self.args.gpu != 0:
            def print_pass(*args):
                pass

            def auto_backup(root='./'):
                pass

            self.logger.info = print_pass
            self.logger.auto_backup = auto_backup
        self.logger.init_info()
        self.logger.auto_backup()
        self.logger.info("Configuration:\n{}".format(json.dumps(self.args.__dict__, indent=4)))
        if self.args.gpu is not None:
            self.logger.info("Use GPU: {} for training".format(self.args.gpu))
        if self.args.distributed:
            if self.args.dist_url == "env://" and self.args.rank == -1:
                self.args.rank = int(os.environ["RANK"])
            
            if self.args.gpu is not None:
                self.set_ddp_batch_size()
                self.set_ddp_num_workers()

            if self.args.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                self.args.rank = self.args.rank * self.ngpus_per_node + self.gpu
            dist.init_process_group(backend=self.args.dist_backend, init_method=self.args.dist_url,
                                    world_size=self.args.world_size, rank=self.args.rank)

    def set_ddp_batch_size(self):
        self.args.batch_size = int(self.args.batch_size / self.ngpus_per_node)

    def set_ddp_num_workers(self):
        self.args.workers = int((self.args.workers + self.ngpus_per_node - 1) / self.ngpus_per_node)

    def set_device(self, model):
        if self.args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if self.args.gpu is not None:
                torch.cuda.set_device(self.args.gpu)
                model.cuda(self.args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
        elif self.args.gpu is not None:
            torch.cuda.set_device(self.args.gpu)
            model = model.cuda(self.args.gpu)

        return model