from .base_runner import BaseRunner
from .lincls_runner import LinClsRunner
from datasets.my_imagefolder import MyImageFolder
from datasets.camelyon17 import CameLyon17
from datasets.debug_dataset import DebugDataset
import os
import time
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
from simmil.utils.core import AverageMeter, ProgressMeter, accuracy
from .cls_runner import ClsRunner


class MoCoRunner(ClsRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)