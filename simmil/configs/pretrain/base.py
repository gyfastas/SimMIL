import argparse
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

class BaseParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='backbone checking base args')
        self.parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
        self.parser.add_argument('--log_dir', type=str, default='./debug/')
        self.parser.add_argument('--dataset', type=str, default='CAMELYON16', choices=['CAMELYON16', 'CAMELYON17', 'NCTCRC', 'NCTCRC-BAGS'])

        self.parser.add_argument('--train-subdir', type=str, default='NCT-CRC-HE-100K',
                            help='the subdirectory inside the data directory that contains the training data')
        self.parser.add_argument('--eval-subdir', type=str, default='CRC-VAL-HE-7K',
                            help='the subdirectory inside the data directory that contains the evaluation data')
                            
        self.parser.add_argument('-b', '--batch-size', default=256, type=int,
                            metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                                'batch size of all GPUs on the current node when '
                                'using Data Parallel or Distributed Data Parallel')
        self.parser.add_argument('--epochs', default=100, type=int, metavar='N',
                            help='number of total epochs to run')
        self.parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        self.parser.add_argument('--no_norm', action='store_true', help='Omitted if not using NCTCRC dataset')
        self.parser.add_argument('--imgnet_pretrained', action='store_true', help='Whether use imagnet pretrain')
        self.parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        self.parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                            choices=model_names,
                            help='model architecture: ' +
                                ' | '.join(model_names) +
                                ' (default: resnet50)')
        self.parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
        self.parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                            help='number of data loading workers (default: 32)')
        self.parser.add_argument('--class_num', '-cn', type=int, default=1000, 
                            help='class number')
        self.parser.add_argument("--save_result", action='store_true',
                            help='save evaluation result')
        self.parser.add_argument("--save_result_name", type=str, default="evaluate_result.pth",
                            help='file name of saved evaluation result. Only useful when save_result is True.')
        self.parser.add_argument('--debug', action='store_true', help='debug use')
        self.parser.add_argument('-p', '--print-freq', default=10, type=int,
                            metavar='N', help='print frequency (default: 10)')
        # distributed learning / multi-processing part
        self.parser.add_argument('--world-size', default=-1, type=int,
                            help='number of nodes for distributed training')
        self.parser.add_argument('--rank', default=-1, type=int,
                            help='node rank for distributed training')
        self.parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                            help='url used to set up distributed training')
        self.parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
        self.parser.add_argument('--seed', default=None, type=int,
                            help='seed for initializing training. ')
        self.parser.add_argument('--gpu', default=None, type=int,
                            help='GPU id to use.')
        self.parser.add_argument('--multiprocessing-distributed', action='store_true',
                            help='Use multi-processing distributed training to launch '
                                'N processes per node, which has N GPUs. This is the '
                                'fastest way to use PyTorch for either single node or '
                                'multi node data parallel training')

    def parse_args(self, args=None):
        return self.parser.parse_args() if args is None else self.parser.parse_args(args)