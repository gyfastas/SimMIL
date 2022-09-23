import torchvision.models as models
from .base import BaseParser

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

class MoCoParser(BaseParser):
    def __init__(self):
        super().__init__()
        self.parser.add_argument('--lr', '--learning-rate', default=0.015, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        self.parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                            help='learning rate schedule (when to drop lr by 10x)')
        self.parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum of SGD solver')
        self.parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        self.parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        # moco specific configs:
        self.parser.add_argument('--moco-dim', default=128, type=int,
                            help='feature dimension (default: 128)')
        self.parser.add_argument('--moco-k', default=4096, type=int,
                            help='queue size; number of negative keys (default: 65536)')
        self.parser.add_argument('--moco-m', default=0.999, type=float,
                            help='moco momentum of updating key encoder (default: 0.999)')
        self.parser.add_argument('--moco-t', default=0.07, type=float,
                            help='softmax temperature (default: 0.07)')

        # options for moco v2
        self.parser.add_argument('--mlp', action='store_true',
                            help='use mlp head')
        self.parser.add_argument('--aug-plus', action='store_true',
                            help='use moco v2 data augmentation')
        self.parser.add_argument('--cos', action='store_true',
                            help='use cosine lr schedule')
