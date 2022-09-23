from .base import BaseParser
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


class MeanTeacherParser(BaseParser):
    def __init__(self):
        super().__init__()
        self.parser.add_argument('--labels', default=None, type=str, metavar='FILE',
                            help='list of image labels (default: based on directory structure)')
        self.parser.add_argument('--labeled-batch-size', default=64, type=int,
                            metavar='N', help="labeled examples per minibatch (default: no constrain)")
        self.parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        self.parser.add_argument('--initial-lr', default=0.0001, type=float,
                            metavar='LR', help='initial learning rate when using linear rampup')
        self.parser.add_argument('--lr-rampup', default=2, type=int, metavar='EPOCHS',
                            help='length of learning rate rampup in the beginning')
        self.parser.add_argument('--lr-rampdown-epochs', default=75, type=int, metavar='EPOCHS',
                            help='length of learning rate cosine rampdown (>= length of training)')
        self.parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                            help='learning rate schedule (when to drop lr by 10x)')
        self.parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum of SGD solver')
        self.parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        self.parser.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA',
                            help='ema variable decay rate (default: 0.999)')
        self.parser.add_argument('--consistency', default=None, type=float, metavar='WEIGHT',
                            help='use consistency loss with given weight (default: None)')
        self.parser.add_argument('--consistency-type', default="mse", type=str, metavar='TYPE',
                            choices=['mse', 'kl'],
                            help='consistency loss type to use')
        self.parser.add_argument('--consistency-rampup', default=30, type=int, metavar='EPOCHS',
                            help='length of the consistency loss ramp-up')
        self.parser.add_argument('--logit-distance-cost', default=-1, type=float, metavar='WEIGHT',
                            help='let the student model have two outputs and use an MSE loss between the logits with the given weight (default: only have one output)')
        self.parser.add_argument('--checkpoint-epochs', default=1, type=int,
                            metavar='EPOCHS', help='checkpoint frequency in epochs, 0 to turn checkpointing off (default: 1)')
        self.parser.add_argument('--evaluation-epochs', default=1, type=int,
                            metavar='EPOCHS', help='evaluation frequency in epochs, 0 to turn evaluation off (default: 1)')
        self.parser.add_argument('--aug-plus', action='store_true',
                            help='use moco v2 data augmentation')