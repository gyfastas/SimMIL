from .base import BaseParser
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

class LinClsParser(BaseParser):
    def __init__(self):
        super().__init__()
        self.parser.add_argument('--sample_file', type=str, default=' ',
                            help='sampling file used, omitted when not using NCTCRC ')
        self.parser.add_argument('--bag_label_dir', '-bld', default=' ', type=str, help='Omitted if not using NCTCRC-BAGS')
        self.parser.add_argument('--no_fixed_trunk', '-nft', action='store_true', help='Do not fix the trunk')
        self.parser.add_argument('--validate_interval', type=int, default=1, 
                            help='validation interval')
        self.parser.add_argument('--lr', '--learning-rate', default=30., type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        self.parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                            help='learning rate schedule (when to drop lr by a ratio)')
        self.parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        self.parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                            metavar='W', help='weight decay (default: 0.)',
                            dest='weight_decay')
        self.parser.add_argument('--load', default='', type=str, metavar='PATH',
                            help='load checkpoint directly')
        self.parser.add_argument('--evaluate_before_train', '-ebt', action='store_true',
                            help='evaluate model before training')
        self.parser.add_argument('--pretrained', default='', type=str,
                            help='path to moco pretrained checkpoint')
        self.parser.add_argument('--eval_class_num', '-ecn',  type=int, default=-1,
                            help='class num when eval, if default, same as class num')
        self.parser.add_argument('--selected_test', action='store_true', default=False)
        self.parser.add_argument("--test_selected_images", type=str, default="/data/samples/camelyon16/test_dataset.npy")
        self.parser.add_argument("--test_selected_labels", type=str, default="/data/samples/camelyon16/test_label.npy")
