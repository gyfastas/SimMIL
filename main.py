import argparse
import os
import random
import warnings
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from simmil.runners.pretrain import LinClsRunner, MoCoRunner, MeanTeacherRunner, ClsRunner
from simmil.runners.bag_cls import AssumptionRunner
from simmil.configs.pretrain import LinClsParser, MoCoParser, MeanTeacherParser
from simmil.configs.bag_cls import AssumptionParser 
from simmil.utils.misc import get_timestamp
from simmil.utils.path import mkdir_or_exist

runner_dict = dict(lincls=LinClsRunner,
                   moco=MoCoRunner,
                   mt=MeanTeacherRunner,
                   oracle=ClsRunner,
                   assumption=AssumptionRunner
)
parser_dict = dict(lincls=LinClsParser,
                   moco=MoCoParser,
                   mt=MeanTeacherParser,
                   oracle=LinClsParser,
                   assumption=AssumptionParser)
def parse_args():
    parser = argparse.ArgumentParser('Backbone Experiment argument parser')
    parser.add_argument('--runner_type', type=str, default='lincls')
    parser.add_argument('--ddp', action='store_true', default=False)

    return parser.parse_known_args()

def main():
    known_args, args = parse_args()
    runner_type = known_args.runner_type
    args = parser_dict[runner_type]().parse_args(args)
    args.timestamp = get_timestamp()
    mkdir_or_exist(args.log_dir)
    if not args.ddp:
        runner = runner_dict[runner_type](args)
        runner.run()
    else:
        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. '
                        'This will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! '
                        'You may see unexpected behavior when restarting '
                        'from checkpoints.')

        if args.gpu is not None:
            warnings.warn('You have chosen a specific GPU. This will completely '
                        'disable data parallelism.')

        if args.dist_url == "env://" and args.world_size == -1:
            args.world_size = int(os.environ["WORLD_SIZE"])

        args.distributed = args.world_size > 1 or args.multiprocessing_distributed

        ngpus_per_node = torch.cuda.device_count()
        if args.multiprocessing_distributed:
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            args.world_size = ngpus_per_node * args.world_size
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # main_worker process function
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, runner_type))
        else:
            # Simply call main_worker function
            main_worker(args.gpu, ngpus_per_node, args, runner_type)

def main_worker(gpu, ngpus_per_node, args, runner_type):
    runner = runner_dict[runner_type](gpu, ngpus_per_node, args)
    runner.run()


if __name__=="__main__":
    main()