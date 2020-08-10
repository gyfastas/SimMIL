import os
from shutil import copy, rmtree
import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
class Logger(object):
    def __init__(self, logdir):
        self.logdir = logdir
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.logfile_path = os.path.join(logdir, 'log.txt')
        self.summary_writer = SummaryWriter(os.path.join(logdir, 'tensorboard'))
        self.global_step = 0
        
    def log_string(self, out_str):
        with open(self.logfile_path, 'a+') as f:
            f.write(str(out_str) + '\n')
            print(out_str)
    
    def log_config(self, args, prefix=''):
        '''
        args: (dict)
        '''
        # if isinstance(args,dict):
        #     args_key = list(args.keys())
        #     args_value = list(args.values())
        #     for k in range(len(args_key)):
        #         self.logfile.write('{:30} {}\n'.format(args_key[k],args_value[k]))
        #         print('{:30} {}\n'.format(args_key[k],args_value[k]))

        for k, v in args.items():
            if isinstance(v, dict):
                self.log_config(v, prefix+k+'.')
            else:
                self.log_string(prefix + '{:30} {}\n'.format(k, v))



    def backup_files(self, file_list):
        for filepath in file_list:
            copy(filepath, self.logdir)

    def close(self):
        self.logfile.close()
        self.summary_writer.close()



    def log_scalars(self, mode, tag, kv_dict, global_step):
        '''

        :param mode:
        :param tag:
        :param kv_dict: {'xsinx':i*np.sin(i/r),
                                    'xcosx':i*np.cos(i/r),
                                    'tanx': np.tan(i/r)}
        :param global_step:
        :return:
        '''

        self.summary_writer.add_scalars(mode+tag, kv_dict, global_step)
        self.summary_writer.flush()
    def log_scalar(self, mode, tag, value, global_step):
        self.summary_writer.add_scalar(mode+tag, value, global_step)
        self.summary_writer.flush()

    def log_scalar_eval(self, tag, value, global_step):
        self.summary_writer.add_scalar('eval/'+tag, value, global_step)
        self.summary_writer.flush()

    def log_scalar_train(self, tag, value, global_step):
        self.summary_writer.add_scalar('train/' + tag, value, global_step)
        self.summary_writer.flush()

    def log_scalar_partial(self, tag, value, global_step):
        self.summary_writer.add_scalar('partial/'+tag, value, global_step)
        self.summary_writer.flush()

    def log_histogram_train(self, tag, value, global_step):
        self.summary_writer.add_histogram('train/'+tag, value, global_step)

    def plot_model(self, net):
        self.summary_writer.add_graph(net, input_to_model=None, verbose=False)
        self.summary_writer.flush()

    def save(self, net, optimizer, lrsch, criterion, global_step):
        '''
        save the model/optimizer checkpoints with global step
        param net: (nn.Module) network
        param optimizer: (nn.optim.optimizer)
        param lrsch: (nn.optim.lr_scheduler)
        param criterion: (nn.Module) loss function
        '''
        self.log_string('Saving{}'.format(global_step))
        state_net = {'net': net.state_dict()}
        state_optim = {
            'opt': optimizer.state_dict(),
            'epoch': global_step,
            'loss': criterion.state_dict(),
            'sch': lrsch.state_dict(),
        }
        save_dir = os.path.join(self.logdir, 'ckp')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        try:
            torch.save(state_net,
                       os.path.join(save_dir, 'net.ckpt{}.pth'.format(global_step)))
            torch.save(state_optim,
                       os.path.join(save_dir, 'optim.ckpt{}.pth'.format(global_step)))
            return True
        except:
            print('save failed!')
            return False

    def load(self, net, optimizer, lrsch, criterion, global_step):
        '''

        :param net:
        :param optimizer:
        :param lrsch:
        :param criterion:
        :param global_step:
        :return:
        '''
        load_dir = os.path.join(self.logdir,'ckp')
        try:
            optim_path = os.path.join(load_dir, 'optim.ckpt{}.pth'.format(global_step))
            self.log_string('==> Resuming Optimim of epoch {}'.format(
                optim_path)
            )
            kwargs = {'map_location': lambda storage, loc: storage}
            ckpt_optim = torch.load(
                optim_path,
                **kwargs
            )
            optimizer.load_state_dict(ckpt_optim['opt'])
            start_epoch = ckpt_optim['epoch']
            lrsch.load_state_dict(ckpt_optim['sch'])
            criterion.load_state_dict(ckpt_optim['loss'])

            ckpt_path = os.path.join(load_dir, 'net.ckpt{}.pth'.format(global_step))
            self.log_string('==> Resuming net of epoch {}'.format(
                ckpt_path)
            )
            kwargs = {'map_location': lambda storage, loc: storage}
            ckpt_net = torch.load(
                ckpt_path,
                **kwargs
            )
            ckpt_net['net'] = {k.replace('module.', ''): v for k, v in ckpt_net['net'].items()}
            net.load_state_dict(ckpt_net['net'], strict=False)
            return net, optimizer, lrsch, criterion, start_epoch
        except:
            self.log_string('load failed!')
            return net, optimizer, lrsch, criterion, start_epoch

    def update_step(self, global_step):
        self.global_step = global_step

    def log_pd(self, ids, preds, global_step):
        dataframe = pd.DataFrame({'id': ids, 'categories': preds})
        save_dir = os.path.join(self.logdir, 'csv')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        dataframe.to_csv(os.path.join(save_dir, '{}.csv'.format(global_step)), index=False, sep=',')
