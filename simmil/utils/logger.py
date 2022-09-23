#!/usr/bin/env
# -*- coding: utf-8 -*-
import os
import json
import time
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from shutil import copy, copytree, rmtree


__all__ = ['Logger']


class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


class Logger(object):
    def __init__(self, args, filename=None, mode=None):
        self.args = args
        self.workdir = args.log_dir
        self.timestamp = args.timestamp
        if filename is None:
            filename = '{}.log.txt'.format(self.timestamp)
        self.filename = os.path.join(self.workdir, filename)
        logging.root.handlers = []
        
        if mode is None:
            mode = 'a' if os.path.exists(self.filename) else 'w'
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s -   %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
            handlers=[
                TqdmHandler(),
                logging.FileHandler(
                    self.filename, mode=mode)
            ]
        )
        self.logger = logging.getLogger()
        
    def init_info(self, save_config=True):
        self.info(
            '------------------- * {} * -------------------'.format(self.timestamp))
        self.info(
            'Working directory: {}'.format(self.workdir))
        self.info('Experiment log saves to {}'.format(self.filename))
        if save_config:
            self.info('Experiment configuration saves to {}:'.format(
                os.path.join(self.workdir, 'config.json')))
            with open(os.path.join(self.workdir, 'config.json'), 'w') as f:
                json.dump(self.args.__dict__, f, indent=4)

    def info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def to_csv(self, name, value):
        pd.DataFrame(value).to_csv(os.path.join(self.workdir, name+".csv"))

    def warning(self, *args, **kwargs):
        self.logger.warning(*args, **kwargs)

    def backup_files(self, file_list):
        for filepath in file_list:
            copy(filepath, self.workdir)

    def auto_backup(self, root='./'):
        for f_name in os.listdir(root):
            if os.path.isdir(os.path.join(root, f_name)):
                self.auto_backup(os.path.join(root, f_name))
            elif f_name.endswith('.py'):
                save_path = os.path.join(os.path.join(self.workdir,'src'), root)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                copy(os.path.join(root,f_name), save_path)
