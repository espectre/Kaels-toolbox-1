#!/usr/bin/env python
# -*- coding: utf-8 -*-
# created 2018/09/29 @Northrend
#
# Testing script
# for pytorch image classification
#

from __future__ import print_function, division
import sys, os, time, math, re, copy
import logging,pprint,docopt

import torch
import numpy as np

cur_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(cur_path,'../lib'))
from io_util import load_checkpoint
from net_util import get_avail_models, init_model
from train_util import generic_train, LRScheduler
from config import merge_cfg_from_file
from config import cfg as _
cfg = _.TEST

# init global logger
log_format = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger()  # sync with mxnet internal logger
fhandler = None     # log to file

def _init_():
    '''
    Training script for image-classification task on mxnet
    Update: 2018-09-29
    Author: @Northrend
    Contributor:

    Changelog:
    2018/09/29      v1.0              basic functions

    Usage:
        pytorch_train.py              <input-cfg>
        pytorch_train.py              -v | --version
        pytorch_train.py              -h | --help

    Arguments:
        <input-cfg>                 path to customized config file

    Options:
        -h --help                   show this help screen
        -v --version                show current version

    '''
    # merge configuration
    merge_cfg_from_file(args["<input-cfg>"])

    # config logger
    logger.setLevel(eval('logging.' + cfg.LOG_LEVEL))
    assert cfg.LOG_PATH, logger.error('Missing LOG_PATH!')
    fhandler = logging.FileHandler(cfg.LOG_PATH, mode=cfg.LOG_MODE)
    logger.addHandler(fhandler)

    # print arguments
    logger.info('=' * 80 + '\nCalled with arguments:')
    for key in sorted(args.keys()):
        logger.info('{:<20}= {}'.format(key.replace('--', ''), args[key]))
    logger.info('=' * 80)

    # reset logger format
    fhandler.setFormatter(logging.Formatter(log_format))


def main():
    logger.info('Configuration:')
    logger.info(pprint.pformat(_))

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in cfg.GPU_IDX])
    num_gpus = len(cfg.GPU_IDX)
    batch_size = num_gpus * cfg.BATCH_SIZE  # on all gpus
    
    USE_CUDA = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    

    
