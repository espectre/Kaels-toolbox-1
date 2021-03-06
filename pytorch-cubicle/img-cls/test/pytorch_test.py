#!/usr/bin/env python
# -*- coding: utf-8 -*-
# created 2018/11/14 @Northrend
#
# Training script
# for pytorch image classification
#

from __future__ import print_function
import sys, os, time, math, re, copy
import logging,pprint,docopt
import torch
import numpy as np

cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_path,'../lib'))
from io_util import inst_data_loader 
from net_util import get_avail_models, init_forward_net 
from test_util import test_wrapper
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
    Update: 2018-11-14
    Author: @Northrend
    Contributor:

    Changelog:
    2018/11/14      v1.0            basic functions 

    Usage:
        pytorch_test.py             <input-cfg>
        pytorch_test.py             -v | --version
        pytorch_test.py             -h | --help

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
    logger.info(pprint.pformat(_.TEST))
    logger.info('PyTorch version: {}'.format(torch.__version__))

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in cfg.GPU_IDX])
    num_gpus = len(cfg.GPU_IDX)
    batch_size = num_gpus * cfg.BATCH_SIZE  # on all gpus

    # use_cuda = torch.cuda.is_available()
    pin_memory = True

    available_models, available_models_names = get_avail_models()
    if cfg.NETWORK not in available_models_names:
        logger.error("Network architecture not supported, should be in:\n{}".format(available_models_names))
        logger.info("Aborting...")
        return 0
    model = init_forward_net(available_models) 

    if cfg.USE_GPU:
        torch.backends.cudnn.benchmark = True
    
    data_loader, data_size = inst_data_loader(cfg.INPUT_IMG_LST, cfg.INPUT_IMG_LST, batch_size, batch_size, test_only=True)
    logger.info("Start testing:")
    test_wrapper(data_loader, data_size, model, use_gpu=cfg.USE_GPU) 

if __name__ == '__main__':
    version = re.compile('.*\d+/\d+\s+(v[\d.]+)').findall(_init_.__doc__)[0]
    args = docopt.docopt(
        _init_.__doc__, version='General pytorch test script {}'.format(version))
    _init_()
    logger.info('Start test job...')
    main()
    logger.info('...Done')
