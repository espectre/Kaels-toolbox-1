#!/usr/bin/env python
# -*- coding: utf-8 -*-
# created 2018/09/17 @Northrend
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
from io_util import inst_data_loader, load_checkpoint 
from net_util import get_avail_models, init_model 
from train_util import generic_train, LRScheduler 
from config import merge_cfg_from_file
from config import cfg as _
cfg = _.TRAIN

# init global logger
log_format = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger()  # sync with mxnet internal logger
fhandler = None     # log to file


def _init_():
    '''
    Training script for image-classification task on mxnet
    Update: 2018-10-10
    Author: @Northrend
    Contributor:

    Changelog:
    2018/10/10      v1.4              fix cuda-oom  
    2018/10/09      v1.3              support check nets mode 
                                      support resnet-v2
    2018/09/26      v1.2              optimize logging info 
    2018/09/25      v1.1              support finetune & scratch training
                                      support xavier initialization
    2018/09/13      v1.0              basic functions 

    Usage:
        pytorch_train.py              <input-cfg> [-c|--check-nets]
        pytorch_train.py              -v | --version
        pytorch_train.py              -h | --help

    Arguments:
        <input-cfg>                 path to customized config file

    Options:
        -h --help                   show this help screen
        -v --version                show current version
        -------------------------------------------------------------
        -c --check-nets             check available network arch only

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


def tensorboard():
    '''
    '''
    from pycrayon import CrayonClient
    cc = CrayonClient(hostname=TENSORBOARD_SERVER)
    try:
        cc.remove_experiment(EXP_NAME)
    except:
        pass
    foo = cc.create_experiment(EXP_NAME)


def main():
    logger.info('Configuration:')
    logger.info(pprint.pformat(_))

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in cfg.GPU_IDX])
    num_gpus = len(cfg.GPU_IDX)
    train_batch_size = num_gpus * cfg.BATCH_SIZE  # on all gpus
    dev_batch_size = num_gpus * cfg.DEV_BATCH_SIZE

    # use_cuda = torch.cuda.is_available()
    pin_memory = True

    available_models, available_models_names = get_avail_models()
    if args['--check-nets']:
        logger.info("Currently available nets:\n{}".format("\n".join(available_models_names)))
        return 0

    if cfg.NETWORK not in available_models_names:
        logger.error("Network architecture not supported, should be in:\n{}".format(available_models_names))
        logger.info("Aborting...")
        return 0
    model = init_model(available_models) 
    if cfg.LOG_NET_PARAMS:
        logger.info('Network params:')
        for name,params in model.named_parameters():
            logger.info('{}: {}'.format(name, [x for x in params.size()]))
        logger.info('---------------------')

    criterion = torch.nn.CrossEntropyLoss()
    criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.BASE_LR, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)

    if cfg.USE_GPU:
        torch.backends.cudnn.benchmark = True

    lr_scheduler = LRScheduler(cfg.BASE_LR, cfg.LR_FACTOR, cfg.STEP_EPOCHS, cfg.MAX_EPOCHS)
    
    data_loader, data_size = inst_data_loader(cfg.TRAIN_LST, cfg.DEV_LST, train_batch_size, dev_batch_size)
    logger.info("Start training:")
    generic_train(data_loader, data_size, model, criterion, optimizer, lr_scheduler, max_epoch=cfg.MAX_EPOCHS, pre_eval=cfg.PRE_EVALUATION) 


if __name__ == '__main__':
    version = re.compile('.*\d+/\d+\s+(v[\d.]+)').findall(_init_.__doc__)[0]
    args = docopt.docopt(
        _init_.__doc__, version='General pytorch training script {}'.format(version))
    _init_()
    logger.info('Start training job...')
    main()
    logger.info('...Done')
