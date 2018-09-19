#!/usr/bin/env python
# -*- coding: utf-8 -*-
# created 2018/09/17 @Northrend
#
# Training script
# for pytorch image classification
#

from __future__ import print_function, division
import sys, os, time, math, re, copy
import logging,pprint,docopt

import torch
from torch.autograd import Variable
import numpy as np
import torchvision
# from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_path,'../lib'))
from io_util import inst_data_loader 
# from net_util import *
from train_util import generic_train, LRScheduler 
# from cam_util import *
import model as extra_models 
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
    Update: 2018-09-13
    Author: @Northrend
    Contributor:

    Changelog:
    2018/09/13      v1.0              basic functions 

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
    # default_model_names = sorted(name for name in torchvision.models.__dict__
    # if name.islower() and not name.startswith("__")
    # and callable(torchvision.models.__dict__[name]))
    # # print(default_model_names)

    # extra_models_names = sorted(name for name in extra_models.__dict__
    # if name.islower() and not name.startswith("__")
    # and callable(extra_models.__dict__[name]))
    # # print(extra_models_names)

    available_models = torchvision.models
    for name in extra_models.__dict__:
        if name.islower() and not name.startswith("__") and callable(extra_models.__dict__[name]):
            available_models.__dict__[name] = extra_models.__dict__[name]

    available_models_names = sorted(name for name in available_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(available_models.__dict__[name]))
    # pprint.pprint(available_models_names)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in cfg.GPU_IDX])
    use_cuda = torch.cuda.is_available()
    pin_memory = True

    # model = available_models.__dict__['resnet18'](pretrained=False)
    # model = available_models.resnext50(baseWidth=32,cardinality=4)
    model = available_models.resnet18(pretrained=True)

    num_filters = model.fc.in_features
    num_cls = 1000
    model.fc = torch.nn.Linear(num_filters, num_cls)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.BASE_LR, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)

    model = torch.nn.DataParallel(model).cuda()
    # model.cuda()
    torch.backends.cudnn.benchmark = True
    print(model)

    lr_scheduler = LRScheduler(cfg.BASE_LR, cfg.LR_FACTOR, cfg.STEP_EPOCHS, cfg.MAX_EPOCHS)
    
    data_loader, data_size = inst_data_loader(cfg.TRAIN_LST, cfg.DEV_LST)
    generic_train(data_loader, data_size, model, criterion, optimizer, lr_scheduler) 


if __name__ == '__main__':
    version = re.compile('.*\d+/\d+\s+(v[\d.]+)').findall(_init_.__doc__)[0]
    args = docopt.docopt(
        _init_.__doc__, version='General pytorch training script {}'.format(version))
    _init_()
    logger.info('Start training job...')
    main()
    logger.info('...Done')
