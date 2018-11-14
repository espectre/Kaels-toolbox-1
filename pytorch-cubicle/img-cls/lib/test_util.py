import time
import logging
import numpy as np
from torch import no_grad
from torch.autograd import Variable
from misc import AvgMeter, accuracy
from io_util import save_checkpoint
from config import cfg


def inst_meter_dict(meter_list, meter_style='avg'): 
    result = dict()
    for meter in meter_list:
        if meter_style == 'avg':
            result[meter] = AvgMeter() 
    return result


def test_wrapper(data_loader, data_size, model, use_gpu=True):
    model.eval()
    accumulator = inst_meter_dict(['top_1_acc','top_5_acc','data_time','batch_time'])
    tic = time.time()
    with no_grad(): # close all grads, operations inside don't track history
        toc = time.time()
        for batch_index, (inputs, labels) in enumerate(data_loader['test']):
            batch_size = inputs.size(0)
            accumulator['data_time'].update(time.time()-toc)
            if use_gpu:
                try:
                    inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda(async=True))
                except:
                    logging.error(inputs,labels)
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            outputs = model(inputs)
            acc_1, acc_5 = accuracy(outputs.data, labels.data, topk=(1, 5))
            accumulator['top_1_acc'].update(acc_1.item(), batch_size)
            accumulator['top_5_acc'].update(acc_5.item(), batch_size)
            accumulator['batch_time'].update(time.time()-toc)
            toc = time.time()

            if (batch_index+1) % 10 == 0: 
                logging.info('[{}/{}] data: {:.4f}s | batch: {:.4f}s'.format(
                    batch_index + 1, len(data_loader["test"]), 
                    accumulator['data_time'].val,
                    accumulator['batch_time'].val
                    ))
                accumulator['data_time'].reset()
                accumulator['batch_time'].reset()

        logging.info('top-1: {:.4f} | top-5: {:.4f} | time: {:.4f}'.format(
                    accumulator['top_1_acc'].avg,
                    accumulator['top_5_acc'].avg,
                    time.time() - tic
                    ))
