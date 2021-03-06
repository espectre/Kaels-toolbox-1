import time
import logging
import numpy as np
from torch import no_grad  
from torch.autograd import Variable
from misc import AvgMeter, accuracy, mixup_accuracy
from io_util import save_checkpoint, mixup_data
from config import cfg

class LRScheduler:
    '''
    '''

    def __init__(self, base_lr, lr_factor, step_epochs, max_epoch, fixed_step=False):
        '''
        '''
        def _precomp_lr_map(self):
            '''
                lr_map: list of specific learning rates, and index=epoch
            '''
            steps = [0]
            steps.extend(self.step_epochs)
            steps.append(self.max_epoch)
            lr_map = [0 for x in range(max_epoch)]
            for i in range(len(steps)-1):
                for j in range(steps[i], steps[i+1]):
                    lr_map[j] = base_lr * pow(lr_factor, i) 
            return lr_map
            
        assert isinstance(step_epochs, list), 'step_epochs should be a list of ints'
        self.base_lr = base_lr
        self.lr_factor = lr_factor
        self.max_epoch = max_epoch
        if fixed_step:  # fixed lr decay interval
            self.step_epochs = range(1, max_epoch, step_epochs[0])
        else:   # customized lr decay epochs
            self.step_epochs = step_epochs
        self.lr_map = _precomp_lr_map(self)

    def get_cur_lr(self, epoch):
        return self.lr_map[epoch]


def update_lr(optimizer, epoch, lr_scheduler):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
    lr = lr_scheduler.get_cur_lr(epoch) 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def inst_meter_dict(meter_list, meter_style='avg'): 
    result = dict()
    for meter in meter_list:
        if meter_style == 'avg':
            result[meter] = AvgMeter() 
    return result

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def generic_train(data_loader, data_size, model, criterion, optimizer, lr_scheduler, max_epoch=100, use_gpu=True, pre_eval=False):
    tic = time.time()

    best_model = model
    best_acc = 0.0

    temporary = inst_meter_dict(['batch_time','data_time','losses','top_1_acc','top_5_acc'])
    accumulator = inst_meter_dict(['losses','top_1_acc','top_5_acc'])

    # pre-evaluation phase to check cuda memory 
    if pre_eval:
        logging.info('Validation [0/{}]:'.format(max_epoch))
        model.eval()
        
        toc = time.time()
        with no_grad(): # close all grads, operations inside don't track history
            batch_size = 0
            for batch_index, (inputs, labels) in enumerate(data_loader['dev']):
                if batch_size == 0:
                    batch_size = inputs.size(0)
                temporary['data_time'].update(time.time()-toc)
                # wrap in Variable
                if use_gpu:
                    try:
                        inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda(async=True))
                    except:
                        logging.error(inputs,labels)
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                acc_1, acc_5 = accuracy(outputs.data, labels.data, topk=(1, 5))

                accumulator['losses'].update(loss.item(), batch_size)
                accumulator['top_1_acc'].update(acc_1.item(), batch_size)
                accumulator['top_5_acc'].update(acc_5.item(), batch_size)

            logging.info('[{}/{}] loss: {:.4f} | top-1: {:.4f} | top-5: {:.4f}'.format(
                        0, max_epoch,
                        accumulator['losses'].avg,
                        accumulator['top_1_acc'].avg,
                        accumulator['top_5_acc'].avg
                        ))
            logging.info('Pre-evaluation done, validation batch-size:{}, everything is ok'.format(batch_size))

    # training and validation
    for epoch in range(max_epoch):
        is_best = False
        use_mixup = cfg.TRAIN.MIXUP 
        if use_mixup:
            logging.info('Mix-up used during training')
            if epoch not in xrange(cfg.TRAIN.MU.ACTIVE_EPOCH_RANGE[0], cfg.TRAIN.MU.ACTIVE_EPOCH_RANGE[1]):
                use_mixup = False
                logging.info('Mix-up switch OFF')
            else:
                logging.info('Mix-up switch ON')

        # Each epoch has a training and validation phase
        # ---- training phase ----
        optimizer = update_lr(optimizer, epoch, lr_scheduler)
        logging.info('Training epoch [{}/{}]: learning rate {}'.format(epoch+1, max_epoch, optimizer.param_groups[0]['lr']))
        model.train()  # Set model to training mode

        # Iterate over data.
        toc = time.time()
        for batch_index, (inputs, labels) in enumerate(data_loader["train"]):
            batch_size = inputs.size(0)
            temporary['data_time'].update(time.time()-toc)
            # wrap in Variable
            if use_gpu:
                try:
                    inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda(async=True))
                    if use_mixup:
                        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, cfg.TRAIN.MU.ALPHA)
                        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
                except:
                    logging.error('\n==> inputs:\n{}\n==> labels:\n{}'.format(inputs,labels))
                    return 0
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # Set gradient to zero to delete history of computations in previous epoch. Track operations so that differentiation can be done automatically.
            optimizer.zero_grad()
            outputs = model(inputs)
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                acc_1, acc_5 = mixup_accuracy(outputs.data, targets_a, targets_b, lam, topk=(1, 5))
            else:
                loss = criterion(outputs, labels)
                acc_1, acc_5 = accuracy(outputs.data, labels.data, topk=(1, 5))

            # losses.update(loss.data[0], inputs.size(0))
            temporary['losses'].update(loss.item(), batch_size)
            temporary['top_1_acc'].update(acc_1.item(), batch_size)
            temporary['top_5_acc'].update(acc_5.item(), batch_size)
            accumulator['losses'].update(loss.item(), batch_size)
            accumulator['top_1_acc'].update(acc_1.item(), batch_size)
            accumulator['top_5_acc'].update(acc_5.item(), batch_size)

            # backward + optimize only if in training phase
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print evaluation statistics
            temporary['batch_time'].update(time.time()-toc)
            toc = time.time()
            
            if (batch_index+1) % cfg.TRAIN.LOG_INTERVAL == 0: 
                logging.info('[{}/{}] [{}/{}] data: {:.4f}s | batch: {:.4f}s | loss: {:.4f} | top-1: {:.4f} | top-5: {:.4f}'.format(
                    epoch+1, max_epoch,
                    batch_index + 1, len(data_loader["train"]), 
                    temporary['data_time'].val,
                    temporary['batch_time'].val,
                    temporary['losses'].avg,
                    temporary['top_1_acc'].avg,
                    temporary['top_5_acc'].avg,
                    ))
                temporary['data_time'].reset()
                temporary['batch_time'].reset()
                temporary['losses'].reset()
                temporary['top_1_acc'].reset()
                temporary['top_5_acc'].reset()

        logging.info('[{}/{}] loss: {:.4f} | top-1: {:.4f} | top-5: {:.4f}'.format(
                    epoch+1, max_epoch,
                    accumulator['losses'].avg,
                    accumulator['top_1_acc'].avg,
                    accumulator['top_5_acc'].avg
                    ))
        accumulator['losses'].reset()
        accumulator['top_1_acc'].reset()
        accumulator['top_5_acc'].reset()
        # -------------------------

        # ---- validation phase ---- 
        logging.info('Validation [{}/{}]:'.format(epoch+1, max_epoch))
        model.eval()

        # Iterate over data.
        toc = time.time()
        with no_grad(): # close all grads, operations inside don't track history
            for batch_index, (inputs, labels) in enumerate(data_loader["dev"]):
                batch_size = inputs.size(0)
                temporary['data_time'].update(time.time()-toc)
                # wrap in Variable
                if use_gpu:
                    try:
                        inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda(async=True))
                    except:
                        logging.error('\n==> inputs:\n{}\n==> labels:\n{}'.format(inputs,labels))
                        return 0
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # Set gradient to zero to delete history of computations in previous epoch. Track operations so that differentiation can be done automatically.
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                acc_1, acc_5 = accuracy(outputs.data, labels.data, topk=(1, 5))

                # losses.update(loss.data[0], inputs.size(0))
                temporary['losses'].update(loss.item(), batch_size)
                temporary['top_1_acc'].update(acc_1.item(), batch_size)
                temporary['top_5_acc'].update(acc_5.item(), batch_size)
                accumulator['losses'].update(loss.item(), batch_size)
                accumulator['top_1_acc'].update(acc_1.item(), batch_size)
                accumulator['top_5_acc'].update(acc_5.item(), batch_size)

            # check if current model is best 
            logging.info('Current validation accuracy: {:.4f}'.format(accumulator['top_1_acc'].avg))
            if accumulator['top_1_acc'].avg > best_acc:
                is_best = True
                best_acc = accumulator['top_1_acc'].avg 
                # best_model = copy.deepcopy(model)
                logging.info('New best accuracy: {:.4f}'.format(best_acc))
            accumulator['losses'].reset()
            accumulator['top_1_acc'].reset()
            accumulator['top_5_acc'].reset()
        # --------------------------

        # ---- save checkpoint ----
        if (epoch+1)%cfg.TRAIN.SAVE_INTERVAL == 0:
            save_checkpoint({
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'acc': accumulator['top_1_acc'].avg,
                    # 'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict()},
                    cfg.TRAIN.OUTPUT_MODEL_PREFIX,
                    is_best=is_best 
                    )
            logging.info('Checkpoint saved to {}-{:0>4}.pth.tar'.format(cfg.TRAIN.OUTPUT_MODEL_PREFIX, epoch+1))
        # ------------------------

    time_elapsed = int(time.time() - tic)
    logging.info('Training job complete in {:d}:{:0>2d}:{:d}'.format(
        time_elapsed // 3600, (time_elapsed - 3600*(time_elapsed//3600)) // 60, (time_elapsed - 60*(time_elapsed//60))))
    logging.info('Best val Acc: {:4f}'.format(best_acc))
    return 0 


if __name__ == '__main__':
    a = LRScheduler(0.1, 0.1, [10,15,20], 30)
    b = LRScheduler(0.1, 0.1, [1,2,3], 30, fixed_step=True)
    print(a.get_cur_lr(18))
    print(b.lr_map)
