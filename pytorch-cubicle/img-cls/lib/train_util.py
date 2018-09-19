import time
import numpy as np
import config as cfg
import torch
from torch.autograd import Variable

class LRScheduler:
    '''
    '''

    def __init__(self, base_lr, lr_factor, step_epochs, max_epoch):
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
        self.step_epochs = step_epochs
        self.max_epoch = max_epoch
        self.lr_map = _precomp_lr_map(self)

    def get_cur_lr(self, epoch):
        return self.lr_map[epoch]


def update_lr(optimizer, epoch, lr_scheduler):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
    lr = lr_scheduler.get_cur_lr(epoch) 
    print('LR: {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def generic_train(data_loader, data_size, model, criterion, optimizer, lr_scheduler, max_epoch=100):
    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(max_epoch):
        print('Epoch {}/{}'.format(epoch, max_epoch - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            if phase == 'train':
                mode='train'
                optimizer = update_lr(optimizer, epoch, lr_scheduler)
                model.train()  # Set model to training mode
            else:
                mode='validation'
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            counter=0
            # Iterate over data.
            for batch_index, (inputs, labels) in enumerate(data_loader[phase]):
                # print(inputs.size())
                # print(type(labels))
                # wrap them in Variable
                # if use_gpu:
                if True:
                    try:
                        inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())
                    except:
                        print(inputs,labels)
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # Set gradient to zero to delete history of computations in previous epoch. Track operations so that differentiation can be done automatically.
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                
                loss = criterion(outputs, labels)
                # print('loss done')                
                # Just so that you can keep track that something's happening and don't feel like the program isn't running.
                # if counter%10==0:
                #     print("Reached iteration ",counter)
                counter+=1

                # backward + optimize only if in training phase
                if phase == 'train':
                    # print('loss backward')
                    loss.backward()
                    # print('done loss backward')
                    optimizer.step()
                    # print('done optim')
                # print evaluation statistics
                try:
                    if (batch_index+1) % cfg.TRAIN.LOG_INTERVAL == 0: 
                        # print('({batch}/{size}) D: {data:.2f}s | B: {bt:.2f}s | T: {total:} | E: {eta:} | L: {loss:.3f} | t1: {top1: .3f} | t5: {top5: .3f}'.format(
                        #         batch=batch_index + 1,
                        #         size=data_size[phase],
                        #         data=data_time.val,
                        #         bt=batch_time.val,
                        #         total=bar.elapsed_td,
                        #         eta=bar.eta_td,
                        #         loss=losses.avg,
                        #         top1=top1.avg,
                        #         top5=top5.avg,
                        #         ))
                        print 'batch: {}, loss: {}'.format(batch_index+1, running_loss/(batch_index+1))
                    running_loss += loss.item()
                    # print(labels.data)
                    # print(preds)
                    running_corrects += torch.sum(preds == labels.data)
                    # print('running correct =',running_corrects)
                except:
                    print('unexpected error, could not calculate loss or do a sum.')
            print('trying epoch loss')
            epoch_loss = running_loss / data_size[phase]
            epoch_acc = running_corrects.item() / float(data_size[phase])
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))


            # deep copy the model
            if phase == 'val':
                print('accuracy:', epoch_acc)
                if USE_TENSORBOARD:
                    foo.add_scalar_value('epoch_loss',epoch_loss,step=epoch)
                    foo.add_scalar_value('epoch_acc',epoch_acc,step=epoch)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(model)
                    print('new best accuracy = ',best_acc)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('returning and looping back')
    return best_model


if __name__ == '__main__':
    a = LRScheduler(0.1, 0.1, [10,15,20], 30)
    print(a.get_cur_lr(18))
