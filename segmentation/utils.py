import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import shutil
import os
import random
import logging

def create_logger(log_path, name='torch', file_name='train.log', fmt='%(asctime)s | %(message)s'):
    logger = logging.getLogger(name)
    formatter = logging.Formatter(fmt)
    logger.setLevel(logging.INFO)
    stream_hander = logging.StreamHandler()
    stream_hander.setFormatter(formatter)
    logger.addHandler(stream_hander)
    file_handler = logging.FileHandler(os.path.join(log_path, file_name))
    logger.addHandler(file_handler)
    file_handler.setFormatter(formatter)
    return logger

def create_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")
        
def get_device(tensor):
    """ get device name from input tensor """
    return "cpu" if tensor.get_device() < 0 else f"cuda:{tensor.get_device()}"
        
def SR2CR(SR):
    """
        Transform segmentation result tensor(N*C*H*W) to classification result tensor(N*C*1*1)
    """
    assert len(SR.shape) == 4
    num_classes = SR.shape[1]
    SR = torch.argmax(SR, dim=1, keepdim=True)
    SR = make_one_hot(SR, num_classes=num_classes)
    CR = SR.sum(dim=[2,3], keepdim=True)
    CR = CR[:, 1:] # except background
    CR = torch.argmax(CR, dim=1, keepdim=True)
    CR = make_one_hot(CR.type(torch.int64), num_classes=num_classes-1)
    return CR

def memory_check():
    for GPU_NUM in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(GPU_NUM))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(GPU_NUM)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(GPU_NUM)/1024**3,1), 'GB')
        print("="*50)
        
def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    assert input.shape[1] == 1
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape, device=get_device(input))
    result = result.scatter_(1, input, 1)
    return result
    
def save_checkpoint(best_score, score, state, is_best, path='', best_metric='loss',verbose=True):
    torch.save(state, path+'_checkpoint.pth.tar')
    if is_best:
        if verbose:
            print(f'Validation {best_metric} got better {best_score} --> {score}.  Saving model ...')
        shutil.copyfile(path+'_checkpoint.pth.tar', path+f'_best-{best_metric}.pth.tar')

def loss_graph(train_loss, valid_loss, best_epoch, model_name, save_path="./png"):
    fig = plt.figure(figsize=(10,5))
    plt.title('Train & Valid Loss Graph', size=17)
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.axvline(best_epoch, color='lightgray', linestyle='--', linewidth=2,
               label=f'best loss 에폭: {best_epoch}')
    plt.xlabel('epoch', size=15)
    plt.ylabel('loss', size=15)
    plt.legend(loc='best', fontsize=15, frameon=True, shadow=True)
    plt.savefig(os.path.join(save_path, f'{model_name}.png'), dpi=50)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count