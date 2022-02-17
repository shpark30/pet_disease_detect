import argparse
import os
import random
import shutil
import time
from datetime import datetime
import warnings
from collections import defaultdict, OrderedDict
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch import optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from utils import *
from evaluation import *
from network import build_model
from dataloader import PetSegDataset
from loss import *

parser=argparse.ArgumentParser(
        description='Training Disease Recognition in Pet CT')

# Dataset
parser.add_argument('root', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--valid-root', default='', type=str)
parser.add_argument('--img-size', default=512, type=int,
                    help='input data will be resized to img_size')
parser.add_argument('--augmentation_prob', default=0.4, type=float,
                    help='')
parser.add_argument('--valid-ratio', default=1/9, type=float,
                    help='')


# DataLoader
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=16, type=int, metavar='N',
                    help='mini-batch size (default: 16), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

# Model
parser.add_argument('-ic', '--img_ch', default=3, type=int,
                    help='number of channels of input images')
parser.add_argument('-nc', '--out_ch', default=3, type=int,
                    help='number of output-channels')
parser.add_argument('--backbone', default='xception', type=str,
                    help='A backbone algorithm. DeepLabv3plus support "xception", "resnet", "mobilenet"')
parser.add_argument('--pretrained', action='store_true',
                    help='whether to use backbone pretrained on ImageNet.')
parser.add_argument('--freeze-bn', action='store_true',
                    help='whether to freeze batch norm layers.')

# Train setting
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

# Validation Setting
parser.add_argument('--iou-threshold', default=0.5, type=float)

# Loss
parser.add_argument('--loss-weight', default='[1.,1.,1.]', type=str,
                    help='weights for computing loss among classes'
                         'default : [1., 1., 1.]')

# Learning rate
parser.add_argument('-lr', '--learning-rate', default=0.02, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-decay', default=0.1, type=float,
                    help='Sets the learning rate to the initial LR decayed by {lr_decay} every {num_epochs_decay} epochs')
parser.add_argument('--num-epochs-decay', default=30, type=int,
                    help='Sets the learning rate to the initial LR decayed by {lr_decay} every {num_epochs_decay} epochs')

# Optimizer
parser.add_argument('--beta1', type=float, default=0.5,
                    help='momentum1 in Adam')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='momentum2 in Adam')
parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                    help='weight_decay')
parser.add_argument('-eps', '--epsilon', type=float, default=1e-8)
parser.add_argument('--amsgrad', action='store_true')

# Early Stop
parser.add_argument('--patience', default=-1, type=int,
                    help='patience for early stop'
                         "default -1 means that don't use early-stop")

# Logging
parser.add_argument('--log-freq', default=0, type=int, metavar='N',
                    help='log loss and metrics every log_freq batch in an epoch.'
                         'default(-1) then print at every epoch')
parser.add_argument('--log-path', default='./log', type=str,
                    help='path where a log file will be saved')
parser.add_argument('--log-name', default='train.log', type=str,
                    help='A log file name')

# Distributed
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distriburted training')
parser.add_argument('--dist-url', default="tcp://localhost:25000", type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training.')

# Single GPU Train
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

# Model Save
parser.add_argument('--model-path', default='./models', type=str,
                    help='')

args=parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(torch.cuda.device_count())))
os.environ["CUDA_LAUNCH_BLOCKING"] = ",".join(map(str, range(torch.cuda.device_count())))
args.loss_weight = eval(args.loss_weight)

best_score = None
best_f1_scores = defaultdict(float)
history = defaultdict(list)

create_directory(args.log_path)
logger = create_logger(args.log_path, file_name=args.log_name)
epochFormat = "%s | %s"

def main(args):
    # fix seed
    if args.seed is not None:
        print("Using Seed Number {}".format(args.seed))
        os.environ["PYTHONHASHSEED"] = str(args.seed)  # set PYTHONHASHSEED env var at fixed value
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)  # pytorch (both CPU and CUDA)
        np.random.seed(args.seed)  # for numpy pseudo-random generator
        random.seed(args.seed)  # set fixed value for python built-in pseudo-random generator
        print("Current pytorch seed : ", torch.initial_seed())
        print("Current pytorch GPU seeds : ", torch.cuda.initial_seed())
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        warnings.warn('You have chosen to seed training. '
                     'This will turn on the CUDNN deterministic setting, '
                     'which can slow down your training considerably! '
                     'You may see unexpected behavior when restarting '
                     'from checkpoints.')
    
    memory_check()
    
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ['WORLD_SIZE'])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    # share variables among distributed processors.
    global best_score
    global best_f1_scores
    global history

    # distributed training set-up
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        print("local rank:", args.rank)
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        print("global rank:", args.rank)
        torch.cuda.set_device(args.rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # build model
    model = build_model(args)

    # CPU, Single GPU, Multi GPUs
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should be always set the single device scope, others,
        # DistributedDataParallel will use all availabe devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # outselved based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(
                (args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will devide and allocate batch_size to all avaialbe GPUs
        model = torch.nn.DataParallel(model).cuda()

    # loss function (criterion)
    weight = torch.tensor(args.loss_weight).cuda(args.gpu)
    criterion = FocalCrossEntropy(weight=weight, reduction='mean')

    # optimizer
    optimizer = optim.Adam(model.parameters(), args.lr,
                           betas=[args.beta1, args.beta2],
                           weight_decay=args.weight_decay)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            # build model
            if checkpoint['args'].distributed:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(checkpoint['state_dict'])
            # build optimizer
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch']+1
            history = history['history']
            best_score = min(history['valid_loss'])
            best_f1_scores = {k: max(v) for k, v in history.items() if 'f1' in k}
            
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            print('start epoch: {}'.format(args.start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data Loading code
    loader = {}
    for mode in ['train', 'valid']:
        # dataset
        if args.valid_root != '': # A validation dataset is already existed
            root = args.root if mode=='train' else args.valid_root
            valid_ratio = 0 if mode=='train' else 1
        else: # to split train dataset into train-data and validation-data
            root = args.root
            valid_ratio = args.valid_ratio
        dataset = PetSegDataset(
            root=root, mode=mode, valid_ratio=valid_ratio, 
            augmentation_prob=args.augmentation_prob, img_size=args.img_size)
        # data loader
        if mode == 'train':
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.distributed else None
            loader[mode] = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        else: # valid
            loader[mode] = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
        # loggin dataset info
        t = [dataset.info['NOR']['C'], dataset.info['ABN']['C'], dataset.info['NOR']['D'], dataset.info['ABN']['D']]
        if not args.distributed or (args.distributed and args.rank!=0):
            strFormat = '%-5s%-5s%-5s%-s'
            logger.info(f"{mode} Dataset Info")
            logger.info(strFormat % ("", "NOR", "ABN", "sum"))
            logger.info(strFormat % ("C", t[0], t[1], t[0]+t[1]))
            logger.info(strFormat % ("D", t[2], t[3], t[2]+t[3]))
            logger.info(strFormat % ("sum", t[0]+t[2], t[1]+t[3], sum(t)))

    # last batch size assertion
    assert len(loader['train'].dataset)%args.batch_size!=1, 'nn.BatchNorm2d require a size of a last batch to be more than 1.'
    
    # model save info
    create_directory(args.model_path)

    model_name = f'{args.root.split("/")[-1]}_{args.backbone}_FCE_{args.epochs}_{args.lr}_{args.num_epochs_decay}_{args.lr_decay}_Adam_{args.beta1}_{args.beta2}_{args.weight_decay}'
    model_name = model_name + '_pretrained' if args.pretrained else model_name
    model_name = model_name + f"_{'-'.join(map(str,args.loss_weight))}"

    # train
    patience = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        step_lr_schedule(optimizer, epoch, args)

        # train for one epoch
        loss_sum, loss_cnt = train(loader['train'], model, criterion, optimizer, args, epoch+1)

        # evaluate on validation set
        score_sum, score_cnt, f1_scores = validate(loader['valid'], model, criterion, args, epoch+1)

        # write history
        if args.distributed:
            epoch_result = torch.tensor([loss_sum, loss_cnt, score_sum, score_cnt],
                                       device=f'cuda:{args.gpu}')
            dist.all_reduce(epoch_result, dist.ReduceOp.SUM, async_op=False)
            loss_sum, loss_cnt, score_sum, score_cnt = temp_tensor.tolist()

        score = score_sum/score_cnt
        history['train_loss'].append(loss_sum/loss_cnt)
        history['valid_loss'].append(score)
        for iou_threshold, score in f1_scores.items():
            history[iou_threshold].append(score)

        # remember best score(loss) and save checkpoint
        is_best = True if best_score is None else score < best_score
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
            save_checkpoint(best_score, score, {
                    'epoch': epoch,
                    'history': history,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'args': args
                }, is_best, path=os.path.join(args.model_path, model_name), verbose=True)
        
        # remember best f1 score and save checkpoint
        for k in f1_scores.keys():
            is_best_f1 = True if epoch == 0 else f1_scores[k] > best_f1_scores[k]
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                save_checkpoint(best_f1_scores[k], f1_scores[k], {
                        'epoch': epoch,
                        'history': history,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'args': args
                    }, is_best_f1, path=os.path.join(args.model_path, model_name), best_metric=k, verbose=True)
            if is_best_f1:
                best_f1_scores[k] = f1_scores[k]
            
            
        # early stopping
        if is_best:
            best_epoch = epoch + 1
            best_score = score
            patience = 0
        else:
            patience += 1
            if args.patience > 0:
                if patience > args.patience:
                    logger.info("Early Stopping")
                    break
                else:
                    logger.info(f'EarlyStopping counter: {patience} out of {args.patience}')
            

def train(train_loader, model, criterion, optimizer, args, epoch):
    losses=AverageMeter('Loss')

    # switch to train mode
    model.train()

    for i, batches in enumerate(train_loader):
        img_names = batches['img_name']
        images = batches['input']
        masks = batches['mask']
        labels = batches['label']

        if args.gpu is not None:
            images=images.cuda(args.gpu, non_blocking=True)
            masks=masks.cuda(args.gpu, non_blocking=True)
            labels=labels.cuda(args.gpu, non_blocking=True)

        # backward
        SR=model(images)
        one_hot_masks=make_one_hot(masks.unsqueeze(1), num_classes=args.out_ch).cuda(args.gpu, non_blocking=True)
        loss=criterion(SR, one_hot_masks)

        losses.update(loss.item(), images.shape[0])

        # comput gradient and do SGD step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # log loss
        i += 1
        if (args.log_freq == 0 and i == len(train_loader)) or (args.log_freq > 0 and i % args.log_freq == 0):
            score=torch.tensor([losses.val*images.shape[0], images.shape[0], losses.sum, losses.count]).cuda(args.gpu)
            if args.distributed:
                dist.all_reduce(score, dist.ReduceOp.SUM, async_op=False)
            if not args.distributed or (args.distributed and args.rank == 0):
                logger.info(epochFormat % (f"Epoch [{epoch}/{args.epochs}]",
                                           f"Batch [{i}/{len(train_loader)}] train loss: {(score[0]/score[1]).item()} (epoch avg {(score[2]/score[3]).item()})"))

    return losses.sum, losses.count


def validate(valid_loader, model, criterion, args, epoch):
    losses=AverageMeter('Loss')
    Eval={"S": SegmentationEvaluation(num_classes=args.out_ch, reduction='sum'),
          "C": ClassificationEvaluation(num_classes=args.out_ch-1, threshold=[0.0, 0.25, 0.5])}
    f1_scores = defaultdict(float)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, batches in enumerate(valid_loader):
            img_names = batches['img_name']
            images = batches['input']
            masks = batches['mask']
            labels = batches['label']
            if args.gpu is not None:
                images=images.cuda(args.gpu, non_blocking=True)
                masks=masks.cuda(args.gpu, non_blocking=True)
                labels=labels.cuda(args.gpu, non_blocking=True)

            # compute output
            SR=model(images)
            one_hot_masks=make_one_hot(masks.unsqueeze(1), num_classes=args.out_ch).cuda(args.gpu, non_blocking=True)
            loss=criterion(SR, one_hot_masks)
            losses.update(loss.item(), images.shape[0])

            # evaluation
            one_hot_SR=make_one_hot(torch.argmax(SR, dim=1, keepdim=True), num_classes=args.out_ch).cuda(args.gpu, non_blocking=True)
            one_hot_CR=SR2CR(SR).cuda(args.gpu, non_blocking=True)
            one_hot_labels=make_one_hot(labels.unsqueeze(1), num_classes=args.out_ch-1).cuda(args.gpu, non_blocking=True)
            Eval['S'](one_hot_SR, one_hot_masks)
            Eval['C'](one_hot_CR, one_hot_labels, one_hot_SR, one_hot_masks)

    
    # logging loss
    titleFormat = '\t%s'
    clsFormat = '\t%-12s%-4s%-4s%-4s%-4s%-12s%-7s%-s'
    segFormat = '\t%-10s%-10s%-10s'
    i += 1
    score=torch.tensor([losses.sum, losses.count]).cuda(args.gpu)
    if args.distributed:
        dist.all_reduce(score, dist.ReduceOp.SUM, async_op=False)
    if not args.distributed or (args.distributed and args.rank == 0):
        logger.info("Validation")
        logger.info(titleFormat % (f"valid loss: {(score[0]/score[1]).item()}"))
    for t in Eval['C'].threshold:
        classes=[0,1]
        TP=torch.tensor([Eval['C'].TP[t][c] for c in classes]).cuda(args.gpu)
        FN=torch.tensor([Eval['C'].FN[t][c] for c in classes]).cuda(args.gpu)
        FP=torch.tensor([Eval['C'].FP[t][c] for c in classes]).cuda(args.gpu)
        TN=torch.tensor([Eval['C'].TN[t][c] for c in classes]).cuda(args.gpu)
        if args.distributed:
            for mt in [TP, FN, FP, TN]:
                dist.all_reduce(mt, dist.ReduceOp.SUM, async_op=False)
        if not args.distributed or (args.distributed and args.rank == 0):
            logger.info(f'\tthreshold: {t}')
            logger.info(clsFormat % ('class id', 'TP', 'FP', 'FN', 'TN', 'precision', 'recall', 'f1_score'))
            for i in classes:
                PC = binary_precision(TP[i], FP[i])
                RC = binary_recall(TP[i], FN[i])
                F1 = binary_f1(PC, RC)
                logger.info(clsFormat % (classes[i], int(TP[i]), int(FP[i]), int(FN[i]), int(TN[i]),
                                         round(PC.item(), 4), round(RC.item(), 4), round(F1.item(), 4)))
                if i == 1:
                    f1_scores[f'f1_{t}'] = float(F1)

    # logging segmentation evaluation
    classes = [0, 1, 2]
    temp_iou = torch.tensor([Eval['S'].IoU['samplewise'][c] for c in classes]).cuda(args.gpu)
    temp_dice = torch.tensor([Eval['S'].DSC['samplewise'][c] for c in classes]).cuda(args.gpu)
    temp_cnt=torch.tensor([Eval['S'].len[c] for c in classes]).cuda(args.gpu)
    if args.distributed:
        for mt in [temp_iou, temp_dice, temp_cnt]:
            dist.all_reduce(mt, dist.ReduceOp.SUM, async_op=False)
    if not args.distributed or (args.distributed and args.rank == 0):
        logger.info(titleFormat % ("segmentation evaluation"))
        logger.info(segFormat % ('class id', 'IoU', 'Dice'))
        for i in range(len(classes)):
            logger.info(segFormat % (classes[i], round((temp_iou[i]/temp_cnt[i]).item(),4), round((temp_dice[i]/temp_cnt[i]).item(), 4)))

    return losses.sum, losses.count, f1_scores


def step_lr_schedule(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr=args.lr * (args.lr_decay ** (epoch // args.num_epochs_decay))
    for param_group in optimizer.param_groups:
        param_group['lr']=lr


if __name__ == '__main__':
    main(args)