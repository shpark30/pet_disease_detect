import argparse
import os
import logging
import warnings
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image

from utils import *
from evaluation import *
from network import build_model
from dataloader import PetSkinDataset

parser=argparse.ArgumentParser(
       description='Testing Disease Recognition in Pet CT')

# Dataset
parser.add_argument('root', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--test-ratio', default=0.2, type=float,
                    help='')
parser.add_argument('--img_size', default=512, type=int,
                    help='input data will be resized to img_size')

# DataLoader Parameter
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=1, type=int, metavar='N',
                    help='mini-batch size (default: 16), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

# Model Parameter
parser.add_argument('-m', '--model-path', required=True, type=str,
                    help='a path of model to test')
parser.add_argument('--log-name', default='test.log', type=str)

# Log
parser.add_argument('--log-path', default='./logs', type=str,
                    help='a path of log file')

# Result
parser.add_argument('--iou-threshold', default=0.5, type=float)

# Use GPU
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

# Seed
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training.')

args = parser.parse_args()

os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

create_directory(args.log_path)
logger = create_logger(args.log_path, file_name=args.log_name)

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
        warnings.warn('You have chosen to seed test.')
        
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU.')
        device = f'cuda:{args.gpu}'
    else:
        warnings.warn('using CPU, this will be slow')
        device = 'cpu'
        
    # load model
    state = torch.load(args.model_path, map_location=device)
    if state['args'].pretrained is True:
        state['args'].pretrained = False
    model = build_model(state['args'])
    if state['args'].distributed == True:
        new_state_dict = OrderedDict()
        for k, v in state['state_dict'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state['state_dict'])
    model = model.to(device)
        
    # get loader
    dataset = PetSkinDataset(root=args.root, mode='test', test_ratio=args.test_ratio, img_size=args.img_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=device=='cuda')
        
    # dataset log
    logger.info(f"model : {args.model_path}")
    logger.info(f"test Dataset Info")
    logger.info(f'test 데이터 수 : {len(dataset)}')
    logger.info(f'test 폴리곤 수 : {dataset.info_polygon.items()}')
    
    # test
    args.out_ch = state['args'].out_ch
    test(loader, model, args, device, logger)
    

def test(loader, model, args, device, logger):
    Eval = SegmentationEvaluation(num_classes=args.out_ch, reduction='sum')
    
    model.eval()
    with torch.no_grad():
        for i, batches in enumerate(loader):
            img_names = batches['img_name']
            images = batches['input']
            masks = batches['mask']
            if args.gpu is not None:
                images=images.cuda(args.gpu, non_blocking=True)
                masks=masks.cuda(args.gpu, non_blocking=True)

            # compute output
            SR=model(images)
            one_hot_masks=make_one_hot(masks.unsqueeze(1), num_classes=args.out_ch).cuda(args.gpu, non_blocking=True)
            one_hot_SR=make_one_hot(torch.argmax(SR, dim=1, keepdim=True), num_classes=args.out_ch).cuda(args.gpu, non_blocking=True)
            Eval(one_hot_SR, one_hot_masks)

    
    # logging loss
    titleFormat = '\t%s'
    segFormat = '\t%-10s%-10s%-10s'

    # logging segmentation evaluation
    classes = [0, 1, 2]
    temp_iou = torch.tensor([Eval.IoU['samplewise'][c] for c in classes]).cuda(args.gpu)
    temp_dice = torch.tensor([Eval.DSC['samplewise'][c] for c in classes]).cuda(args.gpu)
    Eval.len = {c: len([i for i in Eval.IoU[c] if i!=0]) for c in range(3)}
    temp_cnt = torch.tensor([Eval.len[c] for c in classes]).cuda(args.gpu)
    logger.info(titleFormat % ("segmentation evaluation"))
    logger.info(segFormat % ('class id', 'IoU', 'Dice'))
    for i in range(len(classes)):
        logger.info(segFormat % (classes[i], round((temp_iou[i]/temp_cnt[i]).item(),4), round((temp_dice[i]/temp_cnt[i]).item(), 4)))
            
if __name__ == '__main__':
    main(args)
