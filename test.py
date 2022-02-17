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
from dataloader import PetSegDataset

parser=argparse.ArgumentParser(
       description='Testing Disease Recognition in Pet CT')

# Dataset
parser.add_argument('root', metavar='DIR',
                    help='path to dataset')
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
    dataset = PetSegDataset(root=args.root, mode='test', img_size=args.img_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=device=='cuda')
        
    # dataset log
    logger.info(f"model : {args.model_path}")
    strFormat = '%-5s%-5s%-5s%-s'
    logger.info(f"test Dataset Info")
    logger.info(strFormat % ("", "NOR", "ABN", "sum"))
    logger.info(strFormat % ("C", dataset.info['NOR']['C'], dataset.info['ABN']['C'], dataset.info['NOR']['C']+dataset.info['ABN']['C']))
    logger.info(strFormat % ("D", dataset.info['NOR']['D'], dataset.info['ABN']['D'], dataset.info['NOR']['D']+dataset.info['ABN']['D']))
    logger.info(strFormat % ("sum", dataset.info['NOR']['C']+dataset.info['NOR']['D'], dataset.info['ABN']['C']+dataset.info['ABN']['D'],
                             sum([n for k in dataset.info.keys() for n in dataset.info[k].values()])))
    
    # test
    test(loader, model, args, device, logger)
    

def test(loader, model, args, device, logger):
    Eval = ClassificationEvaluation(num_classes=model.out_ch-1, threshold=[args.iou_threshold], return_threshold=args.iou_threshold)
    t = args.iou_threshold
    c = 1
    model.eval()
    with torch.no_grad():
        strFormat = '%-60s%-8s%-8s%-8s%-12s%-12s%-s'
        logger.info("Test Evaluation")
        logger.info(f"iou threshold : {t}")
        logger.info(strFormat % ('image_name', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1-score'))
        for i, batches in enumerate(loader):
            img_names = batches['img_name']
            images = batches['input']
            masks = batches['mask']
            labels = batches['label']

            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            SR = model(images)
            one_hot_CR = SR2CR(SR) # shape: (N,C,H,W)
            one_hot_labels = make_one_hot(labels.unsqueeze(1), num_classes=model.out_ch-1) # shape: (N,C,1,1)
            one_hot_SR=make_one_hot(torch.argmax(SR, dim=1, keepdim=True), num_classes=model.out_ch)
            one_hot_masks=make_one_hot(masks.unsqueeze(1), num_classes=3)
            Eval(one_hot_CR, one_hot_labels, one_hot_SR, one_hot_masks)
            # logging evaluation per image
            PC = binary_precision(Eval.TP[t][c], Eval.FP[t][c])
            RC = binary_recall(Eval.TP[t][c], Eval.FN[t][c])
            F1 = binary_f1(PC, RC)
            logger.info(strFormat % (img_names[0], Eval.TP[t][c], Eval.FP[t][c], Eval.FN[t][c],
                                 round(PC, 4), round(RC, 4), round(F1, 4)))
            
        
        # logging Final Result
        strFormat = '\t%-6s%-6s%-6s%-6s%-11s%-8s%-s'
        logger.info("Test Result")
        t = args.iou_threshold
        logger.info(f'iou_threshold: {t}')
        logger.info(strFormat % ('TP', 'FP', 'FN', 'TN', 'precision', 'recall', 'f1_score'))
        PC = binary_precision(Eval.TP[t][c], Eval.FP[t][c])
        RC = binary_recall(Eval.TP[t][c], Eval.FN[t][c])
        F1 = binary_f1(PC, RC)
        logger.info(strFormat % (Eval.TP[t][c], Eval.FP[t][c], Eval.FN[t][c], Eval.TN[t][c],
                                 round(PC, 4), round(RC, 4), round(F1, 4)))
            
if __name__ == '__main__':
    main(args)
