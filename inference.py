import json
import argparse 
import os
from PIL import Image
import numpy as np
import warnings
import cv2

import torch
from network.deeplab import DeepLabv3plus
from utils import make_one_hot
import albumentations as A
from albumentations.pytorch import ToTensorV2

def main(path, model_path, disease, gpu):
    if gpu is not None:
        warnings.warn('You have chosen a specific GPU.')
        device = f'cuda:{gpu}'
    else:
        warnings.warn('using CPU, this will be slow')
        device = 'cpu'
        
    # get input
    image = np.asarray(Image.open(path))
    h, w = image.shape[:2]
    model_input = preprocessing(image)
    model_input = model_input.to(device)
    
    # build_model
    model = DeepLabv3plus(img_ch=3, out_ch=3, backbone='xception', pretrained=False, freeze_bn=True)
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model = model.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(model_input)
    
    # confidence score
    label, confidence = get_confidence(output[0])
    
    if label == "Y":
        # upsample, one-hot output(original image size)
        post_output = postprocessing(output[0], h, w)
        # get contours
        contours = mask_to_polygon(post_output)
    else:
        contours = None
    
    result = {'DIS_CD': disease,
            'DIS_YN': label,
            'CONF_SCORE': confidence,
            'CLS_TYPE': 'polygon',
            'LABEL_COORD': contours}
    return result
    
    
def preprocessing(image):
    Transform = [A.Resize(height=512, width=512),
                 A.Normalize(mean=[0.0, 0.0, 0.0],
                             std=[1.0, 1.0, 1.0],
                             max_pixel_value=255.0),
                 ToTensorV2()]
    Transform = A.Compose(Transform)
    transformed = Transform(image=image)
    return transformed['image'].unsqueeze(0)


def get_confidence(output):
    output = torch.nn.Softmax(dim=0)(output)
    conf_max = torch.argmax(output, axis=0)
    num_pix1 = (conf_max==1).sum()
    num_pix2 = (conf_max==2).sum()
    if num_pix2 > num_pix1:
        label = 'Y'
        conf = output[2][conf_max==2].mean()
    elif num_pix1==0 and num_pix2==0:
        label = 'N'
        conf = output[0][conf_max==0].mean()
    else:
        label = 'N'
        conf = output[1][conf_max==1].mean()
#     conf = 0. if torch.isnan(conf) else round(float(conf), 4)
    conf = round(float(conf), 4)
    return label, conf
    
    
def postprocessing(output, h, w):
    """
    albumentation은 numpy로 써야되나? HWC????
    output = tensor.shape: CHW
    """
    output = torch.argmax(output, axis=0, keepdim=True).unsqueeze(0)
    output = make_one_hot(output, num_classes=3).squeeze(0)
    output = output.cpu().detach().numpy().transpose(1,2,0)
    Transform = [A.Resize(height=h, width=w)]
    Transform = A.Compose(Transform)
    return Transform(image=output)['image']


def mask_to_polygon(mask):
    contours = cv2.findContours(image=mask[:,:,2].astype(np.uint8), mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    return [contours[0][j].reshape(-1,2).tolist() for j in range(len(contours[0]))]
        
    
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('path', metavar='DIR', help='path to jpg')
    parser.add_argument('--disease', default='피부염', type=str, help='피부염, 구진플라크')
    parser.add_argument('--model-path', default='../output/train/model/피부염/220224/C_FCE_50_0.0002_30_0.1_Adam_0.5_0.999_0.0001_pretrained_0.2-1.4-1.4_best-loss.pth.tar', type=str, help='path to model')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    args=parser.parse_args()
    
    args.gpu = 1
    
    main(args.path, args.model_path, args.disease, args.gpu)