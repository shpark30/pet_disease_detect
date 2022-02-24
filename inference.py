import json
import argparse 
import os
from PIL import Image

import torch
from network import build_model
from utils import make_one_hot
import albumentations as A
from albumentations.pytorch import ToTensorV2

def main(path, model_path):
    # get input
    image = np.asarray(Image.open(path))
    h, w = image.shape[:2]
    model_input = preprocessing(image)
    
    # build_model
    model = build_model(img_ch=3, out_ch=3, backbone='xception', pretrained=False, freeze_bn=True)
    model.load_state_dict(torch.load(model_path)['state_dict'])
    
    model.eval()
    with torch.no_grad():
        output = model(model_input)
    
    # confidence score
    confidence = get_confidence(output[0])
    
    # upsample, one-hot output(original image size)
    post_output = postprocessing(output[0])
    
    # get contours
    contours = mask_to_polygon(post_output)
    
    return ['피부염', 질환여부, confidence, 'polygon', contours]
    
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
    conf_max = torch.argmax(output, axis=0)
    C1_conf = output[1][conf_max==1].mean()
    C6_conf = output[2][conf_max==2].mean()
    return {'C1_감염성피부염': C1_conf,
            'C6_비감염성피부염': C6_conf}
    
def postprocessing(output, h, w):
    """
    albumentation은 numpy로 써야되나? HWC????
    output = tensor.shape: CHW
    """
    Transform = [A.Resize(height=h, width=w)]
    Transform = A.Compose(Transform)
    transformed = Transform(image=output)['image']
    transformed = torch.argmax(transformed, axis=0, keepdim=True).unsqueeze(0)
    return make_one_hot(transformed, num_classes=3).squeeze(0)

def mask_to_polygon(mask):
    contours = []
    for i in range(1, mask.shape[0]):
        contour = cv2.findContours(image=mask[i].numpy().astype(np.uint8), mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
        contours.append({
            'points': [contour[0][j].reshape(-1,2) for j in range(len(contour[0]))]
            'label': i
        })
    return contours
        
    
if __name__="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('path', metavar='DIR', help='path to jpg')
    parser.add_argument('--model-path', default='', type=str, help='path to model')
    args=parser.parse_args()
    
    main(args.path, args.model_path