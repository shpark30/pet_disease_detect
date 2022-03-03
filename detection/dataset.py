import os
import cv2
import json
import random
import numpy as np
from collections import defaultdict, Counter
from PIL import Image, ImageDraw, ImageFile

import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PetEyeDataset(Dataset): 
    def __init__(self, root, mode="train", valid_ratio=0.2, test_ratio=0.2, augmentation_prob=0.4, img_size=512, seed=42):
        random.seed(seed)
        # Assertion
        assert os.path.isdir(root), f"{root} is not existed."
        assert mode in ['train', 'valid', 'test']
        assert (valid_ratio+test_ratio) < 1
        assert img_size % 16 == 0, f'img_size should be a multiple of 16 (4 downsamplings), get {img_size}'
        
        self.root = root
        self.mode = mode
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.augmentation_prob = augmentation_prob
        self.img_size = img_size
        self.seed = seed
        
        self.jsons = []
        self.jsons_mode = {}
        self.label_index = {"무": 1,
                           "유": 2}
        
        jsons = [os.path.join(r, f) for (r, dirs, files) in os.walk(root) for f in filter(lambda f: f[-4:]=="json", files)]
        random.shuffle(jsons)
        
        if mode == 'train':
            self.jsons = jsons[:int(len(jsons)*(1-valid_ratio-test_ratio))]
        elif mode == 'valid':
            self.jsons = jsons[int(len(jsons)*(1-valid_ratio-test_ratio)):int(len(jsons)*(1-test_ratio))]
        elif mode == 'test':
            self.jsons = jsons[int(len(jsons)*(1-test_ratio)):]
            
        # dataset info
        self._get_dataset_info()
            
    def _get_dataset_info(self):
        self.info_box = Counter([])
        for j in self.jsons:
            with open(j, 'r') as f:
                json_data = json.load(f)
            self.info_box.update([json_data['label']['label_disease_lv_3']])
            
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        # get data
        img_name, image, bbox = self.get_data(index)
        
        # transformation
        h, w = image.shape[:2]
        transformed_image, transformed_bbox = self.transform(image, bbox)
        
        # from numpy to tensor
        transformed_image = ToTensorV2().apply(transformed_image) # dtype??
        transformed_bbox = [list(map(int, box)) for box in transformed_bbox]
        transformed_bbox = torch.Tensor(transformed_bbox) # dtype??
        
        return {'img_name':img_name, 'input': transformed_image, 'target': transformed_bbox}
     
    def get_data(self, index):
        # load the image and convert image to numpy array
        img_name = self.jsons[index].replace('json', 'jpg')
        image = np.asarray(Image.open(img_name))
        h, w = image.shape[:2]
        
        # get bbox
        with open(self.jsons[index], 'r') as f:
            json_data =json.load(f)
        bbox = json_data['label']['label_bbox'] # points
        bbox.append(self.label_index[json_data['label']['label_disease_lv_3']]) # label
        
        return img_name, image, [bbox]
    
    def _get_data(self, index):
        # load the image and convert image to numpy array
        img_name = self.jsons[index].replace('json', 'jpg')
        image = np.asarray(Image.open(img_name))
        
        # get bboxes
        with open(self.jsons[index], 'r') as f:
            json_data =json.load(f)
        bboxes = []
        for anno in json_data['labelingInfo']:
            if 'box' in anno.keys():
                label = self.label_index[anno['box']['label']]
                for bbox_i in anno['polygon']['location']:
                    bbox = [bbox_i['x'], bbox_i['y'], bbox_i['width'], bbox_i['height'], label]
                    bboxes += box
        return img_name, image, bboxes
                    
        
    def transform(self, image, bbox, img_name=''):
        Transform = []
        h, w = image.shape[:2]
        aspect_ratio = h / w
        
        # resize
        Transform.append(A.Resize(height=self.img_size, width=self.img_size))
        
        # rotate, flip, color transform
        p = random.random()
        if (self.mode == 'train') and (p < self.augmentation_prob):
            Transform += [A.RandomRotate90(p=1.),
                          A.Rotate(limit=(-10, 10), p=0.5),
                          A.HorizontalFlip(p=0.5),
                          A.VerticalFlip(p=0.5),
                          A.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02, p=1.)]
            
        Transform.append(A.Normalize(mean=[0.0, 0.0, 0.0],
                                     std=[1.0, 1.0, 1.0],
                                     max_pixel_value=255.0))
        Transform = A.Compose(Transform, bbox_params=A.BboxParams(format='pascal_voc')) # pascal_voc format : [x_min, y_min, x_max, y_max]
        transformed = Transform(image=image, bboxes=bbox)
        return transformed['image'], transformed['bboxes']