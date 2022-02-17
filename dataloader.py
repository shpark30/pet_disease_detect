import os
import random
import numpy as np
import cv2
import json
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

class PetSegDataset(Dataset):
    """ """ 
    def __init__(self, root, mode="train", valid_ratio=0.2, augmentation_prob=0.4, img_size=512, seed=42):
        random.seed(seed)
        # Assertion
        assert os.path.isdir(root), f"{root} is not existed."
        assert mode in ['train', 'valid', 'test']
        assert img_size % 16 == 0, f'img_size should be a multiple of 16 (4 downsamplings), get {img_size}'
        
        self.root = root
        self.mode = mode
        self.augmentation_prob = augmentation_prob
        self.img_size = img_size
        self.label_index = {"NOR": 1, "ABN": 2}
        
        json_iter = filter(lambda f: f[-4:]=="json", os.listdir(self.root))
        # test dataset
        if mode == 'test': 
            self.jsons = list(json_iter)
            
        # train & valid dataset
        else: 
            # get data
            jsons = {"NOR": {"C": [], "D": []},
                    "ABN": {"C": [], "D": []}}
            for j in list(json_iter):
                with open(os.path.join(self.root, j), 'r') as f:
                    json_data = json.load(f)
                disease = json_data['metadata']['Disease']
                species = json_data['metadata']['Species']
                jsons[disease][species].append(j)

            # train-valid split
            self.jsons = []
            for disease, species_dict in jsons.items():
                for species, json_files in species_dict.items():
                    random.shuffle(json_files)
                    if mode == 'train':
                        self.jsons += json_files[:int(len(json_files)*(1-valid_ratio))]
                    elif mode == 'valid':
                        self.jsons += json_files[int(len(json_files)*(1-valid_ratio)):]
                    
                    
        # dataset info
        self.info = {"NOR": {"C": 0, "D": 0},
                     "ABN": {"C": 0, "D": 0}}
        for j in self.jsons:
            with open(os.path.join(self.root, j), 'r') as f:
                json_data = json.load(f)
            disease = json_data['metadata']['Disease']
            species = json_data['metadata']['Species']
            self.info[disease][species] += 1
                
    def print_info(self):
        print(f"{self.mode} Dataset Info"+\
             "\n\t\tNOR\tABN\tsum"+\
             f"\n\tC\t{self.info['NOR']['C']}\t{self.info['ABN']['C']}\t{self.info['NOR']['C']+self.info['ABN']['C']}"+\
             f"\n\tD\t{self.info['NOR']['D']}\t{self.info['ABN']['D']}\t{self.info['NOR']['D']+self.info['ABN']['D']}"+\
             f"\n\tsum\t{self.info['NOR']['C']+self.info['NOR']['D']}\t{self.info['ABN']['C']+self.info['ABN']['D']}\t{sum([n for k in self.info.keys() for n in self.info[k].values()])}")
            
    def __len__(self):
        return len(self.jsons)
        
    def __getitem__(self, index):
        # get file paths of the image and json files
        json_name = self.jsons[index]
        img_name = self.jsons[index].replace(".json", ".jpg")
        json_path = os.path.join(self.root, self.jsons[index])
        img_path = os.path.join(self.root, self.jsons[index].replace(".json", ".jpg"))
        
        # get image, class label, polygon coordinates
        image = cv2.imread(img_path) # (height, width, channel)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label, polygon = self.get_annotation(json_path)
        one_hot_label = torch.zeros(1,1, dtype=torch.long) if label==1 else torch.ones(1,1, dtype=torch.long) 
                    
        # (Augmentation) Transform image, polygon
        coord_len = [len(poly) for poly in polygon]
        polygon = np.array([point for poly in polygon for point in poly], dtype = np.int64)
        image, polygon = self.transform(image, polygon)
        # polygon is pts argument of cv2.fillPoly. So it must be like [np.array([x,y], dtype=np.intxx)]
        polygon = np.around(polygon).astype(np.int64)
        poly_num = len(coord_len)
        coord_len = [0] + coord_len
        polygon = [polygon[coord_len[i]:coord_len[i+1]] for i in range(poly_num)]
       
        # get a mask image using transformed polygon coordinates
        h, w = image.shape[:2]
        mask = self.get_mask(label, polygon, h=h, w=w)
       
        # from numpy to tensor
        Transform = ToTensorV2(transpose_mask=True)
        image = Transform.apply(image)
        mask = Transform.apply_to_mask(mask) # one_hot
        mask = torch.argmax(mask, dim=0) # categorical

        return {"img_name": img_name, "input": image, "mask": mask, "label": one_hot_label}
    
    def get_annotation(self, json_path):
        """ get a label and polygon coordinates from a json file """
        # background : 0
        # Normal : 1
        # Abnormal : 2
        with open(json_path, 'r') as f:
            json_data =json.load(f)
        label = self.label_index[json_data['metadata']['Disease']]
        polygon = [anno['points'] for anno in json_data["annotations"] if anno['shape'] == "Polygon"] # [[x1, y1], ... ]
        
        return label, polygon
    
    def get_mask(self, label, polygon, h, w):
        """ get a numpy mask image. The shape is like [H, W, C] """
        mask = np.zeros((h, w, 3), dtype=np.int64)
        target = np.zeros((h, w))
        cv2.fillPoly(target, polygon, 1, cv2.LINE_AA)
        mask[:,:,label] = target # label channel
        mask[:,:,0][mask[:,:,label]==0] = 1 # background channel
        return mask

    def transform(self, image, polygon):
        """ transform a image and  points of polygons repectively """
        Transform = []
        h, w = image.shape[:2]
        aspect_ratio = h / w
        
        # resize
        Transform.append(A.Resize(height=self.img_size, width=self.img_size))
        
        # rotate, flip, color transform
        p = random.random()
        if (self.mode == 'train') and (p < self.augmentation_prob):
            Transform = Transform + [A.RandomRotate90(p=1.),
                                     A.Rotate(limit=(-10, 10), p=0.5),
                                     A.HorizontalFlip(p=0.5),
                                     A.VerticalFlip(p=0.5),
                                     A.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02, p=1.)]
        
        # normalize, totensor
        Transform.append(A.Normalize(mean=[0.0, 0.0, 0.0],
                                     std=[1.0, 1.0, 1.0],
                                     max_pixel_value=255.0))
        
        Transform = A.Compose(Transform, keypoint_params=A.KeypointParams(format='xy'))
        transformed = Transform(image=image, keypoints=polygon)
        transformed_image = transformed['image']
        transformed_polygon = transformed['keypoints']
        
        return transformed_image, transformed_polygon
    