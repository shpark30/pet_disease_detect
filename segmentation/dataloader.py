import os
import random
import numpy as np
import cv2
cv2.setNumThreads(0)
from PIL import Image, ImageDraw, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json
from collections import defaultdict, Counter
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

class _SegmentationDataset(Dataset):
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
        self.label_index = {}
        
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
        self.info_polygon = Counter([])
        for j in self.jsons:
            with open(j, 'r') as f:
                json_data = json.load(f)
            self.info_polygon.update([anno['polygon']['label'][:2]  for anno in json_data['labelingInfo'] if 'polygon' in anno.keys()])
    
    def __len__(self):
        return len(self.jsons)
    
    def __getitem__(self, index):
        img_name, image, labels, polygons = self.get_data(index)
        
        # Transformation
        h, w = image.shape[:2]
        mask = self.get_mask(labels, [np.array(poly, dtype=np.int32) for poly in polygons],
                             h=h, w=w, img_name=img_name)
        transformed_image, transformed_mask = self.transform(image, mask, img_name)
        
        # from numpy to tensor
        Transform = ToTensorV2(transpose_mask=False)
        image = Transform.apply(transformed_image)
        mask = Transform.apply_to_mask(transformed_mask).type(torch.int64) # one_hot
        
        return {"img_name": img_name, "input": image, "mask": mask}

    def get_data(self, index):
#         # image jpg -> np.array
#         img_name = self.jsons[index].replace('json', 'jpg')
#         image = cv2.imread(img_name) # (height, width, channel)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         h, w = image.shape[:2]

        # load the image and convert image to numpy array
        img_name = self.jsons[index].replace('json', 'jpg')
        image = np.asarray(Image.open(img_name))
        h, w = image.shape[:2]
        
        # get polygons
        with open(self.jsons[index], 'r') as f:
            json_data =json.load(f)
        labels, polygons = [], []
        for anno in json_data['labelingInfo']:
            if 'polygon' in anno.keys():
                # polygon label
                labels.append(self.label_index[anno['polygon']['label']])
                # polygon coordinates
                coord_set = anno['polygon']['location']
                for poly_i in range(len(coord_set)):
                    poly = []
                    for coord_i in range(len(coord_set[poly_i].keys())//2):
                        x = coord_set[poly_i][f'x{coord_i+1}']
                        x = x if x < w else w-1
                        y = coord_set[poly_i][f'y{coord_i+1}']
                        y = y if y < h else h-1
                        poly.append([x, y])
                    polygons.append(poly)
        return img_name, image, labels, polygons
    
    def get_mask(self, labels, polygons, h, w, img_name=''):
        # polygon = [(x1,y1),(x2,y2),...] or [x1,y1,x2,y2,...]
        mask = Image.new('L', (w, h), 0)
        for i, lb in enumerate(labels):
            poly = [tuple(xy) for xy in polygons[i].astype('int32')]
            ImageDraw.Draw(mask).polygon(poly, outline=lb, fill=lb)
        mask = np.array(mask).astype('int64')
        return mask
    
    def transform(self, image, mask, img_name=''):
        """ transform a image and  points of polygons repectively """
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
        
        # normalize, totensor
        Transform.append(A.Normalize(mean=[0.0, 0.0, 0.0],
                                     std=[1.0, 1.0, 1.0],
                                     max_pixel_value=255.0))
        Transform = A.Compose(Transform)
        transformed = Transform(image=image, mask=mask)
        return transformed['image'], transformed['mask']
   
    
    
    def __getitem__old(self, index):
        img_name, image, labels, polygons = self.get_data(index)
        
        # Transformation
        num_points_per_poly = [len(poly) for poly in polygons]
        polygons_2d = np.array([point for poly in polygons for point in poly], dtype=np.int64)
        transformed_image, transformed_polygons_2d = self.transform(image, polygons_2d, img_name)
        transformed_polygons_2d = np.around(transformed_polygons_2d).astype(np.int64) # polygon is pts argument of cv2.fillPoly. So it must be like [np.array([x,y], dtype=np.intxx)]
        num_polygons = len(num_points_per_poly)
        transformed_polygons_3d = [transformed_polygons_2d[num_points_per_poly[i-1]:num_points_per_poly[i]] \
                                   if i!=0 else transformed_polygons_2d[:num_points_per_poly[0]] \
                                   for i in range(num_polygons)]
        
        # get a mask image using transformed polygon coordinates
        h, w = transformed_image.shape[:2]
        mask = self.get_mask(labels, transformed_polygons_3d, h=h, w=w, img_name=img_name)
        
        # from numpy to tensor
        Transform = ToTensorV2(transpose_mask=False)
        image = Transform.apply(transformed_image)
        mask = Transform.apply_to_mask(mask) # one_hot

        return {"img_name": img_name, "input": image, "mask": mask}
    
    def _get_mask(self, labels, polygons, h, w, img_name=''):
        """ get a numpy mask image. The shape is like [H, W, C] """
        mask = np.zeros((h, w))
        mask = np.ascontiguousarray(mask, dtype=np.uint8)
        for i, lb in enumerate(labels):
#             poly = polygons[i].reshape((1,) + polygons[i].shape).astype(np.int32)
            poly = polygons[i].astype('int32')
            cv2.fillPoly(mask, [poly], lb, cv2.LINE_AA)
        mask = mask.astype(np.int64)
        return mask
    
    def transform_old(self, image, polygon, img_name=''):
        """ transform a image and  points of polygons repectively """
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
        
        # normalize, totensor
        Transform.append(A.Normalize(mean=[0.0, 0.0, 0.0],
                                     std=[1.0, 1.0, 1.0],
                                     max_pixel_value=255.0))
        Transform = A.Compose(Transform, keypoint_params=A.KeypointParams(format='xy'))
        transformed = Transform(image=image, keypoints=polygon)
        transformed_image = transformed['image']
        transformed_polygon = transformed['keypoints']
        if len(transformed_polygon)==0 or img_name=='../../data/피부염/C1_감염성피부염/CYT_D_C1_002249.jpg':
            print(img_name)
            print(p)
            Transform = [A.Resize(height=self.img_size, width=self.img_size),
                         A.RandomRotate90(p=1.),
                         A.Rotate(limit=(-10, 10), p=1.),
                         A.HorizontalFlip(p=1.),
                         A.VerticalFlip(p=1.),
                         A.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02, p=1.),
                         A.Normalize(mean=[0.0, 0.0, 0.0],
                                     std=[1.0, 1.0, 1.0],
                                     max_pixel_value=255.0)]
            for i in range(len(Transform)):
                transformed = A.Compose(Transform[:i+1], keypoint_params=A.KeypointParams(format='xy'))(image=image, keypoints=polygon)
                print('+'.join([repr(tf).split('(')[0] for tf in Transform[:i+1]]), transformed['image'].shape, transformed['keypoints'], sep='/t')
            transformed = A.Compose([A.Resize(height=self.img_size, width=self.img_size),
                                     A.Normalize(mean=[0.0, 0.0, 0.0],
                                                 std=[1.0, 1.0, 1.0],
                                                 max_pixel_value=255.0)],
                                    keypoint_params=A.KeypointParams(format='xy'))(image=image, keypoints=polygon)
            print('Resize+Normalize', transformed['image'].shape, transformed['keypoints'], sep='\t')
        return transformed_image, transformed_polygon
    
    
class PetSkinDataset(_SegmentationDataset):
    def __init__(self, root, mode="train", valid_ratio=0.2, test_ratio=0.2, augmentation_prob=0.4, img_size=512, seed=42):
        super().__init__(root, mode, valid_ratio, test_ratio, augmentation_prob, img_size, seed)
        self.label_index = {'A7_무증상' : 1,
                           'A1_구진_플라크': 2,
                           'A2_비듬_각질_상피성잔고리': 2,
                           'A3_태선화_과다색소침착': 2,
                           'A4_농포_여드름': 2,
                           'A5_미란_궤양': 2,
                           'A6_결절_종괴': 2,
                           'C6_비감염성피부염': 1,
                           'C1_감염성피부염': 2}
        
class PetDermatitisDataset(_SegmentationDataset):
    def __init__(self, root, mode="train", valid_ratio=0.2, test_ratio=0.2, augmentation_prob=0.4, img_size=512, seed=42):
        super().__init__(root, mode, valid_ratio, test_ratio, augmentation_prob, img_size, seed)
        self.label_index = {'C1_감염성피부염': 1,
                           'C6_비감염성피부염': 2}
