import os
import random
import numpy as np
import cv2
cv2.setNumThreads(0)
from PIL import Image, ImageDraw
import json
from collections import defaultdict, Counter
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

# class _PetDataset(Dataset):
#     def __init__(self, root, mode="train", valid_ratio=0.2, test_ratio=0.2, augmentation_prob=0.4, img_size=512, seed=42):
        
        
        
class PetPapulePlaqueDataset(Dataset):
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
        self.label_index = {'A7_무증상' : 1,
                           'A1_구진_플라크': 2,
                           'C1_감염성피부염': 1,
                           'C6_비감염성피부염': 2}
            
        jsons = [os.path.join(r, f) for (r, dirs, files) in os.walk(root) for f in filter(lambda f: f[-4:]=="json", files)]
#         jsons = [f for (r, dirs, files) in os.walk(root) for f in filter(lambda f: f[-4:]=="json", files)]
        random.shuffle(jsons)
#         jsons = list(filter(lambda f: f[-20:-5] not in ['IMG_D_A1_012566','IMG_D_A1_034516','IMG_D_A7_243207'], jsons))
#         jsons = list(filter(lambda f: f[-20:-5] not in ['IMG_D_A7_217148', 'IMG_D_A7_214072', 'IMG_D_A1_023612'], jsons))
#         jsons = list(filter(lambda f: f[-20:-5] not in ['IMG_D_A7_208516', 'IMG_D_A7_212252', 'IMG_D_A7_239373'], jsons))

        
        
        if mode == 'train':
            self.jsons = jsons[:int(len(jsons)*(1-valid_ratio-test_ratio))]
        elif mode == 'valid':
            self.jsons = jsons[int(len(jsons)*(1-valid_ratio-test_ratio)):int(len(jsons)*(1-test_ratio))]
        elif mode == 'test':
            self.jsons = jsons[int(len(jsons)*(1-test_ratio)):]
        
        # dataset info
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
        
        if img_name == '../../data/피부염/C1_감염성피부염/CYT_D_C1_002249.jpg':
            print('transformed_image\n\t', transformed_image.shape)
            print('transformed_mask\n\t', transformed_mask.shape, (transformed_mask==1).sum(), (transformed_mask==2).sum())
        return {"img_name": img_name, "input": image, "mask": mask}

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
#         mask = torch.argmax(mask, dim=0) # categorical

        return {"img_name": img_name, "input": image, "mask": mask}
#         return {"input": image, "mask": mask}
       
        
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
            if len(poly) < 2:
                print(img_name)
                print(polygons)
                raise
            ImageDraw.Draw(mask).polygon(poly, outline=lb, fill=lb)
        mask = np.array(mask).astype('int64')
        return mask
        
    def get_mask_old(self, labels, polygons, h, w):
        """ get a numpy mask image. The shape is like [H, W, C] """
        mask = np.zeros((h, w))
        # mask poly
        # uint8 uint8 -> 바로 에러 cv2.error: OpenCV(4.5.5) /io/opencv/modules/imgproc/src/drawing.cpp:2396: error: (-215:Assertion failed) p.checkVector(2, CV_32S) >= 0 in function 'fillPoly'
        # uint8 uint64 -> 2epoch 205 배치 다음에 위와 동일한 에러
        # uint8 int32 ->2epoch 205 배치 다음에 위와 동일한 에러
        mask=np.ascontiguousarray(mask, dtype=np.uint8)
        for i, lb in enumerate(labels):
#             poly = polygons[i].reshape((1,) + polygons[i].shape).astype(np.int32)
            poly = polygons[i].astype('int32')
            cv2.fillPoly(mask, [poly], lb, cv2.LINE_AA)
        mask = mask.astype(np.int64)
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
    
