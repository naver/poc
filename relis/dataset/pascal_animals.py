'''
Adapted from https://github.com/naver/oasis/blob/master/dataset/cityscapes_dataset.py
'''
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import glob
import pickle

import torch
import torchvision
from torch.utils import data
import torchvision.transforms as transforms

from PIL import Image
import cv2


'''
Conversion Cityscapes to PASCAL:
    
    Cityscapes classes:
        0 road
        1 sidewalk
        2 building
        3 wall
        4 fence
        5 pole
        6 light
        7 sign
        8 vegetation
        9 terrain # NOT IN IDD!!!
        10 sky
        11 person
        12 rider
        13 car
        14 truck
        15 bus
        16 train
        17 motocycle
        18 bicycle
        
    PASCAL classes:
        'background': 0,
        'aeroplane': 1,
        'bicycle': 2,
        'bird': 3,
        'boat': 4,
        'bottle': 5,
        'bus': 6,
        'car': 7,
        'cat': 8,
        'chair': 9,
        'cow': 10,
        'diningtable': 11,
        'dog': 12,
        'horse': 13,
        'motorbike': 14,
        'person': 15,
        'pottedplant': 16,
        'sheep': 17,
        'sofa': 18,
        'train': 19,
        'tvmonitor': 20
'''

classes = [
    'background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]


class PascalAnimals(data.Dataset):
    def __init__(self, root, scene_list=None, batch_size = 1):
        """
            params

                root : str
                    Path to the data folder.

        """

        self.class_conversion_dict = {3:19, 8:20, 10:21, 12:22, 13:23, 17:24}

        self.root = root
        self.batch_size = batch_size

        self.files = []
        self.label_files = []
        
        root_image_path = f'{root}/JPEGImages/'
        label_path = f'{root}/SegmentationClass/'
        
        
        path = f'{root}/ImageSets/Segmentation/val.txt'
        with open(path, 'r') as f:
            images = [line.rstrip() for line in f]
            
        img_paths = []
        label_paths = []
        
        for img in images:
            label = Image.open(label_path + img + '.png')
            valid = False
            for cl in self.class_conversion_dict.keys():
                if cl in np.array(label):
                    valid = True
            if valid:
                image_path = root_image_path + img + '.jpg'
                img_paths.append(image_path)
                label_paths.append(label_path + img + '.png')

        for img_path in img_paths:
            name = img_path.split('/')[-1]
            self.files.append({
                'img': img_path, # used path
                'name': name # just the end of the path
            })

        for label_img_path in label_paths:
            name = label_img_path.split('/')[-1]
            self.label_files.append({
                'label_img': label_img_path, # used path
                'label_name': name # just the end of the path
            })

        self.annotation_path_list = [_['label_img'] for _ in self.label_files]
        self.img_path_list = [_['img'] for _ in self.files]
        
        print(f'Retrieving PascalAnimals ({len(img_paths)} images)')


    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):

        image = Image.open(self.files[index]['img']).convert('RGB')
        image = np.asarray(image, np.float32)
        name = self.files[index]['img']

        label = Image.open(self.label_files[index]['label_img'])#.convert('RGB')
        label = np.asarray(label)
        label_name = self.label_files[index]['label_name']

        # re-assign labels to filter out non-used ones
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.class_conversion_dict.items():
            label_copy[label == k] = v

        size = image.shape

        return image.copy(), label_copy.copy(), np.array(size), name


if __name__ == '__main__':
    pass

