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


class CS_SYNTH_PASCAL(data.Dataset):
    def __init__(self, root, scene_list):
        """
            params

                root : str
                    Path to the data folder. For CS_SYNTH at NLE, this
                    should be '/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/cs_synth_pascal_val/'

        """

        self.root = root

        self.files = []
        self.label_files = []


        self.img_paths = glob.glob(os.path.join(self.root, 'leftImg8bit/val/*/*.png')) 
        self.img_paths = sorted(self.img_paths)

        self.label_img_paths = glob.glob(os.path.join(self.root, 'gtFine/val/*/*.png'))
        self.label_img_paths = sorted(self.label_img_paths)

        print(f'Retrieving CS_SYNTH_PASCAL ({len(self.img_paths)} images)')

        for img_path in self.img_paths:
            name = img_path.split('/')[-1]
            self.files.append({
                'img': img_path, # used path
                'name': name # just the end of the path
            })

        for label_img_path in self.label_img_paths:
            name = label_img_path.split('/')[-1]
            self.label_files.append({
                'label_img': label_img_path, # used path
                'label_name': name # just the end of the path
            })

        self.annotation_path_list = [_['label_img'] for _ in self.label_files]
        self.img_path_list = [_['img'] for _ in self.files]

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):

        image = Image.open(self.files[index]['img']).convert('RGB')
        image = np.asarray(image, np.float32)
        name = self.files[index]['img']

        label = Image.open(self.label_files[index]['label_img'])#.convert('RGB')
        label = np.asarray(label).astype('float32')
        label_name = self.label_files[index]['label_name']

        size = image.shape

        return image.copy(), label.copy(), np.array(size), name


if __name__ == '__main__':
    pass

