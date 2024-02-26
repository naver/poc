import os
import cv2
import json
import glob

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize, InterpolationMode


def round_to_nearest_multiple(x, p):
    return int(((x - 1) // p + 1) * p)


class RoadAnomaly(Dataset):

    def __init__(self, hparams, transforms):
        super().__init__()

        self.hparams = hparams
        self.transforms = transforms

        self.images = glob.glob(os.path.join(hparams.dataset_root, 'images', '*.jpg'))
        self.images.sort()
        self.num_samples = len(self.images)
        self.labels = [x.replace('.jpg', '.png').replace('/images/', '/labels_masks/') for x in self.images]

    def __getitem__(self, index):

        image = self.read_image(self.images[index])
        label = self.read_image(self.labels[index])

        label = label[:, :, 0]
        label[label == 2] = 1

        aug = self.transforms(image=image, mask=label)
        image = aug['image']
        label = aug['mask']

        if self.hparams.test_image_strategy == 'multi_scale':

            images = []
            for sz in self.hparams.test_image_sizes:
                new_size = [
                    sz,
                    round_to_nearest_multiple(sz * self.hparams.test_hw_ratio, self.hparams.seg_downsample_rate)
                ]
                images.extend([resize(image, size=new_size)])

            image = images

        return image, label.type(torch.LongTensor)

    @staticmethod
    def read_image(path):

        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        return img

    def __len__(self):
        return self.num_samples
