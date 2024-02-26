import torch
import os
import cv2
import glob
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize
import numpy as np

def round_to_nearest_multiple(x, p):
    return int(((x - 1) // p + 1) * p)


def read_image(path):

    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    return img


class ACDCSynth(Dataset):
    """
    Images of ACDC val set augmented with stable diffusion to include both OOD and ID synth objects. 
    """

    def __init__(self, hparams, transforms):
        super().__init__()
        
        self.class_conversion_dict = {7:0, 8:1, 11:2, 12:3, 13:4, 17:5, 19:6, 20:7, 21:8, 22:9,
                                      23: 10, 24:11, 25:12, 26:13, 27:14, 28:15, 31:16, 32:17, 33:18}
        
        self.hparams = hparams
        self.transforms = transforms

        self.images = []
        self.labels = []

        self.images = glob.glob(os.path.join(hparams.dataset_root, 'rgb_anon_trainvaltest/rgb_anon/*/val/*/*.png')) 
        self.labels = [x.replace('rgb_anon_trainvaltest/rgb_anon', 'gt_trainval/gt') for x in self.images]
        self.labels.sort()
        self.images.sort()

        self.num_samples = len(self.images)

        
    def __getitem__(self, index):

        image = read_image(self.images[index])
        label = read_image(self.labels[index])
        label = label[:, :, 0]
        
         # re-assign labels to filter out non-used ones
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.class_conversion_dict.items():
            label_copy[label == k] = v
            
        label_copy[(label_copy < 19)] = 0
        label_copy[(label >= 100) & (label != 255)] = 1 # OOD classes
        
        aug = self.transforms(image=image, mask=label_copy)
        image = aug['image']
        label = aug['mask']

        return image, label.type(torch.LongTensor)


    def __len__(self):
        return self.num_samples

