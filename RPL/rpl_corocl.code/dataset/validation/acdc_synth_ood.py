import os
import torch
from PIL import Image
from collections import namedtuple
import glob
import numpy as np


class ACDCSynth(torch.utils.data.Dataset):
    """
    Images of ACDC val set augmented with stable diffusion to include both OOD and ID synth objects. 
    """
    
    train_id_in = 0
    train_id_out = 1
    
    def __init__(self, root="", transform=None):
        """Load all filenames."""
        
        self.class_conversion_dict = {7:0, 8:1, 11:2, 12:3, 13:4, 17:5, 19:6, 20:7, 21:8, 22:9,
                                      23: 10, 24:11, 25:12, 26:13, 27:14, 28:15, 31:16, 32:17, 33:18}
        
        self.transform = transform
        self.root = root
        
        self.images =  glob.glob(f'{self.root}/rgb_anon_trainvaltest/rgb_anon/*/val/*/*.png')
        self.targets = [x.replace('rgb_anon_trainvaltest/rgb_anon', 'gt_trainval/gt') for x in self.images]
        
        self.images = sorted(self.images)
        self.targets = sorted(self.targets)

    def __len__(self):
        """Return number of images in the dataset split."""
        return len(self.images)

    def __getitem__(self, i):
        """Return raw image, trainIds as torch.Tensor or PIL Image"""
        image = Image.open(self.images[i]).convert('RGB')
        target = Image.open(self.targets[i]).convert('L')
        
        target = np.array(target.copy())
            
        # re-assign targets to filter out non-used ones
        target_copy = 255 * np.ones(target.shape, dtype=np.float32)
        for k, v in self.class_conversion_dict.items():
            target_copy[target == k] = v
            
        target_copy[(target_copy < 19)] = 0
        target_copy[(target >= 100) & (target != 255)] = 1 # OOD classes
        
        target = Image.fromarray(target_copy).convert('L')

        if self.transform is not None:
            image, target = self.transform(image, target)
            
        return image, target

    def __repr__(self):
        """Print some information about dataset."""
        fmt_str = 'ACDC synth: \n'
        fmt_str += '----Number of images: %d\n' % len(self.images)
        return fmt_str.strip()

