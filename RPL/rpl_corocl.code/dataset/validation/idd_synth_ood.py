import os
import torch
from PIL import Image
from collections import namedtuple
import glob
import numpy as np


class IDDSynth(torch.utils.data.Dataset):
    """
    Images of IDD val set augmented with stable diffusion to include both OOD and ID synth objects. 
    """
    
    train_id_in = 0
    train_id_out = 1
    
    def __init__(self, root="", transform=None):
        """Load all filenames."""
        
        self.class_conversion_dict = {0:0, 2:1, 22:2, 14:3, 15:4, 20:5, 19:6, 18:7, 24:8, None:9, 25:10, 4:11,
                                     5:12, 9:13, 10:14, 11:15, None:16, 6:17, 7:18}
        
        self.transform = transform
        self.root = root
        
        self.images =  glob.glob(os.path.join(self.root, 'leftImg8bit/*/*/*.png'))
        self.targets = [x.replace('leftImg8bit/', 'gtFine/') for x in self.images]
        
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
        fmt_str = 'IDD synth: \n'
        fmt_str += '----Number of images: %d\n' % len(self.images)
        return fmt_str.strip()

