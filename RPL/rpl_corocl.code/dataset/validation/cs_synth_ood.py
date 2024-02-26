import os
import torch
from PIL import Image
from collections import namedtuple
import glob
import numpy as np

class CityscapesSynth(torch.utils.data.Dataset):
    """
    Images of Cityscapes val set augmented with stable diffusion to include both OOD and ID synth objects. 
    """
    
    train_id_in = 0
    train_id_out = 1
    
    def __init__(self, root="", transform=None):
        """Load all filenames."""
        self.transform = transform
        self.root = root
        
        self.images = glob.glob(os.path.join(self.root, 'leftImg8bit/val/*/*.png'))
        self.targets = glob.glob(os.path.join(self.root, 'gtFine/val/*/*.png'))
        
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
        target[(target <= 18)] = 0
        target[(target > 18) & (target != 255)] = 1
        target = Image.fromarray(target).convert('L')

        if self.transform is not None:
            image, target = self.transform(image, target)
            
        return image, target

    def __repr__(self):
        """Print some information about dataset."""
        fmt_str = 'Cityscapes synth: \n'
        fmt_str += '----Number of images: %d\n' % len(self.images)
        return fmt_str.strip()

