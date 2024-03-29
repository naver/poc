import os
import torch
from PIL import Image
from collections import namedtuple
import glob
import numpy as np

class RoadAnomaly(torch.utils.data.Dataset):
    RoadAnomaly_class = namedtuple('RoadAnomalyClass', ['name', 'id', 'train_id', 'hasinstances',
                                                        'ignoreineval', 'color'])
    # --------------------------------------------------------------------------------
    # A list of all Lost & Found labels
    # --------------------------------------------------------------------------------
    labels = [
        RoadAnomaly_class('in-distribution', 0, 0, False, False, (144, 238, 144)),
        RoadAnomaly_class('out-distribution', 1, 1, False, False, (255, 102, 102)),
    ]

    train_id_in = 0
    train_id_out = 1
    num_eval_classes = 19
    label_id_to_name = {label.id: label.name for label in labels}
    train_id_to_name = {label.train_id: label.name for label in labels}
    trainid_to_color = {label.train_id: label.color for label in labels}
    label_name_to_id = {label.name: label.id for label in labels}

    def __init__(self, root="/home/yu/yu_ssd/road_anomaly", transform=None):
        """Load all filenames."""
        self.transform = transform
        self.root = root
        self.images = []  # list of all raw input images
        self.targets = []  # list of all ground truth TrainIds images
        filenames = os.listdir(os.path.join(root, 'images'))

        for filename in filenames:
            if os.path.splitext(filename)[1] == '.jpg':
                f_name = os.path.splitext(filename)[0]
                filename_base_img = os.path.join("images", f_name)
                filename_base_labels = os.path.join("labels_masks", f_name)

                self.images.append(os.path.join(self.root, filename_base_img + '.jpg'))
                self.targets.append(os.path.join(self.root, filename_base_labels + '.png'))
        self.images = sorted(self.images)
        self.targets = sorted(self.targets)

    def __len__(self):
        """Return number of images in the dataset split."""
        return len(self.images)

    def __getitem__(self, i):
        """Return raw image, trainIds as torch.Tensor or PIL Image"""
        image = Image.open(self.images[i]).convert('RGB')
        target = Image.open(self.targets[i]).convert('L')
        # Modify ood label to be 1
        target = np.array(target)
        target[target == 2] = 1
        target = Image.fromarray(target).convert('L')
            
        if self.transform is not None:
            image, target = self.transform(image, target)
            

        return image, target

    def __repr__(self):
        """Print some information about dataset."""
        fmt_str = 'Road anomaly Dataset: \n'
        fmt_str += '----Number of images: %d\n' % len(self.images)
        return fmt_str.strip()
