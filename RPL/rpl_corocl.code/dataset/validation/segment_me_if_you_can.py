import os
import torch
from PIL import Image, UnidentifiedImageError
import glob

class SegmentMeIfYouCan(torch.utils.data.Dataset):
    train_id_in = 0
    train_id_out = 1

    def __init__(self, root, split, transform=None):
        """Load all filenames."""
        self.transform = transform
        if split == "road_anomaly":
            self.root = os.path.join(root, "RoadAnomaly21")
        elif split == "road_obstacle":
            self.root = os.path.join(root, "RoadObstacle21")
        else:
            raise FileNotFoundError("there is no subset with name {} in target dataset".format(split))

        self.transform = transform
        if split == 'road_anomaly':
            self.images = glob.glob(f'{self.root}/images/*.png')  # list of all raw input images
        elif split == 'road_obstacle':
            self.images = glob.glob(f'{self.root}/images/*.webp')
            
        self.targets = glob.glob(f'{self.root}/labels_masks/*.png')  # list of all ground truth TrainIds images
        assert len(self.images) == len(self.targets)
        
        self.images = sorted(self.images)
        self.targets = sorted(self.targets)

    def __len__(self):
        """Return number of images in the dataset split."""
        return len(self.images)

    def __getitem__(self, i):
        """Return raw image, trainIds as torch.Tensor or PIL Image"""
        try:
            image = Image.open(self.images[i]).convert('RGB')
        except UnidentifiedImageError:
            # image = webp.load_image(self.images[i], 'RGBA').convert("RGB")
            print('please install the webp with cmd: pip install webp'
                  'and re-install pillow with cmd: pip install --upgrade --force-reinstall Pillow')
            from PIL import features
            assert features.check_module('webp'), "webp error."

        target = Image.open(self.targets[i]).convert('L')
        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target