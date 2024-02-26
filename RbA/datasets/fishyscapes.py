import torch
import os
import cv2
import glob
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize


def round_to_nearest_multiple(x, p):
    return int(((x - 1) // p + 1) * p)


def read_image(path):

    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    return img


class FishyscapesLAF(Dataset):
    """
    The Dataset folder is assumed to follow the following structure. In the given root folder, there must be two
    sub-folders:
    - fishyscapes_lostandfound: contains the mask labels.
    - laf_images: contains the images taken from the Lost & Found Dataset
    """

    def __init__(self, hparams, transforms):
        super().__init__()

        self.hparams = hparams
        self.transforms = transforms

        self.images = []
        self.labels = []

        self.labels = glob.glob(os.path.join(hparams.dataset_root, 'labels_masks', '*.png'))
        self.labels.sort()
        self.images = [x.replace('/labels_masks/', '/images/') for x in self.labels]

        self.num_samples = len(self.images)

    def __getitem__(self, index):

        image = read_image(self.images[index])
        label = read_image(self.labels[index])

        label = label[:, :, 0]

        aug = self.transforms(image=image, mask=label)
        image = aug['image']
        label = aug['mask']

        return image, label.type(torch.LongTensor)

    def __len__(self):
        return self.num_samples


class FishyscapesStatic(Dataset):
    """
    The dataset folder is assumed to follow the following structure. In the given root folder there must be two
    sub-folders:
    - fs_val_v1 (or fs_val_v2): contains the mask labels in .png format
    - fs_static_images_v1 (or fs_static_images_v2): contains the images also in .png format. These images need a processing step to be created from
    cityscapes. the fs_val_v3 file contains .npz files that contain numpy arrays. According to ID of each file, the
    corresponding image from cityscapes should be loaded and then the cityscape image and the image from the .npz file
    should be summed to form the modified image, which should be stored in fs_static_images folder. The images files are
    named using the label file name as follows: img_name = label_name[:-10] + 'rgb.png'
    """

    def __init__(self, hparams, transforms):
        super().__init__()

        self.hparams = hparams
        self.transforms = transforms

        self.labels = glob.glob(os.path.join(hparams.dataset_root, 'labels_masks', '*.png'))
        self.labels.sort()
        self.images = [x.replace('/labels_masks/', '/images/').replace('.png', '.jpg') for x in self.labels]

        self.num_samples = len(self.images)

    def __getitem__(self, index):

        image = read_image(self.images[index])
        label = read_image(self.labels[index])

        label = label[:, :, 0]

        aug = self.transforms(image=image, mask=label)
        image = aug['image']
        label = aug['mask']

        return image, label.type(torch.LongTensor)

    def __len__(self):
        return self.num_samples
