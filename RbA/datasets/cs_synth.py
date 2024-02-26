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


class CityscapesSynth(Dataset):
    """
    Images of Cityscapes val set augmented with stable diffusion to include both OOD and ID synth objects. 
    """

    def __init__(self, hparams, transforms):
        super().__init__()

        self.hparams = hparams
        self.transforms = transforms

        self.images = []
        self.labels = []

        self.images = glob.glob(os.path.join(hparams.dataset_root, 'leftImg8bit/val/*/*.png')) 
        self.labels = glob.glob(os.path.join(hparams.dataset_root, 'gtFine/val/*/*.png')) 
        self.labels.sort()
        self.images.sort()

        self.num_samples = len(self.images)

    def __getitem__(self, index):

        image = read_image(self.images[index])
        label = read_image(self.labels[index])

        label = label[:, :, 0]
        label[(label <= 18)] = 0
        label[(label > 18) & (label != 255)] = 1

        aug = self.transforms(image=image, mask=label)
        image = aug['image']
        label = aug['mask']

        return image, label.type(torch.LongTensor)

    def __len__(self):
        return self.num_samples

