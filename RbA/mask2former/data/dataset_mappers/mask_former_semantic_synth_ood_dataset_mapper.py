# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import os
import glob

import numpy as np
import torch
import random
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances
from zmq import proxy
from .coco import COCO

from collections import namedtuple

__all__ = ["MaskFormerSemanticSynthOodDatasetMapper"]




class MaskFormerSemanticSynthOodDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        ood_label,
        ood_prob,
        size_divisibility,
        repeat_instance_masks,
        labels_mapping,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.ood_label = ood_label
        self.ood_prob = ood_prob
        self.size_divisibility = size_divisibility
        self.repeat_instance_masks = repeat_instance_masks
        self.labels_mapping = labels_mapping
        
        # cs_synth_root = '/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/cs_synth_classes_ood/'
        cs_synth_root = '/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/cs_synth_coco_classes/'
        self.cs_synth_images = glob.glob(os.path.join(cs_synth_root, 'leftImg8bit/train/*/*.png'))
        # self.cs_synth_labels = [x.replace('/leftImg8bit/', '/gtFine/') for x in self.cs_synth_images]
        

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(
            f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")
        
        if ood_prob < 0.0 or ood_prob > 1.0:
            raise ValueError(f"OOD Probability should be between [0,1], given was {ood_prob}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        repeat_instance_masks = cfg.INPUT.REPEAT_INSTANCE_MASKS

        assert repeat_instance_masks >= 1, f"Number of times to repeat a mask cannot be less than one, given was {repeat_instance_masks}"

        if "labels_mapping" in meta.as_dict():
            labels_mapping = np.array(meta.labels_mapping)
        else:
            labels_mapping = None

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "ood_label": cfg.INPUT.OOD_LABEL,
            "ood_prob": cfg.INPUT.OOD_PROB,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "repeat_instance_masks": repeat_instance_masks,
            "labels_mapping": labels_mapping,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerSemanticDatasetMapper should only be used for training!"

        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(
            dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_path = dataset_dict.pop(
                "sem_seg_file_name")
            sem_seg_gt = utils.read_image(sem_seg_path).astype("double")
        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        # Paste OOD object with probability equal to self.ood_prob
        ood_p = np.random.rand()
        if self.labels_mapping is not None:
            # just check that the image is not from cityscapes
            if not (sem_seg_gt.shape[0] == 1024 and sem_seg_gt.shape[1] == 2048):
                sem_seg_gt = self.labels_mapping[(sem_seg_gt).astype("long")].astype("double")
        if ood_p < self.ood_prob:
            img_path = np.random.choice(self.cs_synth_images)
            image = utils.read_image(img_path, format=self.img_format)
            sem_seg_gt = utils.read_image(img_path.replace('/leftImg8bit/', '/gtFine/')).copy().astype("double")
            sem_seg_gt[(sem_seg_gt > 18) & (sem_seg_gt != 255)] = 254 # Convert all OOD classes to 254 label

            
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(
            self.tfm_gens, aug_input)
        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            outlier_mask = torch.zeros_like(sem_seg_gt).contiguous()
            outlier_mask[(sem_seg_gt == self.ood_label) & (sem_seg_gt != self.ignore_label)] = 1
            outlier_mask[sem_seg_gt == self.ignore_label] = self.ignore_label
            dataset_dict["outlier_mask"] = outlier_mask.long()

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size,
                                   value=self.ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError(
                "Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[(classes != self.ignore_label)
                              & (classes != self.ood_label)]
            instances.gt_classes = torch.tensor(classes.repeat(
                self.repeat_instance_masks), dtype=torch.int64)

            masks = []
            for class_id in classes:
                m = sem_seg_gt == class_id
                for _ in range(self.repeat_instance_masks):
                    masks.append(m)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros(
                    (0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack(
                        [torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

        return dataset_dict
