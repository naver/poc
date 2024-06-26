'''
Adapted from https://github.com/naver/oasis/blob/master/main_adapt.py
'''

import sys
import os
import glob
import matplotlib.pyplot as plt
import random
import json
import copy
import argparse
import copy
import pickle
from scipy.io import loadmat

import torch
import torch.nn as nn
from torch.utils import data, model_zoo
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import numpy as np
import numpy.random as npr

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# ours
from dataset.cityscapes_dataset import Cityscapes
from dataset.acdc_dataset import ACDC
from dataset.idd_dataset import IDD
from dataset.pascal_voc_dataset import PASCAL
from dataset.pascal_animals import PascalAnimals
from dataset.pascal_cs_cl import PascalCS
from dataset.cityscapes_synth_pascal_dataset import CS_SYNTH_PASCAL

from metrics_helpers import *
from image_helpers import ImageOps
from uncertainty_helpers import UncertaintyOps

from path_dicts import *

class SolverOps:

    def __init__(self, args):

        self.args = args

        # this is taken from the AdaptSegnet repo
        with open('./dataset/cityscapes_list/info.json', 'r') as f:
            cityscapes_info = json.load(f)

        self.args.num_classes = 19
        self.args.name_classes = cityscapes_info['label']
        if 'PASCAL' in self.args.model_arch or 'VOC' in self.args.model_arch or 'GSAM' in self.args.model_arch:
            self.args.num_classes = 25
            self.args.name_classes = cityscapes_info['label'] + ['bird', 'cat', 'cow', 'dog', 'horse', 'sheep']
        

        print(f'Number of classes: {self.args.num_classes}')

        self.image_ops = ImageOps()
        self.uncertainty_ops = UncertaintyOps()
  
    def compute_final_metrics(self):

        """
        Method to compute final metrics on all predictions/gts
        Note: different from the ones computed at eval, since they
        are sample by sample -- here we use all dataset at once.
        This is the standard way of proceeding in sem.segm. research.
        """

        # label and pred paths
        gt_imgs, pred_imgs = self.retrieve_paired_preds_and_labels_paths()
        assert (len(gt_imgs) == len(pred_imgs))
        
        print('Computing datasets\'s mIoU')
        miou_all = compute_mIoU_fromlist(
                    gt_imgs=gt_imgs, pred_imgs=pred_imgs, args=self.args)

        print('Computing dataset\'s accuracy')
        pixel_acc_all, mean_acc_all = compute_acc_fromlist(
                gt_imgs=gt_imgs, pred_imgs=pred_imgs, args=self.args)

        summary_dict = {
                       'pixel_acc_final': pixel_acc_all,
                       'mean_acc_final': mean_acc_all,
                       'miou_final': miou_all
                       }

        print('Dumping final summary_dict')
        with open(self.args.results_file, 'wb') as f:
            pickle.dump(summary_dict, f, pickle.HIGHEST_PROTOCOL)
            

    def retrieve_paired_preds_and_labels_paths(self):

        """
        Method to retrieve the paths to predicted images based on the condition and scene list.
        """

        scene_list = self.args.scene.split(',')
        cond_list = self.args.cond.split(',')
        
        method_sub_folder = f'uncertainty_{self.args.uncertainty_method}'
        model_arch_sub_folder = self.args.model_arch
        
        # Unsorted all labels list
        all_labels = self.trg_parent_set.annotation_path_list
        # Sorted (paired) preds and labels lists
        pred_imgs_list = []
        label_list = []
        
        for scene, cond in zip(scene_list, cond_list):
            trg_sub_folder = f'{self.args.trg_dataset}_{scene}_{cond}'
        
            self.model_dir = os.path.join(
                    self.args.root_exp_dir, self.args.src_dataset,
                    model_arch_sub_folder, trg_sub_folder, method_sub_folder)

            self.output_images_dir = os.path.join(
                    self.model_dir, 'output_images')

            if cond == 'clean':
                cond = ''
            scene = f'/{scene}/'
            if scene == '/all/':
                scene = ''
                
            pred_imgs = sorted(glob.glob(os.path.join(self.output_images_dir, '*_label.png')))
            labels = sorted([x for x in all_labels if (scene in x and cond in x)])
            
            pred_imgs_list += pred_imgs
            label_list += labels
        
        return label_list, pred_imgs_list
        
    def setup_target_data_loader(self):

        """
        Method to create pytorch dataloaders for the
        target domain selected by the user
        """

        # (can also be a single environment)
        scene_list = self.args.scene.split(',')
        cond_list = self.args.cond.split(',')
            

        if self.args.trg_dataset=='Cityscapes':
            self.trg_parent_set = Cityscapes(
                    CITYSCAPES_ROOT,
                    scene_list, cond_list)

        elif self.args.trg_dataset=='ACDC':
            self.trg_parent_set = ACDC(
                    ACDC_ROOT, scene_list, cond_list,
                    batch_size=self.args.batch_size)
            
        elif self.args.trg_dataset=='IDD':
            self.trg_parent_set = IDD(
                    IDD_ROOT, scene_list, batch_size=self.args.batch_size)
            
        elif self.args.trg_dataset=='PASCAL':
            self.trg_parent_set = PASCAL(
                    PASCAL_ROOT, scene_list, batch_size=self.args.batch_size)
            
        elif self.args.trg_dataset=='PascalAnimals':
            self.trg_parent_set = PascalAnimals(
                    PASCAL_ROOT, scene_list, batch_size=self.args.batch_size)
            
        elif self.args.trg_dataset=='CS_SYNTH_PASCAL':
            self.trg_parent_set = CS_SYNTH_PASCAL(
                    CS_SYNTH_PASCAL_ROOT, scene_list)
            
        elif self.args.trg_dataset=='PascalCS':
            self.trg_parent_set = PascalCS(
                    PASCAL_ROOT, scene_list)
        
        else:
            raise ValueError(f'Unknown dataset {self.args.dataset}')


if __name__ == '__main__':

    # Parse all the arguments provided from the CLI.
    parser = argparse.ArgumentParser()

    # What uncertainty to compute
    parser.add_argument("--uncertainty_method", type=str, default='entropy',
                        help="available options for uncertainty : entropy")

    # Main experiment parameters
    parser.add_argument("--model_arch", type=str, default='SegFormer-B0',
                        help="""Architecture name, see path_dicts.py
                            """)
    parser.add_argument("--src_dataset", type=str, default='Cityscapes',
                        help="Which source dataset to start from {Cityscapes}")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--seed", type=int, default=111,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--root_exp_dir", type=str, default='results/',
                        help="Where to save predictions.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--force_redo", type=int, default=0,
                        help="Whether to re-run even if there is a DONE file in folder")
    parser.add_argument("--results_dir", type=str, default='results/debug/final_metrics/',
                        help="Path where to save the results file")

    # For target
    parser.add_argument("--trg_dataset", type=str, default='Cityscapes',
                        help="Which target dataset to transfer to")
    parser.add_argument("--scene", type=str, default='aachen',
                        help="List of scenes comma separated, e.g. 'aachen, frankfurt'.")
    parser.add_argument("--cond", type=str, default='clean',
                        help="List of conditions comma separated, e.g. 'clean, fog, rain'.")

    args = parser.parse_args()

    args.force_redo = bool(args.force_redo)
    
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
        
    args.results_file = os.path.join(args.results_dir, 'results.pkl')
    if os.path.exists(args.results_file):
        if args.force_redo:
            print(f'File {args.results_file} already exists! Overwriting it.')
        else:
            raise ValueError(f'File {args.results_file} already exists! Stopping here!')

    npr.seed(args.seed)

    solver_ops = SolverOps(args)

    print('Setting up data target loader')
    solver_ops.setup_target_data_loader()

    print('Computing final metrics')
    solver_ops.compute_final_metrics()

