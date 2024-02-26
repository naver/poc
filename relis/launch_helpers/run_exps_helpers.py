"""
Adapted from https://github.com/naver/oasis/blob/master/run_exps_helpers.py
"""

import glob
import numpy as np
import numpy.random as npr
import os
import sys

sys.path.append("..")
sys.path.append(".")
from path_dicts import *

def update_cityscapes_lists(mode, trg_dataset_list, scene_list, cond_list):

    train_cities = ['aachen', 'dusseldorf', 'krefeld', 'ulm',
                'bochum', 'erfurt', 'monchengladbach', 'weimar',
                'bremen', 'hamburg', 'strasbourg', 'zurich',
                'cologne', 'hanover', 'stuttgart',
                'darmstadt', 'jena', 'tubingen']

    # Changing val and test splits for CS since test labels are all non-valid...
    val_cities = ['munster', 'lindau']
    test_cities = ['frankfurt']
    full_val_cities = ['munster', 'lindau', 'frankfurt']

    if mode == 'train-1-clean':
        scene_list_ = train_cities
        cond_list_ = ['clean'] * len(scene_list_)

    elif mode == 'val-1-clean':
        scene_list_ = val_cities
        cond_list_ = ['clean'] * len(scene_list_)
        
    elif mode == 'test-1-clean':
        scene_list_ = test_cities
        cond_list_ = ['clean'] * len(scene_list_)
        
    elif mode == 'val-1-augmented':
        scene_list_ = val_cities
        cond_list_ = ['augmented'] * len(scene_list_)
        
    elif mode == 'full_val':
        scene_list_ = full_val_cities
        cond_list_ = ['clean'] * len(scene_list_)
        
    else:
        raise ValueError('Unknown mode')

    trg_dataset_list_ = ['Cityscapes'] * len(scene_list_)

    trg_dataset_list += trg_dataset_list_
    scene_list += scene_list_
    cond_list += cond_list_

    return None


def update_pascal_animals_lists(mode, trg_dataset_list, scene_list, cond_list):

    if mode == 'val':
        scene_list_ = ['all']
    else:
        raise NotImplementedError
    
    trg_dataset_list_ = ['PascalAnimals'] * len(scene_list_)
    cond_list_ = ['clean'] * len(scene_list_)
    
    trg_dataset_list += trg_dataset_list_
    cond_list += cond_list_
    scene_list += scene_list_

    return None

def update_pascal_cs_cl_lists(mode, trg_dataset_list, scene_list, cond_list):

    if mode == 'val':
        scene_list_ = ['all']
    else:
        raise NotImplementedError
    
    trg_dataset_list_ = ['PascalCS'] * len(scene_list_)
    cond_list_ = ['clean'] * len(scene_list_)
    
    trg_dataset_list += trg_dataset_list_
    cond_list += cond_list_
    scene_list += scene_list_

    return None

def update_cs_synth_pascal_lists(mode, trg_dataset_list, scene_list, cond_list):

    if mode == 'val':
        scene_list_ = ['all']
    else:
        raise NotImplementedError
    
    trg_dataset_list_ = ['CS_SYNTH_PASCAL'] * len(scene_list_)
    cond_list_ = ['clean'] * len(scene_list_)
    
    trg_dataset_list += trg_dataset_list_
    cond_list += cond_list_
    scene_list += scene_list_

    return None