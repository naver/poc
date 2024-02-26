'''
Helper script to create several launch commands that evaluate the accuracy and miou
on different datasets. Require first to launch eval.py 
'''

import glob
import os
import sys

import numpy.random as npr
import numpy as np

import run_exps_helpers

force_redo_ = 0 # Set to 1 to overwrite results. Otherwise it skips done experiments.

for dset in ['PascalCS', 'PascalAnimals', 'CS', 'CS_SYNTH_PASCAL']:
    
    trg_dataset_list, scene_list, cond_list = [], [], []

    if dset == 'CS':
        run_exps_helpers.update_cityscapes_lists('full_val', trg_dataset_list, scene_list, cond_list)
    elif dset == 'PascalAnimals':
        run_exps_helpers.update_pascal_animals_lists('val', trg_dataset_list, scene_list, cond_list)
    elif dset == 'PascalCS':
        run_exps_helpers.update_pascal_cs_cl_lists('val', trg_dataset_list, scene_list, cond_list)
    elif dset == 'CS_SYNTH_PASCAL':
        run_exps_helpers.update_cs_synth_pascal_lists('val', trg_dataset_list, scene_list, cond_list)

    # Which models we want to use. Names should correpond to the keys in path_dicts.py
    model_arch_list = [
                ## Baselines only to be evaluated on CS and PascalCS
                # 'DLV3+ResNet101',
                # 'Segmenter',
                # 'ConvNext',
                'CNXT_SYNTH_PASCAL',
                'CNXT_SYNTH_PASCAL_STITCHED',
                'CNXT_SYNTH_PASCAL_BASELINE',
                'DL_SYNTH_PASCAL',
                'DL_SYNTH_PASCAL_STITCHED',
                'DL_SYNTH_PASCAL_BASELINE',
                'SEGMENTER_SYNTH_PASCAL',
                'SEGMENTER_SYNTH_PASCAL_STITCHED',
                'SEGMENTER_SYNTH_PASCAL_BASELINE',
                ## Baselines only to be evaluated on PascalAnimals and PascalCS
                # 'DLV3_VOC',
                # 'SEGMENTER_VOC',
                # 'CNXT_VOC',
                'CNXT_CS_PASCAL',
                'DLV3_CS_PASCAL',
                'SEGMENTER_CS_PASCAL',
                'GSAM'
                ]

    assert len(set(trg_dataset_list)) == 1 # We compute results for one dset at a time
    trg_dataset_ = trg_dataset_list[0]

    root_exp_dir_ = '/gfs-ssd/project/uss/results/'

    # Format scenes and conditions
    scenes = ''
    conditions = ''
    for scene, cond in zip(scene_list, cond_list):
        scenes = scenes + scene + ','
        conditions = conditions + cond + ','
    scenes = scenes[:-1]
    conditions = conditions[:-1]


    def get_done_filename(trg_dataset_, model_arch_, results_dir):
        # DONE filename (check if experiment has already been done)
        DONE_name = f'results.pkl'
        model_dir = os.path.join(
            results_dir, 'Cityscapes',
            model_arch_, 'miou_metrics_test_set', trg_dataset_)
        return model_dir, os.path.join(model_dir, DONE_name)

    for model_arch_ in model_arch_list:
        # DONE filename (check if experiment has already been done)
        results_dir, done_filename = get_done_filename(trg_dataset_, model_arch_, root_exp_dir_)
        # Check if done file is present 
        if os.path.isfile(done_filename) and not force_redo_:
            pass
        else:
            print(f'python -u compute_final_metrics.py'+
                      f' --force_redo={force_redo_}' +
                      f' --trg_dataset={trg_dataset_}' +
                      f' --scene={scenes}' +
                      f' --cond={conditions}' +
                      f' --model_arch={model_arch_}' +
                      f' --root_exp_dir={root_exp_dir_}'+
                      f' --results_dir={results_dir}')
