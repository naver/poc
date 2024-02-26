import glob
import os
import sys

import numpy.random as npr
import numpy as np

import run_exps_helpers

force_redo_ = 0

trg_dataset_list, scene_list, cond_list = [], [], []

run_exps_helpers.update_cityscapes_lists('test-1-clean', trg_dataset_list, scene_list, cond_list)
run_exps_helpers.update_cityscapes_lists('val-1-clean', trg_dataset_list, scene_list, cond_list)
run_exps_helpers.update_pascal_animals_lists('val', trg_dataset_list, scene_list, cond_list)
run_exps_helpers.update_pascal_cs_cl_lists('val', trg_dataset_list, scene_list, cond_list)
run_exps_helpers.update_cs_synth_pascal_lists('val', trg_dataset_list, scene_list, cond_list)


# Which models we want to use. Names should correpond to the keys in path_dicts.py
model_arch_list = [
                ## Baselines only to be evaluated on cityscapes and pascal_cs
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
                ## Baselines only to be evaluated on pascal_animals and pascal_cs
                # 'DLV3_VOC',
                # 'SEGMENTER_VOC',
                # 'CNXT_VOC',
                'CNXT_CS_PASCAL',
                'DLV3_CS_PASCAL',
                'SEGMENTER_CS_PASCAL',
                'GSAM'
                ]

seed_list=[111]


unc_method_list = ['entropy']


def get_done_filename(trg_dataset_, scene_, cond_, unc_method_, model_arch_, root_exp_dir_):
    # DONE filename (check if experiment has already been done)
    trg_sub_folder = f'{trg_dataset_}_{scene_}_{cond_}'
    method_sub_folder = f'uncertainty_{unc_method_}'
    model_arch_sub_folder = model_arch_
    DONE_name = f'experiment.DONE'
    model_dir = os.path.join(
    root_exp_dir_, 'Cityscapes',
    model_arch_sub_folder, trg_sub_folder, method_sub_folder)
    return os.path.join(model_dir, DONE_name)

counter = 0
for (trg_dataset_, scene_, cond_) in zip(trg_dataset_list[:], scene_list[:], cond_list[:]):
    root_exp_dir_ = '/gfs-ssd/project/uss/results/'
    for seed_ in seed_list:
        for model_arch_ in model_arch_list:
            for unc_method_ in unc_method_list:
                # DONE filename (check if experiment has already been done)
                done_filename = get_done_filename(trg_dataset_, scene_, cond_, unc_method_, model_arch_, root_exp_dir_)
                # Check if done file is present 
                if os.path.isfile(done_filename) and not force_redo_:
                    pass
                else:
                    print(f'python -u eval.py' +
                                f' --force_redo={force_redo_}' +
                                f' --trg_dataset={trg_dataset_}' +
                                f' --scene={scene_}' +
                                f' --cond={cond_}' +
                                f' --uncertainty_method={unc_method_}'+
                                f' --model_arch={model_arch_}' +
                                f' --root_exp_dir={root_exp_dir_}')
                    


                    

