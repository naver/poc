# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import os
from PIL import Image

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on
import random
import time
import torch
import numpy as np

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine.defaults import DefaultPredictor
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from mask2former import add_maskformer2_config

from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# constants
WINDOW_NAME = "mask2former demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="/home/shyam/Mask2Former/configs/cityscapes/semantic-segmentation/test_unk_na.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        default="/scratch/1/project/uss/data/Validation_Dataset/RoadAnomaly21/",
        type=str,
        help="Path to directory with dset",
    )
    parser.add_argument(
        "--output",
        default="/home/shyam/Mask2Former/unk-eval/results/",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--dset",
        default=None,
        type=str,
        help="Dataset name",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--aug",
        default=True,
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    
    cfg = setup_cfg(args)
    model = DefaultPredictor(cfg) 

    anomaly_score_list = []
    ood_gts_list = []
    
    print(args.input)
    if "RoadObstacle21" in args.input:
        path_list = glob.glob(f'{args.input}/images/*.webp')
    elif "fs_static" in args.input:
        path_list = glob.glob(f'{args.input}/images/*.jpg')         
    elif "RoadAnomaly" in  args.input and "RoadAnomaly21" not in args.input:
        path_list = glob.glob(f'{args.input}/images/*.jpg')
    elif 'cs_synth' in args.input:
        path_list = glob.glob(f'{args.input}/leftImg8bit/val/*/*.png')
    elif 'idd_ood' in args.input:
        path_list = glob.glob(f'{args.input}/leftImg8bit/val/*/*.png')
    elif 'acdc_ood' in args.input:
        path_list = glob.glob(f'{args.input}/rgb_anon_trainvaltest/rgb_anon/*/val/*/*.png')
    else:
        path_list = glob.glob(f'{args.input}/images/*.png')
        
    path_list.sort()
        
    for idx, path in enumerate(path_list):
        print(f'Processing image {path}')
        img = read_image(path, format="BGR")
        start_time = time.time()

        img_lr = np.fliplr(img) # Aug 2

        predictions_na = model(img)
        predictions_lr = model(img_lr)
        # predictions_na, _ = demo.run_on_image(img)
        # predictions_lr, _ = demo.run_on_image(img_lr)

        
        predictions_naa =  predictions_na["sem_seg"].unsqueeze(0)
        outputs_na = 1 - torch.max(predictions_naa[0:19,:,:], axis = 1)[0]
        if predictions_na["sem_seg"][19:,:,:].shape[0] > 1:
            outputs_na_mask = torch.max(predictions_na["sem_seg"][19:,:,:].unsqueeze(0),  axis = 1)[0]
            outputs_na_mask[outputs_na_mask < 0.5] = 0
            outputs_na_mask[outputs_na_mask >= 0.5] = 1
            outputs_na_mask = 1 - outputs_na_mask
            outputs_na_save = outputs_na.clone().detach().cpu().numpy().squeeze().squeeze()
            outputs_na = outputs_na*outputs_na_mask.detach()
            outputs_na_mask = outputs_na_mask.detach().cpu().numpy().squeeze().squeeze()
        outputs_na = outputs_na.detach().cpu().numpy().squeeze().squeeze()

        #left-right
        predictions_lrr =  predictions_lr["sem_seg"].unsqueeze(0)
        outputs_lr = 1 - torch.max(predictions_lrr[0:19,:,:], axis = 1)[0]
        if predictions_lr["sem_seg"][19:,:,:].shape[0] > 1:
            outputs_lr_mask = torch.max(predictions_lr["sem_seg"][19:,:,:].unsqueeze(0),  axis = 1)[0]
            outputs_lr_mask[outputs_lr_mask < 0.5] = 0
            outputs_lr_mask[outputs_lr_mask >= 0.5] = 1
            outputs_lr_mask = 1 - outputs_lr_mask
            outputs_lr_save = outputs_lr.clone()
            outputs_lr = outputs_lr*outputs_lr_mask.detach()
        outputs_lr = outputs_lr.detach().cpu().numpy().squeeze().squeeze()
        outputs_lr = np.flip(outputs_lr.squeeze(), 1)

        outputs = np.expand_dims((outputs_lr + outputs_na )/2.0, 0).astype(np.float32)
        pathGT = path.replace("images", "labels_masks")
        
        ##------------------------------------------------------------------------------
        # Save anomaly score map
        model_name = cfg.MODEL.WEIGHTS.split('/')[-2]
        ano_score_path = f'/scratch/1/project/uss/anomaly_scores/M2A/{model_name}/{args.dset}/'
        ano_score_path += f'{idx}'.zfill(4) + '.npy'
        if not os.path.exists(f'/scratch/1/project/uss/anomaly_scores/M2A/{model_name}/{args.dset}'):
            os.makedirs(f'/scratch/1/project/uss/anomaly_scores/M2A/{model_name}/{args.dset}')
        np.save(ano_score_path, outputs)
        ##------------------------------------------------------------------------------

        if "RoadObstacle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if 'cs_synth' in args.input:
            pathGT = path.replace('/leftImg8bit/', '/gtFine/')
        if 'idd_ood' in args.input:
            pathGT = path.replace('/leftImg8bit/', '/gtFine/')
        if 'acdc_ood' in args.input:
            pathGT = path.replace('rgb_anon_trainvaltest/rgb_anon', 'gt_trainval/gt')

        mask = Image.open(pathGT)
        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)
        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)
        if 'cs_synth' in path:
            ood_gts[(ood_gts <= 18)] = 0
            ood_gts[(ood_gts > 18) & (ood_gts != 255)] = 1
        if 'acdc_ood' in path:
            class_conversion_dict = {7:0, 8:1, 11:2, 12:3, 13:4, 17:5, 19:6, 20:7, 21:8, 22:9,
                                      23: 10, 24:11, 25:12, 26:13, 27:14, 28:15, 31:16, 32:17, 33:18}
             # re-assign labels to filter out non-used ones
            label_copy = 255 * np.ones(ood_gts.shape, dtype=np.float32)
            for k, v in class_conversion_dict.items():
                label_copy[ood_gts == k] = v
            label_copy[(label_copy < 19)] = 0
            label_copy[(ood_gts >= 100) & (ood_gts != 255)] = 1 # OOD classes
            ood_gts = label_copy.copy()
        if 'idd_ood' in path:
            class_conversion_dict = {0:0, 2:1, 22:2, 14:3, 15:4, 20:5, 19:6, 18:7, 24:8, None:9, 25:10, 4:11,
                                     5:12, 9:13, 10:14, 11:15, None:16, 6:17, 7:18}
             # re-assign labels to filter out non-used ones
            label_copy = 255 * np.ones(ood_gts.shape, dtype=np.float32)
            for k, v in class_conversion_dict.items():
                label_copy[ood_gts == k] = v
            label_copy[(label_copy < 19)] = 0
            label_copy[(ood_gts >= 100) & (ood_gts != 255)] = 1 # OOD classes
            ood_gts = label_copy.copy()

        # if 1 not in np.unique(ood_gts):
        #     continue              
        # else:
        #     ood_gts_list.append(np.expand_dims(ood_gts, 0))
        #     anomaly_score_list.append(outputs)
        ood_gts_list.append(np.expand_dims(ood_gts, 0))
        anomaly_score_list.append(outputs)
        print(f'Finished processing img: {path} in {time.time() - start_time}s')
        

    print('Computing OOD metrics')
    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)
    
    
    # np.save('/gfs-ssd/project/uss/results_chaos-06.npy', anomaly_scores)
    # drop void pixels
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    fpr, tpr, _ = roc_curve(val_label, val_out)    
    roc_auc = auc(fpr, tpr)
    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'AUROC score: {roc_auc}')
    print(f'AUPRC score: {prc_auc}')
    print(f'FPR@TPR95: {fpr}')