### Here we define paths to datasets, model configs and checkpoints

# Train sets
CITYSCAPES_ROOT = '/gfs-ssd/project/clara/data/Cityscapes/'
CITYSCAPES_AUG_ROOT = 'PATH/TO/DSET' # Not used
ACDC_ROOT = '/gfs-ssd/project/clara/data/ACDC'
IDD_ROOT = '/gfs-ssd/project/uss/data/IDD/IDD_Segmentation/'


# Val sets
PASCAL_ROOT = '/gfs-ssd/project/clara/data-new/PASCAL_2012/VOC2012/'
CS_SYNTH_PASCAL_ROOT = '/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/cs_synth_pascal_val/'


mmseg_models_configs = {
    # Baselines trained on Cityscapes
    'DLV3+ResNet101': 'mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r101-d8_769x769_80k_cityscapes.py',
    'Segmenter': 'mmsegmentation/configs/segmenter/segmenter_vit-l_mask_4x1_769x769_160k_lr0.005_cityscapes.py',
    'ConvNext': 'mmsegmentation/configs/convnext/upernet_convnext_large_fp16_769x769_80k_cityscapes.py',
    # CNXT trained with POC Animals
    'CNXT_SYNTH_PASCAL': 'mmsegmentation/configs/convnext/upernet_convnext_large_fp16_769x769_80k_cityscapes_synth_pascal.py',
    # CNXT trained with T2I crop
    'CNXT_SYNTH_PASCAL_STITCHED': 'mmsegmentation/configs/convnext/upernet_convnext_large_fp16_769x769_80k_cityscapes_synth_pascal_t2i_stitched.py',
    # CNXT trained with T2I full
    'CNXT_SYNTH_PASCAL_BASELINE': 'mmsegmentation/configs/convnext/upernet_convnext_large_fp16_769x769_80k_cityscapes_synth_pascal_t2i_baseline.py',
    'DL_SYNTH_PASCAL': 'mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r101-d8_769x769_80k_pascal_synth.py',
    'DL_SYNTH_PASCAL_STITCHED': 'mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r101-d8_769x769_80k_pascal_synth_t2i_stitched.py',
    'DL_SYNTH_PASCAL_BASELINE': 'mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r101-d8_769x769_80k_pascal_synth_t2i_baseline.py',
    'SEGMENTER_SYNTH_PASCAL': 'mmsegmentation/configs/segmenter/segmenter_vit-l_mask_4x1_769x769_160k_lr0.005_pascal_synth.py',
    'SEGMENTER_SYNTH_PASCAL_STITCHED': 'mmsegmentation/configs/segmenter/segmenter_vit-l_mask_4x1_769x769_160k_lr0.005_pascal_synth_t2i_stitched.py',
    'SEGMENTER_SYNTH_PASCAL_BASELINE': 'mmsegmentation/configs/segmenter/segmenter_vit-l_mask_4x1_769x769_160k_lr0.005_pascal_synth_t2i_baseline.py',
    # Networks trained with full pascal voc dset (as baselines)
    'DLV3_VOC': 'mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r101-d8_512x512_40k_voc12aug.py',
    'SEGMENTER_VOC': 'mmsegmentation/configs/segmenter/segmenter_vit-l_mask_4x1_512x512_160k_lr0.005_voc.py',
    'CNXT_VOC': 'mmsegmentation/configs/convnext/upernet_convnext_large_fp16_512x512_80k_voc12.py',
    # Networks trained with POC Animals + Cityscapes classes
    'CNXT_CS_PASCAL': 'mmsegmentation/configs/convnext/upernet_convnext_large_fp16_769x769_80k_cs_and_pascal_synth.py',
    'DLV3_CS_PASCAL': 'mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r101-d8_769x769_80k_cs_and_pascal_synth.py',
    'SEGMENTER_CS_PASCAL': 'mmsegmentation/configs/segmenter/segmenter_vit-l_mask_4x1_769x769_160k_lr0.005_cs_and_pascal_synth.py',
    # Open vocabulary baseline
    'GSAM': '',
    }


# Paths to model checkpoints: Models should be downloaded or locally trained
mmseg_models_checkpoints = {
    'DLV3+ResNet101': '/gfs-ssd/project/uss/pre_trained_models/DeepLabV3+/R-101-D8_769x769_80K/deeplabv3plus_r101-d8_769x769_80k_cityscapes_20200607_000405-a7573d20.pth',
    'Segmenter': '/gfs-ssd/project/uss/pre_trained_models/Segmenter/segmenter_vit-l_mask_4x1_769x769_160k_lr0.005_cityscapes.pth',
    'ConvNext': '/gfs-ssd/project/uss/pre_trained_models/ConvNext/upernet_convnext_large_4x1_fp16_769x769_80k_cityscapes.pth',
    'CNXT_SYNTH_PASCAL': '/gfs-ssd/project/uss/results/train_convnext_cs_synth_pascal/iter_80000.pth',
    'CNXT_SYNTH_PASCAL_STITCHED': '/gfs-ssd/project/uss/results/train_convnext_cs_synth_pascal_t2i_stitched/iter_80000.pth',
    'CNXT_SYNTH_PASCAL_BASELINE': '/gfs-ssd/project/uss/results/train_convnext_cs_synth_pascal_t2i_baseline/iter_80000.pth',
    'DL_SYNTH_PASCAL': '/gfs-ssd/project/uss/results/train_dlv3p_cs_synth_pascal/iter_80000.pth',
    'DL_SYNTH_PASCAL_STITCHED': '/gfs-ssd/project/uss/results/train_dlv3p_cs_synth_pascal_t2i_stitched/iter_80000.pth',
    'DL_SYNTH_PASCAL_BASELINE': '/gfs-ssd/project/uss/results/train_dlv3p_cs_synth_pascal_t2i_baseline/iter_80000.pth',
    'SEGMENTER_SYNTH_PASCAL': '/gfs-ssd/project/uss/results/train_segmenter_cs_synth_pascal_v2/iter_160000.pth',
    'SEGMENTER_SYNTH_PASCAL_STITCHED': '/gfs-ssd/project/uss/results/train_segmenter_cs_synth_pascal_t2i_stitched/iter_160000.pth',
    'SEGMENTER_SYNTH_PASCAL_BASELINE': '/gfs-ssd/project/uss/results/train_segmenter_cs_synth_pascal_t2i_baseline/iter_160000.pth',
    'DLV3_VOC': '/gfs-ssd/project/uss/results/train_dlv3p_voc/iter_40000.pth',
    'SEGMENTER_VOC': '/gfs-ssd/project/uss/results/train_segmenter_voc_v2/iter_160000.pth',
    'CNXT_VOC': '/gfs-ssd/project/uss/results/train_convnext_voc/iter_80000.pth',
    'CNXT_CS_PASCAL':'/gfs-ssd/project/uss/results/train_convnext_cs_and_pascal_synth/iter_80000.pth',
    'DLV3_CS_PASCAL': '/gfs-ssd/project/uss/results/train_dlv3p_cs_and_pascal_synth_v1/iter_80000.pth',
    'SEGMENTER_CS_PASCAL': '/gfs-ssd/project/uss/results/train_segmenter_cs_and_pascal_synth_v1/iter_160000.pth',
    'GSAM': '',
    }
