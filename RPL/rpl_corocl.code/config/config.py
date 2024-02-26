import os
import numpy
from easydict import EasyDict

C = EasyDict()
config = C
cfg = C

C.seed = 666

"""Root Directory Config"""
C.repo_name = 'ood_seg'
C.root_dir = os.path.realpath("")

"""Data Dir and Weight Dir"""
C.city_root_path = '/gfs-ssd/project/clara/data/Cityscapes/'
C.coco_root_path = '/scratch/2/project/poc_sd/data/coco/'
C.fishy_root_path = '/gfs-ssd/project/clara/data-new/Validation_Dataset/'
C.segment_me_root_path = '/gfs-ssd/project/clara/data-new/Validation_Dataset/'
C.road_anomaly_root_path = '/gfs-ssd/project/clara/data-new/Validation_Dataset/RoadAnomaly/'
# C.cs_synth_root_path = '/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/cs_synth_classes_ood/'
C.cs_synth_root_path = '/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/cs_synth_coco_classes/'
C.cs_synth_ood_root_path = '/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/cs_synth_ood_val/'
C.acdc_synth_ood_root_path = '/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/acdc_ood/'
C.idd_synth_ood_root_path = '/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/idd_ood/'


# C.rpl_corocl_weight_path = os.path.join('/gfs-ssd/project/uss/pre_trained_models/RPL/rev3.pth')
C.rpl_corocl_weight_path = os.path.join('/gfs-ssd/project/uss/results_rpl/rpl.code+corocl/epoch-last.pth')
C.pretrained_weight_path = os.path.join('/gfs-ssd/project/uss/pre_trained_models/RPL/cityscapes_best.pth')

"""Network Config"""
C.fix_bias = True
C.bn_eps = 1e-5
C.bn_momentum = 0.1

"""Image Config"""
C.num_classes = 19
C.outlier_exposure_idx = 254  # NOTE: it starts from 0

C.image_mean = numpy.array([0.485, 0.456, 0.406])  # 0.485, 0.456, 0.406
C.image_std = numpy.array([0.229, 0.224, 0.225])

C.city_image_height = 700
C.city_image_width = 700

C.ood_image_height = C.city_image_height
C.ood_image_width = C.city_image_width

# C.city_train_scale_array = [0.5, 0.75, 1, 1.5, 1.75, 2.0]
C.ood_train_scale_array = [.25, .5, .5, .75, .1, .125]

C.num_train_imgs = 2975
C.num_eval_imgs = 500

"""Train Config"""
C.lr = 7.5e-5
C.batch_size = 8
C.energy_weight = .05

C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 1e-4

C.nepochs = 40 
C.niters_per_epoch = C.num_train_imgs // C.batch_size

C.num_workers = 8
C.void_number = 5
C.warm_up_epoch = 0

# 'coco' for original OOD finetuning with COCO images, 'synth' to use the cs synth dataset
C.ood_source = 'coco'
# Whether to use the ID classes of COCO during training as well
C.use_id_coco = True

"""Eval Config"""
C.eval_iter = int(C.niters_per_epoch)
C.measure_way = "energy"
C.eval_stride_rate = 1 / 3
C.eval_scale_array = [1., ]
C.eval_flip = True
C.eval_crop_size = 1024

"""Display Config"""
C.record_info_iter = C.niters_per_epoch // 2
C.display_iter = C.niters_per_epoch

"""Wandb Config"""
# Specify you wandb environment KEY; and paste here
C.wandb_key = ""

# Your project [work_space] name
C.proj_name = "OoD_Segmentation"

C.experiment_name = "rpl_full_hist"

# half pretrained_ckpts-loader upload images; loss upload every iteration
C.upload_image_step = [0, C.eval_iter]

# False for debug; True for visualize
C.wandb_online = False
# (Pau) Setting also this variable as previous setting is not working
C.WANDB_MODE='offline'

# Wandb local files dirs
C.WANDB_DIR = '/gfs-ssd/project/uss/results_rpl/logs/'


"""Save Config"""
# C.saved_dir = os.path.join("/gfs-ssd/project/uss/results_rpl/", C.experiment_name)
C.saved_dir = os.path.join("/scratch/1/project/uss/results_rpl/", C.experiment_name)

# if not os.path.exists(C.saved_dir):
#     os.mkdir(C.saved_dir)

