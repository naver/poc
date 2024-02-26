## Anomaly Fine-tuning
# Fine-tune with COCO objects
CUDA_VISIBLE_DEVICES=1 python train_net.py --num-gpus 1 --config-file configs/cityscapes/semantic-segmentation/anomaly_ft.yaml OUTPUT_DIR /gfs-ssd/project/uss/results/ft_m2a_coco_ckpt/ MODEL.WEIGHTS /gfs-ssd/project/uss/pre_trained_models/Mask2Anomaly/bt-f-xl.pth

# Fine-tune with POC coco classes
CUDA_VISIBLE_DEVICES=1 python train_net.py --num-gpus 1 --config-file configs/cityscapes/semantic-segmentation/anomaly_ft_poc_coco.yaml OUTPUT_DIR /gfs-ssd/project/uss/results/ft_m2a_synth_coco_classes_thr_0.2/ MODEL.WEIGHTS /gfs-ssd/project/uss/pre_trained_models/Mask2Anomaly/bt-f-xl.pth MODEL.MASK_FORMER.ANOMALY_MIX_RATIO 0.2

# Fine-tune with POC alternative classes
CUDA_VISIBLE_DEVICES=1 python train_net.py --num-gpus 1 --config-file configs/cityscapes/semantic-segmentation/anomaly_ft_poc_alt.yaml OUTPUT_DIR /gfs-ssd/project/uss/results/ft_m2a_synth_ood_classes_thr_0.2/ MODEL.WEIGHTS /gfs-ssd/project/uss/pre_trained_models/Mask2Anomaly/bt-f-xl.pth MODEL.MASK_FORMER.ANOMALY_MIX_RATIO 0.2



## Anomaly Inference
# Note: /PATH/TO/MODEL/ should point to the final model saved in the OUTPUT_DIR of fine-tuning script.
#RoadAnomaly21
path_imgs=/gfs-ssd/project/clara/data-new/Validation_Dataset/RoadAnomaly21/
CUDA_VISIBLE_DEVICES=1 python anomaly_utils/anomaly_inference.py --input $path_imgs --config-file configs/cityscapes/semantic-segmentation/anomaly_inference.yaml --dset $dset_name --opts MODEL.WEIGHTS /PATH/TO/MODEL/

#RoadObsticle21
path_imgs=/gfs-ssd/project/clara/data-new/Validation_Dataset/RoadObstacle21/
CUDA_VISIBLE_DEVICES=1 python anomaly_utils/anomaly_inference.py --input $path_imgs --config-file configs/cityscapes/semantic-segmentation/anomaly_inference.yaml --dset $dset_name --opts MODEL.WEIGHTS /PATH/TO/MODEL/

#FS L&F val
path_imgs=/gfs-ssd/project/clara/data-new/Validation_Dataset/FS_LostFound_full/
CUDA_VISIBLE_DEVICES=1 python anomaly_utils/anomaly_inference.py --input $path_imgs --config-file configs/cityscapes/semantic-segmentation/anomaly_inference.yaml --dset $dset_name --opts MODEL.WEIGHTS /PATH/TO/MODEL/

# FS-Static
path_imgs=/gfs-ssd/project/clara/data-new/Validation_Dataset/fs_static/
CUDA_VISIBLE_DEVICES=1 python anomaly_utils/anomaly_inference.py --input $path_imgs --config-file configs/cityscapes/semantic-segmentation/anomaly_inference.yaml --dset $dset_name --opts MODEL.WEIGHTS /PATH/TO/MODEL/

#RoadAnomaly
path_imgs=/gfs-ssd/project/clara/data-new/Validation_Dataset/RoadAnomaly/
CUDA_VISIBLE_DEVICES=1 python anomaly_utils/anomaly_inference.py --input $path_imgs --config-file configs/cityscapes/semantic-segmentation/anomaly_inference.yaml --dset $dset_name --opts MODEL.WEIGHTS /PATH/TO/MODEL/

