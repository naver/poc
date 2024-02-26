CUDA_HOME=/nfs/core/cuda/11.3/

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mask2former

# To Fine-tune on COCO objects
python train_net.py --num-gpus $num_gpus --config-file ckpts/swin_b_1dl_rba_ood_coco/config.yaml  OUTPUT_DIR /gfs-ssd/project/uss/results/ft_rba_coco/
 
# To Fine-tune on POC data
# Synth data path hardcoded 
python train_net.py --num-gpus $num_gpus --config-file ckpts/swin_b_1dl_rba_ood_synth/config.yaml  OUTPUT_DIR /gfs-ssd/project/uss/results/ft_rba_synth/ INPUT.OOD_PROB 0.2