python evaluate_ood.py \
    --out_path results/ \
    --models_folder /gfs-ssd/project/uss/results/ \
    --model_mode selective \
    --dataset_mode all \
    --store_anomaly_scores \
    --selected_models ft_rba_coco ft_rba_synth_coco ft_rba_synth_alt