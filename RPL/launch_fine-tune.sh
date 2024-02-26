# Fine-tune on COCO data
python rpl_corocl.code/main.py --ood_source coco --use_id_coco False

# Fine-tune on POC data
## Modify `cs_synth_root_path` in rpl_corocl.code/config/config.py
## to choose the OOD dataset, POC coco or POC alt (or custom).
python rpl_corocl.code/main.py --ood_source synth
