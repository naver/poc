# For fine-tuned baselines copy the path to experiment dir folder 
# after running the fine-tuning script `main.py`
python rpl_corocl.code/test.py --model /PATH/TO/MODEL/epoch-last.pth

# For baseline prior to fine-tuning
python rpl_corocl.code/test.py --model /PATH/TO/PRE-TRAINED/MODEL/cityscapes_best.pth