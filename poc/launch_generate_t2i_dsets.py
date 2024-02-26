import os
import glob


force_redo = False

# Folder with reduced Cityscapes
root_path = '/gfs-ssd/project/clara/data/Cityscapes/leftImg8bit/'
partitions = ['train']
image_dir_list = []
for part in partitions:
    image_dir_list += glob.glob(f'{root_path}/{part}/*')
    
# Folder with original labels
labels_root = '/gfs-ssd/project/clara/data/Cityscapes/'
    
stitched_root = '/scratch/1/project/sd_inp/synth_dsets/cs_pascal_synth_t2i_stitched/'
t2i_baseline_root = '/scratch/1/project/sd_inp/synth_dsets/cs_pascal_synth_t2i_baseline/'

location_prompt = 'the road'

objects = [
    'bird flying',
    'cat',
    'cow',
    'dog',
    'horse',
    'sheep',
]
    
results_root = '/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/cs_synth_pascal/'
location_prompt = 'the road'

object_id = {
    'bird flying': 19,
    'cat': 20,
    'cow': 21,
    'dog': 22,
    'horse': 23,
    'sheep': 24,
}

object_configs = {
}

# Potentially we could modify the config for some objects
for obj in objects:
    object_configs[obj] = {'box_min': 200, 'box_max': 500, 'img_crop_size': -1,
                           'stitch_mode': 'object_mask',
                           'prompts': [obj]}


for image_dir in image_dir_list:
    for seed in [1]:
        for object_prompt in objects:
            results_stitched = f"{stitched_root}/{image_dir.split('/Cityscapes/')[-1]}"
            results_baseline = f"{t2i_baseline_root}/{image_dir.split('/Cityscapes/')[-1]}"
            
            labels_dir = f"{labels_root}/{image_dir.split('/Cityscapes/')[-1]}"
            labels_dir = labels_dir.replace('leftImg8bit', 'gtFine')
            
            label_id = object_id[object_prompt]
            t2i_prompts = ','.join(object_configs[object_prompt]['prompts'])
            
            done_name = f"{object_prompt.replace(',', '_').replace(' ', '-')}_{seed}_done.txt"
            done_filename = os.path.join(results_baseline, done_name)
            
            if not os.path.exists(done_filename) or force_redo:
                print(f'python -u generate_t2i_dset.py' +
                      f' --image_folder=\"{image_dir}\"' +
                      f' --labels_folder=\"{labels_dir}\"' +
                      f' --t2i_prompt=\"{t2i_prompts}\"' +
                      f' --object_prompt=\"{object_prompt}\"' +
                      f' --label_id=\"{label_id}\"' +
                      f' --results_dir_t2i_stitched=\"{results_stitched}\"' +
                      f' --results_dir_t2i_baseline=\"{results_baseline}\"' +
                      f' --seed={seed}' +
                      f' --done_filename={done_filename}'
                     )
        
        

    
        

