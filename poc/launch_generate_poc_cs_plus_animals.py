import os
import glob
import numpy as np

force_redo = False

# Folder with reduced Cityscapes
root_path = '/gfs-ssd/project/clara/data/Cityscapes/leftImg8bit/'
partitions = ['val', 'train']
image_dir_list = []
for part in partitions:
    image_dir_list += glob.glob(f'{root_path}/{part}/*')
    
# Folder with original labels
labels_root = '/gfs-ssd/project/clara/data/Cityscapes/'
    
objects = ['rider,bicycle', 'rider,motorcycle', 'train', 'bus', 'person', 'car']

non_cs_objects = [
    'bird flying',
    'cat',
    'cow',
    'dog',
    'horse',
    'sheep',
]

objects += non_cs_objects
    
results_root = '/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/cs_and_pascal_synth/'
location_prompt = 'the road'

object_id = {
    'rider,bicycle': '12,18',
    'rider,motorcycle': '12,17',
    'train': 16,
    'bus': 15,
    'person': 11,
    'car': 13,
    'truck': 14,
}

idx = 19
for obj in non_cs_objects:
    object_id[obj] = idx
    idx += 1


object_configs = {
    'train': {'box_min': 550, 'box_max': 700, 'img_crop_size': -1, 'blur_std': 1,
              'stitch_mode': 'object_mask',
              'prompts': ['train']}, # The final inpaint prompt will be 
                                     # 'A good foto of {prompt}'
                                     # WARNING: prompts can not have commas!
    
    'bus': {'box_min': 550, 'box_max': 700, 'img_crop_size': -1, 'blur_std': 1,
            'stitch_mode': 'object_mask',
            'prompts': ['bus']},
    
    'rider,motorcycle': {'box_min': 300, 'box_max': 450, 'img_crop_size': -1, 'blur_std': 1,
                         'stitch_mode': 'object_mask',
                         'prompts': ['motorbike', 'motorbike with rider',
                                     'scooter', 'scooter with rider']},
    
    'rider,bicycle': {'box_min': 300, 'box_max': 450, 'img_crop_size': -1, 'blur_std': 1,
                      'stitch_mode': 'object_mask', 
                      'prompts': ['bicycle', 'bicycle with rider', 'person riding a bicycle']},
    
    'person': {'box_min': 350, 'box_max': 450, 'img_crop_size': 512, 'blur_std': 1,
               'stitch_mode': 'object_mask',
               'prompts': ['man walking in the street', 'a woman walking in the street']},
    
    'car': {'box_min': 350, 'box_max': 650, 'img_crop_size': -1, 'blur_std': 1,
            'stitch_mode': 'object_mask',
            'prompts': ['car', 'suv', 'van']},
    
    'truck': {'box_min': 500, 'box_max': 700, 'img_crop_size': -1, 'blur_std': 1,
              'stitch_mode': 'object_mask',
              'prompts': ['european semi-truck']},
}


for obj in non_cs_objects:
    object_configs[obj] = {'box_min': 200, 'box_max': 500, 'img_crop_size': -1, 'blur_std': 1,
                           'stitch_mode': 'object_mask',
                           'prompts': [obj]}

    
for image_dir in image_dir_list:
    for seed in [1]:
        np.random.seed(seed)
        images = glob.glob(f'{image_dir}/*leftImg8bit.png')
        labels = [x.replace('/leftImg8bit/', '/gtFine/').replace('leftImg8bit.png', 'gtFine_labelTrainIds.png') for x in images]
        object_assignment = [np.random.choice(objects, size=3, replace=False) for _ in images]
        assert len(images) == len(labels)
        
        for object_prompt in objects:
            selected_imgs = [x[0].split('/')[-1] for x in zip(images, object_assignment) if object_prompt in x[1]]
            selected_imgs_str = ','.join(selected_imgs)
            
            selected_labels = [x[0].split('/')[-1] for x in zip(labels, object_assignment) if object_prompt in x[1]]
            selected_labels_str = ','.join(selected_labels)
            
            if object_prompt == 'bird flying':
                location = ''
            else:
                location = location_prompt
            results_dir = f"{results_root}/{image_dir.split('/Cityscapes/')[-1]}"
            
            labels_dir = f"{labels_root}/{image_dir.split('/Cityscapes/')[-1]}"
            labels_dir = labels_dir.replace('leftImg8bit', 'gtFine')
            
            labels_results_dir = results_dir.replace('leftImg8bit', 'gtFine')
            
            label_id = object_id[object_prompt]
            inpaint_prompts = ','.join(object_configs[object_prompt]['prompts'])
            
            done_filename = f"{object_prompt.replace(',', '_').replace(' ', '-')}_{seed}_done.txt"
            done_filename = os.path.join(results_dir, done_filename)
            if (not os.path.exists(done_filename) or force_redo) and len(selected_imgs) > 0:
                print(f'python -u generate_poc_dset.py' +
                      f' --image_folder=\"{image_dir}\"' +
                      f' --labels_folder=\"{labels_dir}\"' +
                      f' --img_name_list=\"{selected_imgs_str}\"' +
                      f' --labels_name_list=\"{selected_labels_str}\"' +
                      f' --inpaint_prompt=\"{inpaint_prompts}\"' +
                      f' --object_prompt=\"{object_prompt}\"' +
                      f' --label_id=\"{label_id}\"' +
                      f' --location_prompt=\"{location}\"' +
                      f' --results_dir=\"{results_dir}\"' +
                      f' --results_labels=\"{labels_results_dir}\"' +
                      f' --box_min={object_configs[object_prompt]["box_min"]}' +
                      f' --box_max={object_configs[object_prompt]["box_max"]}' +
                      f' --img_crop_size={object_configs[object_prompt]["img_crop_size"]}' +
                      f' --stitch_mode=\"{object_configs[object_prompt]["stitch_mode"]}\"' +
                      f' --blur_std={object_configs[object_prompt]["blur_std"]}' +
                      f' --seed={seed}' +
                      f' --done_filename={done_filename}'
                     )