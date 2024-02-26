import os
import glob
import numpy as np
from PIL import Image

np.random.seed(0)

force_redo = False

root_path = '/gfs-ssd/project/uss/data/IDD/IDD_Segmentation/leftImg8bit'
label_path = '/gfs-ssd/project/uss/data/IDD/IDD_Segmentation/gtFine'

image_dir_list = glob.glob(f'{root_path}/val/*')
full_image_list = glob.glob(f'{root_path}/val/*/*leftImg8bit.*')
full_label_list = glob.glob(f'{label_path}/val/*/*gtFine_labellevel3Ids.*')
assert len(full_image_list) == len(full_label_list)

large_image_list = [x for x in full_image_list if Image.open(x).size == (1920, 1080)]
large_label_list = [x for x in full_label_list if Image.open(x).size == (1920, 1080)]
large_image_list.sort()
large_label_list.sort()
assert len(large_image_list) == len(large_label_list)

valid_idx = np.random.choice(range(len(large_image_list)), 200, replace=False)
valid_image_list = np.array(large_image_list)[valid_idx]
valid_label_list = np.array(large_label_list)[valid_idx]

    
# Folder with original labels
labels_root = '/gfs-ssd/project/uss/data/IDD/IDD_Segmentation/'

results_root = '/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/idd_ood/'
    
objects = ['rider,bicycle', 'rider,motorcycle', 'bus', 'person', 'car']

non_cs_objects = [
           'stroller',
           'trolley',
           'garbage bag',
           'wheelie bin',
           'suitcase',
           'skateboard',
           'chair dumped on the street',
           'sofa dumped on the street',
           'furniture dumped on the street',
           'matress dumped on the street',
           'garbage dumped on the street',
           'clothes dumped on the street',
           'cement mixer on the street',
           'cat',
           'dog',
           'bird flying',
           'horse',
           'skunk',
           'sheep',
           'crocodile',
           'alligator',
           'bear',
           'llama',
           'tiger',
           'monkey',
]

objects += non_cs_objects
    
location_prompt = 'the road'

object_id = {
    'rider,bicycle': '5,7',
    'rider,motorcycle': '5,6',
    'bus': 11,
    'person': 4,
    'car': 9,
    'truck': 10,
    'stroller': 100,
   'trolley': 101,
   'garbage bag': 102,
   'wheelie bin': 103,
   'suitcase': 104,
   'skateboard': 105,
   'chair dumped on the street': 106,
   'sofa dumped on the street': 107,
   'furniture dumped on the street': 108,
   'matress dumped on the street': 109,
   'garbage dumped on the street': 110,
   'clothes dumped on the street': 111,
   'cement mixer on the street': 112,
   'cat': 113,
   'dog': 114,
   'bird flying': 115,
   'horse': 116,
   'skunk': 117,
   'sheep': 118,
   'crocodile': 119,
   'alligator': 120,
   'bear': 121,
   'llama': 122,
   'tiger': 123,
   'monkey': 124,
}

object_configs = {
    'train': {'box_min': 500, 'box_max': 600, 'img_crop_size': -1, 'blur_std': 1,
              'stitch_mode': 'object_mask',
              'prompts': ['train']}, # The final inpaint prompt will be 
                                     # 'A good foto of {prompt}'
                                     # WARNING: prompts can not have commas!
    
    'bus': {'box_min': 500, 'box_max': 600, 'img_crop_size': -1, 'blur_std': 1,
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
    
    'car': {'box_min': 350, 'box_max': 600, 'img_crop_size': -1, 'blur_std': 1,
            'stitch_mode': 'object_mask',
            'prompts': ['car', 'suv', 'van']},
    
    'truck': {'box_min': 500, 'box_max': 600, 'img_crop_size': -1, 'blur_std': 1,
              'stitch_mode': 'object_mask',
              'prompts': ['european semi-truck']},
}


for obj in non_cs_objects:
    object_configs[obj] = {'box_min': 250, 'box_max': 400, 'img_crop_size': 512, 'blur_std': 1,
                           'stitch_mode': 'object_mask',
                           'prompts': [obj]}

    
    
for image_dir in image_dir_list:
    for seed in [1]:
        
        np.random.seed(seed)
        images = glob.glob(f'{image_dir}/*leftImg8bit.*')
        images = [x for x in images if x in valid_image_list]
        images.sort()
        labels = glob.glob(f'{image_dir.replace("/leftImg8bit/", "/gtFine/")}/*gtFine_labellevel3Ids.*')
        labels = [x for x in labels if x in valid_label_list]
        labels.sort()
        assert len(images) == len(labels)
        
        object_assignment = [np.random.choice(objects, size=1, replace=False) for _ in images]
        for object_prompt in objects:
            selected_imgs = [x[0].split('/')[-1] for x in zip(images, object_assignment) if object_prompt in x[1]]
            selected_imgs_str = ','.join(selected_imgs)
            
            selected_labels = [x[0].split('/')[-1] for x in zip(labels, object_assignment) if object_prompt in x[1]]
            selected_labels_str = ','.join(selected_labels)
            
            if object_prompt == 'bird flying':
                location = ''
            else:
                location = location_prompt
            results_dir = f"{results_root}/{image_dir.split('/IDD_Segmentation/')[-1]}"
            
            labels_dir = f"{labels_root}/{image_dir.split('/IDD_Segmentation/')[-1]}"
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
                      f' --seed={seed}' +
                      f' --done_filename={done_filename}'
                     )
        
        

    
        

