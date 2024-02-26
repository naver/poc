import os
import glob
import numpy as np

np.random.seed(0)

force_redo = False

root_path = '/gfs-ssd/project/clara/data/ACDC'

image_dir_list = sorted(glob.glob(f'{root_path}/rgb_anon_trainvaltest/rgb_anon/*/val/*'))
labels_dir_list = sorted(glob.glob(f'{root_path}/gt_trainval/gt/*/val/*'))
assert len(image_dir_list) == len(labels_dir_list)

full_img_list = sorted(glob.glob(f'{root_path}/rgb_anon_trainvaltest/rgb_anon/*/val/*/*_rgb_anon.*'))
full_label_list = sorted(glob.glob(f'{root_path}/gt_trainval/gt/*/val/*/*_gt_labelIds.*'))
assert len(full_img_list) == len(full_label_list)

valid_idx = np.random.choice(range(len(full_img_list)), 200, replace=False)
valid_img_list = np.array(full_img_list)[valid_idx]
valid_label_list = np.array(full_label_list)[valid_idx]



results_root = '/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/acdc_ood/'
    
objects = ['rider,bicycle', 'rider,motorcycle', 'train', 'bus', 'person', 'car', 'truck']

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
    
location_labels = '7'

object_id = {
    'rider,bicycle': '25,33',
    'rider,motorcycle': '25,32',
    'train': 31,
    'bus': 28,
    'person': 24,
    'car': 26,
    'truck': 27,
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
    'train': {'box_min': 400, 'box_max': 500, 'img_crop_size': -1, 'blur_std': 1,
              'stitch_mode': 'object_mask',
              'prompts': ['train']}, # The final inpaint prompt will be 
                                     # 'A good foto of {prompt}'
                                     # WARNING: prompts can not have commas!
    
    'bus': {'box_min': 400, 'box_max': 500, 'img_crop_size': -1, 'blur_std': 1,
            'stitch_mode': 'object_mask',
            'prompts': ['bus']},
    
    'rider,motorcycle': {'box_min': 300, 'box_max': 400, 'img_crop_size': -1, 'blur_std': 1,
                         'stitch_mode': 'object_mask',
                         'prompts': ['motorbike', 'motorbike with rider',
                                     'scooter', 'scooter with rider']},
    
    'rider,bicycle': {'box_min': 300, 'box_max': 400, 'img_crop_size': -1, 'blur_std': 1,
                      'stitch_mode': 'object_mask', 
                      'prompts': ['bicycle', 'bicycle with rider', 'person riding a bicycle']},
    
    'person': {'box_min': 300, 'box_max': 400, 'img_crop_size': 512, 'blur_std': 1,
               'stitch_mode': 'object_mask',
               'prompts': ['man walking in the street', 'a woman walking in the street']},
    
    'car': {'box_min': 350, 'box_max': 500, 'img_crop_size': -1, 'blur_std': 1,
            'stitch_mode': 'object_mask',
            'prompts': ['car', 'suv', 'van']},
    
    'truck': {'box_min': 400, 'box_max': 500, 'img_crop_size': -1, 'blur_std': 1,
              'stitch_mode': 'object_mask',
              'prompts': ['european semi-truck']},
}

for obj in non_cs_objects:
    object_configs[obj] = {'box_min': 250, 'box_max': 400, 'img_crop_size': 512, 'blur_std': 1,
                           'stitch_mode': 'object_mask',
                           'prompts': [obj]}

    
    
for image_dir, labels_dir in zip(image_dir_list, labels_dir_list):
    for seed in [1]:
        np.random.seed(seed)
        images = glob.glob(f'{image_dir}/*_rgb_anon.*')
        images = [x for x in images if x in valid_img_list]
        images.sort()
        labels = glob.glob(f"{labels_dir}/*_gt_labelIds.*")
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
                location = None
            else:
                location = location_labels
            results_dir = f"{results_root}/{image_dir.split('/ACDC/')[-1]}"
            results_labels = f"{results_root}/{labels_dir.split('/ACDC/')[-1]}"
            
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
                      f' --location_labels={location}' +
                      f' --results_dir=\"{results_dir}\"' +
                      f' --results_labels=\"{results_labels}\"' +
                      f' --box_min={object_configs[object_prompt]["box_min"]}' +
                      f' --box_max={object_configs[object_prompt]["box_max"]}' +
                      f' --img_crop_size={object_configs[object_prompt]["img_crop_size"]}' +
                      f' --stitch_mode=\"{object_configs[object_prompt]["stitch_mode"]}\"' +
                      f' --seed={seed}' +
                      f' --done_filename={done_filename}'
                     )
        
        

    
        

