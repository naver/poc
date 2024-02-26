"""
Script to generate a variant of a given dset where some of the classes are 
synthetic. Generated with an inpainting pipeline. 
The script has two main components:
1) Calling the generative pipeline to obtain the modified images
2) Modifying the corresponding labels
"""

import os
import glob
import random
import copy
import argparse
import time

import torch
import numpy as np
from PIL import Image
from torchvision.io import read_image

from path_utils import *
from poc_pipeline import PlacingObjectsInContextPipeline


# Parse all the arguments
parser = argparse.ArgumentParser(description="Inpaint objects with SD")

parser.add_argument("--image_folder", type=str, default='/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/cs_reduced_classes/leftImg8bit/train/aachen/',
                    help="Folder with image files. All images within folder will be used.")
parser.add_argument("--labels_folder", type=str, default='/gfs-ssd/project/clara/data/Cityscapes/gtFine/train/aachen/',
                    help="Folder with label files.")
parser.add_argument("--img_name_list", type=str, default='',
                    help="Comma separated images within folder to inpaint.")
parser.add_argument("--labels_name_list", type=str, default='',
                    help="Comma separated labels corresponding to inpainted images.")
parser.add_argument("--inpaint_prompt", type=str, default='dog',
                    help="Prompt to describe the object e.g. dog, wheelie bin, flying bird.")
parser.add_argument("--box_min", type=int, default=300,
                    help="Param of inpainting pipeline.")
parser.add_argument("--box_max", type=int, default=450,
                    help="Param of inpainting pipeline.")
parser.add_argument("--img_crop_size", type=int, default=512,
                    help="Param of inpainting pipeline.")
parser.add_argument("--blur_std", type=int, default=None,
                    help="Param of inpainting pipeline.")
parser.add_argument("--stitch_mode", type=str, default='full_crop',
                    help="Param of inpainting pipeline. One of 'full_crop' or 'object_mask'")
parser.add_argument("--object_prompt", type=str, default='dog',
                    help="Prompt to describe the object e.g. dog, wheelie bin, flying bird.")
parser.add_argument("--label_id", type=str, default='-1',
                    help="Label of the added object.")
parser.add_argument("--location_prompt", type=str, default=None,
                    help="Prompt of the location e.g. road, sky, tree.")
parser.add_argument("--location_labels", type=str, default=None,
                    help="Comma separated labels of classes to be used as location. Can not be used if --location_prompt is specified.")
parser.add_argument("--seed", type=int, default=123,
                    help="Random seed to reproduce results.")
parser.add_argument("--results_dir", type=str, 
                    default='/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/cs_synth_classes/leftImg8bit/train/aachen/',
                    help="Directory where to save the predictions.")
parser.add_argument("--results_labels", type=str, 
                    default='/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/cs_synth_classes/gtFine/train/aachen/',
                    help="Directory where to save the labels.")
parser.add_argument("--done_filename", type=str, default='/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/cs_synth_classes/leftImg8bit/train/aachen/dog_1_done.txt',
                    help="Empty file to indicate that a job is complete so it is not re-submitted.")


args = parser.parse_args()

# Retrieve image list
input_image_names = args.img_name_list.split(',')
input_labels_names = args.labels_name_list.split(',')

print(f'Found {len(input_image_names)} images.')

print(f'Saving images in {args.results_dir}')

print(f'Saving labels in {args.results_labels}')

if not os.path.exists(args.results_dir):
    time.sleep(np.random.randint(0,100)/50)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    
if not os.path.exists(args.results_labels):
    time.sleep(np.random.randint(0,100)/50)
    if not os.path.exists(args.results_labels):
        os.makedirs(args.results_labels)

# Load inpainting pipeline
print('Loading inpainting pipeline.')
poc_pipeline = PlacingObjectsInContextPipeline(
                                 gdino_ckpt_path,
                                 gdino_config,
                                 sam_ckpt,
                                 inpainting_model_pipeline=sd_inp_v2)
print('Done')


args.location_labels = None if args.location_labels=='None' else args.location_labels

if args.location_labels is not None:
    assert args.location_prompt is None # Can not be used together
    
if args.location_prompt == '':
    args.location_prompt = None
    

prompt_list = args.inpaint_prompt.split(',')
prompt_list = [f'A good photo of a {x}' for x in prompt_list]

print(f'Using prompts: {prompt_list}')

original_seed = args.seed # For done filename
np.random.seed(original_seed)
sampled_prompts = np.random.choice(prompt_list, size=len(input_image_names))

object_list = args.object_prompt.split(',')
id_list = args.label_id.split(',')
assert len(id_list) == len(object_list)


# Begin inpainting loop
for image_name, labels_name, inpaint_prompt in zip(input_image_names, input_labels_names, sampled_prompts):
    
    image_path = os.path.join(args.image_folder, image_name)
    labels_path = os.path.join(args.labels_folder, labels_name)
    
    inp_img_name = f"{image_name.split('.')[0]}_{inpaint_prompt.replace(' ', '-')}_{args.seed}"
    inp_img_path = os.path.join(args.results_dir, inp_img_name+'.png')

    inp_labels_name = f"{image_name.split('.')[0]}_{inpaint_prompt.replace(' ', '-')}_{args.seed}"
    inp_labels_path = os.path.join(args.results_labels, inp_labels_name + '.png')

    if os.path.exists(inp_img_path) and os.path.exists(inp_labels_path):
        try:
            _ = read_image(inp_img_path)
            _ = Image.open(inp_labels_path)
        
            print(f'Image {inp_img_path} already present, skipping it.')
            args.seed += 1
            continue
        except:
            print(f'Image {inp_img_path} was corrputed, rewritting it.')

    input_image = Image.open(image_path)
    input_labels = Image.open(labels_path)
    
    if args.location_labels is not None:
        ids = args.location_labels.split(',')
        np_labels = np.array(input_labels)
        valid_area = np.zeros_like(np_labels)
        for valid_id in ids:
            valid_area[np_labels == int(valid_id)] = 1
    else:
        valid_area = None
    
    done = False
    while not done:
        try: 
            (inpainted_image,
             object_bbox,
             object_masks) = poc_pipeline.inpaint_image(input_image,
                                                inpaint_prompt,
                                                object_prompt=object_list,
                                                valid_area=valid_area,
                                                img_crop_size=args.img_crop_size, 
                                                guidance_scale=7.5,
                                                strength=0.1,
                                                location_prompt=args.location_prompt,
                                                box_min=args.box_min,
                                                stitch_mode=args.stitch_mode,
                                                box_max=args.box_max,
                                                mask_thr=0.5,
                                                bottom_overlap=False,
                                                blur_std=args.blur_std,
                                                sd_pipeline='inp',
                                                box_margin=50,
                                                seed=args.seed)
        except: # If the inpainting pipeline can not find a box within the location we
                # add the object in a random location in the image.
            print('Could not find space for object, will try smaller bbox')
            try:
                (inpainted_image,
                 object_bbox,
                 object_masks) = poc_pipeline.inpaint_image(input_image,
                                                    inpaint_prompt,
                                                    object_prompt=object_list,
                                                    valid_area=valid_area,
                                                    img_crop_size=args.img_crop_size, 
                                                    guidance_scale=7.5,
                                                    strength=0.1,
                                                    location_prompt=None,
                                                    box_min=args.box_min * 0.7,
                                                    stitch_mode=args.stitch_mode,
                                                    box_max=args.box_max * 0.7,
                                                    mask_thr=0.3,
                                                    bottom_overlap=False,
                                                    blur_std=args.blur_std,
                                                    sd_pipeline='inp',
                                                    box_margin=50,
                                                    seed=args.seed)
            except:
                print('Could not find space for object either, will place it in a random location')
                (inpainted_image,
                 object_bbox,
                 object_masks) = poc_pipeline.inpaint_image(input_image,
                                                    inpaint_prompt,
                                                    object_prompt=object_list,
                                                    valid_area=None, # We do not force the location
                                                    img_crop_size=args.img_crop_size, 
                                                    guidance_scale=7.5,
                                                    strength=0.1,
                                                    location_prompt=None, # We do not force the location
                                                    box_min=args.box_min,
                                                    stitch_mode=args.stitch_mode,
                                                    box_max=args.box_max,
                                                    mask_thr=0.5,
                                                    bottom_overlap=False,
                                                    blur_std=args.blur_std,
                                                    sd_pipeline='inp',
                                                    box_margin=50,
                                                    seed=args.seed)
                    


        if object_list[-1] in object_masks.keys():
            for prompt in object_masks.keys():
                object_masks[prompt] = (object_masks[prompt].sum(dim=0) > 0).float()
            
            if len(object_list) == 1:
                object_mask = object_masks[object_list[0]][0].numpy().astype('uint8')
                # Augment image labels with inpainted object
                input_labels = np.array(input_labels)
                input_labels[object_mask > 0] = id_list[0]
                input_labels = Image.fromarray(input_labels)
            elif len(object_list) == 2:
                if object_list[0] not in object_masks.keys():
                    object_masks[object_list[0]] = torch.zeros_like(object_masks[object_list[1]])
                intersection = object_masks[object_list[0]] * object_masks[object_list[1]]
                union = (object_masks[object_list[0]] + object_masks[object_list[1]] > 0).float()
                # If both rider and bike have highly overlapping masks
                # Usually means gsam detects the bike as rider
                if intersection.sum() / union.sum() > 0.8: 
                    object_masks[object_list[0]] *= 0
                else: # Otherwise we give priority to rider over bike class.
                    object_masks[object_list[1]] -= intersection
                # Augment image labels with inpainted objects
                input_labels = np.array(input_labels)
                object_mask = object_masks[object_list[0]][0].numpy().astype('uint8')
                input_labels[object_mask > 0] = id_list[0]
                object_mask = object_masks[object_list[1]][0].numpy().astype('uint8')
                input_labels[object_mask > 0] = id_list[1]
                input_labels = Image.fromarray(input_labels)
            else:
                raise NotImplementedError
            
            ## Save inpainted labels and image
            # Update seed in filename (in case more than one seed was used)
            print('Saving image and mask...')
            inp_img_name = f"{image_name.split('.')[0]}_{inpaint_prompt.replace(' ', '-')}_{args.seed}"
            inp_img_path = os.path.join(args.results_dir, inp_img_name+'.png')
            inp_labels_name = f"{image_name.split('.')[0]}_{inpaint_prompt.replace(' ', '-')}_{args.seed}"
            inp_labels_path = os.path.join(args.results_labels, inp_labels_name + '.png')

            input_labels.save(inp_labels_path)
            inpainted_image.save(inp_img_path)
            
            args.seed += 1  
            done = True
            print('Done!')
            
        else:
            print('Generation failed, trying different seed')
            args.seed += 1
            
    
# Save done file
with open(args.done_filename,'wb') as f:
    print('Saving end of training file')
        
            
        





