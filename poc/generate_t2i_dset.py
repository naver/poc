"""
Script to generate a variant of Cityscapes with t2i inpainting pipeline.
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

from path_utils import *
from diffusers import DiffusionPipeline
from open_vocab_segmentation_pipeline import OpenSegmentor


# Parse all the arguments
parser = argparse.ArgumentParser(description="Inpaint objects with SD")

parser.add_argument("--image_folder", type=str, default='/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/cs_reduced_classes/leftImg8bit/train/aachen/',
                    help="Folder with image files. All images within folder will be used.")
parser.add_argument("--labels_folder", type=str, default='/gfs-ssd/project/clara/data/Cityscapes/gtFine/train/aachen/',
                    help="Folder with label files.")
parser.add_argument("--t2i_prompt", type=str, default='a dog',
                    help="Prompt to generate the image.")

parser.add_argument("--object_prompt", type=str, default='dog',
                    help="Prompt to describe the object e.g. dog, wheelie bin, flying bird.")
parser.add_argument("--label_id", type=str, default='-1',
                    help="Label of the added object.")
parser.add_argument("--seed", type=int, default=123,
                    help="Random seed to reproduce results.")
parser.add_argument("--results_dir_t2i_stitched", type=str, 
                    default='/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/cs_synth_t2i_stitched_v2/leftImg8bit/train/aachen/',
                    help="Directory where to save the predictions.")
parser.add_argument("--results_dir_t2i_baseline", type=str, 
                    default='/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/cs_synth_t2i_baseline_v2/leftImg8bit/train/aachen/',
                    help="Directory where to save the predictions.")
parser.add_argument("--done_filename", type=str, default='/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/cs_synth_t2i_baseline_v2/leftImg8bit/train/aachen/dog_1_done.txt',
                    help="Empty file to indicate that a job is complete so it is not re-submitted.")


args = parser.parse_args()

# Retrieve image list
input_image_list = glob.glob(f'{args.image_folder}/*leftImg8bit.png')
input_image_list.sort()
    
print(f'Found {len(input_image_list)} images.')

print(f'Saving images in {args.results_dir_t2i_baseline}')
print(f'Saving images in {args.results_dir_t2i_stitched}')

# Retrieve label list
labels_dir_base = args.results_dir_t2i_baseline.replace('leftImg8bit', 'gtFine')
labels_dir_stit = args.results_dir_t2i_stitched.replace('leftImg8bit', 'gtFine')

print(f'Saving labels in {labels_dir_base}')
print(f'Saving labels in {labels_dir_stit}')

if not os.path.exists(args.results_dir_t2i_baseline):
    time.sleep(np.random.randint(0,100)/50)
    if not os.path.exists(args.results_dir_t2i_baseline):
        os.makedirs(args.results_dir_t2i_baseline)
        
if not os.path.exists(args.results_dir_t2i_stitched):
    time.sleep(np.random.randint(0,100)/50)
    if not os.path.exists(args.results_dir_t2i_stitched):
        os.makedirs(args.results_dir_t2i_stitched)
    
if not os.path.exists(labels_dir_base):
    time.sleep(np.random.randint(0,100)/50)
    if not os.path.exists(labels_dir_base):
        os.makedirs(labels_dir_base)
        
if not os.path.exists(labels_dir_stit):
    time.sleep(np.random.randint(0,100)/50)
    if not os.path.exists(labels_dir_stit):
        os.makedirs(labels_dir_stit)

# Load inpainting pipeline
print('Loading pipelines.')
count = 0
while count < 10:
    try:
        t2i_pipeline = DiffusionPipeline.from_pretrained(sd_v2,
                                             torch_dtype=torch.float16).to('cuda')
        count = 100
    except:
        print(f'Loading diffusion pipeline failed {count + 1} times, trying again')
        count += 1
print('Done loading diffusion pipeline')

count = 0
while count < 10:
    try:
        seg_pipeline = OpenSegmentor(gdino_ckpt_path, gdino_config, sam_ckpt)
        count = 100
    except:
        print(f'Loading gsam failed {count + 1} times, trying again')
        count += 1
print('Done loading gsam')


prompt_list = args.t2i_prompt.split(',')
prompt_list = [f'A good photo of a {x}' for x in prompt_list]

print(f'Using prompts: {prompt_list}')

original_seed = args.seed # For done filename
np.random.seed(original_seed)
sampled_prompts = np.random.choice(prompt_list, size=len(input_image_list))

object_list = args.object_prompt.split(',')
id_list = args.label_id.split(',')
assert len(id_list) == len(object_list)


# Begin inpainting loop
for image_path, inpaint_prompt in zip(input_image_list, sampled_prompts):
    image_name = image_path.split('/')[-1].split('.')[0]
    labels_name = image_name.replace('leftImg8bit', 'gtFine_labelTrainIds')

    inp_img_name = f"{image_name}_{inpaint_prompt.replace(' ', '-')}_{args.seed}"
    inp_img_path_stit = os.path.join(args.results_dir_t2i_stitched, inp_img_name+'.png')
    inp_img_path_base = os.path.join(args.results_dir_t2i_baseline, inp_img_name+'.png')

    inp_labels_name = f"{image_name}_{inpaint_prompt.replace(' ', '-')}_{args.seed}"
    inp_labels_path_base = os.path.join(labels_dir_base, inp_labels_name + '.png')
    inp_labels_path_stit = os.path.join(labels_dir_stit, inp_labels_name + '.png')

    if (os.path.exists(inp_img_path_stit) and os.path.exists(inp_labels_path_stit) and
        os.path.exists(inp_img_path_base) and os.path.exists(inp_labels_path_base)):
        try:
            _ = Image.open(inp_img_path_stit)
            _ = Image.open(inp_labels_path_stit)
            _ = Image.open(inp_img_path_base)
            _ = Image.open(inp_labels_path_base)
        
            print(f'Image {inp_img_path_stit} already present, skipping it.')
            args.seed += 1
            continue
        except:
            print(f'Image {inp_img_path_stit} was corrputed, rewritting it.')

    input_image = Image.open(image_path)
    input_labels = Image.open(os.path.join(args.labels_folder, labels_name + '.png'))
    
    done = False
    while not done:
        # To do: add inpainting arguments as script options.
        g = torch.Generator('cuda').manual_seed(args.seed)
        syn_img = t2i_pipeline(args.t2i_prompt, generator=g).images[0]
        object_prompt = 'train'
        
        object_masks = {}
        for prompt in object_list:
            object_masks[prompt] = seg_pipeline.segment_image(syn_img, prompt, box_threshold=0.3)

        if object_masks[object_list[-1]].sum() > 30: # At least 50 pixels in the mask
            if len(object_list) == 1:
                object_mask = object_masks[object_list[0]].astype('uint8')
                # Augment image labels with inpainted object
                label_mask = 255 + np.zeros_like(object_mask)
                label_mask[object_mask > 0] = id_list[0]
            elif len(object_list) == 2:
                intersection = object_masks[object_list[0]] * object_masks[object_list[1]]
                union = (object_masks[object_list[0]] + object_masks[object_list[1]] > 0).astype('float')
                # If both rider and bike have highly overlapping masks
                # Usually means gsam detects the bike as rider
                if intersection.sum() / union.sum() > 0.8: 
                    object_masks[object_list[0]] *= 0
                else: # Otherwise we give priority to rider over bike class.
                    object_masks[object_list[1]] -= intersection
                # Augment image labels with inpainted objects
                label_mask = 255 + np.zeros_like(intersection)
                object_mask = object_masks[object_list[0]].astype('uint8')
                label_mask[object_mask > 0] = id_list[0]
                object_mask = object_masks[object_list[1]].astype('uint8')
                label_mask[object_mask > 0] = id_list[1]                
            else:
                raise NotImplementedError
            
            syn_img = np.array(syn_img)
            input_labels = np.array(input_labels)
            input_image = np.array(input_image)
            
            np.random.seed(args.seed)
            # Where to stitch the object
            h = np.random.randint(50, input_labels.shape[0] - label_mask.shape[0] - 50)
            w = np.random.randint(50, input_labels.shape[1] - label_mask.shape[1] - 50)
            
            # Base image
            base_img = input_image.copy()
            base_img[h:h+label_mask.shape[0], w:w+label_mask.shape[1]] = syn_img
            
            base_label = input_labels.copy()
            base_label[h:h+label_mask.shape[0], w:w+label_mask.shape[1]] = label_mask
            
            base_img = Image.fromarray(base_img)
            base_label = Image.fromarray(base_label)
            
            # Stitched image
            st_img = input_image.copy()
            st_img[h:h+label_mask.shape[0], w:w+label_mask.shape[1], :][label_mask < 255, :] = syn_img[label_mask < 255, :]
            
            st_label = input_labels.copy()
            st_label[h:h+label_mask.shape[0], w:w+label_mask.shape[1]][label_mask < 255] = label_mask[label_mask < 255]
            
            st_img = Image.fromarray(st_img)
            st_label = Image.fromarray(st_label)
            
            ## Save labels and image
            # Update seed in filename (in case more than one seed was used)
            print('Saving image and mask...')
            
            inp_img_name = f"{image_name}_{inpaint_prompt.replace(' ', '-')}_{args.seed}"
            inp_img_path_stit = os.path.join(args.results_dir_t2i_stitched, inp_img_name+'.png')
            inp_img_path_base = os.path.join(args.results_dir_t2i_baseline, inp_img_name+'.png')

            inp_labels_name = f"{image_name}_{inpaint_prompt.replace(' ', '-')}_{args.seed}"
            inp_labels_path_base = os.path.join(labels_dir_base, inp_labels_name + '.png')
            inp_labels_path_stit = os.path.join(labels_dir_stit, inp_labels_name + '.png')
    
            base_img.save(inp_img_path_base)
            base_label.save(inp_labels_path_base)
            st_img.save(inp_img_path_stit)
            st_label.save(inp_labels_path_stit)
            
            args.seed += 1  
            done = True
            print('Done!')
            
        else:
            print('Generation failed, trying different seed')
            args.seed += 1
            
    
# Save done file
with open(args.done_filename,'wb') as f:
    print('Saving end of training file')
        
            
        





