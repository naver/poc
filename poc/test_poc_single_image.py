import numpy as np
import os
import torch
import glob
import matplotlib.pyplot as plt
import warnings
from PIL import Image
import time
import cv2


from grounded_sam_utils import *
from utils_sd import *
from poc_pipeline import PlacingObjectsInContextPipeline

from path_utils import *
import argparse


parser = argparse.ArgumentParser(
	description='Apply POC on a single image.')
parser.add_argument('--image_path', default='my_image.png', help='path of the image POC will modify')
parser.add_argument('--obj', type=str, default='dog', help='what we want POC to insert')
parser.add_argument('--loc', type=str, default='road', help='where we want POC to modify the image')
parser.add_argument('--seed', type=int, default=213, help='random seed')
args = parser.parse_args()

poc_pipeline = PlacingObjectsInContextPipeline(
                                         gdino_ckpt_path,
                                         gdino_config,
                                         sam_ckpt,
                                         inpainting_model_pipeline=sd_inp_v2)

input_image = Image.open(args.image_path).convert("RGB")

object_prompt = [args.obj]
inpaint_prompt = f'A good photo of a {args.obj}'
negative_prompt = None
box_min = 200
box_max = 400
mask_thr = 0.5
blur_std = 1
img_crop_size = 512

init_time = time.time()
inpainted_image, object_bbox, object_masks = poc_pipeline.inpaint_image(input_image, inpaint_prompt,
                                                                negative_prompt=negative_prompt,
                                                                object_prompt=object_prompt,
                                                                img_crop_size=img_crop_size,
                                                                location_prompt=args.loc,
                                                                box_min=box_min,
                                                                box_max=box_max,
                                                                mask_thr=mask_thr,
                                                                blur_std=blur_std,
                                                                seed=args.seed)

eta = time.time() - init_time
print(eta)

image_ext = args.image_path.split(".")[-1]
output_path = args.image_path.replace(f'.{image_ext}', f'_{args.obj}_{args.loc}_{args.seed}.{image_ext}')
inpainted_image.save(output_path)

# get inpainting region for reference
mask = poc_pipeline.inpainting_full_mask.astype('uint8')

contours,_ = cv2.findContours(mask.copy(), 1, 1)
rect = cv2.minAreaRect(contours[0])

box = cv2.boxPoints(rect)
box = np.int0(box)
rect2 = cv2.drawContours(np.array(inpainted_image),[box],0,(0,0,255),5)

colors = [np.array([30/255, 144/255, 255/255, 0.6]),
          np.array([255/255, 144/255, 10/255, 0.6]),
          np.array([114/255, 144/255, 114/255, 0.6])]

for ii, prompt in enumerate(object_masks.keys()):
    plt.figure(figsize=(15,15))
    plt.imshow(rect2)
    mask = object_masks[prompt].sum(dim=0).cpu().numpy()
    mask = (mask > 0).astype('uint8')

    show_mask(mask, plt.gca(), color=colors[ii])
    plt.axis('off')

output_path = args.image_path.replace(f'.{image_ext}', f'_{args.obj}_{args.loc}_{args.seed}_ann.{image_ext}')
plt.savefig(output_path, bbox_inches='tight')





