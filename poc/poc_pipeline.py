import numpy as np
import os
import torch
import glob
import matplotlib.pyplot as plt
import warnings
from PIL import Image, ImageOps
import time

from diffusers import StableDiffusionInpaintPipeline
from grounded_sam_utils import *
from utils_sd import *

class PlacingObjectsInContextPipeline:
    
    def __init__(self,
                 gdino_ckpt=None,
                 gdino_config=None,
                 sam_ckpt=None,
                 inpainting_model_pipeline="stabilityai/stable-diffusion-2-inpainting",
                 device='cuda'
                ):
        
        # We need the three in order to run groundedSAM
        if gdino_ckpt is None:
            assert gdino_config is None
            assert sam_ckpt is None
            
        self.device = device
        # Load inpainting pipeline form HuggingFace
        # Alternative model: "runwayml/stable-diffusion-inpainting"
        self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                inpainting_model_pipeline,
                torch_dtype=torch.float16,
#                 revision="fp16",
                ).to(self.device)
        
        
        if gdino_ckpt is not None:
            # Load GroundingDINO model
            self.gdino_model = load_model(gdino_config, gdino_ckpt, device='cpu')

            # SAM model
            self.sam_model = SamPredictor(build_sam(checkpoint=sam_ckpt).to(self.device))
        
        
    def get_random_mask(self, img_size=(1024, 2048), box_min=(140,140), box_max=(220,220),
                    valid_area=None, mask_thr=1.0, bottom_overlap=False, seed=None,
                    box_margin=50):
        '''
        Descr: Function to produce a random box-shaped mask over an image overlapping with a given class.
        Params: 
            -img_size: Tuple with size in pixels of image to which the mask will be applied (height, width).
            -box_min: Tuple or int with minimum size in pixels of the box (height, width).
            -box_max: Tuple or int with maximum size in pixels of the box (height, width).
            -valid_area: numpy array with a binary mask of the valid_area for the mask (height, width).
            -mask_thr: float between 0 and 1 describing the minimum overlap percentage. 
            -bottom_overlap: bool, if true then the overlap has to be on the bottom of the mask;
                in many cases we consider an object is in an area if it "stands" there. 
            -seed: optional random seed for reproducibility.
            -box_margin: Tuple or int with margin from inpainting region to img edges (height, width).
        Returns:
            random_mask (binary mask), tuple with box coordinates (bottom_h, bottom_w, height, width)
        '''

        if seed is not None:
            np.random.seed(seed)
            
        if valid_area is None:
            valid_area = np.ones(img_size) # all image is valid area

        if not isinstance(box_min, tuple):
            box_min = (box_min, box_min)
        if not isinstance(box_max, tuple):
            box_max = (box_max, box_max)
        if not isinstance(box_margin, tuple):
            box_margin = (box_margin, box_margin)
            
        box_h = np.random.randint(box_min[0], box_max[0])
        box_w = np.random.randint(max(box_h * 0.8, box_min[1]), min(box_h * 1.2, box_max[1]))

        img_h, img_w = img_size
        
        assert 2 * box_margin[0] + box_max[0] <= img_h
        assert 2 * box_margin[1] + box_max[1] <=img_w
        
        # Find enclosing rectangles for valid area and generate mask
        # This is an upper bound of the pixels where a valid box could start
        cnts = cv2.findContours(valid_area.astype('uint8'),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        valid_boxes = np.zeros_like(valid_area)
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            left = max(x - box_w, 0)
            bottom = max(y - box_h, 0)
            valid_boxes[bottom:y+h, left:x+w] = 1
            
        self.valid_area = valid_area
        self.valid_boxes = valid_boxes
            
        # Set margins for valid area
        min_h = box_margin[0]
        min_w = box_margin[1]
        max_h = img_h - box_h - box_margin[0]
        max_w = img_w - box_w - box_margin[1]

        margins = np.zeros_like(valid_area)
        margins[min_h:max_h, min_w:max_w] = 1
        self.search_area = valid_boxes * margins

        valid_pixels = np.where(self.search_area > 0)

        # Go through all potentially valid boxes randomly until one is found valid
        random_h = random_w = None
        if len(valid_pixels[0]) > 0 and len(valid_pixels[1]) > 0:
            indxs = list(range(len(valid_pixels[0])))
            np.random.shuffle(indxs)
            valid_h = valid_pixels[0][indxs]
            valid_w = valid_pixels[1][indxs]
            for h, w in zip(valid_h, valid_w):
                if bottom_overlap:
                    bottom_thr = int(h + box_h * (1 - mask_thr))
                    area = valid_area[bottom_thr:int(h+box_h), w:(w+box_w)].mean()
                    area = area * mask_thr
                else:
                    area = valid_area[h:h+box_h, w:w+box_w].mean()
                if area >= mask_thr:
                    random_mask = np.zeros(img_size, dtype='uint8')
                    random_mask[h:h+box_h, w:w+box_w] = 1
                    return random_mask, (h, w, box_h, box_w)
        
        raise ValueError(f'Could not find box with mask_thr: {mask_thr} overlap.')

        
    @staticmethod
    def get_crop_around_mask(image, inpaint_mask, inpaint_box, img_crop_size):
        '''
        Descr: Function to crop image around inpainting mask.
        Params: 
            -image: PIL image to crop and apply mask.
            -inpaint_mask: np array with inpaint full mask.
            -inapint_box: Bounding box of inpainting region (y, x, h, w)
            -img_crop_size: size of the img_crop.
        Returns:
            img_crop (PIL), img_mask (PIL), img_crop_bottom (bottom left coordinates of cropped image)
        '''
        
        img_w, img_h = image.size
        img_size = np.array([img_h, img_w])
        img_crop_size = np.array(img_crop_size)
        bottom_h, bottom_w, box_h, box_w = inpaint_box
        box_center = np.array([bottom_h + box_h//2 , bottom_w + box_w//2])
        img_crop_bottom = box_center - img_crop_size // 2
        img_crop_top = img_crop_bottom + img_crop_size
        # Check in case crop is out of the image
        bottom_offset =  -np.minimum(0, img_crop_bottom) # If crop bottom is negative we subtract it.
        top_offset = np.minimum(0, img_size - img_crop_top) # If top is larger than img size
        img_crop_bottom = img_crop_bottom + bottom_offset + top_offset
        img_crop_top = img_crop_top + bottom_offset + top_offset
                                 
                                 
        img_crop = np.array(image)[img_crop_bottom[0]:img_crop_top[0],
                              img_crop_bottom[1]:img_crop_top[1]]

        cropped_inpaint_mask = inpaint_mask[img_crop_bottom[0]:img_crop_top[0],
                              img_crop_bottom[1]:img_crop_top[1]]
        
        return (Image.fromarray(img_crop), Image.fromarray(cropped_inpaint_mask*255),
                img_crop_bottom)

    
    # @staticmethod
    def stitch_image(self, original_image, modified_crop, img_crop_bottom, mask,
                     blur_std=0, full_mask=True, img_crop_size=(512,512)):
        '''
        Descr: Function to stitch a modified crop of an image into the original full image.
        Params: 
            -original_image: PIL image that was cropped.
            -modified_crop: PIL image of the cropped region that was modified. 
            -img_crop_bottom: bottom left coordinates of cropped image - (height, width)
            -mask: binary np array of the mask of the region to stitch. If mask is the area
                of the crop, then the full cropped image is included.
            -blur_std: std for the gaussian blur applied to smooth the stitching boundaries.
            -img_crop_size: size of the img_crop.
            -full_mask: bool. If True, pixels within the masked area are not blended.
        Returns:
            modified_image (PIL)
        '''
        
        modified_image = np.array(original_image)
        
        modified_image[
            img_crop_bottom[0]:img_crop_bottom[0]+img_crop_size[0],
            img_crop_bottom[1]:img_crop_bottom[1]+img_crop_size[1], :] = np.array(modified_crop)
        
        self.img_before_blur = modified_image
        if blur_std > 0:
            print('Blurring edges')
            fusion_mask = cv2.GaussianBlur(mask*255, (2*blur_std+1, 2*blur_std+1), sigmaX=blur_std)/255
            if full_mask:
                fusion_mask = np.maximum(mask, fusion_mask)
        else:
            fusion_mask = mask

        fusion_mask = fusion_mask[..., np.newaxis]
        modified_image = modified_image * fusion_mask + (1 - fusion_mask) * original_image
        self.fusion_mask = fusion_mask
        return Image.fromarray(modified_image.astype('uint8'))
    

    def run_grounded_sam(self, input_image, prompt, box_threshold=0.3, text_threshold=0.25,
                        device='cuda'):
        '''
        Descr: Function to run GroundedSAM model to extract mask and bounding boxes based on prompts.
        Params: 
            -input_image: PIL image to be segmented/detected.
            -prompt: text prompt to be used. 
            -box_threshold: Hyperparam from GroundingDINO. Using their default.
            -text_threshold: Hyperparam from GroundingDINO. Using their default.
        Returns:
            bounding_boxes: tensor list of bounding boxes [N, W, H, W, H] where W, H are coordinates
                            of bottom left and top right vertex and N is num of boxes.
            masks: tensor list of boolean masks of shape [1, N, H, W] where 1 is the number of 
            images in the batch, N num of masks and H, W are the image dimensions.
        '''

        # Transform image to feed to model
        torch_image = transform_PIL_image(input_image) 

        # run grounding dino model
        boxes_filt, pred_phrases = get_grounding_output(
            self.gdino_model, torch_image, prompt,
            box_threshold, text_threshold, device='cpu' # Problem with CUDA installation. Need to check.
        )

        # filter bounding boxes
        size = input_image.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()

        # compute SAM embedding of image
        self.sam_model.set_image(np.array(input_image))

        transformed_boxes = self.sam_model.transform.apply_boxes_torch(
                                boxes_filt,
                                np.array(input_image).shape[:2]
                            ).to('cuda')

        masks, _, _ = self.sam_model.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        )
        
        return boxes_filt, masks
    
    
    @staticmethod
    def compute_bbox_mask_full_image(boxes_filt, masks, img_h, img_w, img_crop_bottom):
        '''
        Descr: Function to compute the bbox and masks corresponding to the full image given
            the ones extracted from the cropped image.
        Params: 
            -boxes_filt: original bounding boxes corresponding to the crop
            -masks: original masks corresponding to the crop.
            -img_h: height of the full image.
            -img_w: width of the full image.
            -img_crop_bottom: full image coordinates of the bottom left corner of the crop.
        Returns:
            bounding_boxes: tensor list of bounding boxes [N, W, H, W, H] where W, H are coordinates
                            of bottom left and top right vertex and N is num of boxes.
            masks: tensor list of boolean masks of shape [1, N, H, W] where 1 is the number of 
                   images in the batch, N num of masks and H, W are the image dimensions.
        '''
        
        for ii, _ in enumerate(boxes_filt):
            boxes_filt[ii][0] += img_crop_bottom[1] # Add width offset
            boxes_filt[ii][2] += img_crop_bottom[1] 
            boxes_filt[ii][1] += img_crop_bottom[0] # Add height offset
            boxes_filt[ii][3] += img_crop_bottom[0]

        full_masks = torch.zeros(masks.shape[0], masks.shape[1], img_h, img_w)
        for ii in range(full_masks.shape[0]):
            for jj in range(full_masks.shape[1]):
                full_masks[ii, jj,
                           img_crop_bottom[0]:img_crop_bottom[0]+masks.shape[2],
                           img_crop_bottom[1]:img_crop_bottom[1]+masks.shape[3]] = masks[ii,jj,...]
                
        return boxes_filt, full_masks
    
    
    def get_valid_area_from_grounded_sam(self, input_image, location_prompt, box_threshold=0.3,
                                        text_threshold=0.25):
        '''
        Descr: Function to find the valid area based on grounded sam and a location prompt.
        Params: 
            -input_image: PIL image to be segmented/detected.
            -location_prompt: text prompt to be used to extract the valid area for inpainting.
                if neither 'location prompt' nor 'valid_area' are specified anywhere is valid.
            -box_threshold: Hyperparam from GroundingDINO. Using their default.
            -text_threshold: Hyperparam from GroundingDINO. Using their default.
        Returns:
            valid_area: np array with binary mask corresponding to location prompt.
        '''
        if location_prompt is not None:
            _, masks = self.run_grounded_sam(input_image, location_prompt,
                                            box_threshold=box_threshold,
                                            text_threshold=text_threshold,
                                            device=self.device)
            # Join masks if more than one
            self.valid_area = (masks.float().sum(dim=0)[0] > 0).cpu().numpy().astype('float')
        else:
            self.valid_area = np.ones((self.img_h, self.img_w))
        
        return self.valid_area
            
    
    def inpaint_image(self, input_image, inpaint_prompt, negative_prompt=None, object_prompt=None,
                      valid_area=None, location_prompt=None, guidance_scale=7.5, strength=0.1,
                      box_min=None, box_max=None, mask_thr=1.0, bottom_overlap=False,
                      img_crop_size=(512,512), box_threshold=0.3, text_threshold=0.25, 
                      stitch_mode='object_mask', blur_std=None, box_margin=0, sd_pipeline='inp', seed=None):
        '''
        Descr: Function to inpaint an object in a region of the image. Will return the inpainted image and
            the bounding box(es) / mask(s) for the inpainted object(s).
            Several inpainting / img2img passes can be done in case some of the relevant params are a list.
        Params: 
            -input_image: PIL image to be segmented/detected.
            -inpaint_prompt: text prompt to be used for inpainting.
            -negative_prompt: text negative prompt to be used for inpainting.
            -guidance_scale: hyperparam from stable diffusion inpainting pipeline. Using their default.
                    Can be a list for multiple inpaint passes.
            -strength: hyperparam from stable diffusion img2img pipeline.
                    Can be a list for multiple inpaint passes.
            -object_prompt: List of text prompts to be used to detect the added object(s). 
                    If 'None', no objects will be detected.
            -valid_area: numpy array with a binary mask of the valid_area for inpainting (height, width).
            -location_prompt: text prompt to be used to extract the valid area for inpainting.
                if neither 'location prompt' nor 'valid_area' are specified anywhere is valid.
            -box_min: Tuple with minimum size in pixels of the box (height, width).
            -box_max: Tuple with maximum size in pixels of the box (height, width).
            -mask_thr: float between 0 and 1 describing the minimum overlap percentage. 
            -bottom_overlap: bool, if true then the overlap has to be on the bottom of the mask;
                in many cases we consider an object is in an area if it "stands" there. 
            -img_crop_size: Tuple with size of the img_crop (H, W).
                    Can be a list for multiple inpaint passes.
            -box_threshold: Hyperparam from GroundingDINO. Using their default.
            -text_threshold: Hyperparam from GroundingDINO. Using their default.
            -stitch_mode: one of 'object_mask', 'full_crop'.
                    > object_mask: use the mask of the segmented added object.
                    > full_crop: use the mask of the full crop.
                    Can be a list for multiple inpaint passes.
            -blur_std: std for the gaussian blur applied to smooth the stitching boundaries.
                    Can be a list for multiple inpaint passes.
            -box_margin: margin from the inpainting box to the image edges.
            -sd_pipeline: one of 'inp' or 'i2i' for inpainting and image2image respectively.
                    Can be a list for multiple inpaint passes.
            -seed: random seed.
        Returns:
            modified_image: PIL image with the inpainted object.
            bounding_boxes: tensor list of bounding boxes [N, W, H, W, H] where W, H are coordinates
                    of top left and bottom right vertex and N is num of boxes.
            masks: tensor list of boolean masks of shape [1, N, H, W] where 1 is the number of 
                    images in the batch, N num of masks and H, W are the image dimensions.
        '''
        
        self.img_w, self.img_h = input_image.size
        
        if seed is None:
            seed = np.random.randint(100000)
        
        # If the valid area is not user-defined, find it with location prompt or all image is valid.
        self.valid_area = valid_area
        if self.valid_area is None:
            self.get_valid_area_from_grounded_sam(input_image, location_prompt, box_threshold,
                                        text_threshold)
            
        # If box size is not user-defined, this is a reasonable heuristic   
        if box_min is None or box_max is None:
            box_min = int(max(self.img_h, self.img_w) * 0.1)
            box_max = int(max(self.img_h, self.img_w) * 0.3)
            
        # Randomly select mask for inpainting
        start = time.time()
        self.inpainting_full_mask, inpaint_box = self.get_random_mask((self.img_h, self.img_w),
                                                      box_min,
                                                      box_max,
                                                      valid_area=self.valid_area,
                                                      mask_thr=mask_thr,
                                                      bottom_overlap=bottom_overlap,
                                                      box_margin=box_margin,
                                                      seed=seed)
        print(f'Found mask in {time.time() - start} seconds.')
        
        if img_crop_size == -1:
            img_crop_size = int(1.2 * np.max(inpaint_box[-2:]))
            
        if blur_std is None:
            blur_std = min(int(0.2 * np.max(inpaint_box[-2:])), 50)
            
        if isinstance(img_crop_size, int):
            img_crop_size = (img_crop_size, img_crop_size)
        if isinstance(box_min, int):
            box_min = (box_min, box_min)
        if isinstance(box_max, int):
            box_max = (box_max, box_max)
            
        # Convert relevant params to list
        guidance_scale = ensure_list(guidance_scale)
        strength = ensure_list(strength)
        img_crop_size = ensure_list(img_crop_size)
        stitch_mode = ensure_list(stitch_mode)
        blur_std = ensure_list(blur_std)
        sd_pipeline = ensure_list(sd_pipeline)
        
        # If some param requires multiple passes, arrange others accordingly
        list_params = [guidance_scale, strength, img_crop_size, stitch_mode, blur_std, sd_pipeline]
        num_passes = max([len(x) for x in list_params])
        if num_passes > 1:
            guidance_scale = even_list_len(guidance_scale, num_passes)
            strength = even_list_len(strength, num_passes)
            img_crop_size = even_list_len(img_crop_size, num_passes)
            stitch_mode = even_list_len(stitch_mode, num_passes)
            blur_std = even_list_len(blur_std, num_passes)
            sd_pipeline = even_list_len(sd_pipeline, num_passes)
            
        # The inpainting box can not be larger than the crop
        for crop_size in img_crop_size:
            assert np.max(inpaint_box[-2:]) <= np.min(crop_size)
            
        # Fix random seed for inpainting
        g = torch.Generator('cuda').manual_seed(seed)
            
        self.modified_imgs = []
        self.modified_crops = []
        
        object_masks = {}
        object_bbox = {}
        
        tmp_inpainting_full_mask = self.inpainting_full_mask
        # Loop of SD to inpaint the image
        for ii in range(num_passes):
            # Get img_crop
            (img_crop,
             inpainting_mask,
             img_crop_bottom) = self.get_crop_around_mask(input_image,
                                                        tmp_inpainting_full_mask,
                                                        inpaint_box,
                                                        img_crop_size[ii])
            
            if sd_pipeline[ii] == 'inp':
                # Make sure image is the right size for SD
                original_crop_size = img_crop.size
                img_crop = img_crop.resize((512,512), Image.LANCZOS)
                inpainting_mask = inpainting_mask.resize((512,512), Image.BICUBIC)
                
                inpainting_mask = Image.fromarray((np.array(inpainting_mask) > 0).astype('uint8')*255)
                modified_crop = self.inpaint_pipeline(prompt=inpaint_prompt, image=img_crop,
                                 mask_image=inpainting_mask, guidance_scale=guidance_scale[ii],
                                 negative_prompt=negative_prompt,
                                 generator=g).images[0]
                self.img_crop = img_crop
                # Resize modified crop to original size 
                modified_crop = modified_crop.resize(original_crop_size, Image.BICUBIC)
                inpainting_mask = inpainting_mask.resize(original_crop_size, Image.BICUBIC)
                inpainting_mask = (np.array(inpainting_mask) > 0).astype('int')
            else:
                # This is in case other generative models like image2image want to be added.
                raise ValueError(f'Value: "{sd_pipeline[ii]}" is invalid.')
                
            self.modified_crops.append(modified_crop)
        
            
            if object_prompt is not None:
                object_masks, object_bbox = {}, {}
                for prompt in object_prompt:
                    try: # If no bbox is found grounded sam raises an error
                        boxes_filt, masks = self.run_grounded_sam(modified_crop, prompt,
                                                                  box_threshold=box_threshold,
                                                                  text_threshold=text_threshold,
                                                                  device=self.device)
                        
                        masks = masks.float()
                        for jj in range(len(masks)):
                            mask = masks[jj].cpu().numpy().astype('int')
                            # Make sure mask is not outside the inpainting box.
                            # If it is it's likely the mask is wrong.
                            # Also if mask is very small it's likely wrong.
                            if (mask * inpainting_mask).sum() / mask.sum() < 0.95:
                                print(f'Mask for {prompt} is outside the inpainting area')
                                masks[jj] = 0
                            if mask.mean() < 0.01:
                                print(f'Mask for {prompt} is too small - {mask.mean()}')
                                masks[jj] = 0
                                
                        if masks.sum() == 0:
                            print(f'All masks were invalid for {prompt}')
                            raise ValueError
                            

                        (boxes_filt_full,
                         masks_full) = self.compute_bbox_mask_full_image(
                                                                        boxes_filt,
                                                                        masks,
                                                                        self.img_h,
                                                                        self.img_w,
                                                                        img_crop_bottom
                                                                        )
                        object_masks[prompt] = masks_full
                        object_bbox[prompt] = boxes_filt_full
                        
                    except:
                        warnings.warn(f'No object was detected corresponding to: {prompt}')
                        
            # Stitch inpainted crop back to the full image     
            if stitch_mode[ii] == 'object_mask' and object_masks.keys():
                join_mask = None
                for prompt in object_masks.keys():
                    # Join all potential masks of the added object(s)
                    mask = (object_masks[prompt].float().sum(dim=0)[0] > 0).cpu().numpy().astype('float')
                    if join_mask is not None:
                        join_mask += mask
                        join_mask = (join_mask > 0).astype('float')
                    else:
                        join_mask = mask
                full_mask = True
            else:
                full_mask = False
                if stitch_mode[ii] == 'object_mask':
                    warnings.warn('Can not stitch from object mask, no added object was detected. '+
                                  'Did you specify "object_prompt"? '+
                                  'If yes, this is likely a stable diffusion generation failure.')
                join_mask = self.inpainting_full_mask

            tmp_inpainting_full_mask = join_mask
            
            input_image = self.stitch_image(input_image,
                                            modified_crop,
                                            img_crop_bottom,
                                            join_mask,
                                            blur_std=blur_std[ii],
                                            full_mask=full_mask,
                                            img_crop_size=img_crop_size[ii])
            
            self.modified_imgs.append(input_image)

        
        return input_image, object_bbox, object_masks