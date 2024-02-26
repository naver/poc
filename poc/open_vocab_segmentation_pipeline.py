import numpy as np
import torch

from grounded_sam_utils import *
from utils_sd import *
from path_utils import *

class OpenSegmentor:
    
    def __init__(self,
                 gdino_ckpt=None,
                 gdino_config=None,
                 sam_ckpt=None,
                 device='cuda'
                ):
        
        self.device = device
        # Load GroundingDINO model
        self.gdino_model = load_model(gdino_config, gdino_ckpt, device='cpu')

        # SAM model
        self.sam_model = SamPredictor(build_sam(checkpoint=sam_ckpt).to(self.device))
        

    def run_grounded_sam(self, input_image, prompt, box_threshold=0.3, text_threshold=0.25, device=None):
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
        boxes_filt, pred_phrases, logits = get_grounding_output_with_logits(
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
                            ).to(self.device)

        masks, _, _ = self.sam_model.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        )
        
        return boxes_filt, masks, logits
    
    
    def segment_image(self, input_image, prompt, box_threshold=0.3,
                                        text_threshold=0.25):
        '''
        Descr: Function to find the valid area based on grounded sam and a location prompt.
        Params: 
            -input_image: PIL image to be segmented/detected.
            -prompt: text prompt to segment.
            -box_threshold: Hyperparam from GroundingDINO. Using their default.
            -text_threshold: Hyperparam from GroundingDINO. Using their default.
        Returns:
            valid_area: np array with binary mask corresponding to location prompt.
        '''
        try:
            _, masks, logits = self.run_grounded_sam(input_image, prompt,
                                            box_threshold=box_threshold,
                                            text_threshold=text_threshold,
                                            device=self.device)
            # Join masks if more than one
            self.mask = (masks.float().cpu() * torch.tensor(logits).reshape(len(logits), 1, 1, 1))
            self.mask = self.mask.max(dim=0)[0].cpu().numpy().astype('float')
        except:
            self.mask = np.zeros((input_image.size[1], input_image.size[0]))
        
        return self.mask