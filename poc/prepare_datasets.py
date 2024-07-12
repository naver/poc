from tqdm import tqdm
import shutil
import os
import glob
from PIL import Image
import numpy as np


def prepare_images_cs_and_idd(root_poc: str, root_original: str, root_results: str):
    poc_images = glob.glob(f"{root_poc}/leftImg8bit/*/*/*.npz")
    for poc_img_path in tqdm(poc_images):

        original_img_path = poc_img_path.split(root_poc)[-1].split("_leftImg8bit_")[0]
        original_img_path = f"{root_original}/{original_img_path}_leftImg8bit"
        
        try:
            original_img = np.array(Image.open(original_img_path+".png"))
        except:
            original_img = np.array(Image.open(original_img_path+".jpg"))
            
        poc_img = np.load(poc_img_path)["img"]
        
        new_img = Image.fromarray(poc_img + original_img)
        
        new_img_path = poc_img_path.split(root_poc)[-1].replace(".npz", ".png")
        new_img_path = f"{root_results}/{new_img_path}"
        os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
        
        new_img.save(new_img_path)

        poc_gt_path = poc_img_path.replace("/leftImg8bit/", "/gtFine/")
        poc_gt_path = poc_gt_path.replace(".npz", ".png")
        
        new_gt_path = new_img_path.replace("/leftImg8bit/", "/gtFine/")
        
        os.makedirs(os.path.dirname(new_gt_path), exist_ok=True)
        shutil.copyfile(poc_gt_path, new_gt_path)


def prepare_images_acdc(root_poc: str, root_acdc: str, root_results: str):
    poc_images = glob.glob(f"{root_poc}/rgb_anon_trainvaltest/*/*/*/*/*.npz")
    for poc_img_path in tqdm(poc_images):

        acdc_img_path = poc_img_path.split(root_poc)[-1].split("_rgb_anon_")[0]
        acdc_img_path = f"{root_acdc}/{acdc_img_path}_rgb_anon.png"
        
        acdc_img = np.array(Image.open(acdc_img_path))
        poc_img = np.load(poc_img_path)["img"]
        
        new_img = Image.fromarray(poc_img + acdc_img)
        
        new_img_path = poc_img_path.split(root_poc)[-1].replace(".npz", ".png")
        new_img_path = f"{root_results}/{new_img_path}"
        os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
        
        new_img.save(new_img_path)

        poc_gt_path = poc_img_path.replace("/rgb_anon_trainvaltest/", "/gt_trainval/")
        poc_gt_path = poc_gt_path.replace("/rgb_anon/", "/gt/")
        poc_gt_path = poc_gt_path.replace(".npz", ".png")
        
        new_gt_path = new_img_path.replace("/rgb_anon_trainvaltest/", "/gt_trainval/")
        new_gt_path = new_gt_path.replace("/rgb_anon/", "/gt/")
        
        os.makedirs(os.path.dirname(new_gt_path), exist_ok=True)
        shutil.copyfile(poc_gt_path, new_gt_path)


if __name__ == "__main__":

    root_data = "PATH/TO/FOLDER/WITH/DOWNLOADED/DATA/"
    root_cs = "PATH/TO/CITYSCAPES/FOLDER/"
    root_idd = "PATH/TO/IDD/FOLDER/"
    root_acdc = "PATH/TO/ACDC/FOLDER/"
    
    print("Preparing ACDC POC")
    root_poc = f"{root_data}/POC_ACDC_VAL/"
    root_results = f"{root_data}/POC_ACDC_VAL_RECONSTRUCTED/"
    prepare_images_acdc(root_poc, root_acdc, root_results)

    print("Preparing CS POC")
    root_poc = f"{root_data}/POC_CS_VAL/"
    root_results = f"{root_data}/POC_CS_VAL_RECONSTRUCTED/"
    prepare_images_cs_and_idd(root_poc, root_cs, root_results)

    # print("Preparing IDD POC")
    # root_poc = f"{root_data}/POC_IDD_VAL/"
    # root_results = f"{root_data}/POC_IDD_VAL_RECONSTRUCTED/"
    # prepare_images_cs_and_idd(root_poc, root_idd, root_results)

    print("Preparing POC Alt finetuning")
    root_poc = f"{root_data}/POC_CS_ALT_FT/"
    root_results = f"{root_data}/POC_CS_ALT_RECONSTRUCTED/"
    prepare_images_cs_and_idd(root_poc, root_cs, root_results)

    print("Preparing POC COCO finetuning")
    root_poc = f"{root_data}/POC_CS_COCO_FT/"
    root_results = f"{root_data}/POC_CS_COCO_RECONSTRUCTED/"
    prepare_images_cs_and_idd(root_poc, root_cs, root_results)

    print("Preparing POC Animal")
    root_poc = f"{root_data}/POC_ANIMAL/"
    root_results = f"{root_data}/POC_ANIMAL_RECONSTRUCTED/"
    prepare_images_cs_and_idd(root_poc, root_cs, root_results)

    print("Preparing POC CS + Animal")
    root_poc = f"{root_data}/POC_CS_ANIMAL/"
    root_results = f"{root_data}/POC_CS_ANIMAL_RECONSTRUCTED/"
    prepare_images_cs_and_idd(root_poc, root_cs, root_results)

    print("Preparing POC Animal eval")
    root_poc = f"{root_data}/POC_ANIMAL_VAL/"
    root_results = f"{root_data}/POC_ANIMAL_VAL_RECONSTRUCTED/"
    prepare_images_cs_and_idd(root_poc, root_cs, root_results)

    

    
    