from tqdm import tqdm
import shutil
import os
import glob
from PIL import Image
import numpy as np

def extract_diff_images_cs_and_idd(root_poc: str, root_original: str, root_results: str, subfolder: str="*"):
    poc_images = glob.glob(f"{root_poc}/leftImg8bit/{subfolder}/*/*.png")
    for poc_img_path in tqdm(poc_images):
        
        original_img_path = poc_img_path.split(root_poc)[-1].split("_leftImg8bit_")[0]
        original_img_path = f"{root_original}/{original_img_path}_leftImg8bit"

        try:
            original_img = np.array(Image.open(original_img_path+".png"))
        except:
            original_img = np.array(Image.open(original_img_path+".jpg"))
        poc_img = np.array(Image.open(poc_img_path))
        
        diff_img = poc_img - original_img
        
        diff_img_path = poc_img_path.split(root_poc)[-1].replace(".png", ".npz")
        diff_img_path = f"{root_results}/{diff_img_path}"
        os.makedirs(os.path.dirname(diff_img_path), exist_ok=True)
        
        np.savez_compressed(diff_img_path, img=diff_img)

        poc_gt_path = poc_img_path.replace("/leftImg8bit/", "/gtFine/")
        
        diff_gt_path = diff_img_path.replace("/leftImg8bit/", "/gtFine/")
        diff_gt_path = diff_gt_path.replace(".npz", ".png")
        os.makedirs(os.path.dirname(diff_gt_path), exist_ok=True)
        shutil.copyfile(poc_gt_path, diff_gt_path)
        

def extract_diff_images_acdc(root_poc: str, root_acdc: str, root_results: str):
    poc_images = glob.glob(f"{root_poc}/rgb_anon_trainvaltest/*/*/*/*/*.png")
    for poc_img_path in tqdm(poc_images):

        acdc_img_path = poc_img_path.split(root_poc)[-1].split("_rgb_anon_")[0]
        acdc_img_path = f"{root_acdc}/{acdc_img_path}_rgb_anon.png"
        
        acdc_img = np.array(Image.open(acdc_img_path))
        poc_img = np.array(Image.open(poc_img_path))
        
        diff_img = poc_img - acdc_img
        
        diff_img_path = poc_img_path.split(root_poc)[-1].replace(".png", ".npz")
        diff_img_path = f"{root_results}/{diff_img_path}"
        os.makedirs(os.path.dirname(diff_img_path), exist_ok=True)
        
        np.savez_compressed(diff_img_path, img=diff_img)

        poc_gt_path = poc_img_path.replace("/rgb_anon_trainvaltest/", "/gt_trainval/")
        poc_gt_path = poc_gt_path.replace("/rgb_anon/", "/gt/")
        
        diff_gt_path = diff_img_path.replace("/rgb_anon_trainvaltest/", "/gt_trainval/")
        diff_gt_path = diff_img_path.replace("/rgb_anon/", "/gt/")
        
        diff_gt_path = diff_gt_path.replace(".npz", ".png")
        os.makedirs(os.path.dirname(diff_gt_path), exist_ok=True)
        shutil.copyfile(poc_gt_path, diff_gt_path)

if __name__ == "__main__":
    
    # print("Preparing ACDC POC")
    # root_poc = "/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/acdc_ood/"
    # root_acdc = "/gfs-ssd/project/clara/data/ACDC/"
    # root_results = "/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/POC_ACDC_VAL/"
    # extract_diff_images_acdc(root_poc, root_acdc, root_results)

    # print("Preparing CS POC")
    # root_poc = "/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/cs_synth_ood_val/"
    # root_cs = "/gfs-ssd/project/clara/data/Cityscapes/"
    # root_results = "/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/POC_CS_VAL/"
    # extract_diff_images_cs_and_idd(root_poc, root_cs, root_results)

    # print("Preparing IDD POC")
    # root_poc = "/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/idd_ood/"
    # root_idd = "/gfs-ssd/project/uss/data/IDD/IDD_Segmentation/"
    # root_results = "/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/POC_IDD_VAL/"
    # extract_diff_images_cs_and_idd(root_poc, root_idd, root_results)

    # print("Preparing CS POC COCO finetuning")
    # root_poc = "/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/cs_synth_coco_classes/"
    # root_cs = "/gfs-ssd/project/clara/data/Cityscapes/"
    # root_results = "/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/POC_CS_COCO_FT/"
    # subfolder = "train"
    # extract_diff_images_cs_and_idd(root_poc, root_cs, root_results, subfolder)

    # print("Preparing CS POC Alt finetuning")
    # root_poc = "/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/cs_synth_classes_all/"
    # root_cs = "/gfs-ssd/project/clara/data/Cityscapes/"
    # root_results = "/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/POC_CS_ALT_FT/"
    # subfolder = "train"
    # extract_diff_images_cs_and_idd(root_poc, root_cs, root_results, subfolder)

    # print("Preparing CS POC Pascal")
    # root_poc = "/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/cs_synth_pascal"
    # root_cs = "/gfs-ssd/project/clara/data/Cityscapes/"
    # root_results = "/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/POC_ANIMAL/"
    # subfolder = "train"
    # extract_diff_images_cs_and_idd(root_poc, root_cs, root_results, subfolder)

    # print("Preparing CS POC CS and Pascal")
    # root_poc = "/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/cs_and_pascal_synth"
    # root_cs = "/gfs-ssd/project/clara/data/Cityscapes/"
    # root_results = "/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/POC_CS_ANIMAL/"
    # subfolder = "train"
    # extract_diff_images_cs_and_idd(root_poc, root_cs, root_results, subfolder)

    # print("Preparing CS POC Pascal val")
    # root_poc = "/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/cs_synth_pascal_val"
    # root_cs = "/gfs-ssd/project/clara/data/Cityscapes/"
    # root_results = "/gfs-ssd/project/clara/data-new/Cityscapes_SD_aug/POC_ANIMAL_VAL/"
    # subfolder = "val"
    # extract_diff_images_cs_and_idd(root_poc, root_cs, root_results, subfolder)


    