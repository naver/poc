# Installation

The project is based on the pytorch 1.11.0 with python 3.8.

## 1. Clone the Git  repo

``` shell
$ git clone https://github.com/yyliu01/RPL.git
$ cd RPL
```

## 2. Install dependencies

1) create conda env
    ```shell
    $ conda env create -f rpl.yml
    ```
2) install the torch 1.11.0
    ```shell
    $ conda activate rpl
    # IF cuda version < 11.0
    $ pip install torch==1.11.0+cu102 torchvision==0.12.0+cu102 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu102
    # IF cuda version >= 11.0 (e.g., 30x or above)
    $ pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
    ```

## 3. Prepare dataset

### cityscapes

1) please download the Cityscapes dataset (gt_Fine).
2) (optional) you might need to preprocess Cityscapes dataset
   in [here](https://github.com/mcordts/cityscapesScripts/tree/master/cityscapesscripts/preparation), as we follow the
   common setting with **19** classes. 
3) specify the Cityscapes dataset path in **code/config/config.py** file, which is **C.city_root_path**.

### fishyscapes

1) for the time being, you can download from the official website in [here](https://fishyscapes.com/dataset).
2) specify the coco dataset path in **code/config/config.py** file, which is **C.fishy_root_path**.

*You can alternatively download both preprocessed fishyscapes & cityscapes datasets* from [here](https://drive.google.com/drive/u/0/folders/1_5rZgWEDT9VbWch5dTkXWAOddUGqMmgE).

### coco (for outlier exposures)

1) please follow [Meta-OoD](https://github.com/robin-chan/meta-ood/tree/master/preparation) to prepare the COCO images
   serving as OoD proxy for OoD
   training. [This script](https://github.com/robin-chan/meta-ood/blob/master/preparation/prepare_coco_segmentation.py)
   generates binary segmentation masks for COCO images not containing any instances that could also be assigned to one
   of the Cityscapes (train-)classes. Execute via:
   ```shell 
   $ python preparation/prepare_coco_segmentation.py
   ```
2) specify the coco dataset path in **code/config/config.py** file, which is **C.coco_root_path**.

## 4. Dataset Structure

1) the tree structures of the training datasets (including both cityscapes, and coco) are shown below.

```shell
city_scape/
├── annotation
│   └── city_gt_fine
│       ├── train
│       └── val
└── images
    └── city_gt_fine
        ├── train
        └── val
coco/
├── annotations
│   └── ood_seg_train2017
└── train2017
```
2) the tree sturcutres of all the validation datasets are shown in below.
```shell
fishyscapes/
├── LostAndFound
│   ├── entropy
│   ├── labels
│   ├── labels_with_ROI
│   ├── logit_distance
│   ├── mae_features
│   ├── original
│   ├── semantic
│   └── synthesis
└── Static
    ├── entropy
    ├── labels
    ├── labels_with_ROI
    ├── logit_distance
    ├── mae_features
    ├── original
    └── semantic
    
segment_me
├── dataset_AnomalyTrack
│   ├── images
│   └── labels_masks
└── dataset_ObstacleTrack
    ├── images
    └── labels_masks

road_anomaly/
├── entropy
├── labels
├── mae_features
├── original
├── semantic
└── synthesis

lost_and_found/
├── gtCoarse
│   ├── test
│   │   ├── 02_Hanns_Klemm_Str_44
│   │   ├── 04_Maurener_Weg_8
│   │   ├── 05_Schafgasse_1
│   │   ├── 07_Festplatz_Flugfeld
│   │   └── 15_Rechbergstr_Deckenpfronn
│   └── train
│       ├── 01_Hanns_Klemm_Str_45
│       ├── 03_Hanns_Klemm_Str_19
│       ├── 06_Galgenbergstr_40
│       ├── 10_Schlossberg_9
│       ├── 11_Parkplatz_Flugfeld
│       ├── 12_Umberto_Nobile_Str
│       ├── 13_Elly_Beinhorn_Str
│       └── 14_Otto_Lilienthal_Str_24
└── leftImg8bit
    ├── test
    │   ├── 02_Hanns_Klemm_Str_44
    │   ├── 04_Maurener_Weg_8
    │   ├── 05_Schafgasse_1
    │   ├── 07_Festplatz_Flugfeld
    │   └── 15_Rechbergstr_Deckenpfronn
    └── train
        ├── 01_Hanns_Klemm_Str_45
        ├── 03_Hanns_Klemm_Str_19
        ├── 06_Galgenbergstr_40
        ├── 10_Schlossberg_9
        ├── 11_Parkplatz_Flugfeld
        ├── 12_Umberto_Nobile_Str
        ├── 13_Elly_Beinhorn_Str
        └── 14_Otto_Lilienthal_Str_24

```


