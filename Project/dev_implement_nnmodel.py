# Work in progress
# For the sake of convenience, just single file for now.
# ============
## TIMESTAMP @ 2024-11-07T16:34:02
## start

## end --
# Import necessary libs
import os
import pandas as pd

import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Sequential

# Debug flags
isDebug = True
isVerbose = True

# Dataset Preparation
# TODO: should be dedicated classes/functions

# manual configuration
# chest-xray-small:
# link: https://www.kaggle.com/datasets/pcbreviglieri/pneumonia-xray-images/data
# chest-xray-full:
# link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data
datasets  = ['chest_xray_small', 'chest-xray-full']
select_dataset = 0
dataset_original_path = f"./datasets/{datasets[select_dataset]}"


if isVerbose: print(dataset_original_path)

# Setup labels, data paths, etc...
num_classes = 2
labels = ['normal', 'opacity']
#
# data_paths = {'normal': [
#                     f"{dataset_original_path}/train/normal",
#                     f"{dataset_original_path}/val/normal",
#                     f"{dataset_original_path}/test/normal"
#                         ],
#              'opacity': [
#                     f"{dataset_original_path}/train/opacity",
#                     f"{dataset_original_path}/val/opacity",
#                     f"{dataset_original_path}/test/opacity"
#              ]}
#
# for k,v in data_paths.items():
#     print(k, v)
datapaths = {}
for label in labels:
    datapaths[label] = [
        f"{dataset_original_path}/train/{label}",
        f"{dataset_original_path}/val/{label}",
        f"{dataset_original_path}/test/{label}"
    ]

if isVerbose:
    for k,v in datapaths.items():
        print(k, v)
        for v_item in v:
            print(f"{v_item} is {os.path.exists(v_item)}")
            files_list = os.listdir(v_item)
            files_fullpaths = [os.path.join(v_item, fpath) for fpath in files_list]
            print(files_fullpaths)






