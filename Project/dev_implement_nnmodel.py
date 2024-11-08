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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential

# Dataset Preparation
# TODO: should be dedicated classes/functions

# manual configuration
# chest-xray-small:
# link: https://www.kaggle.com/datasets/pcbreviglieri/pneumonia-xray-images/data
# chest-xray-full:
# link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data
datasets  = ['chest-xray-small', 'chest-xray-full']
select_dataset = 0
dataset_original_path = f"./datasets/{datasets[select_dataset]}"

print(dataset_original_path)

# Setup labels, data paths, etc...
num_classes = 2
labels = ['Normal', 'Opacity']

data_paths = {'normal': [
                    f"{dataset_original_path}/train/normal",
                    f"{dataset_original_path}/val/normal",
                    f"{dataset_original_path}/test/normal"
                        ],
             'opacity': [
                    f"{dataset_original_path}/train/opacity",
                    f"{dataset_original_path}/val/opacity",
                    f"{dataset_original_path}/test/opacity"
             ]}

for k,v in data_paths.items():
    print(k, v)
data2_paths = {}
for label in labels:
    data2_paths[label] = [
        f"{dataset_original_path}/train/{label}",
        f"{dataset_original_path}/val/{label}",
        f"{dataset_original_path}/test/{label}"
    ]

for k,v in data2_paths.items():
    print(k, v)








