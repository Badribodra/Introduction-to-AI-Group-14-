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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Debug flags
isDebug = True
isVerbose = True
randomState_value = 2024

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
test_split_value = 0.2
val_split_value = 0.3
default_xcol_value = 'filepath'
default_ycol_value = 'label'
batch_size = 4
isShuffle = False

if isVerbose: print(dataset_original_path)

# Setup labels, data paths, etc...
num_classes = 2
labels = ['normal', 'opacity']

datapaths = {}
for label in labels:
    datapaths[label] = [
        f"{dataset_original_path}/train/{label}",
        f"{dataset_original_path}/val/{label}",
        f"{dataset_original_path}/test/{label}"
    ]
files_fullpaths = []
labels_list = []
if isVerbose:
    for k,v in datapaths.items():
        for v_item in v:
            print(f"label {k} : {v_item} is {os.path.exists(v_item)}")
            files_list = os.listdir(v_item)
            ffulpaths = [os.path.join(v_item, fpath) for fpath in files_list]
            #TODO: Need to check file valid
            files_fullpaths.extend(ffulpaths)
            labels_list.extend([k] * len(ffulpaths))
            # print(f"size of files {len(ffulpaths)}")

        #print(labels_list)

pddata = pd.DataFrame(list(zip(files_fullpaths, labels_list)), columns=['filepath', 'label'])

print(len(pddata))
print(pddata.head())
print(pddata.shape)
print(pddata["label"].value_counts())

# Train test split
train_val_images, test_images = train_test_split(
    pddata,
    test_size=test_split_value,
    random_state=randomState_value)
train_set, val_set = train_test_split(
    train_val_images,
    test_size=val_split_value,
    random_state=randomState_value)

print(train_set.shape)
print(test_images.shape)
print(val_set.shape)
print(train_val_images.shape)

image_gen = ImageDataGenerator(preprocessing_function= tf.keras.applications.mobilenet_v2.preprocess_input)
train = image_gen.flow_from_dataframe(dataframe= train_set,
                                      x_col=default_xcol_value,
                                      y_col=default_ycol_value,
                                      target_size=(244,244),
                                      color_mode='rgb',
                                      class_mode="categorical",
                                      batch_size=batch_size,
                                      shuffle=isShuffle
                                     )
test = image_gen.flow_from_dataframe(dataframe= test_images,
                                     x_col=default_xcol_value,
                                     y_col=default_ycol_value,
                                     target_size=(244,244),
                                     color_mode='rgb',
                                     class_mode="categorical",
                                     batch_size=batch_size,
                                     shuffle= isShuffle
                                    )
val = image_gen.flow_from_dataframe(dataframe= val_set,
                                    x_col=default_xcol_value,
                                    y_col=default_ycol_value,
                                    target_size=(244,244),
                                    color_mode= 'rgb',
                                    class_mode="categorical",
                                    batch_size=batch_size,
                                    shuffle=isShuffle
                                   )




