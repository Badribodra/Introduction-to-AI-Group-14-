import os
import pandas as pd
import dev_configuration
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def load_dataset(
        preconfigured_dataset = 'Chest',
        datapaths = None,
        labels = None,
        dataset_original_path = "",
        isverbose = False,
):
    if preconfigured_dataset != 'Chest':
        return print(f"Not implemented")
    if preconfigured_dataset == 'Chest':
        labels = dev_configuration.labels
        dataset_original_path = f"./datasets/{dev_configuration.datasets[dev_configuration.select_dataset]}"
        isverbose = dev_configuration.isverbose
        print(f"Processing {dataset_original_path}")


    datapaths = {}
    for label in labels:
        datapaths[label] = [
            f"{dataset_original_path}/train/{label}",
            f"{dataset_original_path}/val/{label}",
            f"{dataset_original_path}/test/{label}"
        ]
    files_fullpaths = []
    labels_list = []
    if isverbose:
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

    return pddata

def data_generators(datasource: None, data_generator:'keras'):

    if datasource is None:
        return print('Datasource is None')

    # default configuration
    test_split_value = dev_configuration.test_split_value
    val_split_value = dev_configuration.val_split_value
    randomState_value = dev_configuration.randomState_value
    default_xcol_value = dev_configuration.default_xcol_value
    default_ycol_value = dev_configuration.default_ycol_value
    batch_size = dev_configuration.batch_size
    isShuffle = dev_configuration.isShuffle

    print(len(datasource))
    print(datasource.head())
    print(datasource.shape)
    print(datasource["label"].value_counts())

    # Train val test split
    # Split train_val:test
    train_val_images, test_images = train_test_split(
        datasource,
        test_size=test_split_value,
        random_state=randomState_value)
    # Split train:val
    train_set, val_set = train_test_split(
        train_val_images,
        test_size=val_split_value,
        random_state=randomState_value)

    print(train_set.shape)
    print(test_images.shape)
    print(val_set.shape)
    print(train_val_images.shape)

    if data_generator == 'keras':
        image_gen = ImageDataGenerator()
        train_data = image_gen.flow_from_dataframe(dataframe=train_set,
                                                   x_col=default_xcol_value,
                                                   y_col=default_ycol_value,
                                                   target_size=(244, 244),
                                                   color_mode='rgb',
                                                   class_mode="categorical",
                                                   batch_size=batch_size,
                                                   shuffle=isShuffle
                                                   )
        test_data = image_gen.flow_from_dataframe(dataframe=test_images,
                                                  x_col=default_xcol_value,
                                                  y_col=default_ycol_value,
                                                  target_size=(244, 244),
                                                  color_mode='rgb',
                                                  class_mode="categorical",
                                                  batch_size=batch_size,
                                                  shuffle=isShuffle
                                                  )
        val_data = image_gen.flow_from_dataframe(dataframe=val_set,
                                                 x_col=default_xcol_value,
                                                 y_col=default_ycol_value,
                                                 target_size=(244, 244),
                                                 color_mode='rgb',
                                                 class_mode="categorical",
                                                 batch_size=batch_size,
                                                 shuffle=isShuffle
                                                 )
        return train_data, val_data, test_data



