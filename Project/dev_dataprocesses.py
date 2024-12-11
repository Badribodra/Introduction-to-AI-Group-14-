import os
import pandas as pd
import dev_configuration
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def load_dataset(
        preconfigured_dataset = 'sport-balls',
        datapaths =dict(),
        labels = None,
        dataset_original_path = "",
        isverbose = False,
):
    """

    :param preconfigured_dataset:
    :param datapaths:
    :param labels:
    :param dataset_original_path:
    :param isverbose:
    :return:
    """
    isverbose = dev_configuration.isverbose
    if preconfigured_dataset in dev_configuration.datasets:
        dataset_original_path = f"./datasets/{preconfigured_dataset}"
    else:
        return print(f"Invalid preconfigured dataset: {preconfigured_dataset}")

    print(f"Processing {dataset_original_path}")
    datasource_dict = {
        "train": None,
        "test": None,
        "val": None
    }

    data_splits = os.listdir(dataset_original_path)
    for split in data_splits:
        files_fullpaths2 = []
        labels_list2 = []
        print(f"Processing {split} set")
        split_path = os.path.join(dataset_original_path, split)
        if os.path.isdir(f"{split_path}"):
            for label in os.listdir(f"{split_path}"):
                label_path = os.path.join(split_path, label)
                files_list = os.listdir(label_path)
                files_fullpaths2.extend([os.path.join(label_path, fpath) for fpath in files_list])
                labels_list2.extend([label] * len(files_list))

        # put in pandas dataframe
        pd_data = pd.DataFrame(list(zip(files_fullpaths2, labels_list2)), columns=['filepath', 'label'])
        if split in datasource_dict.keys():
            datasource_dict[split] = pd_data
        else:
            print(f"split {split} not in datasource_dict")


    #pd_data = pd.DataFrame(list(zip(files_fullpaths2, labels_list2)), columns=['filepath', 'label'])
    if isverbose:
        print(f" processes completed"
              f"\n {len(datasource_dict)}")
        for split in datasource_dict:
            pd_data_item = datasource_dict[split]
            if pd_data_item is not None:
                print(f"\n Label: {split} "
                      f"\nLength is : {len(pd_data_item)}"
                      f"\nSample head: \n {pd_data_item.head()}"
                      f"\nSample shape: \n {pd_data_item.shape}"
                      f"\nValue counts: \n {pd_data_item['label'].value_counts()}"
                      f"==================")
            else:
                print(f"\n Label: {split} is None")

    return datasource_dict

"""
Data generators helpers
1. Keras data generator
2. Sklearn generator

"""

def data_generators(datasource: None, data_generator:'keras'):
    """

    :param datasource:
    :param data_generator:
    :return:
    """
    if datasource is None:
        return print('Datasource is None')

    is_train_test_only = False
    # default configuration
    test_split_value = dev_configuration.test_split_value
    val_split_value = dev_configuration.val_split_value
    randomState_value = dev_configuration.randomState_value
    default_xcol_value = dev_configuration.default_xcol_value
    default_ycol_value = dev_configuration.default_ycol_value
    batch_size = dev_configuration.batch_size
    isShuffle = dev_configuration.isShuffle
    train_set = None

    print(f" processing from datasources:"
          f"\n datasource: {len(datasource)}")
    print(len(datasource))
    for key_item in datasource.keys():
        if datasource[key_item] is not None:
            print(f"data split {key_item}")
            print(datasource[key_item].head())
            print(datasource[key_item].shape)
            print(datasource[key_item]["label"].value_counts())
        else:
            print(f"data split {key_item} is None")
            is_train_test_only = True


    # Train val test split
    # Split train_val:test
    # train_val_images, test_images = train_test_split(
    #     datasource,
    #     test_size=test_split_value,
    #     random_state=randomState_value)
    # Split train:val
    if is_train_test_only:
        train_val_images = datasource["train"]
        test_images = datasource["test"]
        train_set, val_set = train_test_split(
            datasource["train"],
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



