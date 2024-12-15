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
    val_test_split_value = dev_configuration.val_test_split_value
    randomState_value = dev_configuration.randomState_value
    default_xcol_value = dev_configuration.default_xcol_value
    default_ycol_value = dev_configuration.default_ycol_value
    batch_size = dev_configuration.batch_size
    isShuffle = dev_configuration.isShuffle
    train_set = None
    train_data = val_data = test_data = None

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


    # [Train val test] split
    # if all in one dataset:
    # train:val:test = _:val_split_value:test_split_value
    # train_val_images, test_images = train_test_split(
    #     datasource,
    #     test_size=test_split_value,
    #     random_state=randomState_value)
    # if only train_test dataset:
    # train:test => train:[val:test]. Split value is val_test_split_value.
    if is_train_test_only:
        train_val_images = datasource["train"]
        test_images = datasource["test"]
        train_set, val_set = train_test_split(
            datasource["train"],
            test_size=val_split_value,
            random_state=randomState_value)

        print(f"\nTrain length: {train_val_images.shape}")
        print(f"\n test length: {test_images.shape}")
        print(f"\n Train length {train_set.shape}")
        print(f"\n Val length: {val_set.shape}")
        print(f"\n Test length: {test_images.shape}")
    else:
        print(f"Not implemented")

    if data_generator == 'keras':
        ## TIMESTAMP @ 2024-12-12T01:32:02
        ## author: phuocddat
        ## start
        # try augmentation
        ## end --
        # image_gen = ImageDataGenerator()
        #

        train_gen = ImageDataGenerator(
            rescale=1. / 255,
            #                            rotation_range=0.45,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.35,
                            horizontal_flip=True,
                            vertical_flip=True
            )
        val_gen = ImageDataGenerator(
            rescale=1. / 255,
            #                          rotation_range=0.45,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     zoom_range=0.35,
                                     horizontal_flip=True,
                                     vertical_flip=True
                                     )
        ## TIMESTAMP @ 2024-12-13T17:53:21
        ## author: phuocddat
        ## start
        # Disable augmentation for test gen
        ## end --
        ts_gen = ImageDataGenerator(
            rescale=1. / 255,
                                    # # rotation_range=0.45,
                                    # width_shift_range=0.1,
                                    # height_shift_range=0.1,
                                    # zoom_range=0.1,
                                    # horizontal_flip=True,
                                    # vertical_flip=True
                                    )

        train_data = train_gen.flow_from_dataframe(dataframe=train_set,
                                                   x_col=default_xcol_value,
                                                   y_col=default_ycol_value,
                                                   target_size=(224, 224),
                                                   color_mode='rgb',
                                                   class_mode="categorical",
                                                   batch_size=batch_size,
                                                   shuffle=isShuffle
                                                   )
        val_data = val_gen.flow_from_dataframe(dataframe=val_set,
                                                 x_col=default_xcol_value,
                                                 y_col=default_ycol_value,
                                                 target_size=(224, 224),
                                                 color_mode='rgb',
                                                 class_mode="categorical",
                                                 batch_size=batch_size,
                                                 shuffle=isShuffle
                                                 )
        test_data = ts_gen.flow_from_dataframe(dataframe=test_images,
                                                  x_col=default_xcol_value,
                                                  y_col=default_ycol_value,
                                                  target_size=(224, 224),
                                                  color_mode='rgb',
                                                  class_mode="categorical",
                                                  batch_size=batch_size,
                                                  shuffle=isShuffle
                                                  )

    else:
        print(f"Not implemented")

    return train_data, val_data, test_data



