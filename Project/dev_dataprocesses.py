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
        # Dynamically determine the base directory (where the script is located)
        base_dir = os.path.dirname(os.path.abspath(__file__))  # Extracts the directory where the script resides and Gets the absolute path of the current script.
        if os.path.exists(os.path.join(base_dir, 'datasets')):
            dataset_original_path = os.path.join(base_dir, 'datasets', preconfigured_dataset)
        else:
            dataset_original_path = os.path.join(base_dir, 'archive (sport Balls)')
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


isDebug = True
isVerbose = True
randomState_value = 2024
# hard code for now.
dataset_name = 'sport-balls'
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image, ImageChops
from skimage import io
loaded_datasets = load_dataset(preconfigured_dataset=dataset_name)
plt.figure(figsize=(12,4))

check_dataset = loaded_datasets["train"]
check_dataset_labels = check_dataset["label"].value_counts()
print(check_dataset_labels)
pal = sns.color_palette("rocket_r", len(check_dataset_labels))
rank = check_dataset_labels.argsort().argsort()
# for label in check_dataset.index:
#     print(label)
labels_dict = check_dataset_labels.keys()
ax = check_dataset_labels.plot(kind='bar', stacked=True, color=pal)
# ax = sns.barplot(x=labels_dict, y=check_dataset, palette=pal)
#ax = sns.barplot(x=[label_dict[label] for label in check_dataset.index], y=check_dataset.values, palette=np.array(pal[::-1])[rank])
plt.title("Distribution of classes within training dataset")
plt.ylabel("# of images")
plt.xlabel("Label")
ax.set_xticklabels(ax.get_xticklabels(), rotation=60);
images_list = []
for loc_id, row_value in check_dataset.iterrows():
    print(f"**loc_id: {loc_id} \n row: -{row_value}")
    read_image = cv2.imread(row_value["filepath"])
    images_list.append(read_image.shape)

check_dataset["imagesize"] = images_list
print(check_dataset.head())

plt.figure(figsize=(12,7))

check_dataset["size"] = training_data["Image"].apply(np.size)

ax = sns.boxplot(y=[label_dict[label] for label in training_data["Label"]], x="size", data=training_data, orient="h")
plt.title("Distribution of image sizes")
plt.xticks(np.arange(0, 1900000, 200000), rotation=45)
plt.xlim(0, 1900000)
plt.ylabel("Label");
plt.ylabel("Size");

