# manual configuration
# chest-xray-small:
# link: https://www.kaggle.com/datasets/pcbreviglieri/pneumonia-xray-images/data
# chest-xray-full:
# link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data
randomState_value = 2024
isverbose = True
datasets  = ['chest_xray_small', 'chest-xray-full']
select_dataset = 0
dataset_original_path = f"./datasets/{datasets[select_dataset]}"
test_split_value = 0.2
val_split_value = 0.3
default_xcol_value = 'filepath'
default_ycol_value = 'label'
batch_size = 4
isShuffle = False


#Model configuration
model_loss_opt = 'categorical_crossentropy'
learning_rate = 0.001

# Setup labels, data paths, etc...
num_classes = 2
labels = ['normal', 'opacity']
