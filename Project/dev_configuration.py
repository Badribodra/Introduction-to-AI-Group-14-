# default configuration

randomState_value = 2024
isverbose = True
datasets  = ['chest_xray_small', 'chest-xray-full', 'sport-balls']
select_dataset = 2
dataset_original_path = f"./datasets/{datasets[select_dataset]}"
test_split_value = 0.2
is_train_test_only = True
val_split_value = 0.3
val_test_split_value = 0.5
default_xcol_value = 'filepath'
default_ycol_value = 'label'
batch_size = 8
isShuffle = False

#Model configuration
model_loss_opt = 'categorical_crossentropy'
learning_rate = 0.001
## TIMESTAMP @ 2024-12-13T17:53:55
## author: phuocddat
## start
# Test new strategies with moderate epochs
epochs = 200
## end --

