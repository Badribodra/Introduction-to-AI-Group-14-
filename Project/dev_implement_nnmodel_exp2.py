# Work in progress
# For the sake of convenience, just single file for now.
# ============
## TIMESTAMP @ 2024-11-07T16:34:02
## start

## TIMESTAMP @ 2024-12-07T11:05:00
## author: phuocddat
## start

## end --
## end --
## TIMESTAMP @ 2024-12-11T09:01:29
## author: phuocddat
## start
# Exp with transfer learning approach
# Pretrained weight from Imagenet
# Arch: VGG16/19
# Optimizer: AdamW, Adam
## end --
# Import necessary libs
import os
import pandas as pd

import datetime

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

import dev_configuration
import dev_dataprocesses
from dev_configuration import learning_rate, model_loss_opt, epochs
import time

# Debug flags
isDebug = True
isVerbose = True
randomState_value = 2024
dataset_name = 'sport-balls'

loaded_datasets = dev_dataprocesses.load_dataset(preconfigured_dataset=dataset_name)

train_data, val_data, test_data = dev_dataprocesses.data_generators(datasource=loaded_datasets,
                                                                    data_generator='keras')

# model configuration
models_archs = ['VGG16', 'VGG19', 'EfficientNetB3', 'ModelCNNExp1']
model_sel = 'EfficientNetB3'
pretrained_weights = 'imagenet'
batch_size = 32
frozen_weights = True

# Model building
# Customized model arch

ModelCNNExp1 = keras.models.Sequential([
    keras.layers.Conv2D(filters=128, kernel_size=(8, 8), strides=(3, 3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3, 3)),

    keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2, 2)),

    keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
])


print(f"Training {model_sel} model ")

if model_sel == 'VGG16':
    base_model = keras.applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(224,224,3),
    )
elif model_sel == 'VGG19':
    base_model = keras.applications.VGG19(
        include_top=False,
        weights='imagenet',
        input_shape=(224,224,3),
    )
elif model_sel == 'EfficientNetB3':
    base_model = keras.applications.EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_shape=(224,224,3),
    )
elif model_sel == 'ModelCNNExp1':
    base_model = ModelCNNExp1
else:
    base_model = keras.applications.VGG19(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
    )
if model_sel != 'ModelCNNExp1':
# Freeze pretrained weight from base_model layers.
    base_model.trainable = not frozen_weights

    model_new = keras.models.Sequential([
        base_model,
        keras.layers.Flatten(),
        keras.layers.Dense(1024,activation='relu'),
        keras.layers.Dense(256,activation='relu'),
        keras.layers.Dense(64,activation='relu'),
        keras.layers.Dense(32,activation='relu'),

        keras.layers.Dense(15,activation='softmax')
    ])
else:
    model_new = keras.models.Sequential([
        base_model,
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(15, activation='softmax')
    ])


total_steps = len(train_data)*dev_configuration.epochs
decay_steps = total_steps * 0.7


cosine_decay_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate = learning_rate,
    decay_steps = decay_steps,
    alpha=0.1
)


model_new.compile(optimizer=tf.optimizers.AdamW(learning_rate=cosine_decay_scheduler),
                                            loss='categorical_crossentropy',
                                            metrics=['accuracy'])

model_new.summary()
os.makedirs('./checkpoints', exist_ok=True)
current_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_path = f"{current_timestamp}_aug_model_{model_sel}_AdamW_bs_{batch_size}_frozen_{str(frozen_weights)}_cosine_decay_scheduler.h5"
print(f"epochs {dev_configuration.epochs}"
      f"batchsize = {batch_size}"
      f"\n augmentation: rotation_range=0.45 "
      f"width_shift_range=0.1,"
      f"height_shift_range=0.1,"
      f"zoom_range=0.1,"
      f"horizontal_flip=True,"
      f"vertical_flip=True"
      f"\n Checkpoint: {checkpoint_path}")

my_callbacks = [
#    keras.callbacks.EarlyStopping(patience=10),
    keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    ),
    keras.callbacks.TensorBoard(log_dir='./logs'),
]
# Exp model training
start_time = time.time()

training_exper_model_eff = model_new.fit(train_data,
                                         epochs=dev_configuration.epochs,
                                         #steps_per_epoch=len(train_data) // batch_size,
                                         validation_data=val_data,
                                         verbose=1,
                                         callbacks=my_callbacks)
# Evaluate model
print(f"Evaluating on test set")
model_new.evaluate(test_data, verbose=1)

print(f"Export plot results")
current_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
fig_name = (f"{current_timestamp}_"
            f"aug_"
            f"model_{model_sel}_imagenet_frozen_{frozen_weights}_"
            f"ep{dev_configuration.epochs}_bs_{dev_configuration.batch_size}_cosine_decay_scheduler.png")

plt.plot(training_exper_model_eff.history['accuracy'])
plt.plot(training_exper_model_eff.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
# plt.show()
plt.savefig(fig_name)
elapsed_time = time.time() - start_time
print(f"{current_timestamp}: Processing time: {elapsed_time} seconds. Saved plot to {fig_name} \n")
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
