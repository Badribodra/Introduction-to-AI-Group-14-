# Work in progress
# For the sake of convenience, just single file for now.
# ============
## TIMESTAMP @ 2024-11-07T16:34:02
## start

## end --
## TIMESTAMP @ 2024-12-07T19:01:46
## author: phuocddat
## start

## end --
# Import necessary libs
import os
import pandas as pd


from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import dev_dataprocesses
from dev_configuration import learning_rate, model_loss_opt

# Debug flags
isDebug = True
isVerbose = True
randomState_value = 2024
# hard code for now.
dataset_name = 'sport-balls'

loaded_datasets = dev_dataprocesses.load_dataset(preconfigured_dataset=dataset_name)

train_data, val_data, test_data = dev_dataprocesses.data_generators(datasource=loaded_datasets,
                                                                    data_generator='keras')

# Model building
# experimental model architecture number 1
"""
ModelArchName: CNN_exp_no1
Note: number of classes is harcoded. Should be variable. 
"""
# training from scratch....

model_exp_1 = keras.models.Sequential([
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
    keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(15, activation='softmax')
])

## TIMESTAMP @ 2024-12-07T19:02:58
## author: phuocddat
## start
"""
model compile 
arch: CNN_exp_no1
loss: model_loss_opt (from dev_configuration)
optimizer: SGD with lr from dev_configuration
metrics: ['accuracy']
"""
## end --
model_exp_1.compile(
    loss=model_loss_opt,
    optimizer=tf.optimizers.SGD(learning_rate=learning_rate),
    metrics=['accuracy']
)

# get summary of model.
model_exp_1.summary()


# Exp model training
# hardcode everything.
training_exper = model_exp_1.fit(train_data,
                                 epochs=10,
                                 validation_data=val_data,
                                 verbose=1)
# Evaluate model
model_exp_1.evaluate(test_data, verbose=1)

plt.plot(training_exper.history['accuracy'])
plt.plot(training_exper.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
