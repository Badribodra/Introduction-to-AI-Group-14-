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
## end --
## TIMESTAMP @ 2024-12-13T00:20:35
## end --
# Import necessary libs
import os, io
import datetime
import time
import logging
import random
import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

import dev_configuration
import dev_dataprocesses
from dev_configuration import learning_rate


# Generate unique exp string
digits = random.choices(string.digits, k=2)
letters = random.choices(string.ascii_uppercase, k=9)
uniq_sample = f"".join(random.sample(digits + letters, 9))

models_archs = ['ModelCNNExp1', 'ModelCNNExp2', 'ModelCNNExp3', 'ModelCNNExp4']
# Setting up folders and logging
exp_base_dir = f"./experiments/" #default folder to save artifacts.
model_sel = 'ModelCNNExp1' # Selection of model architecture
# Options configuration for model architecture
config_output_arc = 'config_1' # last layers settings
kernel_l_value = 0.001 # l2 value for kernel reg.
act_l_value = 0.001 # value for activity reg.

# options strings preformat for logging purpose.
augmentation_options = f"rescale width_shift_range height_shift_range zoom_range horizontal_flip=True vertical_flip=True"
# augmentation_options = f"augs_rescale rot=no horizontal_flip=True vertical_flip=True"
#augmentation_options = f"augs_rescale rot=no horizontal_flip=no vertical_flip=no"
other_options = f"_{config_output_arc}_activity_reg_{act_l_value}_kernel_reg_{kernel_l_value}_dropout_0.3_0.3"

current_datestamp = datetime.datetime.now().strftime("%Y%m%d")
exp_folder = f"{model_sel}-{current_datestamp}_{uniq_sample}"

#logging
logs_text_dir = os.path.join(exp_base_dir, exp_folder, "logs")
checkpoints_dir = os.path.join(exp_base_dir, exp_folder, "checkpoints")

current_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_filename = f"{logs_text_dir}/{model_sel}-{current_timestamp}.log"

if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir, exist_ok=True)

if not os.path.exists(logs_text_dir):
    os.makedirs(logs_text_dir, exist_ok=True)

# Create a logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

# Create a formatter to define the log format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Create a file handler to write logs to a file
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Create a stream handler to print logs to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # You can set the desired log level for console output
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info(f"Datetime: {current_timestamp}"
            f"\n Setup all folders and files:"
            f"\n Unique expname: {uniq_sample}"
            f"\n exp_folder: {exp_folder}"
            f"\n -logs_text_dir dir: {logs_text_dir}"
            f"\n -checkpoints_dir : {checkpoints_dir}")

# Debug flags
isDebug = True
isVerbose = True
randomState_value = 2024
dataset_name = 'sport-balls'
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)
# model configuration

pretrained_weights = 'imagenet'
batch_size = 32
frozen_weights = False

loaded_datasets = dev_dataprocesses.load_dataset(preconfigured_dataset=dataset_name)

train_data, val_data, test_data = dev_dataprocesses.data_generators(datasource=loaded_datasets,
                                                                    data_generator='keras')

def get_model_summary(model):
    """
    Print model summary details to console and logging.
    :param model: model instance to be logged.
    :return: model information flush.
    """
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string


class LearningRateLogger(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        """
        Callbacks function on epoch end
        Get up-to-date learning rate of current epoch and return it to the logging of model training process. (History)
        :param epoch: current epoch
        :param logs: set current log
        """
        logs = logs or {}
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model.optimizer.iterations)
        logs['learning_rate'] = float(tf.keras.backend.get_value(lr))
        print(f" Learning rate is {logs['learning_rate']}")


# Model building
# Customized model arch
ModelCnnExp3 = keras.models.Sequential([
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=img_shape),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2, 2)),

    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2, 2)),

    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2, 2)),

    keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2, 2)),

    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.1),

])

ModelCnnExp4 = keras.models.Sequential([
    keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=img_shape),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2, 2)),

    keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2, 2)),

    keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
])

ModelCNNExp1 = keras.models.Sequential([
    keras.layers.Conv2D(filters=128, kernel_size=(8, 8), strides=(3, 3), activation='relu', input_shape=img_shape),
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

ModelCNNExp2 = keras.models.Sequential([
    keras.layers.Conv2D(filters=128, kernel_size=(8, 8), strides=(3, 3), activation='relu', input_shape=img_shape),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3, 3)),

    keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),

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

    keras.layers.MaxPool2D(pool_size=(2, 2))
])

## TIMESTAMP @ 2024-12-12T10:41:52
## author: phuocddat
## start

## end --
logger.info(f"Training {model_sel} model  with new configuration")

if model_sel == 'VGG16':
    base_model = keras.applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=img_shape,
    )
elif model_sel == 'VGG19':
    base_model = keras.applications.VGG19(
        include_top=False,
        weights='imagenet',
        input_shape=img_shape,
    )
elif model_sel == 'EfficientNetB3':
    base_model = keras.applications.EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='max'
    )
elif model_sel == 'ModelCNNExp1':
    base_model = ModelCNNExp1
elif model_sel == 'ModelCNNExp2':
    base_model = ModelCNNExp1
elif model_sel == 'ModelCNNExp3':
    base_model = ModelCnnExp3
elif model_sel == 'ModelCNNExp4':
    base_model = ModelCnnExp4
else:
    base_model = keras.applications.VGG19(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
    )

if model_sel == 'ModelCNNExp1' or model_sel == 'ModelCNNExp2':
    logger.info(f"Training {model_sel} model  from scratch")
    model_new = keras.models.Sequential([
        # base_model,
        keras.layers.Conv2D(filters=128, kernel_size=(8, 8), strides=(3, 3), activation='relu', input_shape=img_shape),
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

        keras.layers.Flatten(),

        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.3),
        ## TIMESTAMP @ 2024-12-13T17:52:43
        ## author: phuocddat
        ## start
        # Add regularizers to reduce overfitting

        keras.layers.Dense(512, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(l=kernel_l_value),
                           activity_regularizer=keras.regularizers.l1(l=act_l_value)),
        keras.layers.Dropout(0.3),
        ## end --
        keras.layers.Dense(15, activation='softmax')
    ])
elif model_sel == 'ModelCNNExp3':
    logger.info(f"Training {model_sel} model  from scratch")
    model_new = keras.models.Sequential([
        keras.layers.Conv2D(46, kernel_size=(3, 3), activation='relu', input_shape=img_shape),
        keras.layers.Conv2D(46, kernel_size=(3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Dropout(0.15),

        keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Dropout(0.3),

        keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Dropout(0.5),

        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Dropout(0.1),

        keras.layers.Flatten(),
        keras.layers.Dense(15, activation='softmax', kernel_regularizer=keras.regularizers.l2(l=0.16))

    ])
elif model_sel == 'ModelCNNExp4':
    logger.info(f"Training {model_sel} model  from scratch")
    model_new = keras.models.Sequential([
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=img_shape),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(2, 2)),

        keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3, 3)),
        keras.layers.Dropout(0.3),

        keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.3),

        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(2, 2)),

        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        # keras.layers.Dense(1024, activation='relu'),
        # keras.layers.Dropout(0.5),
        keras.layers.Dense(15, activation='softmax')
    ])

elif model_sel == 'EfficientNetB3':
    if frozen_weights:
        logger.info(f"Perform training with frozen weight with {model_sel}")
    else:
        logger.info(f"Perform training with no pretrained-weights {model_sel}")

    # Freeze pretrained weight from base_model layers.
    base_model.trainable = False
    logger.info(f"Processing {model_sel} with {config_output_arc} for output block")
    if config_output_arc == 'config_1':
        model_new = keras.models.Sequential([
            base_model,
            # Config 1
            keras.layers.Flatten(),
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(15, activation='softmax')
        ])
    elif config_output_arc == 'config_2':
        model_new = keras.models.Sequential([
            base_model,
            # Config 2
            ## TIMESTAMP @ 2024-12-12T10:38:52
            ## author: phuocddat
            ## start
            ## end --
            keras.layers.BatchNormalization(),
            keras.layers.Dense(256,
                               # kernel_regularizer=keras.regularizers.l2(0.001),
                               # activity_regularizer=keras.regularizers.l1(0.001),
                               activation='relu', ),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(15, activation="softmax")
        ])
    elif config_output_arc == 'config_3':
        model_new = keras.models.Sequential([
            base_model,
            # Config 3
            # Add regularizers to reduce overfitting
            keras.layers.Dense(512, activation='relu',
                               kernel_regularizer=keras.regularizers.l2(l=kernel_l_value),
                               activity_regularizer=keras.regularizers.l1(l=act_l_value)),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(15, activation='softmax')
            ## end --
        ])
    elif config_output_arc == 'config_4':
        model_new = keras.models.Sequential([
            base_model,
            # Config 4
            # Add regularizes to reduce overfitting
            keras.layers.Flatten(),
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(512, activation='relu',
                               #kernel_regularizer=keras.regularizers.l2(l=kernel_l_value),
                               #activity_regularizer=keras.regularizers.l1(l=act_l_value)
                               ),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(15, activation='softmax')
        ])
    elif config_output_arc == 'config_5':
        model_new = keras.models.Sequential([
            # Config 5
            base_model,
            keras.layers.Flatten(),
            keras.layers.Dense(1024, activation='relu'),
            # keras.layers.Dropout(0.3),
            keras.layers.Dense(512, activation='relu',
                               # kernel_regularizer=keras.regularizers.l2(l=kernel_l_value),
                               # activity_regularizer=keras.regularizers.l1(l=act_l_value)
                               ),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(256, activation='relu',
                               # kernel_regularizer=keras.regularizers.l2(l=kernel_l_value),
                               # activity_regularizer=keras.regularizers.l1(l=act_l_value)
                               ),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation='relu',
                               kernel_regularizer=keras.regularizers.l2(l=kernel_l_value),
                               activity_regularizer=keras.regularizers.l1(l=act_l_value)
                               ),
            keras.layers.Dropout(0.3),
            ## end --
            keras.layers.Dense(15, activation='softmax')

        ])
    else:
        logger.info(f"Not implemented {config_output_arc}")
        logger.info(f"Using one of drafting version below")
        model_new = keras.models.Sequential([
            base_model,
            # Config 1
            # keras.layers.Flatten(),
            # keras.layers.Dense(1024,activation='relu'),
            # keras.layers.Dense(256,activation='relu'),
            # keras.layers.Dense(64,activation='relu'),
            # keras.layers.Dense(32,activation='relu'),
            # keras.layers.Dense(15,activation='softmax'),
            # Config 2
            ## TIMESTAMP @ 2024-12-12T10:38:52
            ## author: phuocddat
            ## start
            ## end --
                # keras.layers.BatchNormalization(),
                # keras.layers.Dense(256,
                #                    # kernel_regularizer=keras.regularizers.l2(0.001),
                #                    # activity_regularizer=keras.regularizers.l1(0.001),
                #                    activation='relu',),
                # keras.layers.Dropout(0.3),
                # keras.layers.Dense(15, activation="softmax" )
            # Config 3
            # Add regularizers to reduce overfitting

            # keras.layers.Dense(512, activation='relu',
            #                    kernel_regularizer=keras.regularizers.l2(l=kernel_l_value),
            #                    activity_regularizer=keras.regularizers.l1(l=act_l_value)),
            # keras.layers.Dropout(0.5),
            # ## end --
            # keras.layers.Dense(15, activation='softmax')
            # Config 4
            # Add regularizers to reduce overfitting
            # keras.layers.Flatten(),
            # keras.layers.Dense(1024, activation='relu'),
            # keras.layers.Dropout(0.3),
            # keras.layers.Dense(512, activation='relu',
            #                    #kernel_regularizer=keras.regularizers.l2(l=kernel_l_value),
            #                    #activity_regularizer=keras.regularizers.l1(l=act_l_value)
            #                    ),
            # keras.layers.Dropout(0.3),
            # ## end --
            # keras.layers.Dense(15, activation='softmax')
            # Config 5

            keras.layers.Flatten(),
            keras.layers.Dense(1024, activation='relu'),
            # keras.layers.Dropout(0.3),
            keras.layers.Dense(512, activation='relu',
                               # kernel_regularizer=keras.regularizers.l2(l=kernel_l_value),
                               # activity_regularizer=keras.regularizers.l1(l=act_l_value)
                               ),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(256, activation='relu',
                               # kernel_regularizer=keras.regularizers.l2(l=kernel_l_value),
                               # activity_regularizer=keras.regularizers.l1(l=act_l_value)
                               ),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation='relu',
                               kernel_regularizer=keras.regularizers.l2(l=kernel_l_value),
                               activity_regularizer=keras.regularizers.l1(l=act_l_value)
                               ),
            keras.layers.Dropout(0.3),
            ## end --
            keras.layers.Dense(15, activation='softmax')
        ])
else:
    if frozen_weights:
        logger.info(f"Perform training with frozen weight with {model_sel}")
    else:
        logger.info(f"Perform training with no pretrained-weights {model_sel}")
    # Freeze pretrained weight from base_model layers.
    base_model.trainable = not frozen_weights
    model_new = keras.models.Sequential([
        base_model,
        # Config 1
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(15, activation='softmax')
        # Config 2
        ## TIMESTAMP @ 2024-12-12T10:38:52
        ## author: phuocddat
        ## start
        ## end --
        # keras.layers.BatchNormalization(),
        # keras.layers.Dense(256,
        #                    # kernel_regularizer=keras.regularizers.l2(0.001),
        #                    # activity_regularizer=keras.regularizers.l1(0.001),
        #                    activation='relu', ),
        # keras.layers.Dropout(0.3),
        # keras.layers.Dense(15, activation="softmax")
        # Config 3
        # Add regularizers to reduce overfitting
        #
        # keras.layers.Dense(512, activation='relu',
        #                    kernel_regularizer=keras.regularizers.l2(l=kernel_l_value),
        #                    activity_regularizer=keras.regularizers.l1(l=act_l_value)),
        # keras.layers.Dropout(0.3),
        # ## end --
        # keras.layers.Dense(15, activation='softmax')
    ])

total_steps = len(train_data) * dev_configuration.epochs
decay_steps = total_steps * 0.9
logger.info(f"Total steps: {total_steps}"
      f"\__decay steps: {decay_steps}")

cosine_decay_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=learning_rate,
    decay_steps=decay_steps,
    alpha=0.08
)

# model_new.compile(optimizer=tf.optimizers.AdamW(learning_rate=cosine_decay_scheduler),
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])

model_new.compile(optimizer=tf.optimizers.Adamax(learning_rate=cosine_decay_scheduler),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# logger.info(model_new.summary())
model_summary_string = get_model_summary(model_new)
logger.info(model_summary_string)

current_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
exp_save_name = (f"{current_timestamp}___"
                 f"model_{model_sel}_{config_output_arc}_{pretrained_weights}_frozen_"
                 f"Adamax_bs_{batch_size}_frozen_{str(frozen_weights)}_cosine__aug")
checkpoint_path = f"{exp_save_name}.h5"
logger.info(f"epochs {dev_configuration.epochs}"
            f"\n batch size = {batch_size}"
            f"\n other options: {other_options} "
            f"\n augmentation: {augmentation_options}"            
            f"\n Checkpoint: {checkpoint_path}")

my_callbacks = [
    #    keras.callbacks.EarlyStopping(patience=10),
    LearningRateLogger(),
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
                                         validation_data=val_data,
                                         verbose=1,
                                         callbacks=my_callbacks)
# Evaluate model
logger.info(f"Evaluating on test set")
model_new.evaluate(test_data, verbose=1)

# Full Evaluate model

ts_length = len(test_data)
test_batch_size = max(
    sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length % n == 0 and ts_length / n <= 80]))
test_steps = ts_length // test_batch_size

train_score = model_new.evaluate(train_data, steps=test_steps, verbose=1)
valid_score = model_new.evaluate(val_data, steps=test_steps, verbose=1)
test_score = model_new.evaluate(test_data, steps=test_steps, verbose=1)

print("Train Loss: ", train_score[0])
print("Train Accuracy: ", train_score[1])
print('-' * 20)
print("Validation Loss: ", valid_score[0])
print("Validation Accuracy: ", valid_score[1])
print('-' * 20)
print("Test Loss: ", test_score[0])
print("Test Accuracy: ", test_score[1])
metrics_string = f"acc_tr{train_score[1]}_va{valid_score[1]}_te{test_score[1]}"

current_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
fig_name = (f"{logs_text_dir}/{current_timestamp}_"
            f"aug_"
            f"model_{model_sel}_{pretrained_weights}_frozen_{frozen_weights}_"
            f"ep{dev_configuration.epochs}_bs_{dev_configuration.batch_size}_cosine_decay_scheduler_"
            f"{metrics_string}_"
            f"accuracy_plot.png")

plt.plot(training_exper_model_eff.history['accuracy'])
plt.plot(training_exper_model_eff.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(fig_name)
logger.info(f"Export plot results: {fig_name}")

fig_name = (f"{logs_text_dir}/{current_timestamp}_"
            f"aug_"
            f"model_{model_sel}_{pretrained_weights}_frozen_{frozen_weights}_"
            f"ep{dev_configuration.epochs}_bs_{dev_configuration.batch_size}_cosine_decay_scheduler_accuracy_loss_"
            f"{metrics_string}"
            f"_plots.png")
# Define needed variables
tr_acc = training_exper_model_eff.history['accuracy']
tr_loss = training_exper_model_eff.history['loss']
val_acc = training_exper_model_eff.history['val_accuracy']
val_loss = training_exper_model_eff.history['val_loss']

learning_rate_logs = training_exper_model_eff.history['learning_rate']

index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]
index_acc = np.argmax(val_acc)
acc_highest = val_acc[index_acc]
# index_lr = np.argmin(learning_rate)

loss_label = f'best epoch= {str(index_loss + 1)}'
acc_label = f'best epoch= {str(index_acc + 1)}'

Epochs = [i + 1 for i in range(len(tr_acc))]

# Plot training history
plt.figure(figsize=(20, 8))

plt.subplot(1, 3, 1)
plt.plot(Epochs, tr_loss, 'orange', label='Training loss')
plt.plot(Epochs, val_loss, label='Validation loss')
plt.scatter(index_loss + 1, val_lowest, s=150, c='red', label=loss_label)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(Epochs, tr_acc, 'orange', label='Training Accuracy')
plt.plot(Epochs, val_acc, label='Validation Accuracy')
plt.scatter(index_acc + 1, acc_highest, s=150, c='red', label=acc_label)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(Epochs, learning_rate_logs, 'orange', label='Learning rate')
plt.title('Learning rate scheduler')
plt.xlabel('Epochs')
plt.ylabel('Learning rate')
plt.legend()

plt.savefig(fig_name)
logger.info(f"Export plot results: {fig_name}")

try:
    logger.info(f"Train loss: {train_score[0]}"
                f"\nTrain Accuracy: {train_score[1]}"
                f"\nValidation loss: {valid_score[0]}"
                f"\nValidation Accuracy: {valid_score[1]}"
                f"\nTest Loss: {test_score[0]}"
                f"\nTest Accuracy: {test_score[1]}")
except Exception as e:
    print(f"Unable to log evaluation score"
          f"\n Reason: {e}")

preds = model_new.predict_generator(test_data)
y_pred = np.argmax(preds, axis=1)
print(y_pred)

target_names = ['american_football', 'baseball', 'basketball', 'billiard_ball', 'bowling_ball', 'cricket_ball',
                'football', 'golf_ball', 'hockey_ball', 'hockey_puck', 'rugby_ball', 'shuttlecock',
                'table_tennis_ball', 'tennis_ball', 'volleyball']

# Classification report
logger.info(classification_report(test_data.classes, y_pred, target_names=target_names))

acc = test_score[1] * 100
save_path = checkpoints_dir

# Save history
history_filename = f"{logs_text_dir}/{model_sel}-{current_timestamp}.npy"
np.save(history_filename, training_exper_model_eff.history)

logger.info(f'model training history was saved as {history_filename}')

# Save model
save_id = str(f'{exp_save_name}-{"%.2f" % round(acc, 2)}.h5')
model_save_loc = os.path.join(save_path, save_id)
model_new.save(model_save_loc)
logger.info(f'model was saved as {model_save_loc}')

# Save weights
weight_save_id = str(f'{exp_save_name}-{"%.2f" % round(acc, 2)}-weights.h5')
weights_save_loc = os.path.join(save_path, weight_save_id)
model_new.save_weights(weights_save_loc)
logger.info(f'weights were saved as {weights_save_loc}')

elapsed_time = time.time() - start_time
logger.info(f"{current_timestamp}: Processing time: {elapsed_time} seconds. Saved plot to {fig_name} \n")
logger.info(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
logger.info(f"Finished!")

import itertools


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Oranges,
                          figname='figname.png'):
    """
    Plot confusion matrix of classification report.
    This functions is borrowed from scikit-learn and
    https://www.kaggle.com/code/abdallahwagih/efficientnetb3-sports-balls-classification-94
    :param cm:confusion matrix
    :param classes: classes name
    :param normalize:
    :param title:
    :param cmap:
    :param figname:
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized Confusion Matrix')
    else:
        print('Confusion Matrix, Without Normalization')
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(fig_name)

fig_name = (f"{logs_text_dir}/{current_timestamp}_"
            f"aug_"
            f"model_{model_sel}_{pretrained_weights}_frozen_{frozen_weights}_"
            f"ep{dev_configuration.epochs}_bs_{dev_configuration.batch_size}_cosine_decay_scheduler_accuracy_loss_"
            f"{metrics_string}"
            f"_confusion_mat.png")
# Confusion matrix
cm = confusion_matrix(test_data.classes, y_pred)
plot_confusion_matrix(cm=cm, classes=target_names, title='Confusion Matrix', figname=fig_name)
