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

# Dataset processing
# TODO: should be dedicated classes/functions

# manual configuration
datasets  = ['chest-xray-small', 'chest-xray-full']
select_dataset = 0
dataset_original_path = f"./datasets/{datasets[0]}"


