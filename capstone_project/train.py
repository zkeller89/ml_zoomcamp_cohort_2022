import os
import re

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import CenterCrop
from tensorflow.keras.layers import Rescaling

img_size = (256, 256)
batch_size = 32

ds = tf.keras.utils.image_dataset_from_directory(
    "images_by_class/train/",
    label_mode='categorical',
    image_size=img_size,
    batch_size=batch_size,
)

img_height = img_size[0]
img_width = img_size[1]
class_names = ds.class_names
num_classes = len(class_names)

base_model = keras.applications.Xception(
    weights='imagenet',
    input_shape=(img_height, img_width, 3),
    include_top=False
)

base_model.trainable = False

inputs = keras.Input(shape=(img_height, img_width, 3))

x = tf.cast(inputs, tf.float32)
x = keras.applications.xception.preprocess_input(x)
x = base_model(inputs, training=False)
x = layers.MaxPooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.GlobalAveragePooling2D()(x)

outputs = layers.Dense(num_classes, activation='softmax')(x)

model = keras.Model(inputs, outputs)

learning_rate = 0.001
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
loss = keras.losses.CategoricalCrossentropy()

model.compile(
    optimizer=optimizer,
    loss=loss
)

epochs = 10

history = model.fit(
    ds,
    batch_size=32,
    epochs=epochs
)

model.save_weights('model_v1.h5', save_format='h5')