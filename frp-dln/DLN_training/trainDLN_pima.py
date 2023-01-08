#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Train DLN on pima-modified.csv
#
################################################################################
import logging
import numpy as np
import pandas as pd
import sys
import tensorflow_lattice as tfl
import tensorflow as tf
logging.disable(sys.maxsize)
################################################################################
LEARNING_RATE = 0.1
BATCH_SIZE = 16
NUM_EPOCHS = 200

################################################################################
train_df = pd.read_csv("../datasets/pima-modified/train.csv")
test_df = pd.read_csv("../datasets/pima-modified/test.csv")
df = pd.concat([train_df, test_df], ignore_index=True)
feature_names = list(train_df.columns)
class_name = feature_names.pop()
feature_name_indices = {name: index for index, name in enumerate(feature_names)}

attr_domain = dict()
for feat in feature_names:
    vals = list(df[feat].unique())
    vals.sort()
    attr_domain.update({feat: vals})

class_vals = list(df[class_name].unique())
class_vals.sort()
attr_domain.update({class_name: class_vals})

for item in attr_domain:
    print(item, len(attr_domain[item]), min(attr_domain[item]), max(attr_domain[item]))

################################################################################
model_inputs = []
# feature 1 to 4
lattice_inputs1 = []
# feature 5 to 8
lattice_inputs2 = []

################################################################################
Preg_input = tf.keras.layers.Input(shape=[1], name='Preg')
model_inputs.append(Preg_input)
Preg_calibrator = tfl.layers.PWLCalibration(
    name='Preg_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['Preg']),
        max(attr_domain['Preg']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(Preg_input)
lattice_inputs1.append(Preg_calibrator)

Plas_input = tf.keras.layers.Input(shape=[1], name='Plas')
model_inputs.append(Plas_input)
Plas_calibrator = tfl.layers.PWLCalibration(
    name='Plas_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['Plas']),
        max(attr_domain['Plas']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(Plas_input)
lattice_inputs1.append(Plas_calibrator)

Pres_input = tf.keras.layers.Input(shape=[1], name='Pres')
model_inputs.append(Pres_input)
Pres_calibrator = tfl.layers.PWLCalibration(
    name='Pres_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['Pres']),
        max(attr_domain['Pres']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(Pres_input)
lattice_inputs1.append(Pres_calibrator)

Skin_input = tf.keras.layers.Input(shape=[1], name='Skin')
model_inputs.append(Skin_input)
Skin_calibrator = tfl.layers.PWLCalibration(
    name='Skin_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['Skin']),
        max(attr_domain['Skin']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(Skin_input)
lattice_inputs1.append(Skin_calibrator)

Insu_input = tf.keras.layers.Input(shape=[1], name='Insu')
model_inputs.append(Insu_input)
Insu_calibrator = tfl.layers.PWLCalibration(
    name='Insu_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['Insu']),
        max(attr_domain['Insu']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(Insu_input)
lattice_inputs2.append(Insu_calibrator)

Mass_input = tf.keras.layers.Input(shape=[1], name='Mass')
model_inputs.append(Mass_input)
Mass_calibrator = tfl.layers.PWLCalibration(
    name='Mass_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['Mass']),
        max(attr_domain['Mass']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(Mass_input)
lattice_inputs2.append(Mass_calibrator)

Pedi_input = tf.keras.layers.Input(shape=[1], name='Pedi')
model_inputs.append(Pedi_input)
Pedi_calibrator = tfl.layers.PWLCalibration(
    name='Pedi_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['Pedi']),
        max(attr_domain['Pedi']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(Pedi_input)
lattice_inputs2.append(Pedi_calibrator)

Age_input = tf.keras.layers.Input(shape=[1], name='Age')
model_inputs.append(Age_input)
Age_calibrator = tfl.layers.PWLCalibration(
    name='Age_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['Age']),
        max(attr_domain['Age']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(Age_input)
lattice_inputs2.append(Age_calibrator)

################################################################################
rtl1 = tfl.layers.RTL(
    name='rtl',
    num_lattices=3,
    lattice_rank=4,
    lattice_size=3,
    output_min=0.0,
    output_max=1.0,
    separate_outputs=False,
)({
    'increasing': lattice_inputs1,
})

rtl2 = tfl.layers.RTL(
    name='rtl2',
    num_lattices=3,
    lattice_rank=4,
    lattice_size=3,
    output_min=0.0,
    output_max=1.0,
    separate_outputs=False,
)({
    'increasing': lattice_inputs2,
})

################################################################################
concatted = tf.keras.layers.Concatenate()([rtl1, rtl2])

lin = tfl.layers.Linear(
    num_input_dims=6,
    monotonicities=['increasing'] * 6,
)(concatted)

model_output = tf.keras.layers.Dense(
    1, activation=tf.nn.sigmoid
)(lin)

model = tf.keras.models.Model(
    inputs=model_inputs,
    outputs=model_output)
tf.keras.utils.plot_model(model, to_file="model_pima.png", rankdir='LR')

################################################################################
train_xs = np.split(
    train_df[feature_names].values.astype(np.float32),
    indices_or_sections=len(feature_names),
    axis=1)
train_ys = train_df[class_name].values.astype(np.float32)

test_xs = np.split(
    test_df[feature_names].values.astype(np.float32),
    indices_or_sections=len(feature_names),
    axis=1)
test_ys = test_df[class_name].values.astype(np.float32)

################################################################################
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy',dtype=None,threshold=0.5)],
    optimizer=tf.keras.optimizers.Adagrad(LEARNING_RATE))
model.fit(train_xs, train_ys,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=True)

print('Test Set Evaluation...')
print(model.evaluate(test_xs, test_ys))

################################################################################
model.save('../DLNs/pima-modified')
loaded_model = tf.keras.models.load_model('../DLNs/pima-modified')
assert np.allclose(model.predict(train_xs), loaded_model.predict(train_xs))
assert np.allclose(model.predict(test_xs), loaded_model.predict(test_xs))
