#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Train DLN on nursery.csv
#
################################################################################
import tensorflow as tf
import logging
import numpy as np
import pandas as pd
import sys
import tensorflow_lattice as tfl
logging.disable(sys.maxsize)
################################################################################
LEARNING_RATE = 0.1
BATCH_SIZE = 128
NUM_EPOCHS = 200

################################################################################
train_df = pd.read_csv("../datasets/nursery/train.csv")
test_df = pd.read_csv("../datasets/nursery/test.csv")
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
# feature 1 to 6
lattice_inputs1 = []

################################################################################
A1_input = tf.keras.layers.Input(shape=[1], name='parents')
model_inputs.append(A1_input)
A1_calibrator = tfl.layers.PWLCalibration(
    name='A1_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['parents']),
        max(attr_domain['parents']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(A1_input)
lattice_inputs1.append(A1_calibrator)

A2_input = tf.keras.layers.Input(shape=[1], name='has_nurs')
model_inputs.append(A2_input)
A2_calibrator = tfl.layers.PWLCalibration(
    name='A2_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['has_nurs']),
        max(attr_domain['has_nurs']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(A2_input)
lattice_inputs1.append(A2_calibrator)

A3_input = tf.keras.layers.Input(shape=[1], name='form')
model_inputs.append(A3_input)
A3_calibrator = tfl.layers.PWLCalibration(
    name='A3_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['form']),
        max(attr_domain['form']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(A3_input)
lattice_inputs1.append(A3_calibrator)

A4_input = tf.keras.layers.Input(shape=[1], name='children')
model_inputs.append(A4_input)
A4_calibrator = tfl.layers.PWLCalibration(
    name='A4_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['children']),
        max(attr_domain['children']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(A4_input)
lattice_inputs1.append(A4_calibrator)

A5_input = tf.keras.layers.Input(shape=[1], name='housing')
model_inputs.append(A5_input)
A5_calibrator = tfl.layers.PWLCalibration(
    name='A5_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['housing']),
        max(attr_domain['housing']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(A5_input)
lattice_inputs1.append(A5_calibrator)

A6_input = tf.keras.layers.Input(shape=[1], name='finance')
model_inputs.append(A6_input)
A6_calibrator = tfl.layers.PWLCalibration(
    name='A6_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['finance']),
        max(attr_domain['finance']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(A6_input)
lattice_inputs1.append(A6_calibrator)

A7_input = tf.keras.layers.Input(shape=[1], name='social')
model_inputs.append(A7_input)
A7_calibrator = tfl.layers.PWLCalibration(
    name='A7_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['social']),
        max(attr_domain['social']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(A7_input)
lattice_inputs1.append(A7_calibrator)

A8_input = tf.keras.layers.Input(shape=[1], name='health')
model_inputs.append(A8_input)
A8_calibrator = tfl.layers.PWLCalibration(
    name='A8_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['health']),
        max(attr_domain['health']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(A8_input)
lattice_inputs1.append(A8_calibrator)

################################################################################
rtl1 = tfl.layers.RTL(
    name='rtl1',
    num_lattices=3,
    lattice_rank=4,
    lattice_size=3,
    output_min=0.0,
    output_max=1.0,
    separate_outputs=False,
)({
    'increasing': lattice_inputs1,
})

################################################################################

lin = tfl.layers.Linear(
    num_input_dims=3,
    monotonicities=['increasing'] * 3,
)(rtl1)

model_output = tf.keras.layers.Dense(
    len(attr_domain[class_name]), activation=tf.nn.softmax
)(lin)

model = tf.keras.models.Model(
    inputs=model_inputs,
    outputs=model_output)
tf.keras.utils.plot_model(model, to_file="model_nursery.png", rankdir='LR')

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
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'],
    optimizer=tf.keras.optimizers.Adagrad(LEARNING_RATE))
model.fit(train_xs, train_ys,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    verbose=True)

print('Test Set Evaluation...')
print(model.evaluate(test_xs, test_ys))

################################################################################
model.save('../DLNs/nursery')
loaded_model = tf.keras.models.load_model('../DLNs/nursery')
assert np.allclose(model.predict(train_xs), loaded_model.predict(train_xs))
assert np.allclose(model.predict(test_xs), loaded_model.predict(test_xs))
