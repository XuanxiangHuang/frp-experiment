#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Train DLN on australian.csv
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
BATCH_SIZE = 16
NUM_EPOCHS = 200

################################################################################
train_df = pd.read_csv("../datasets/australian/train.csv")
test_df = pd.read_csv("../datasets/australian/test.csv")
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
# feature 1 to 7
lattice_inputs1 = []
# feature 8 to 14
lattice_inputs2 = []

################################################################################
A1_input = tf.keras.layers.Input(shape=[1], name='A1')
model_inputs.append(A1_input)
A1_calibrator = tfl.layers.PWLCalibration(
    name='A1_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['A1']),
        max(attr_domain['A1']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(A1_input)
lattice_inputs1.append(A1_calibrator)

A2_input = tf.keras.layers.Input(shape=[1], name='A2')
model_inputs.append(A2_input)
A2_calibrator = tfl.layers.PWLCalibration(
    name='A2_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['A2']),
        max(attr_domain['A2']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(A2_input)
lattice_inputs1.append(A2_calibrator)

A3_input = tf.keras.layers.Input(shape=[1], name='A3')
model_inputs.append(A3_input)
A3_calibrator = tfl.layers.PWLCalibration(
    name='A3_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['A3']),
        max(attr_domain['A3']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(A3_input)
lattice_inputs1.append(A3_calibrator)

A4_input = tf.keras.layers.Input(shape=[1], name='A4')
model_inputs.append(A4_input)
A4_calibrator = tfl.layers.PWLCalibration(
    name='A4_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['A4']),
        max(attr_domain['A4']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(A4_input)
lattice_inputs1.append(A4_calibrator)

A5_input = tf.keras.layers.Input(shape=[1], name='A5')
model_inputs.append(A5_input)
A5_calibrator = tfl.layers.PWLCalibration(
    name='A5_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['A5']),
        max(attr_domain['A5']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(A5_input)
lattice_inputs1.append(A5_calibrator)

A6_input = tf.keras.layers.Input(shape=[1], name='A6')
model_inputs.append(A6_input)
A6_calibrator = tfl.layers.PWLCalibration(
    name='A6_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['A6']),
        max(attr_domain['A6']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(A6_input)
lattice_inputs1.append(A6_calibrator)

A7_input = tf.keras.layers.Input(shape=[1], name='A7')
model_inputs.append(A7_input)
A7_calibrator = tfl.layers.PWLCalibration(
    name='A7_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['A7']),
        max(attr_domain['A7']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(A7_input)
lattice_inputs1.append(A7_calibrator)

A8_input = tf.keras.layers.Input(shape=[1], name='A8')
model_inputs.append(A8_input)
A8_calibrator = tfl.layers.PWLCalibration(
    name='A8_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['A8']),
        max(attr_domain['A8']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(A8_input)
lattice_inputs2.append(A8_calibrator)

A9_input = tf.keras.layers.Input(shape=[1], name='A9')
model_inputs.append(A9_input)
A9_calibrator = tfl.layers.PWLCalibration(
    name='A9_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['A9']),
        max(attr_domain['A9']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(A9_input)
lattice_inputs2.append(A9_calibrator)

A10_input = tf.keras.layers.Input(shape=[1], name='A10')
model_inputs.append(A10_input)
A10_calibrator = tfl.layers.PWLCalibration(
    name='A10_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['A10']),
        max(attr_domain['A10']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(A10_input)
lattice_inputs2.append(A10_calibrator)

A11_input = tf.keras.layers.Input(shape=[1], name='A11')
model_inputs.append(A11_input)
A11_calibrator = tfl.layers.PWLCalibration(
    name='A11_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['A11']),
        max(attr_domain['A11']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(A11_input)
lattice_inputs2.append(A11_calibrator)

A12_input = tf.keras.layers.Input(shape=[1], name='A12')
model_inputs.append(A12_input)
A12_calibrator = tfl.layers.PWLCalibration(
    name='A12_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['A12']),
        max(attr_domain['A12']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(A12_input)
lattice_inputs2.append(A12_calibrator)

A13_input = tf.keras.layers.Input(shape=[1], name='A13')
model_inputs.append(A13_input)
A13_calibrator = tfl.layers.PWLCalibration(
    name='A13_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['A13']),
        max(attr_domain['A13']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(A13_input)
lattice_inputs2.append(A13_calibrator)

A14_input = tf.keras.layers.Input(shape=[1], name='A14')
model_inputs.append(A14_input)
A14_calibrator = tfl.layers.PWLCalibration(
    name='A14_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['A14']),
        max(attr_domain['A14']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(A14_input)
lattice_inputs2.append(A14_calibrator)

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
tf.keras.utils.plot_model(model, to_file="model_aus.png", rankdir='LR')

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
    metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', dtype=None, threshold=0.5)],
    optimizer=tf.keras.optimizers.Adagrad(LEARNING_RATE))
model.fit(train_xs, train_ys,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    verbose=True)

print('Test Set Evaluation...')
print(model.evaluate(test_xs, test_ys))

################################################################################
model.save('../DLNs/australian')
loaded_model = tf.keras.models.load_model('../DLNs/australian')
assert np.allclose(model.predict(train_xs), loaded_model.predict(train_xs))
assert np.allclose(model.predict(test_xs), loaded_model.predict(test_xs))
