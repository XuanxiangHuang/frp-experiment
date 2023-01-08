#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Train DLN on breast_cancer.csv
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
train_df = pd.read_csv("../datasets/breast_cancer/train.csv")
test_df = pd.read_csv("../datasets/breast_cancer/test.csv")
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
lattice_inputs = []

################################################################################
age_input = tf.keras.layers.Input(shape=[1], name='age')
model_inputs.append(age_input)
age_calibrator = tfl.layers.PWLCalibration(
    name='age_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['age']),
        max(attr_domain['age']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(age_input)
lattice_inputs.append(age_calibrator)

menopause_input = tf.keras.layers.Input(shape=[1], name='menopause')
model_inputs.append(menopause_input)
menopause_calibrator = tfl.layers.PWLCalibration(
    name='menopause_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['menopause']),
        max(attr_domain['menopause']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(menopause_input)
lattice_inputs.append(menopause_calibrator)

tumor_size_input = tf.keras.layers.Input(shape=[1], name='tumor-size')
model_inputs.append(tumor_size_input)
tumor_size_calibrator = tfl.layers.PWLCalibration(
    name='tumor-size_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['tumor-size']),
        max(attr_domain['tumor-size']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(tumor_size_input)
lattice_inputs.append(tumor_size_calibrator)

inv_nodes_input = tf.keras.layers.Input(shape=[1], name='inv-nodes')
model_inputs.append(inv_nodes_input)
inv_nodes_calibrator = tfl.layers.PWLCalibration(
    name='inv-nodes_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['inv-nodes']),
        max(attr_domain['inv-nodes']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(inv_nodes_input)
lattice_inputs.append(inv_nodes_calibrator)

node_caps_input = tf.keras.layers.Input(shape=[1], name='node-caps')
model_inputs.append(node_caps_input)
node_caps_calibrator = tfl.layers.PWLCalibration(
    name='node-caps_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['node-caps']),
        max(attr_domain['node-caps']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(node_caps_input)
lattice_inputs.append(node_caps_calibrator)

deg_malig_input = tf.keras.layers.Input(shape=[1], name='deg-malig')
model_inputs.append(deg_malig_input)
deg_malig_calibrator = tfl.layers.PWLCalibration(
    name='deg-malig_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['deg-malig']),
        max(attr_domain['deg-malig']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(deg_malig_input)
lattice_inputs.append(deg_malig_calibrator)

breast_input = tf.keras.layers.Input(shape=[1], name='breast')
model_inputs.append(breast_input)
breast_calibrator = tfl.layers.PWLCalibration(
    name='breast_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['breast']),
        max(attr_domain['breast']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(breast_input)
lattice_inputs.append(breast_calibrator)

breast_quad_input = tf.keras.layers.Input(shape=[1], name='breast-quad')
model_inputs.append(breast_quad_input)
breast_quad_calibrator = tfl.layers.PWLCalibration(
    name='breast-quad_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['breast-quad']),
        max(attr_domain['breast-quad']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(breast_quad_input)
lattice_inputs.append(breast_quad_calibrator)

irradiat_input = tf.keras.layers.Input(shape=[1], name='irradiat')
model_inputs.append(irradiat_input)
irradiat_calibrator = tfl.layers.PWLCalibration(
    name='irradiat_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['irradiat']),
        max(attr_domain['irradiat']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(irradiat_input)
lattice_inputs.append(irradiat_calibrator)

################################################################################
rtl = tfl.layers.RTL(
    name='rtl',
    num_lattices=3,
    lattice_rank=4,
    lattice_size=3,
    output_min=0.0,
    output_max=1.0,
    separate_outputs=False,
)({
    'increasing': lattice_inputs,
})

################################################################################
lin = tfl.layers.Linear(
    num_input_dims=3,
    monotonicities=['increasing'] * 3,
)(rtl)

model_output = tf.keras.layers.Dense(
    1, activation=tf.nn.sigmoid
)(lin)

model = tf.keras.models.Model(
    inputs=model_inputs,
    outputs=model_output)
tf.keras.utils.plot_model(model, to_file="model_bc.png", rankdir='LR')

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
model.save('../DLNs/breast_cancer')
loaded_model = tf.keras.models.load_model('../DLNs/breast_cancer')
assert np.allclose(model.predict(train_xs), loaded_model.predict(train_xs))
assert np.allclose(model.predict(test_xs), loaded_model.predict(test_xs))
