#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Train DLN on heart_c.csv
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
BATCH_SIZE = 32
NUM_EPOCHS = 200

################################################################################
train_df = pd.read_csv("../datasets/heart_c/train.csv")
test_df = pd.read_csv("../datasets/heart_c/test.csv")
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
# feature 7 to 13
lattice_inputs2 = []

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
lattice_inputs1.append(age_calibrator)

sex_input = tf.keras.layers.Input(shape=[1], name='sex')
model_inputs.append(sex_input)
sex_calibrator = tfl.layers.PWLCalibration(
    name='sex_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['sex']),
        max(attr_domain['sex']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(sex_input)
lattice_inputs1.append(sex_calibrator)

cp_input = tf.keras.layers.Input(shape=[1], name='cp')
model_inputs.append(cp_input)
cp_calibrator = tfl.layers.PWLCalibration(
    name='cp_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['cp']),
        max(attr_domain['cp']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(cp_input)
lattice_inputs1.append(cp_calibrator)

trestbps_input = tf.keras.layers.Input(shape=[1], name='trestbps')
model_inputs.append(trestbps_input)
trestbps_calibrator = tfl.layers.PWLCalibration(
    name='trestbps_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['trestbps']),
        max(attr_domain['trestbps']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(trestbps_input)
lattice_inputs1.append(trestbps_calibrator)

chol_input = tf.keras.layers.Input(shape=[1], name='chol')
model_inputs.append(chol_input)
chol_calibrator = tfl.layers.PWLCalibration(
    name='chol_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['chol']),
        max(attr_domain['chol']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(chol_input)
lattice_inputs1.append(chol_calibrator)

fbs_input = tf.keras.layers.Input(shape=[1], name='fbs')
model_inputs.append(fbs_input)
fbs_calibrator = tfl.layers.PWLCalibration(
    name='fbs_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['fbs']),
        max(attr_domain['fbs']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(fbs_input)
lattice_inputs1.append(fbs_calibrator)

restecg_input = tf.keras.layers.Input(shape=[1], name='restecg')
model_inputs.append(restecg_input)
restecg_calibrator = tfl.layers.PWLCalibration(
    name='restecg_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['restecg']),
        max(attr_domain['restecg']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(restecg_input)
lattice_inputs2.append(restecg_calibrator)

thalach_input = tf.keras.layers.Input(shape=[1], name='thalach')
model_inputs.append(thalach_input)
thalach_calibrator = tfl.layers.PWLCalibration(
    name='thalach_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['thalach']),
        max(attr_domain['thalach']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(thalach_input)
lattice_inputs2.append(thalach_calibrator)

exang_input = tf.keras.layers.Input(shape=[1], name='exang')
model_inputs.append(exang_input)
exang_calibrator = tfl.layers.PWLCalibration(
    name='exang_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['exang']),
        max(attr_domain['exang']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(exang_input)
lattice_inputs2.append(exang_calibrator)

oldpeak_input = tf.keras.layers.Input(shape=[1], name='oldpeak')
model_inputs.append(oldpeak_input)
oldpeak_calibrator = tfl.layers.PWLCalibration(
    name='oldpeak_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['oldpeak']),
        max(attr_domain['oldpeak']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(oldpeak_input)
lattice_inputs2.append(oldpeak_calibrator)

slope_input = tf.keras.layers.Input(shape=[1], name='slope')
model_inputs.append(slope_input)
slope_calibrator = tfl.layers.PWLCalibration(
    name='slope_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['slope']),
        max(attr_domain['slope']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(slope_input)
lattice_inputs2.append(slope_calibrator)

ca_input = tf.keras.layers.Input(shape=[1], name='ca')
model_inputs.append(ca_input)
ca_calibrator = tfl.layers.PWLCalibration(
    name='ca_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['ca']),
        max(attr_domain['ca']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(ca_input)
lattice_inputs2.append(ca_calibrator)

thal_input = tf.keras.layers.Input(shape=[1], name='thal')
model_inputs.append(thal_input)
thal_calibrator = tfl.layers.PWLCalibration(
    name='thal_cab',
    monotonicity='increasing',
    input_keypoints=np.linspace(
        min(attr_domain['thal']),
        max(attr_domain['thal']),
        num=20,
        dtype=np.float32),
    dtype=tf.float32,
    output_min=0.0,
    output_max=1.0,
)(thal_input)
lattice_inputs2.append(thal_calibrator)

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

################################################################################
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
tf.keras.utils.plot_model(model, to_file="model_heart_c.png", rankdir='LR')

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
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    verbose=True)

print('Test Set Evaluation...')
print(model.evaluate(test_xs, test_ys))

################################################################################
model.save('../DLNs/heart_c')
loaded_model = tf.keras.models.load_model('../DLNs/heart_c')
assert np.allclose(model.predict(train_xs), loaded_model.predict(train_xs))
assert np.allclose(model.predict(test_xs), loaded_model.predict(test_xs))
