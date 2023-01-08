#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   DLN model info, parameters
#
################################################################################
import tensorflow as tf
import pandas as pd
################################################################################


if __name__ == '__main__':
    datasets = ["australian", "breast_cancer", "heart_c", "nursery", "pima-modified"]
    for data_name in datasets:
        print(f"###### {data_name} ######")

        train_df = pd.read_csv(f"../datasets/{data_name}/train.csv")
        test_df = pd.read_csv(f"../datasets/{data_name}/test.csv")
        feature_names = list(train_df.columns)
        class_name = feature_names.pop()
        feature_name_indices = {name: index for index, name in enumerate(feature_names)}

        train_dict = dict(train_df)
        test_dict = dict(test_df)
        train_xs = [train_dict[feature_name] for feature_name in feature_names]
        test_xs = [test_dict[feature_name] for feature_name in feature_names]
        train_ys = train_dict[class_name]
        test_ys = test_dict[class_name]

        loaded_model = tf.keras.models.load_model(f'../DLNs/{data_name}')
        print(loaded_model.evaluate(test_xs, test_ys))

        loaded_model.summary()
        print()
