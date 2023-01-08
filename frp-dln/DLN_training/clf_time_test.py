#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Testing classification time (the CPU time of prediction function)
#
################################################################################
import pandas as pd
import time
import tensorflow as tf
from statistics import median
################################################################################


def get_pred_timer(mclf, datapoint):
    time_start = time.perf_counter()
    raw_prediction = mclf.predict(datapoint, verbose=0)
    time_end = time.perf_counter()
    return time_end - time_start


if __name__ == '__main__':
    datasets = ["australian", "breast_cancer", "heart_c", "nursery", "pima-modified"]
    for ds in datasets:
        all_df = pd.read_csv(f"../samples/clf_time_test/{ds}.csv")
        feature_names = list(all_df.columns)
        loaded_model = tf.keras.models.load_model(f'../DLNs/{ds}')

        d_len = len(all_df)
        T_time = []
        for i in range(d_len):
            x_i = all_df.loc[i][feature_names]
            input_x = []
            assert list(x_i.index) == feature_names
            for name, val in zip(feature_names, x_i.values):
                input_x.append(pd.Series(data=val, name=name))

            time_i = get_pred_timer(loaded_model, input_x)
            T_time.append(time_i)
        print(f"########## Dataset: {ds} ##########")
        print(f"Number of tested data points: {len(T_time)}")
        print(f"Max predict() time: {max(T_time):.4f}")
        print(f"Median predict() time: {median(T_time):.4f}")
        print(f"Avg predict() time: {sum(T_time) / d_len:.4f}")
        print()
