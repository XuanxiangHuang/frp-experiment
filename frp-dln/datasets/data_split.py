#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Splitting datasets
#
################################################################################
import pandas as pd
from sklearn.model_selection import train_test_split
################################################################################


if __name__ == '__main__':
    datasets = ["australian", "breast_cancer", "heart_c", "nursery", "pima-modified"]
    for ds in datasets:
        input_file = f'{ds}/{ds}.csv'
        df = pd.read_csv(input_file, sep=',')
        train_df, test_df = train_test_split(df, test_size=0.2)
        train_df.to_csv(f'{ds}/train.csv', sep=',', index=False)
        test_df.to_csv(f'{ds}/test.csv', sep=',', index=False)
