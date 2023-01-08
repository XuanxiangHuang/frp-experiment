#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Generate tested features
#
################################################################################
import random
import csv
import numpy as np
import pandas as pd
################################################################################


if __name__ == '__main__':
    datasets = ["australian", "breast_cancer", "heart_c", "nursery", "pima-modified"]
    for ds in datasets:
        df = pd.read_csv(f"test_insts/{ds}.csv")
        feature_names = list(df.columns)
        feature_names.pop()
        feature_name_indices = {name: index for index, name in enumerate(feature_names)}
        nv = len(feature_names)
        ids = [i for i in range(nv)]
        num_test = len(df)
        print(ds)
        print(f"Test instances: ", num_test)
        print("Features: ", nv)
        if num_test < nv:
            feat = np.array(random.sample(ids, num_test))
        else:
            feat = np.array(np.random.choice(ids, num_test))
        test_feat = feat.reshape(-1, 1)

        with open(f"test_feats/{ds}.csv", 'w') as f:
            write = csv.writer(f)
            write.writerows(test_feat)