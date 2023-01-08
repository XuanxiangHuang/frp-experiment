#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Generate 10000 datapoints (without class label) for testing classification time (CPU time of prediction function).
#
################################################################################
import random
import csv
import pandas as pd
################################################################################


if __name__ == '__main__':
    datasets = ["australian", "breast_cancer", "heart_c", "nursery", "pima-modified"]
    for ds in datasets:
        train_df = pd.read_csv(f"../datasets/{ds}/train.csv")
        test_df = pd.read_csv(f"../datasets/{ds}/test.csv")
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

        tested = set()
        d_len = 10000

        round_i = 0
        while round_i < d_len:
            tmp_sample = []
            for fid, feat in enumerate(feature_names):
                val_idx = random.randint(0, len(attr_domain[feat])-1)
                tmp_sample.append(attr_domain[feat][val_idx])
            while tuple(tmp_sample) in tested:
                tmp_sample = []
                for fid, feat in enumerate(feature_names):
                    val_idx = random.randint(0, len(attr_domain[feat])-1)
                    tmp_sample.append(attr_domain[feat][val_idx])

            assert tuple(tmp_sample) not in tested
            tested.add(tuple(tmp_sample))
            round_i += 1

        assert len(tested) == d_len
        data = []
        for item in tested:
            csv_item = list(item)
            assert len(csv_item) == len(feature_names)
            data.append(csv_item)

        with open(f"clf_time_test/{ds}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(data)
