#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   pick tested instances
#
################################################################################
import pandas as pd
################################################################################


if __name__ == '__main__':
    datasets = ["australian", "breast_cancer", "heart_c", "nursery", "pima-modified"]
    num_test = 200
    for ds in datasets:
        df = pd.read_csv(f"../datasets/{ds}/{ds}.csv")
        print(f"############ {ds} ############")
        if len(df.index) > num_test:
            save_df = df.sample(n=num_test)
            save_df.to_csv(f'test_insts/{ds}.csv', sep=',', index=False)
        else:
            save_df = df.sample(n=len(df.index))
            save_df.to_csv(f'test_insts/{ds}.csv', sep=',', index=False)