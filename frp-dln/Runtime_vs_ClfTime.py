#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   kappa(v) / runtime %
#
################################################################################


if __name__ == '__main__':
    datasets = ["australian", "breast_cancer", "heart_c", "nursery", "pima-modified"]
    for data_name in datasets:
        runtime_file = f"results/runtime/{data_name}_time.txt"
        clf_time_file = f"results/runtime/{data_name}_mclf_call_time.txt"

        with open(runtime_file, 'r') as fp:
            runtime_lines = fp.readlines()

        with open(clf_time_file, 'r') as fp:
            clf_time_lines = fp.readlines()

        avg = []
        for t1, t2 in zip(runtime_lines, clf_time_lines):
            time1 = float(t1)
            time2 = float(t2)
            assert time1 >= time2
            avg.append((time2 / time1) * 100)
        print(f"{data_name}: ", round(sum(avg) / len(avg), 1))

