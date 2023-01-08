#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Compute average time/calls
#
################################################################################
from statistics import median
################################################################################


if __name__ == '__main__':
    datasets = ["australian", "breast_cancer", "heart_c", "nursery", "pima-modified"]
    for ds in datasets:
        file_time = f"runtime/{ds}_time.txt"
        file_clf_time = f"runtime/{ds}_mclf_call_time.txt"
        file_sat_time = f"runtime/{ds}_sat_call_time.txt"
        file_clf_calls = f"calls/{ds}_num_mclf_calls.txt"
        file_sat_calls = f"calls/{ds}_num_sat_calls.txt"

        with open(file_time) as f:
            T_time = []
            lines = f.readlines()
            d_len = len(lines)
            for line in lines:
                T_time.append(float(line))
            print(f"########## Dataset: {ds} ##########")
            print(f"Number of tested data points: {len(T_time)}")
            print(f"Max runtime: {max(T_time):.4f}")
            print(f"Median runtime: {median(T_time):.4f}")
            print(f"Avg runtime: {sum(T_time) / d_len:.4f}")
            print()

        with open(file_sat_time) as f:
            T_time = []
            lines = f.readlines()
            d_len = len(lines)
            for line in lines:
                T_time.append(float(line))
            print(f"Number of tested data points: {len(T_time)}")
            print(f"Max sat time: {max(T_time):.4f}")
            print(f"Median sat time: {median(T_time):.4f}")
            print(f"Avg sat time: {sum(T_time) / d_len:.4f}")
            print()

        with open(file_sat_calls) as f:
            T_calls = []
            lines = f.readlines()
            d_len = len(lines)
            for line in lines:
                T_calls.append(float(line))
            print(f"Number of tested data points: {len(T_calls)}")
            print(f"Max sat calls: {max(T_calls):.4f}")
            print(f"Median sat calls: {median(T_calls):.4f}")
            print(f"Avg sat calls: {sum(T_calls) / d_len:.4f}")
            print()

        with open(file_clf_time) as f:
            T_time = []
            lines = f.readlines()
            d_len = len(lines)
            for line in lines:
                T_time.append(float(line))
            print(f"Number of tested data points: {len(T_time)}")
            print(f"Max clf time: {max(T_time):.4f}")
            print(f"Median clf time: {median(T_time):.4f}")
            print(f"Avg clf time: {sum(T_time) / d_len:.4f}")
            print()

        with open(file_clf_calls) as f:
            T_calls = []
            lines = f.readlines()
            d_len = len(lines)
            for line in lines:
                T_calls.append(float(line))
            print(f"Number of tested data points: {len(T_calls)}")
            print(f"Max clf calls: {max(T_calls):.4f}")
            print(f"Median clf calls: {median(T_calls):.4f}")
            print(f"Avg clf calls: {sum(T_calls) / d_len:.4f}")
            print()

