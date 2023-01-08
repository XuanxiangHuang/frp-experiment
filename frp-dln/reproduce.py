#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Reproduce experiments
#
################################################################################
from math import ceil
from statistics import median
import logging, time, sys
import pandas as pd
import numpy as np
import tensorflow as tf
from mono_feature_relevancy import FeatureRelevancy as frp
logging.disable(sys.maxsize)
################################################################################


clf_calls_count = 0
clf_cpu_time = 0
features = None


def get_pred_binary_timer(mclf, datapoint):
    global clf_calls_count
    global clf_cpu_time
    time_start = time.perf_counter()
    raw_prediction = mclf.predict(datapoint, verbose=0)
    time_end = time.perf_counter()
    clf_cpu_time += (time_end - time_start)
    clf_calls_count += 1
    return round(raw_prediction[0][0])


def get_pred_multi_timer(mclf, datapoint):
    global clf_calls_count
    global clf_cpu_time
    time_start = time.perf_counter()
    raw_prediction = mclf.predict(datapoint, verbose=0)
    time_end = time.perf_counter()
    clf_cpu_time += (time_end - time_start)
    clf_calls_count += 1
    return np.argmax(raw_prediction[0])


def free_feats_complex(inst, univ, mv_Mv):
    assert len(mv_Mv) == len(features)
    lower_bound = inst.copy()
    upper_bound = inst.copy()
    dis_inst_low = 0
    dis_inst_up = 0
    for ii in univ:
        dis_i = mv_Mv[ii][1] - mv_Mv[ii][0]
        assert dis_i != 0.0
        cval = inst[ii].values[0]
        if cval == mv_Mv[ii][0]:
            assert cval < mv_Mv[ii][1]
        elif cval == mv_Mv[ii][1]:
            assert mv_Mv[ii][0] < cval
        else:
            assert mv_Mv[ii][0] < cval < mv_Mv[ii][1]
        dis_inst_low += ( (cval - mv_Mv[ii][0]) / dis_i )
        dis_inst_up += ( (mv_Mv[ii][1] - cval) / dis_i )
        lower_bound[ii] = pd.Series(data=mv_Mv[ii][0], name=features[ii])
        upper_bound[ii] = pd.Series(data=mv_Mv[ii][1], name=features[ii])
    ret_dis_inst_low = round(dis_inst_low, 4)
    ret_dis_inst_up = round(dis_inst_up, 4)
    return lower_bound, upper_bound, ret_dis_inst_low, ret_dis_inst_up


if __name__ == '__main__':
    datasets = ["australian", "breast_cancer", "heart_c", "nursery", "pima-modified"]
    for data_name in datasets:
        test_insts_df = pd.read_csv(f"samples/test_insts/{data_name}.csv")
        test_feat = f"samples/test_feats/{data_name}.csv"

        with open(test_feat, 'r') as fp:
            feat_lines = fp.readlines()

        feature_names = list(test_insts_df.columns)
        class_name = feature_names.pop()
        feature_name_indices = {name: index for index, name in enumerate(feature_names)}

        min_max_val = dict()
        attr_domain = dict()
        for feat in feature_names:
            vals = list(test_insts_df[feat].unique())
            vals.sort()
            attr_domain.update({feature_name_indices[feat]: vals})
            min_max_val.update({feature_name_indices[feat]: (min(vals), max(vals))})

        class_vals = list(test_insts_df[class_name].unique())
        class_vals.sort()
        attr_domain.update({class_name: class_vals})

        features = feature_names
        print(f'########## {data_name} ##########')
        print(f'# num of features={len(features)}')
        print(f'# feature names: {features}')
        print(f'# num of classes={len(class_vals)}')
        print(f'# ordered classes: {class_vals}')

        if len(class_vals) == 2:
            get_pred_timer = get_pred_binary_timer
        else:
            get_pred_timer = get_pred_multi_timer

        ################################################################################
        axps = []
        mclf_N_calls = []
        mclf_T_time = []
        sat_N_calls = []
        sat_T_time = []
        T_time = []
        runtime = ""
        sat_call_time = ""
        num_sat_calls = ""
        mclf_call_time = ""
        num_mclf_calls = ""

        assert len(test_insts_df) == len(feat_lines)
        d_len = len(test_insts_df)
        nv = len(feature_names)
        mn_clf = tf.keras.models.load_model(f'DLNs/{data_name}')
        frp_monoc = frp(nv=nv, free_feats=free_feats_complex,
                        get_pred=get_pred_timer, classes=class_vals, verb=1)

        for i in range(d_len):
            x_i = test_insts_df.loc[i][feature_names]
            inst_ = []
            assert list(x_i.index) == feature_names
            for name, val in zip(feature_names, x_i.values):
                inst_.append(pd.Series(data=val, name=name))
            target_feat = int(feat_lines[i])

            clf_cpu_time = 0
            clf_calls_count = 0
            time_i = 0
            print(f"{i}-th instance out of {d_len}")
            print(f"Query on feature {target_feat} out of {nv} features:")
            weakaxp, solving_time, sat_time, sat_calls = frp_monoc.cegar(
                clf=mn_clf, inst=inst_, feat_id=target_feat,
                min_max_val=min_max_val)
            time_i += solving_time
            axp = []
            if weakaxp:
                print(f"Weak AXp (size {len(weakaxp)}): {weakaxp}")
                axp, extracting_time = frp_monoc.extract(
                    clf=mn_clf, inst=inst_, feat_id=target_feat,
                    weakaxp=weakaxp, min_max_val=min_max_val)
                time_i += extracting_time
                axps.append(axp)
            else:
                print('=============== no AXp exists ===============')

            print(f"Calling Mono-Clf time: {clf_cpu_time:.2f}")
            print(f"Number of Mono-Clf calls: {clf_calls_count}")
            sat_T_time.append(sat_time)
            sat_N_calls.append(sat_calls)
            mclf_T_time.append(clf_cpu_time)
            mclf_N_calls.append(clf_calls_count)
            runtime += f"{time_i:.2f}\n"
            sat_call_time += f"{sat_time:.4f}\n"
            num_sat_calls += f"{sat_calls}\n"
            mclf_call_time += f"{clf_cpu_time:.2f}\n"
            num_mclf_calls += f"{clf_calls_count}\n"

            T_time.append(time_i)

        exp_results = f"{data_name} & {d_len} & "
        exp_results += f"{nv} & "
        exp_results += f"{ceil(len(axps) / d_len * 100):.0f} & "
        exp_results += f"{max([len(x) for x in axps]):.0f} & "
        exp_results += f"{median([len(x) for x in axps]):.0f} & "
        exp_results += "{0:.2f} & {1:.2f} & {2:.2f} & {3:.2f} & " \
            .format(sum(T_time), max(T_time), min(T_time), median(T_time))
        exp_results += "{0:.2f} & {1:.2f} & {2:.0f} & {3:.0f} & " \
            .format(max(sat_T_time), median(sat_T_time), max(sat_N_calls), median(sat_N_calls))
        exp_results += "{0:.2f} & {1:.2f} & {2:.0f} & {3:.0f}\n" \
            .format(max(mclf_T_time), median(mclf_T_time), max(mclf_N_calls), median(mclf_N_calls))

        print(exp_results)

        with open(f'results/runtime/{data_name}_time.txt', 'w') as f:
            f.write(runtime)
        with open(f'results/runtime/{data_name}_sat_call_time.txt', 'w') as f:
            f.write(sat_call_time)
        with open(f'results/calls/{data_name}_num_sat_calls.txt', 'w') as f:
            f.write(num_sat_calls)
        with open(f'results/runtime/{data_name}_mclf_call_time.txt', 'w') as f:
            f.write(mclf_call_time)
        with open(f'results/calls/{data_name}_num_mclf_calls.txt', 'w') as f:
            f.write(num_mclf_calls)
        with open('results/mono_frp.txt', 'a') as f:
            f.write(exp_results)