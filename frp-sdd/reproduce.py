#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Reproduce experiments
#
################################################################################
import resource
import sys
from math import ceil
from pysdd.sdd import Vtree, SddManager, SddNode
from sdd_feature_relevancy import FeatureRelevancy as FRPSDD
################################################################################


def support_vars(sdd: SddManager):
    """
        Given a SDD manager, return support variables,
        i.e. variables that used/referenced by SDD node.
        :param sdd: SDD manager
        :return:
    """
    all_vars = [_ for _ in sdd.vars]
    nv = len(all_vars)
    sup_vars = [None] * nv

    for i in range(nv):
        lit = all_vars[i].literal
        assert (lit == i + 1)
        neglit = -all_vars[i].literal
        if sdd.is_var_used(lit) or sdd.is_var_used(neglit):
            sup_vars[i] = all_vars[i]
    return sup_vars


def to_lits(sup_vars, inst):
    lits = [None] * len(inst)

    for j in range(len(inst)):
        if sup_vars[j]:
            if int(inst[j]):
                lits[j] = sup_vars[j].literal
            else:
                lits[j] = -sup_vars[j].literal
    return lits


def prediction(root: SddNode, lits):
    out = root
    for item in lits:
        if item:
            out = out.condition(item)
    assert out.is_true() or out.is_false()
    return True if out.is_true() else False


def sdd_get_one_axp(sdd_file, vtree_file, circuit, insts_file, feats_file, method='kc'):
    name = circuit
    ######################  Pre-processing: original #####################
    # string to bytes
    sdd_file = bytes(sdd_file, 'utf-8')
    vtree_file = bytes(vtree_file, 'utf-8')
    vtree = Vtree.from_file(vtree_file)
    sdd = SddManager.from_vtree(vtree)
    # Disable gc and minimization
    sdd.auto_gc_and_minimize_off()
    root = sdd.read_sdd_file(sdd_file)
    root.ref()
    sdd.garbage_collect()
    assert not root.garbage_collected()
    # obtain all variables (don't cared variables are None)
    tmp_sup_vars = support_vars(sdd)
    tmp_nv = len(tmp_sup_vars)
    # get all features
    tmp_features = [f"x_{ii}" for ii in range(1, tmp_nv + 1)]
    assert (len(tmp_features) == tmp_nv)

    # extract cared features and variables
    sup_vars = []
    features = []
    for jj in range(tmp_nv):
        if tmp_sup_vars[jj]:
            sup_vars.append(tmp_sup_vars[jj])
            features.append(tmp_features[jj])
    nv = len(sup_vars)
    assert (len(features) == nv)
    ######################  Pre-processing: original #####################
    T_time = []
    tested = set()
    sat_axps = []
    cnf_nv = []
    cnf_claus = []
    ###################### read instance file ######################
    with open(insts_file, 'r') as fp:
        inst_lines = fp.readlines()
    ###################### read instance file ######################
    ########### read feature file ###########
    with open(feats_file, 'r') as fp:
        feat_lines = fp.readlines()
    ########### read feature file ###########
    assert len(inst_lines) == len(feat_lines)
    d_len = len(inst_lines)
    failed_count = 0

    for i, s in enumerate(inst_lines):
        tmp_inst = [int(v.strip()) for v in s.split(',')]
        # extract value of cared features
        assert tuple(tmp_inst) not in tested
        tested.add(tuple(tmp_inst))

        assert len(tmp_inst) == tmp_nv

        inst = []
        for ii in range(tmp_nv):
            if tmp_sup_vars[ii]:
                inst.append(tmp_inst[ii])

        lits = to_lits(sup_vars, inst)
        pred = prediction(root, lits)

        assert pred is False

        feat_mem = FRPSDD(root, nv, features, sup_vars, verb=1)

        time_i_start = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                       resource.getrusage(resource.RUSAGE_SELF).ru_utime

        f_id = int(feat_lines[i])
        assert 0 <= f_id <= nv - 1
        print(f"{name}, {i}-th inst file out of {d_len}")
        print(f"SAT encoding: query on feature {f_id} out of {nv} features:")
        sat_axp = []
        sat_weakaxp, nv_cnf, claus_cnf, failed = feat_mem.answer(lits, f_id)
        if failed:
            failed_count += 1
        elif sat_weakaxp:
            print("Answer Yes")
            sat_axp, nv_cnf_, claus_cnf_ = feat_mem.extract(lits, f_id, sat_weakaxp, method)
            if method == 'enc':
                nv_cnf += nv_cnf_
                claus_cnf += claus_cnf_
            sat_axps.append(sat_axp)
        else:
            print('=============== no AXp exists ===============')
        cnf_nv.append(nv_cnf)
        cnf_claus.append(claus_cnf)

        time_i = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                 resource.getrusage(resource.RUSAGE_SELF).ru_utime - time_i_start

        if sat_axp:
            assert f_id in sat_axp
            print('=== check AXp ===')
            assert feat_mem.check_one_axp(lits, sat_axp) is True
            print('=== check succeed ===')

        if method == 'kc':
            print('=== SDD garbage collection ===')
            sdd.garbage_collect()
            assert not root.garbage_collected()

        T_time.append(time_i)

    exp_results = f"{name} & {d_len} & "
    exp_results += f"{nv} & nn & "
    exp_results += f"{ceil(len(sat_axps) / d_len * 100):.0f} ({ceil(failed_count / d_len * 100):.0f}) & "
    exp_results += f"{max([len(x) for x in sat_axps]):.0f} & "
    exp_results += f"{ceil(sum([len(x) for x in sat_axps]) / len(sat_axps)):.0f} & "
    exp_results += "{0:.0f} & {1:.0f} & " \
        .format(sum(cnf_nv) / d_len, sum(cnf_claus) / d_len)
    exp_results += "{0:.1f} & {1:.1f} & {2:.1f} & {3:.1f}\n" \
        .format(sum(T_time), max(T_time), min(T_time), sum(T_time) / d_len)

    print(exp_results)

    with open('results/sdd_frp.txt', 'a') as f:
        f.write(exp_results)

    return


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) >= 1 and args[0] == '-bench':
        bench_name = args[1]

        with open(bench_name, 'r') as fp:
            name_list = fp.readlines()

        for item in name_list:
            name = item.strip()
            print(f"############ {name} ############")
            circuit_sdd = f"sdds/{name}.sdd"
            circuit_sdd_vtree = f"sdds/{name}.vtree"
            test_insts = f"samples/test_insts/{name}.csv"
            test_feats = f"samples/test_feats/{name}.csv"
            sdd_get_one_axp(circuit_sdd, circuit_sdd_vtree, name, test_insts, test_feats)
    exit(0)
