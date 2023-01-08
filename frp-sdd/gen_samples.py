#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Generate tested instances/features
#
################################################################################
import sys
import random
import csv
import numpy as np
from pysdd.sdd import Vtree, SddManager, SddNode
################################################################################


def support_vars(sdd: SddManager):
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


def sdd_gen_tested_insts(sdd_file, vtree_file, circuit, num_test):
    name = circuit
    ######################  Pre-processing:  #####################
    sdd_file = bytes(sdd_file, 'utf-8')
    vtree_file = bytes(vtree_file, 'utf-8')
    vtree = Vtree.from_file(vtree_file)
    sdd = SddManager.from_vtree(vtree)
    sdd.auto_gc_and_minimize_off()
    root = sdd.read_sdd_file(sdd_file)
    tmp_sup_vars = support_vars(sdd)
    tmp_nv = len(tmp_sup_vars)
    tmp_features = [f"x_{ii}" for ii in range(1, tmp_nv + 1)]
    assert (len(tmp_features) == tmp_nv)

    sup_vars = []
    features = []
    for jj in range(tmp_nv):
        if tmp_sup_vars[jj]:
            sup_vars.append(tmp_sup_vars[jj])
            features.append(tmp_features[jj])
    nv = len(sup_vars)
    assert (len(features) == nv)
    ######################  Pre-processing:  #####################

    tested = set()
    d_len = num_test
    round_i = 0
    while round_i < d_len:
        tmp_sample = []
        for ii in range(tmp_nv):
            tmp_sample.append(random.randint(0, 1))
        while tuple(tmp_sample) in tested:
            tmp_sample = []
            for ii in range(tmp_nv):
                tmp_sample.append(random.randint(0, 1))

        assert tuple(tmp_sample) not in tested

        sample = []
        for ii in range(tmp_nv):
            if tmp_sup_vars[ii]:
                sample.append(tmp_sample[ii])

        lits = to_lits(sup_vars, sample)
        pred = prediction(root, lits)

        if pred:
            continue

        tested.add(tuple(tmp_sample))
        round_i += 1

    assert len(tested) == num_test
    data = []
    for item in tested:
        csv_item = list(item)
        assert len(csv_item) == tmp_nv
        sample = []
        for ii in range(tmp_nv):
            if tmp_sup_vars[ii]:
                sample.append(int(csv_item[ii]))

        lits = to_lits(sup_vars, sample)
        pred = prediction(root, lits)
        assert pred is False
        data.append(csv_item)

    with open(f"samples/test_insts/{name}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)

    return


def sdd_gen_tested_feats(sdd_file, vtree_file, circuit, num_test):
    name = circuit
    ######################  Pre-processing:  #####################
    sdd_file = bytes(sdd_file, 'utf-8')
    vtree_file = bytes(vtree_file, 'utf-8')
    vtree = Vtree.from_file(vtree_file)
    sdd = SddManager.from_vtree(vtree)
    sdd.auto_gc_and_minimize_off()
    root = sdd.read_sdd_file(sdd_file)
    tmp_sup_vars = support_vars(sdd)
    tmp_nv = len(tmp_sup_vars)
    tmp_features = [f"x_{ii}" for ii in range(1, tmp_nv + 1)]
    assert (len(tmp_features) == tmp_nv)

    sup_vars = []
    features = []

    for jj in range(tmp_nv):
        if tmp_sup_vars[jj]:
            sup_vars.append(tmp_sup_vars[jj])
            features.append(tmp_features[jj])
    nv = len(sup_vars)
    assert (len(features) == nv)

    ######################  Pre-processing:  #####################
    if num_test < nv:
        sample_seed_row = np.array(random.sample(list(range(nv)), num_test))
    else:
        sample_seed_row = np.array(np.random.choice(list(range(nv)), num_test))
    sample_seed_col = sample_seed_row.reshape(-1, 1)

    with open(f"samples/test_feats/{name}.csv", "w", newline="") as f:
        write = csv.writer(f)
        write.writerows(sample_seed_col)

    return


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 4 and args[0] == '-bench':
        bench_name = args[1]

        with open(bench_name, 'r') as fp:
            name_list = fp.readlines()

        for item in name_list:
            name = item.strip()
            print(name)
            circuit_sdd = f"sdds/{name}.sdd"
            circuit_sdd_vtree = f"sdds/{name}.vtree"
            if args[2] == '-inst':
                sdd_gen_tested_insts(circuit_sdd, circuit_sdd_vtree, name, int(args[3]))
            elif args[2] == '-feat':
                sdd_gen_tested_feats(circuit_sdd, circuit_sdd_vtree, name, int(args[3]))
    exit(0)
