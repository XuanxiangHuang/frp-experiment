#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Deciding feature relevancy problem for monotonic classifier
#
################################################################################
import time
from pysat.formula import IDPool
from pysat.solvers import Solver
################################################################################


class FeatureRelevancy(object):
    """
        Feature relevancy problem.
    """
    def __init__(self, nv, free_feats, get_pred, classes, verb=0):
        self.nv = nv
        self.free_feats = free_feats
        self.get_pred = get_pred
        self.classes = classes
        self.verbose = verb

    def same_pred(self, clf, pred, low, up, dis_inst_low, dis_inst_up):
        assert pred in self.classes, f"prediction of tested instance not in {self.classes}"
        if self.classes[0] == pred:
            pred_low = pred
            pred_up = self.get_pred(clf, up)
            assert pred_low <= pred_up
            if pred_low == pred_up:
                return 1
            else:
                return 0
        elif self.classes[-1] == pred:
            pred_low = self.get_pred(clf, low)
            pred_up = pred
            assert pred_low <= pred_up
            if pred_low == pred_up:
                return 1
            else:
                return 0
        else:
            if dis_inst_low < dis_inst_up:
                pred_up = self.get_pred(clf, up)
                assert pred <= pred_up
                if pred != pred_up:
                    return 0
                pred_low = self.get_pred(clf, low)
                assert pred_low <= pred
                if pred != pred_low:
                    return 0
            else:
                pred_low = self.get_pred(clf, low)
                assert pred_low <= pred
                if pred != pred_low:
                    return 0
                pred_up = self.get_pred(clf, up)
                assert pred <= pred_up
                if pred != pred_up:
                    return 0
            assert pred_low == pred == pred_up
            return 1

    def reduce_pick(self, clf, inst, pred, min_max_val, fixed, times):
        fix = fixed.copy()
        for i in range(self.nv):
            if fix[i]:
                fix[i] = not fix[i]
                low, up, dis_inst_low, dis_inst_up = \
                    self.free_feats(inst, [k for k in range(self.nv) if not fix[k]], min_max_val)
                if not self.same_pred(clf=clf, pred=pred, low=low, up=up,
                                      dis_inst_low=dis_inst_low, dis_inst_up=dis_inst_up):
                    fix[i] = not fix[i]
                times -= 1
                if times == 0:
                    break
        new_pick = [i for i in range(self.nv) if fix[i]]
        return new_pick

    def reduce_drop(self, clf, inst, pred, min_max_val, universal, times):
        univ = universal.copy()
        for i in range(self.nv):
            if univ[i]:
                univ[i] = not univ[i]
                low, up, dis_inst_low, dis_inst_up = \
                    self.free_feats(inst, [k for k in range(self.nv) if univ[k]], min_max_val)
                if self.same_pred(clf=clf, pred=pred, low=low, up=up,
                                  dis_inst_low=dis_inst_low, dis_inst_up=dis_inst_up):
                    univ[i] = not univ[i]
                times -= 1
                if times == 0:
                    break
        new_drop = [i for i in range(self.nv) if univ[i]]
        return new_drop

    def cegar(self, clf, inst, feat_id, min_max_val, times=0):
        #########################################
        vpool = IDPool()

        def new_var(name):
            return vpool.id(f'{name}')
        #########################################
        if self.verbose:
            print('(Cegar) Feature Relevancy of Monotonic Classifiers ...')
        ###############################################################
        if self.verbose:
            print('Start solving...')
        time_solving_start = time.perf_counter()
        inst_ = inst.copy()
        slts = [new_var(f's_{i}') for i in range(self.nv)]
        pred_inst = self.get_pred(clf, inst_)
        ###############################################################
        weakaxp = []
        slv = Solver(name="Glucose4", use_timer=True)
        slv_calls = 0
        while slv.solve(assumptions=[slts[feat_id]]):
            slv_calls += 1
            ##############################
            pick = []
            model = slv.get_model()
            assert model
            for lit in model:
                name = vpool.obj(abs(lit)).split(sep='_')
                if name[0] == 's':
                    if lit > 0 and int(name[1]) != feat_id:
                        pick.append(int(name[1]))
            assert feat_id not in pick
            drop = []
            for i in range(self.nv):
                if i == feat_id:
                    continue
                if i not in pick:
                    drop.append(i)
            assert feat_id not in drop
            ##############################
            low, up, dis_inst_low, dis_inst_up = \
                self.free_feats(inst_, drop, min_max_val)
            if not self.same_pred(clf=clf, pred=pred_inst, low=low, up=up,
                                  dis_inst_low=dis_inst_low, dis_inst_up=dis_inst_up):
                ###########################
                # try to reduce drop set (optional)
                if times > 0:
                    univ = [False] * self.nv
                    for j in drop:
                        univ[j] = True
                    assert not univ[feat_id]
                    new_drop = self.reduce_drop(clf=clf, inst=inst_,
                                                pred=pred_inst, min_max_val=min_max_val, universal=univ,
                                                times=times)
                    if len(new_drop) < len(drop):
                        drop = new_drop
                ###########################
                ct_examp = [slts[i] for i in drop]
                assert ct_examp != []
                slv.add_clause(ct_examp)
            else:
                low, up, dis_inst_low, dis_inst_up = \
                    self.free_feats(inst_, drop+[feat_id], min_max_val)
                if not self.same_pred(clf=clf, pred=pred_inst, low=low, up=up,
                                      dis_inst_low=dis_inst_low, dis_inst_up=dis_inst_up):
                    weakaxp = pick + [feat_id]
                    break
                ###########################
                # try to reduce pick set (optional)
                if times > 0:
                    fix = [False] * self.nv
                    for j in pick:
                        fix[j] = True
                    assert not fix[feat_id]
                    new_pick = self.reduce_pick(clf=clf, inst=inst_,
                                                pred=pred_inst, min_max_val=min_max_val, fixed=fix,
                                                times=times)
                    if len(new_pick) < len(pick):
                        pick = new_pick
                ###########################
                slv.add_clause([-slts[i] for i in pick])
        weakaxp.sort()
        time_solving_end = time.perf_counter()
        solving_time = time_solving_end-time_solving_start
        sat_time = slv.time_accum()
        if self.verbose:
            print(f"Solving (CPU) time: {solving_time:.2f} secs")
            print('Calling SAT (CPU) time: {0:.4f} secs'.format(sat_time))
            print(f'Number of SAT calls: {slv_calls}')
        slv.delete()
        return weakaxp, solving_time, sat_time, slv_calls

    def extract(self, clf, inst, feat_id, weakaxp, min_max_val):
        assert feat_id in weakaxp
        time_solving_start = time.perf_counter()

        inst_ = inst.copy()
        pred_inst = self.get_pred(clf, inst_)
        univ = []
        fix = []
        for i in range(self.nv):
            if i not in weakaxp:
                univ.append(i)
            else:
                fix.append(i)

        for k in fix:
            if k == feat_id:
                continue
            low, up, dis_inst_low, dis_inst_up = \
                self.free_feats(inst_, univ+[k], min_max_val)
            if self.same_pred(clf=clf, pred=pred_inst, low=low, up=up,
                              dis_inst_low=dis_inst_low, dis_inst_up=dis_inst_up):
                univ.append(k)

        axp = []
        for i in range(self.nv):
            if i not in univ:
                axp.append(i)
        if self.verbose:
            print(f"AXp (size {len(axp)}): {axp}")
        time_solving_end = time.perf_counter()
        extracting_time = time_solving_end-time_solving_start
        if self.verbose:
            print(f"Extracting (CPU) time: {extracting_time:.2f} secs")
        return axp, extracting_time

