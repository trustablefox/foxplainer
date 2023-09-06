import pandas as pd
import pickle

from .pysat.solvers import Solver

class LRExplainer(object):
    def __init__(self, data, options):

        with open(options.classifier, 'rb') as f:
            self.model = pickle.load(f)
        self.options = options
        self.fnames = data.feature_names
        self.label = data.names[-1]
        self.data = data
        self.extract_bounds()

    def extract_bound(self, i):
        values = list(map(lambda l: l[i], self.data.X))
        return max(values), min(values)

    def extract_bounds(self):
        self.lbounds = []
        self.ubounds = []
        coefs = self.model.coef_[0]
        for i in range(len(self.data.extended_feature_names_as_array_strings)):
            coef = coefs[i]
            max_value, min_value = self.extract_bound(i)
            if coef >= 0:
                self.lbounds.append(min_value)
                self.ubounds.append(max_value)
            else:
                self.lbounds.append(max_value)
                self.ubounds.append(min_value)
        self.lbounds = pd.Series(self.lbounds, index=self.fnames)
        self.ubounds = pd.Series(self.ubounds, index=self.fnames)

    def free_attr(self, i, inst, lbounds, ubounds, deset, inset):
        self.inst = inst
        lbounds[i] = self.lbounds[i]
        ubounds[i] = self.ubounds[i]
        deset.remove(i)
        inset.add(i)

    def fix_attr(self, i, inst, lbounds, ubounds, deset, inset):
        lbounds[i] = inst[i]
        ubounds[i] = inst[i]
        deset.remove(i)
        inset.add(i)

    def equal_pred(self, lbounds, ubounds):
        return self.model.predict([lbounds])[0] == self.model.predict([ubounds])[0]

    def explain(self, inst):
        self.hypos = list(range(len(inst)))
        pred = self.model.predict([inst])[0]
        self.pred = pred
        self.time = {'abd': 0, 'con': 0}
        self.exps = {'abd': [], 'con': []}
        if self.options.xnum not in (-1, 'all'):
            if self.options.xtype in ['abd', 'abductive']:
                self.exps['abd'].append(self.extract_AXp(inst))
            else:
                self.exps['con'].append(self.extract_CXp(inst))
        else:
            self.exps = self.enumrate(inst)

        preamble = ['{0} = {1}'.format(self.fnames[i], inst[i]) for i in self.hypos]
        explained_instance = 'IF {0} THEN {1} = {2}'.format(' AND '.join(preamble), self.label, pred)

        explanation_list =  {'abd': [], 'con': []}
        explanation_size_list =  {'abd': [], 'con': []}

        #xtype = 'abd' if self.options.xtype in ['abd', 'abductive'] else 'con'
        for xtype in ['abd', 'con']:
            for exp in self.exps[xtype]:
                preamble = ['{0} {1} {2}'.format(self.fnames[i], '=' if xtype == 'abd' else '!=', inst[i])
                            for i in sorted(exp)]
                explanation = 'IF {} THEN {} {} {}'.format(' AND '.join(preamble),
                                                                self.label,
                                                                '=' if xtype == 'abd' else '!=',
                                                                pred)
                explanation_size = 'Number of Explained Features: {0}'.format(len(exp))
                explanation_list[xtype].append(explanation)
                explanation_size_list[xtype].append(explanation_size)

                """
                xtype_ = 'abd' if xtype == 'con' else 'con'
                
                for exp_ in self.exps[xtype_]:
                    preamble = ['{0} {1} {2}'.format(self.fnames[i], '=' if xtype_ == 'abd' else '!=', inst[i])
                                for i in sorted(exp_)]
                    print_xtype = 'Abductive Explanation' if xtype_ == 'abd' else 'Contrastive Explanation'
                    print('{}:\nIF {}\nTHEN {} {} {}'.format(print_xtype,
                                                             ' AND \n'.join(preamble),
                                                             self.label,
                                                             '=' if xtype_ == 'abd' else '!=',
                                                             pred))
                    print('Explanation Size: {0}'.format(len(exp_)))
                """
        return self.exps, self.time, explained_instance, explanation_list, explanation_size_list

    def extract_AXp(self, inst, seed=set()):
        lbounds = inst.copy()
        ubounds = inst.copy()
        candidate, drop, pick = set(self.hypos), set(), set()
        for i in seed:
            self.free_attr(i, inst, lbounds, ubounds, candidate, drop)
        potential = list(filter(lambda l: l not in seed, self.hypos))
        for i in potential:
            self.free_attr(i, inst, lbounds, ubounds, candidate, drop)
            if not self.equal_pred(lbounds, ubounds):
                self.fix_attr(i, inst, lbounds, ubounds, drop, pick)
        return pick

    def extract_CXp(self, inst, seed=set()):
        lbounds = self.lbounds.copy()
        ubounds = self.ubounds.copy()
        candidate, drop, pick = set(self.hypos), set(), set()
        for i in seed:
            self.fix_attr(i, inst, lbounds, ubounds, candidate, drop)
        potential = list(filter(lambda l: l not in seed, self.hypos))
        for i in potential:
            self.fix_attr(i, inst, lbounds, ubounds, candidate, drop)
            if self.equal_pred(lbounds, ubounds):
                self.free_attr(i, inst, lbounds, ubounds, drop, pick)
        return pick

    def enumrate(self, inst):
        oracle = Solver(name=self.options.solver)
        exps = {'abd': [], 'con': []}
        self.hit = set()
        while True:
            if not oracle.solve():
                return exps
            assignment = oracle.get_model()
            lbounds = self.lbounds.copy()
            ubounds = self.ubounds.copy()
            for i in self.hit:
                if assignment[i] > 0:
                    lbounds[i] = inst[i]
                    ubounds[i] = inst[i]
            if self.equal_pred(lbounds, ubounds):
                seed = set(self.hypos).difference(set(filter(lambda i: assignment[i] > 0, self.hit)))
                exp = self.extract_AXp(inst, seed)
                exps['abd'].append(exp)
                oracle.add_clause([-(i + 1) for i in sorted(exp)])
            else:
                seed = set(filter(lambda i: assignment[i] > 0, self.hit))
                exp = self.extract_CXp(inst, seed)
                exps['con'].append(exp)
                oracle.add_clause([i + 1 for i in sorted(exp)])
            self.hit.update(exp)
