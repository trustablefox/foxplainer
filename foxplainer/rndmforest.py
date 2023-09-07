from .data import Data
from .tree import Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import collections
from six.moves import range
import six
import math

from .pysat.formula import CNF, IDPool
from .pysat.solvers import Solver
from .pysat.card import CardEnc, EncType
from .pysat.examples.hitman import Hitman

import pickle


def pickle_load_file(filename):
    f =  open(filename, "rb")
    data = pickle.load(f)
    f.close()
    return data


class Dataset(Data):
    """
        Class for representing dataset (transactions).
    """
    def __init__(self, filename=None, fpointer=None, mapfile=None,
            separator=',', use_categorical = False):
        super().__init__(filename, fpointer, mapfile, separator, use_categorical)

        # split data into X and y
        self.feature_names = self.names[:-1]
        self.nb_features = len(self.feature_names)
        self.use_categorical = use_categorical

        samples = np.asarray(self.samps)
        if not all(c.isnumeric() for c in samples[:, -1]):
            le = LabelEncoder()
            le.fit(samples[:, -1])
            samples[:, -1]= le.transform(samples[:, -1])
            self.class_names = le.classes_

        samples = np.asarray(samples, dtype=np.float32)
        self.X = samples[:, 0: self.nb_features]
        self.y = samples[:, self.nb_features]
        self.num_class = len(set(self.y))
        self.target_name = list(range(self.num_class))

        # check if we have info about categorical features
        self.categorical_features = []
        self.categorical_names = []
        self.binarizer = []

        #feat map
        self.mapping_features()

    def transform(self, x):
        if (len(x.shape) == 1):
            x = np.expand_dims(x, axis=0)
        return x

    def mapping_features(self):
        self.extended_feature_names = {}
        self.extended_feature_names_as_array_strings = []
        counter = 0
        for i in range(self.nb_features):
            self.extended_feature_names.update({counter: (self.feature_names[i], None)})
            self.extended_feature_names_as_array_strings.append("f{}".format(i))#(self.feature_names[i])
            counter = counter + 1

    def readable_sample(self, x):
        readable_x = []
        for i, v in enumerate(x):
            readable_x.append(v)
        return np.asarray(readable_x)


"""
class VotingRF(VotingClassifier):
    def fit(self, X, y, sample_weight=None):
        self.estimators_ = []
        for _, est in self.estimators:
            self.estimators_.append(est)

        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_

    def predict(self, X):
        # 'hard' voting
        predictions = self._predict(X)
        predictions =  np.asarray(predictions, np.int64) #NEED TO BE CHECKED
        maj = np.apply_along_axis(
            lambda x: np.argmax(
                np.bincount(x, weights=self._weights_not_none)),
            axis=1, arr=predictions)
        maj = self.le_.inverse_transform(maj)
        return maj
"""

class RF2001(object):
    """
        The main class to train Random Forest Classifier (RFC).
    """
    def __init__(self, options):
        """
            Constructor.
        """
        self.forest = None
        self.voting = None
        self.opt = options
        param_dist = {'n_estimators': options.n_estimators,
                      'max_depth':options.maxdepth,
                      'random_state': 0,
                      'n_jobs': -1}
        self.forest = RandomForestClassifier(**param_dist)
    """
    def train(self, dataset, outfile=None):
        X_train, X_test, y_train, y_test = dataset.train_test_split()
        if self.opt.verb:
            dataset.test_encoding_transformes(X_train)
        X_train = dataset.transform(X_train)
        X_test = dataset.transform(X_test)
        print("Build a random forest.")
        self.forest.fit(X_train,y_train)
        rtrees = [ ('dt', dt) for i, dt in enumerate(self.forest.estimators_)]
        self.voting = VotingRF(estimators=rtrees)
        self.voting.fit(X_train,y_train)
        self.update_trees(dataset.extended_feature_names_as_array_strings)
        train_acc = accuracy_score(self.predict(X_train), y_train)
        test_acc = accuracy_score(self.predict(X_test), y_test)
        if self.opt.verb > 1:
            self.print_acc_vote(X_train, X_test, y_train, y_test)
            self.print_acc_prob(X_train, X_test, y_train, y_test)

        return train_acc, test_acc

    def update_trees(self, feature_names):
        self.trees = [build_tree(dt.tree_, feature_names) for dt in self.forest.estimators_]

    def predict(self, X):
        majs = []
        for id, inst in enumerate(X):
            scores = [predict_tree(dt, inst) for dt in self.trees]
            scores = np.asarray(scores)
            maj = np.argmax(np.bincount(scores))
            majs.append(maj)
        majs = np.asarray(majs)
        return majs

    def predict_prob(self, X):
        self.forest.predict(X)
    """

    def estimators(self):
        assert(self.forest.estimators_ is not None)
        return self.forest.estimators_
    
    """
    def n_estimators(self):
        return self.forest.n_estimators
    
    def print_acc_vote(self, X_train, X_test, y_train, y_test):
        train_acc = accuracy_score(self.predict(X_train), y_train)
        test_acc = accuracy_score(self.predict(X_test), y_test)
        print("----------------------")
        print("RF2001:")
        print("Train accuracy RF2001: {0:.2f}".format(100. * train_acc))
        print("Test accuracy RF2001: {0:.2f}".format(100. * test_acc))
        print("----------------------")

    def print_acc_prob(self, X_train, X_test, y_train, y_test):
        train_acc = accuracy_score(self.forest.predict(X_train), y_train)
        test_acc = accuracy_score(self.forest.predict(X_test), y_test)
        print("RF-scikit:")
        print("Train accuracy RF-scikit: {0:.2f}".format(100. * train_acc))
        print("Test accuracy RF-scikit: {0:.2f}".format(100. *  test_acc))
        print("----------------------")

    def print_accuracy(self, data, X_test, y_test):
        X_test = data.transform(X_test)
        test_acc = accuracy_score(self.predict(X_test), y_test)
        print("c Cross-Validation: {0:.2f}".format(100. * test_acc))
    """


class XRF(object):
    """
        class to encode and explain Random Forest classifiers.
    """
    def __init__(self, dataset, options):
        self.cls = RF2001(options)
        self.cls.forest = pickle_load_file(options.classifier)
        self.data = dataset
        self.label = dataset.names[-1]
        self.verbose = options.verb
        self.options = options
        assert (options.encode in [None, "sat", "maxsat"])
        self.opt_encoding = options.encode
        self.f = Forest(self.cls, dataset.extended_feature_names_as_array_strings)
        if options.verb > 2:
            self.f.print_trees()

    def encode(self, inst):
        """
            Encode a tree ensemble trained previously.
        """
        if 'f' not in dir(self):
            self.f = Forest(self.cls, self.data.extended_feature_names_as_array_strings)
        self.enc = SATEncoder(self.f, self.data.feature_names, self.data.num_class, \
                                  self.data.extended_feature_names_as_array_strings)
        inst = self.data.transform(np.array(inst))[0]
        _, _, _, _ = self.enc.encode(inst)

    def explain(self, inst):
        """
            Explain a prediction made for a given sample with a previously
            trained RF.
        """
        if 'enc' not in dir(self):
            self.encode(inst)
        inpvals = self.data.readable_sample(inst)
        preamble = []
        for f, v in zip(self.data.feature_names, inpvals):
            if f not in str(v):
                preamble.append('{0} = {1}'.format(f, v))
            else:
                preamble.append(v)
        inps = self.data.extended_feature_names_as_array_strings # input (feature value) variables
        self.x = SATExplainer(self.enc, inps, preamble, self.label, self.data.class_names, options=self.options, verb=self.verbose)
        inst = self.data.transform(np.array(inst))[0]
        expls, time, explained_instance, explanation, explanation_size = self.x.explain(np.array(inst))
        pred = self.cls.forest.predict([inst])[0]
        return expls, time, explained_instance, explanation, explanation_size, pred


class SATEncoder(object):
    """
        Encoder of Random Forest classifier into SAT.
    """
    def __init__(self, forest, feats, nof_classes, extended_feature_names,  from_file=None):
        self.feats = feats
        self.from_file = from_file
        self.forest = forest
        self.num_class = nof_classes
        self.vpool = IDPool()
        self.extended_feature_names = extended_feature_names
        #encoding formula
        self.cnf = None
        # for interval-based encoding
        self.intvs, self.imaps, self.ivars, self.thvars = None, None, None, None

    def newVar(self, name):
        if name in self.vpool.obj2id: #var has been already created
            return self.vpool.obj2id[name]
        var = self.vpool.id('{0}'.format(name))
        return var

    def traverse(self, tree, k, clause):
        """
            Traverse a tree and encode each node.
        """
        if tree.children:
            f = tree.name
            v = tree.threshold
            pos = neg = []
            if f in self.intvs:
                d = self.imaps[f][v]
                pos, neg = self.thvars[f][d], -self.thvars[f][d]
            else:
                var = self.newVar(tree.name)
                pos, neg = var, -var
            assert (pos and neg)
            self.traverse(tree.children[0], k, clause + [-neg])
            self.traverse(tree.children[1], k, clause + [-pos])
        else:  # leaf node
            cvar = self.newVar('class{0}_tr{1}'.format(tree.values,k))
            self.cnf.append(clause + [cvar])

    def compute_intervals(self):
        """
            Traverse all trees in the ensemble and extract intervals for each
            feature.

            At this point, the method only works for numerical datasets!
        """
        def traverse_intervals(tree):
            """
                Auxiliary function. Recursive tree traversal.
            """
            if tree.children:
                f = tree.name
                v = tree.threshold
                if f in self.intvs:
                    self.intvs[f].add(v)
                traverse_intervals(tree.children[0])
                traverse_intervals(tree.children[1])
        # initializing the intervals
        self.intvs = {'{0}'.format(f): set([]) for f in self.extended_feature_names if '_' not in f}
        for tree in self.forest.trees:
            traverse_intervals(tree)
        # OK, we got all intervals; let's sort the values
        self.intvs = {f: sorted(self.intvs[f]) + ([math.inf] if len(self.intvs[f]) else []) for f in six.iterkeys(self.intvs)}
        def int_feat(intvs):
            for intv in intvs[:-1]:
                if intv % 0.5 != 0:
                    return False
            return True
        self.imaps, self.ivars = {}, {}
        self.thvars = {}
        for feat, intvs in six.iteritems(self.intvs):
            self.imaps[feat] = {}
            self.ivars[feat] = []
            self.thvars[feat] = []
            is_int_feat = int_feat(intvs)
            for i, ub in enumerate(intvs):
                self.imaps[feat][ub] = i
                ivar = self.newVar('{0}_intv{1}'.format(feat, i))
                self.ivars[feat].append(ivar)
                if is_int_feat and ub % 1 == 0.5 and i > 0:
                    # non-existing ivars due to integer data types
                    if ub - intvs[i-1] == 0.5:
                        self.cnf.append([-ivar])
                if ub != math.inf:
                    thvar = self.newVar('{0}_th{1}'.format(feat, i))
                    self.thvars[feat].append(thvar)

    def maj_vote_const(self, ctvars):
        """
            capture majority class vote with cardinality constraints (Pseudo Boolean..)
        """
        # define Tautology var
        # vtaut = self.newVar('Tautology')
        num_tree = len(self.forest.trees)
        if(self.num_class == 2):
            rhs = math.floor(num_tree / 2) + 1
            if(self.cmaj==1 and not num_tree%2):
                rhs = math.floor(num_tree / 2)
            lhs = [ctvars[k][1 - self.cmaj] for k in range(num_tree)]
            atls = CardEnc.atleast(lits = lhs, bound = rhs, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(atls)
        """
        else:
            zvars = []
            zvars.append([self.newVar('z_0_{0}'.format(k)) for k in range (num_tree) ])
            zvars.append([self.newVar('z_1_{0}'.format(k)) for k in range (num_tree) ])
            rhs = num_tree
            lhs0 = zvars[0] + [ - ctvars[k][self.cmaj] for k in range(num_tree)]
            atls = CardEnc.atleast(lits = lhs0, bound = rhs, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(atls)
            rhs = num_tree + 1
            lhs1 =  zvars[1] + [ - ctvars[k][self.cmaj] for k in range(num_tree)]
            atls = CardEnc.atleast(lits = lhs1, bound = rhs, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(atls)
            pvars = [self.newVar('p_{0}'.format(k)) for k in range(self.num_class + 1)]
            for k,p in enumerate(pvars):
                for i in range(num_tree):
                    if k == 0:
                        z = zvars[0][i]
                        self.cnf.append([-p, z, -vtaut])
                    elif k == self.cmaj+1:
                        z = zvars[1][i]
                        self.cnf.append([-p, z, -vtaut])
                    else:
                        z = zvars[0][i] if (k<self.cmaj+1) else zvars[1][i]
                        self.cnf.append([-p, -z, ctvars[i][k-1] ])
                        self.cnf.append([-p, z, -ctvars[i][k-1] ])
            self.cnf.append([-pvars[0], -pvars[self.cmaj+1]])
            lhs1 =  pvars[:(self.cmaj+1)]
            eqls = CardEnc.equals(lits = lhs1, bound = 1, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(eqls)
            lhs2 = pvars[(self.cmaj + 1):]
            eqls = CardEnc.equals(lits = lhs2, bound = 1, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(eqls)
        """

    def encode(self, sample):
        """
            Do the job.
        """
        self.cnf = CNF()
        num_tree = len(self.forest.trees)
        # introducing class-tree variables
        ctvars = [[] for t in range(num_tree)]
        for k in range(num_tree):
            for j in range(self.num_class):
                var = self.newVar('class{0}_tr{1}'.format(j,k))
                ctvars[k].append(var)
        # traverse all trees and extract all possible intervals
        # for each feature
        self.compute_intervals()
        # traversing and encoding each tree
        for k, tree in enumerate(self.forest.trees):
            # encoding the tree
            self.traverse(tree, k, [])
            # exactly one class var is true
            card = CardEnc.atmost(lits=ctvars[k], vpool=self.vpool,encoding=EncType.cardnetwrk)
            self.cnf.extend(card.clauses)
        # calculate the majority class
        self.cmaj = self.forest.predict_inst(sample)
        #Cardinality constraint AtMostK to capture a j_th class
        self.maj_vote_const(ctvars)
        # enforce exactly one of the feature values to be chosen
        # (for categorical features)
        categories = collections.defaultdict(lambda: [])
        for f in self.extended_feature_names:
            if '_' in f:
                categories[f.split('_')[0]].append(self.newVar(f))
        """ categorical features 
        for c, feats in six.iteritems(categories):
            # exactly-one feat is True
            self.cnf.append(feats)
            card = CardEnc.atmost(lits=feats, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(card.clauses)
        """
        # lits of intervals
        for f, intvs in six.iteritems(self.ivars):
            if not len(intvs):
                continue
            self.cnf.append(intvs)
            card = CardEnc.atmost(lits=intvs, vpool=self.vpool, encoding=EncType.cardnetwrk)
            self.cnf.extend(card.clauses)
        for f, threshold in six.iteritems(self.thvars):
            for j, thvar in enumerate(threshold):
                d = j+1
                pos, neg = self.ivars[f][d:], self.ivars[f][:d]
                if j == 0:
                    assert(len(neg) == 1)
                    self.cnf.append([thvar, neg[-1]])
                    self.cnf.append([-thvar, -neg[-1]])
                else:
                    self.cnf.append([thvar, neg[-1], -threshold[j-1]])
                    self.cnf.append([-thvar, threshold[j-1]])
                    self.cnf.append([-thvar, -neg[-1]])
                if j == len(threshold) - 1:
                    assert(len(pos) == 1)
                    self.cnf.append([-thvar, pos[0]])
                    self.cnf.append([thvar, -pos[0]])
                else:
                    self.cnf.append([-thvar, pos[0], threshold[j+1]])
                    self.cnf.append([thvar, -pos[0]])
                    self.cnf.append([thvar, -threshold[j+1]])
        return self.cnf, self.intvs, self.imaps, self.ivars


"""
class MaxSATEncoder(SATEncoder):
    def __init__(self, forest, feats, nof_classes, extended_feature_names,  from_file=None):
        super(MaxSATEncoder, self).__init__(forest, feats, nof_classes, extended_feature_names, from_file)

    def maj_vote_const(self, ctvars):
        num_tree = len(self.forest.trees)
        self.soft = dict()
        for j in range(self.num_class):
            self.soft[j] = [ctvars[i][j] for i in range(num_tree)]
            assert any([(f"class{j}" in self.vpool.obj(abs(v))) for v in self.soft[j]])
"""


class SATExplainer(object):
    """
        An SAT-inspired minimal explanation extractor for Random Forest models.
    """

    def __init__(self, sat_enc, inps, preamble, label, target_name, options, verb=1):
        """
            Constructor.
        """
        self.enc = sat_enc
        self.label = label
        self.inps = inps  # input (feature value) variables
        self.target_name = target_name
        self.preamble = preamble
        self.options = options
        self.verbose = verb
        self.slv = None
        # number of oracle calls
        self.calls = 0

    def prepare(self, sample):
        """
            Prepare the oracle for computing an explanation.
        """
        self.assums = []  # var selectors to be used as assumptions
        self.sel2fid = {}  # selectors to original feature ids
        self.sel2vid = {}  # selectors to categorical feature ids
        self.sel2v = {} # selectors to (categorical/interval) values
        # preparing the selectors
        for i, (inp, val) in enumerate(zip(self.inps, sample), 1):
            """ TODO
            if '_' in inp:
                assert (inp not in self.enc.intvs)
                feat = inp.split('_')[0]
                selv = self.enc.newVar('selv_{0}'.format(feat))
                self.assums.append(selv)
                if selv not in self.sel2fid:
                    self.sel2fid[selv] = int(feat[1:])
                    self.sel2vid[selv] = [i - 1]
                else:
                    self.sel2vid[selv].append(i - 1)
                p = self.enc.newVar(inp)
                if not val:
                    p = -p
                else:
                    self.sel2v[selv] = p
                self.enc.cnf.append([-selv, p])
            """
            if len(self.enc.intvs[inp]):
                v = next((intv for intv in self.enc.intvs[inp] if intv >= val), None)
                assert(v is not None)
                selv = self.enc.newVar('selv_{0}'.format(inp))
                self.assums.append(selv)
                assert (selv not in self.sel2fid)
                self.sel2fid[selv] = int(inp[1:])
                self.sel2vid[selv] = [i - 1]
                for j,p in enumerate(self.enc.ivars[inp]):
                    cl = [-selv]
                    if j == self.enc.imaps[inp][v]:
                        cl += [p]
                        self.sel2v[selv] = p
                    else:
                        cl += [-p]
                    self.enc.cnf.append(cl)

    def explain(self, sample):
        """
            Hypotheses minimization.
        """
        explained_instance = 'IF {0} THEN {1} = {2}'.format(' AND '.join(self.preamble),
                                                            self.label,
                                                            self.target_name[self.enc.cmaj])
        #create a SAT solver
        self.slv = Solver(name="glucose3")
        # adapt the solver to deal with the current sample
        self.prepare(sample)
        self.assums = sorted(set(self.assums))
        # pass a CNF formula
        self.slv.append_formula(self.enc.cnf)
        self.time = {'abd': 0, 'con': 0}

        if self.options.xtype in ('abductive', 'abd') and self.options.xnum not in (-1, 'all'):
            self.expls = [self.extract_mus()]
        else:
            self.mhs_mus_enumeration()
        self.expls = list(map(lambda l: sorted([self.sel2fid[h] for h in l ]), self.expls))
        # delete sat solver
        self.slv.delete()
        self.slv = None

        explanation_list = {'abd': [], 'con': []}
        explanation_size_list = {'abd': [], 'con': []}

        #if self.options.xtype in ('abductive', 'abd'):
        #xtype = 'abd'
        if self.options.xtype in ('abductive', 'abd'):
            for expl in self.expls:
                preamble = [self.preamble[i] for i in expl]
                explanation = 'IF {0} THEN {1} {2} {3}'.format(' AND '.join(preamble),
                                                                self.label,
                                                                '=' if self.options.xtype in ('abductive', 'abd') else '!=',
                                                                self.target_name[self.enc.cmaj])
                explanation_size = 'Number of Explained Features: {0}'.format(len(expl))
                explanation_list['abd'].append(explanation)
                explanation_size_list['abd'].append(explanation_size)

        if not (self.options.xtype in ('abductive', 'abd') and self.options.xnum not in (-1, 'all')):
            xtype = 'con'
            expls_ = list(map(lambda l: sorted([self.sel2fid[h] for h in l]), self.duals))
            for expl in expls_:
                preamble = [self.preamble[i].replace(' = ', ' != ') for i in expl]
                explanation = "IF {0} THEN {1} {2} {3}".format( 'AND '.join(preamble),
                                                                self.label,
                                                                '=' if xtype == 'abd' else '!=',
                                                                self.target_name[self.enc.cmaj])
                explanation_size = 'Number of Explained Features: {0}'.format(len(expl))
                explanation_list['con'].append(explanation)
                explanation_size_list['con'].append(explanation_size)

        self.expls = {self.options.xtype: self.expls}
        return self.expls, self.time, explained_instance, explanation_list, explanation_size_list

    def extract_mus(self, start_from=None):
        """
            Compute any subset-minimal explanation.
        """
        self.nsat, self.nunsat = 0, 0
        self.stimes, self.utimes = [], []
        vtaut = self.enc.newVar('Tautology')
        def _do_linear(core):
            """
                Do linear search.
            """
            def _assump_needed(a):
                if len(to_test) > 1:
                    to_test.remove(a)
                    self.calls += 1
                    sat = self.slv.solve(assumptions=[vtaut] + sorted(to_test))
                    if not sat:
                        self.nunsat += 1
                        return False
                    to_test.add(a)
                    self.nsat += 1
                    return True
                else:
                    return True
            to_test = set(core)
            return list(filter(lambda a: _assump_needed(a), core))
        if start_from is None:
            # this call must be unsatisfiable!
            assert self.slv.solve(assumptions=[vtaut] + self.assums) == False
        else:
            assert self.slv.solve(assumptions=[vtaut] + start_from) == False
        # this is our MUS over-approximation
        core = self.slv.get_core()
        core = list(filter(lambda l: l != vtaut, core))
        expl = _do_linear(core)
        return expl

    def mhs_mus_enumeration(self):
        """
            Enumerate subset- and cardinality-minimal explanations.
        """
        # result
        self.expls = []
        # just in case, let's save dual (contrastive) explanations
        self.duals = []
        vtaut = self.enc.newVar('Tautology')
        with Hitman(bootstrap_with=[self.assums], htype='sorted' if self.options.smallest else 'lbx') as hitman:
            # computing unit-size MCSes
            for i, hypo in enumerate(self.assums):
                self.calls += 1
                if self.slv.solve(assumptions=[vtaut] + self.assums[:i] + self.assums[(i + 1):]):
                    hitman.hit([hypo])
                    self.duals.append([hypo])

                    if self.options.xtype in ('con', 'contrastive') and self.options.xnum not in (-1, 'all'):
                        return

                else:
                    pass
            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1
                if hset is None:
                    break
                self.calls += 1
                if self.slv.solve(assumptions=[vtaut] + hset):
                    to_hit = []
                    satisfied, unsatisfied = [], []
                    removed = list(set(self.assums).difference(set(hset)))
                    model = self.slv.get_model()
                    for h in removed:
                        if model[abs(h) - 1] != h:
                            unsatisfied.append(h)
                        else:
                            hset.append(h)
                    # computing an MCS (expensive)
                    for h in unsatisfied:
                        self.calls += 1
                        if self.slv.solve(assumptions=[vtaut] + hset + [h]):
                            hset.append(h)
                        else:
                            to_hit.append(h)
                    hitman.hit(to_hit)
                    self.duals.append(to_hit)

                    if self.options.xtype in ('con', 'contrastive') and self.options.xnum not in (-1, 'all'):
                        break

                else:
                    self.expls.append(hset)
                    if len(self.expls) != self.options.xnum:
                        hitman.block(hset)
                    else:
                        break
