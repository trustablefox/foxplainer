#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## hitman.py
##
##      A minimum/minimal hitting set enumerator based on MaxSAT solving
##      and also MCS enumeration (LBX- or CLD-like). MaxSAT-based hitting
##      set enumeration computes hitting sets in a sorted manner, e.g. from
##      smallest size to largest size. MCS-based hitting set solver computes
##      arbitrary hitting sets, with no respect to their size.
##
##  Created on: Aug 23, 2018
##      Author: Alexey S. Ignatiev
##      E-mail: aignatiev@ciencias.ulisboa.pt
##

"""
    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        Hitman

    ==================
    Module description
    ==================

    A SAT-based implementation of an implicit minimal hitting set [1]_
    enumerator. The implementation is capable of computing/enumerating
    cardinality- and subset-minimal hitting sets of a given set of sets.
    Cardinality-minimal hitting set enumeration can be seen as ordered (sorted
    by size) subset-minimal hitting enumeration.

    The minimal hitting set problem is trivially formulated as a MaxSAT formula
    in WCNF, as follows. Assume :math:`E=\{e_1,\ldots,e_n\}` to be a universe
    of elements. Also assume there are :math:`k` sets to hit:
    :math:`s_i=\{e_{i,1},\ldots,e_{i,j_i}\}` s.t. :math:`e_{i,l}\in E`. Every
    set :math:`s_i=\{e_{i,1},\ldots,e_{i,j_i}\}` is translated into a hard
    clause :math:`(e_{i,1} \\vee \ldots \\vee e_{i,j_i})`. This results in the
    set of hard clauses having size :math:`k`. The set of soft clauses
    comprises unit clauses of the form :math:`(\\neg{e_{j}})` s.t.
    :math:`e_{j}\in E`, each having weight 1.

    Taking into account this problem formulation as MaxSAT, ordered hitting
    enumeration is done with the use of the state-of-the-art MaxSAT solver
    called :class:`.RC2` [2]_ [3]_ [4]_ while unordered hitting set enumeration
    is done through the *minimal correction subset* (MCS) enumeration, e.g.
    using the :class:`.LBX`- [5]_ or :class:`.MCSls`-like [6]_ MCS enumerators.

    .. [1] Erick Moreno-Centeno, Richard M. Karp. *The Implicit Hitting Set
        Approach to Solve Combinatorial Optimization Problems with an
        Application to Multigenome Alignment*. Operations Research 61(2). 2013.
        pp. 453-468

    .. [2] António Morgado, Carmine Dodaro, Joao Marques-Silva. *Core-Guided
        MaxSAT with Soft Cardinality Constraints*. CP 2014. pp. 564-573

    .. [3] António Morgado, Alexey Ignatiev, Joao Marques-Silva. *MSCG: Robust
        Core-Guided MaxSAT Solving*. JSAT 9. 2014. pp. 129-134

    .. [4] Alexey Ignatiev, António Morgado, Joao Marques-Silva. *RC2: a
        Python-based MaxSAT Solver*. MaxSAT Evaluation 2018. p. 22

    .. [5] Carlos Mencía, Alessandro Previti, Joao Marques-Silva.
        *Literal-Based MCS Extraction*. IJCAI. 2015. pp. 1973-1979

    .. [6] Joao Marques-Silva, Federico Heras, Mikolás Janota,
        Alessandro Previti, Anton Belov. *On Computing Minimal Correction
        Subsets*. IJCAI. 2013. pp. 615-622

    :class:`Hitman` supports hitting set enumeration in the *implicit* manner,
    i.e. when sets to hit can be added on the fly as well as hitting sets can
    be blocked on demand.

    An example usage of :class:`Hitman` through the Python ``import`` interface
    is shown below. Here we target unordered subset-minimal hitting set
    enumeration.

    .. code-block:: python

        >>> from pysat.examples.hitman import Hitman
        >>>
        >>> h = Hitman(solver='m22', htype='lbx')
        >>> # adding sets to hit
        >>> h.hit([1, 2, 3])
        >>> h.hit([1, 4])
        >>> h.hit([5, 6, 7])
        >>>
        >>> h.get()
        [1, 5]
        >>>
        >>> h.block([1, 5])
        >>>
        >>> h.get()
        [2, 4, 5]
        >>>
        >>> h.delete()

    Enumerating cardinality-minimal hitting sets can be done as follows:

    .. code-block:: python

        >>> from pysat.examples.hitman import Hitman
        >>>
        >>> sets = [[1, 2, 3], [1, 4], [5, 6, 7]]
        >>> with Hitman(bootstrap_with=sets, htype='sorted') as hitman:
        ...     for hs in hitman.enumerate():
        ...         print(hs)
        ...
        [1, 5]
        [1, 6]
        [1, 7]
        [3, 4, 7]
        [2, 4, 7]
        [3, 4, 6]
        [3, 4, 5]
        [2, 4, 6]
        [2, 4, 5]

    Finally, implicit hitting set enumeration can be used in practical problem
    solving. As an example, let us show the basic flow of a MaxHS-like [7]_
    algorithm for MaxSAT:

    .. code-block:: python

        >>> from pysat.examples.hitman import Hitman
        >>> from pysat.solvers import Solver
        >>>
        >>> hitman = Hitman(htype='sorted')
        >>> oracle = Solver()
        >>>
        >>> # here we assume that the SAT oracle
        >>> # is initialized with a MaxSAT formula,
        >>> # whose soft clauses are extended with
        >>> # selector literals stored in "sels"
        >>> while True:
        ...     hs = hitman.get()  # hitting the set of unsatisfiable cores
        ...     ts = set(sels).difference(set(hs))  # soft clauses to try
        ...
        ...     if oracle.solve(assumptions=ts):
        ...         print('s OPTIMUM FOUND')
        ...         print('o', len(hs))
        ...         break
        ...     else:
        ...         core = oracle.get_core()
        ...         hitman.hit(core)

    .. [7] Jessica Davies, Fahiem Bacchus. *Solving MAXSAT by Solving a
        Sequence of Simpler SAT Instances*. CP 2011. pp. 225-239

    ==============
    Module details
    ==============
"""

#
#==============================================================================
from ..formula import IDPool, WCNFPlus
from .rc2 import RC2, RC2Stratified
from .lbx import LBX
from .mcsls import MCSls
import six


#
#==============================================================================
class Atom(object):
    """
        Atoms are elementary (signed) objects necessary when dealing with
        hitting sets subject to hard constraints.
    """

    def __init__(self, obj, sign=True):
        """
            Simple atom initialiser.
        """

        self.obj = obj
        self.sign = sign


#
#==============================================================================
class Hitman(object):
    """
        A cardinality-/subset-minimal hitting set enumerator. The enumerator
        can be set up to use either a MaxSAT solver :class:`.RC2` or an MCS
        enumerator (either :class:`.LBX` or :class:`.MCSls`). In the former
        case, the hitting sets enumerated are ordered by size (smallest size
        hitting sets are computed first), i.e. *sorted*. In the latter case,
        subset-minimal hitting are enumerated in an arbitrary order, i.e.
        *unsorted*.

        This is handled with the use of parameter ``htype``, which is set to be
        ``'sorted'`` by default. The MaxSAT-based enumerator can be chosen by
        setting ``htype`` to one of the following values: ``'maxsat'``,
        ``'mxsat'``, or ``'rc2'``. Alternatively, by setting it to ``'mcs'`` or
        ``'lbx'``, a user can enforce using the :class:`.LBX` MCS enumerator.
        If ``htype`` is set to ``'mcsls'``, the :class:`.MCSls` enumerator is
        used.

        In either case, an underlying problem solver can use a SAT oracle
        specified as an input parameter ``solver``. The default SAT solver is
        Glucose3 (specified as ``g3``, see :class:`.SolverNames` for details).

        Objects of class :class:`Hitman` can be bootstrapped with an iterable
        of iterables, e.g. a list of lists. This is handled using the
        ``bootstrap_with`` parameter. Each set to hit can comprise elements of
        any type, e.g. integers, strings or objects of any Python class, as
        well as their combinations. The bootstrapping phase is done in
        :func:`init`.

        Another optional parameter ``subject_to`` can be used to specify
        arbitrary hard constraints that must be respected when computing
        hitting sets of the given sets. Note that ``subject_to`` should be an
        iterable containing pure clauses and/or native AtMostK constraints.
        Note that native cardinality constraints supported only by
        MiniCard-like solvers. Finally, note that these hard constraints must
        be defined over the set of signed atomic objects, i.e. instances of
        class :class:`.Atom`.

        A few other optional parameters include the possible options for RC2
        as well as for LBX- and MCSls-like MCS enumerators that control the
        behaviour of the underlying solvers.

        :param bootstrap_with: input set of sets to hit
        :param weights: a mapping from objects to their weights (if weighted)
        :param subject_to: hard constraints (either clauses or native AtMostK constraints)
        :param solver: name of SAT solver
        :param htype: enumerator type
        :param mxs_adapt: detect and process AtMost1 constraints in RC2
        :param mxs_exhaust: apply unsatisfiable core exhaustion in RC2
        :param mxs_minz: apply heuristic core minimization in RC2
        :param mxs_trim: trim unsatisfiable cores at most this number of times
        :param mcs_usecld: use clause-D heuristic in the MCS enumerator

        :type bootstrap_with: iterable(iterable(obj))
        :type weights: dict(obj)
        :type subject_to: iterable(iterable(Atom))
        :type solver: str
        :type htype: str
        :type mxs_adapt: bool
        :type mxs_exhaust: bool
        :type mxs_minz: bool
        :type mxs_trim: int
        :type mcs_usecld: bool
    """

    def __init__(self, bootstrap_with=[], weights=None, subject_to=[],
            solver='g3', htype='sorted', mxs_adapt=False, mxs_exhaust=False,
            mxs_minz=False, mxs_trim=0, mcs_usecld=False):
        """
            Constructor.
        """

        # hitting set solver
        self.oracle = None

        # name of SAT solver
        self.solver = solver

        # various oracle options
        self.adapt    = mxs_adapt
        self.exhaust  = mxs_exhaust
        self.minz     = mxs_minz
        self.trim     = mxs_trim
        self.usecld   = mcs_usecld

        # hitman type: either a MaxSAT solver or an MCS enumerator
        if htype in ('maxsat', 'mxsat', 'rc2', 'sorted'):
            self.htype = 'rc2'
        elif htype in ('mcs', 'lbx'):
            self.htype = 'lbx'
        else:  # 'mcsls'
            self.htype = 'mcsls'

        # pool of variable identifiers (for objects to hit)
        self.idpool = IDPool()

        # initialize hitting set solver
        self.init(bootstrap_with, weights=weights, subject_to=subject_to)

    def __del__(self):
        """
            Destructor.
        """

        self.delete()

    def __enter__(self):
        """
            'with' constructor.
        """

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
            'with' destructor.
        """

        self.delete()

    def init(self, bootstrap_with, weights=None, subject_to=[]):
        """
            This method serves for initializing the hitting set solver with a
            given list of sets to hit. Concretely, the hitting set problem is
            encoded into partial MaxSAT as outlined above, which is then fed
            either to a MaxSAT solver or an MCS enumerator.

            An additional optional parameter is ``weights``, which can be used
            to specify non-unit weights for the target objects in the sets to
            hit. This only works if ``'sorted'`` enumeration of hitting sets
            is applied.

            Another optional parameter is available, namely, ``subject_to``.
            It can be used to specify arbitrary hard constraints that must be
            respected when computing hitting sets of the given sets. Note that
            ``subject_to`` should be an iterable containing pure clauses
            and/or native AtMostK constraints. Finally, note that these hard
            constraints must be defined over the set of signed atomic objects,
            i.e. instances of class :class:`.Atom`.

            :param bootstrap_with: input set of sets to hit
            :param weights: weights of the objects in case the problem is weighted
            :param subject_to: hard constraints (either clauses or native AtMostK constraints)
            :type bootstrap_with: iterable(iterable(obj))
            :type weights: dict(obj)
            :type subject_to: iterable(iterable(Atom))
        """

        # formula encoding the sets to hit
        formula = WCNFPlus()

        # hard clauses
        for to_hit in bootstrap_with:
            to_hit = list(map(lambda obj: self.idpool.id(obj), to_hit))

            formula.append(to_hit)

        # additional hard constraints
        for cl in subject_to:
            if not len(cl) == 2 or not type(cl[0]) in (list, tuple, set):
                # this is a pure clause
                formula.append(list(map(lambda a: self.idpool.id(a.obj) * (2 * a.sign - 1), cl)))
            else:
                # this is a native AtMostK constraint
                formula.append([list(map(lambda a: self.idpool.id(a.obj) * (2 * a.sign - 1), cl[0])), cl[1]], is_atmost=True)

        # soft clauses
        for obj_id in six.iterkeys(self.idpool.id2obj):
            formula.append([-obj_id],
                    weight=1 if not weights else weights[self.idpool.obj(obj_id)])

        if self.htype == 'rc2':
            if not weights or min(weights.values()) == max(weights.values()):
                self.oracle = RC2(formula, solver=self.solver, adapt=self.adapt,
                        exhaust=self.exhaust, minz=self.minz, trim=self.trim)
            else:
                self.oracle = RC2Stratified(formula, solver=self.solver,
                        adapt=self.adapt, exhaust=self.exhaust, minz=self.minz,
                        nohard=True, trim=self.trim)
        elif self.htype == 'lbx':
            self.oracle = LBX(formula, solver_name=self.solver,
                    use_cld=self.usecld)
        else:
            self.oracle = MCSls(formula, solver_name=self.solver,
                    use_cld=self.usecld)

    def add_hard(self, clause, weights=None):
        """
            Add a hard constraint, which can be either a pure clause or an
            AtMostK constraint.

            Note that an optional parameter that can be passed to this method
            is ``weights``, which contains a mapping the objects under
            question into weights. Also note that the weight of an object must
            not change from one call of :meth:`hit` to another.

            :param clause: hard constraint (either a clause or a native AtMostK constraint)
            :param weights: a mapping from objects to weights

            :type clause: iterable(obj)
            :type weights: dict(obj)
        """

        if not len(clause) == 2 or not type(clause[0]) in (list, tuple, set):
            # this is a pure clause
            clause = list(map(lambda a: self.idpool.id(a.obj) * (2 * a.sign - 1), clause))

            # a soft clause should be added for each new object
            new_obj = filter(lambda vid: abs(vid) not in self.oracle.vmap.e2i, clause)
        else:
            # this is a native AtMostK constraint
            clause = [list(map(lambda a: self.idpool.id(a.obj) * (2 * a.sign - 1), clause[0])), clause[1]]

            # a soft clause should be added for each new object
            new_obj = filter(lambda vid: abs(vid) not in self.oracle.vmap.e2i, clause[0])

            # there may be duplicate literals if the constraint is weighted
            new_obj = list(set(new_obj))

        # some of the literals may also have the opposite polarity
        new_obj = [l if l in self.idpool.obj2id else -l for l in new_obj]

        # adding the hard clause
        self.oracle.add_clause(clause)

        # new soft clauses
        for vid in new_obj:
            self.oracle.add_clause([-vid], 1 if not weights else weights[self.idpool.obj(vid)])

    def delete(self):
        """
            Explicit destructor of the internal hitting set oracle.
        """

        if self.oracle:
            self.oracle.delete()
            self.oracle = None

    def get(self):
        """
            This method computes and returns a hitting set. The hitting set is
            obtained using the underlying oracle operating the MaxSAT problem
            formulation. The computed solution is mapped back to objects of the
            problem domain.

            :rtype: list(obj)
        """

        model = self.oracle.compute()

        if model is not None:
            if self.htype == 'rc2':
                # extracting a hitting set
                self.hset = filter(lambda v: v > 0, model)
            else:
                self.hset = model

            return list(map(lambda vid: self.idpool.id2obj[vid], self.hset))

    def hit(self, to_hit, weights=None):
        """
            This method adds a new set to hit to the hitting set solver. This
            is done by translating the input iterable of objects into a list of
            Boolean variables in the MaxSAT problem formulation.

            Note that an optional parameter that can be passed to this method
            is ``weights``, which contains a mapping the objects under
            question into weights. Also note that the weight of an object must
            not change from one call of :meth:`hit` to another.

            :param to_hit: a new set to hit
            :param weights: a mapping from objects to weights

            :type to_hit: iterable(obj)
            :type weights: dict(obj)
        """

        # translating objects to variables
        to_hit = list(map(lambda obj: self.idpool.id(obj), to_hit))

        # a soft clause should be added for each new object
        new_obj = list(filter(lambda vid: vid not in self.oracle.vmap.e2i, to_hit))

        # new hard clause
        self.oracle.add_clause(to_hit)

        # new soft clauses
        for vid in new_obj:
            self.oracle.add_clause([-vid], 1 if not weights else weights[self.idpool.obj(vid)])

    def block(self, to_block, weights=None):
        """
            The method serves for imposing a constraint forbidding the hitting
            set solver to compute a given hitting set. Each set to block is
            encoded as a hard clause in the MaxSAT problem formulation, which
            is then added to the underlying oracle.

            Note that an optional parameter that can be passed to this method
            is ``weights``, which contains a mapping the objects under
            question into weights. Also note that the weight of an object must
            not change from one call of :meth:`hit` to another.

            :param to_block: a set to block
            :param weights: a mapping from objects to weights

            :type to_block: iterable(obj)
            :type weights: dict(obj)
        """

        # translating objects to variables
        to_block = list(map(lambda obj: self.idpool.id(obj), to_block))

        # a soft clause should be added for each new object
        new_obj = list(filter(lambda vid: vid not in self.oracle.vmap.e2i, to_block))

        # new hard clause
        self.oracle.add_clause([-vid for vid in to_block])

        # new soft clauses
        for vid in new_obj:
            self.oracle.add_clause([-vid], 1 if not weights else weights[self.idpool.obj(vid)])

    def enumerate(self):
        """
            The method can be used as a simple iterator computing and blocking
            the hitting sets on the fly. It essentially calls :func:`get`
            followed by :func:`block`. Each hitting set is reported as a list
            of objects in the original problem domain, i.e. it is mapped back
            from the solutions over Boolean variables computed by the
            underlying oracle.

            :rtype: list(obj)
        """

        done = False
        while not done:
            hset = self.get()

            if hset != None:
                self.block(hset)
                yield hset
            else:
                done = True

    def oracle_time(self):
        """
            Report the total SAT solving time.
        """

        return self.oracle.oracle_time()
