# -*- coding: utf-8 -*-

import time
import numpy as np
from scipy.optimize import OptimizeResult
from itertools import permutations
from joblib import Parallel, delayed

from tsp_solvers import rand_init_guess


## exact brute force
# def solve_brute_force(fun, D, seq0=None, random_state=42):
#     # not sustainable for high seq0.size
#     # seq0: assumed [0,...,0] array_like

#     # ncity = seq0.size - 1  # number of cities in the path
#     # f_seq = np.empty(maxiter + 1)  # objective function sequence
#     # time_seq = np.zeros_like(f_seq)  # runtime for each iteration

#     ncity = D.shape[0]
#     _rng = np.random.default_rng(random_state)  # generation seed

#     if seq0 is None:
#         # generate random hamiltonian cycle [0,...,0]
#         seq0 = rand_init_guess(ncity, _rng)

#     best_seq = seq0.copy()  # starting solution, assume City0 in seq0[0]
#     best_f = fun(seq0, D)  # starting objective function

#     # f_seq[0] = fk
#     # time_seq[0] = 0.
#     _start = time.time()
#     # warnflag = 0

#     # consider parallelization
#     for partial_seq in permutations(seq0[1:-1]):

#         seqt = seq0.copy()
#         seqt[1:-1] = np.array(partial_seq)

#         ft = fun(seqt, D)

#         if ft < best_f:
#             best_f = ft
#             best_seq = seqt

#     result = OptimizeResult(fun=best_f, x=best_seq, solver="brute-force",
#                             runtime=time.time() - _start)

#     return result


def solve_brute_force(fun, D, seq0=None, random_state=42, n_jobs=2):
    # not sustainable for high seq0.size
    # seq0: assumed [0,...,0] array_like

    # ncity = seq0.size - 1  # number of cities in the path
    # f_seq = np.empty(maxiter + 1)  # objective function sequence
    # time_seq = np.zeros_like(f_seq)  # runtime for each iteration

    _rng = np.random.default_rng(random_state)  # generation seed

    if seq0 is None:
        # generate random hamiltonian cycle [0,...,0]
        seq0 = rand_init_guess(D.shape[0], _rng)

    middle = seq0[1:-1].copy()
    _start = time.time()
    # warnflag = 0


    with Parallel(n_jobs=n_jobs, backend="loky") as parallel:

        results = parallel(
            delayed(_check_seq)(fun, D, seq0, partial_seq)
            for partial_seq in permutations(middle))

    results.sort(reverse=True, key=lambda x: x[0])

    res = OptimizeResult(fun=results[0][0], x=results[0][1],
                         solver="brute-force",
                         runtime=(time.time() - _start))

    return res


def _check_seq(fun, D, seq0, partial_seq):

    # new permutation
    seq0[1:-1] = np.array(partial_seq)
    # objective function value
    ft = fun(seq0, D)

    return ft, seq0
