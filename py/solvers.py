# -*- coding: utf-8 -*-

import time
import numpy as np
from scipy.optimize import OptimizeResult
from itertools import permutations

from tsp import tsp_fun
from solvers_utils import rand_city_idx, rand_init_guess


solvers_list = ["swap", "swap-rev", "brute-force"]


def relax(D, seq0=None, solver="swap", maxiter=100, random_state=42):
    # seq0: assumed [0,...,0]

    ncity = D.shape[0]  # number of cities in the path
    f_seq = np.empty(maxiter + 1)  # objective function sequence
    time_seq = np.zeros_like(f_seq)  # runtime for each iteration

    _rng = np.random.default_rng(random_state)  # generation seed

    if seq0 is None:
        # generate random hamiltonian cycle [0,...,0]
        seq0 = rand_init_guess(ncity, _rng)

    seqk = seq0.copy()  # starting solution, assume City0 in seq0[0]
    fk = tsp_fun(seq0, D)  # starting objective function

    f_seq[0] = fk
    time_seq[0] = 0.
    _start = time.time()
    # warnflag = 0

    k = 0

    while k < maxiter:

        seqt = seqk.copy()  # attempt guess

        # *********************** #
        ##### swap heuristics ##### (2-exchange)

        # draw 2 non-consecutive random city indices
        i, j = rand_city_idx(ncity, _rng)

        if solver == "swap":

            # split the two selected cities
            seqt[i], seqt[j] = seqt[j], seqt[i]

        elif solver == "swap-rev":

            # split the two selected cities
            seqt[i], seqt[j] = seqt[j], seqt[i]
            # access the elements between the two selected cities
            # reverse the cities between
            seqt[i+1:j] = seqt[i+1:j][::-1]

        # compute objective function
        ft = tsp_fun(seqt, D)

        if ft < fk:
            # if the objective function decreases update the sequence
            seqk = seqt  # new sequence
            fk = ft  # new best objective function value

        # *********************** #

        k += 1

        f_seq[k] = fk
        time_seq[k] = time.time() - _start

    result = OptimizeResult(fun=fk, x=seqk, nit=k, solver=solver,
                            runtime=time_seq[k], fun_seq=f_seq)

    return result


def brute_force(D, seq0=None, random_state=42):
    # not sustainable for high seq0.size
    # seq0: assumed [0,...,0] array_like

    # ncity = seq0.size - 1  # number of cities in the path
    # f_seq = np.empty(maxiter + 1)  # objective function sequence
    # time_seq = np.zeros_like(f_seq)  # runtime for each iteration

    ncity = D.shape[0]
    _rng = np.random.default_rng(random_state)  # generation seed

    if seq0 is None:
        # generate random hamiltonian cycle [0,...,0]
        seq0 = rand_init_guess(ncity, _rng)

    best_seq = seq0.copy()  # starting solution, assume City0 in seq0[0]
    best_f = tsp_fun(seq0, D)  # starting objective function

    # f_seq[0] = fk
    # time_seq[0] = 0.
    _start = time.time()
    # warnflag = 0

    # consider parallelization
    for partial_seq in permutations(seq0[1:-1]):

        seqt = seq0.copy()
        seqt[1:-1] = np.array(partial_seq)

        ft = tsp_fun(seqt, D)

        if ft < best_f:
            best_f = ft
            best_seq = seqt

    result = OptimizeResult(fun=best_f, x=best_seq, solver="brute-foce",
                            runtime=time.time() - _start)

    return result


# %% 2-exchange

# smooth the hard constraint
# def two_exchange(seqt, fk, swap_type, generator)

