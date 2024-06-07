# -*- coding: utf-8 -*-

import time
import numpy as np
from scipy.optimize import OptimizeResult

from tsp import tsp_fun
from solvers_utils import stopping, rand_idx


def relax(seq0, D, solver="swap", maxiter=100, random_state=42):
    # seq0: assumed [0,...,0]

    ncity = seq0.size - 1  # number of cities in the path
    f_seq = np.empty(maxiter + 1)  # objective function sequence
    time_seq = np.zeros_like(f_seq)  # runtime for each iteration

    seqk = seq0.copy()  # starting solution, assume City0 in seq0[0]
    fk = tsp_fun(seq0, D)  # starting objective function

    f_seq[0] = fk
    time_seq[0] = 0.
    _start = time.time()
    # warnflag = 0

    _rng = np.random.default_rng(random_state)  # generation seed
    k = 0

    while stopping(maxiter, k):

        seqt = seqk.copy()  # attempt guess

        # *********************** #
        ##### swap heuristics ##### (2-exchange)

        # draw 2 non-consecutive random city indices
        i, j = rand_idx(ncity, _rng)

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


# %% 2-exchange

# smooth the hard constraint
# def two_exchange(seqt, fk, swap_type, generator)

