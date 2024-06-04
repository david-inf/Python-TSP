# -*- coding: utf-8 -*-

import time
import numpy as np
from scipy.optimize import OptimizeResult

from tsp import tsp_fun, adjacency
from solvers_utils import _stopping


# smooth the problem
def relax(seq0, D, solver="swap", maxiter=100, random_state=42):

    N = seq0.size
    f_seq = np.empty(maxiter+1)
    time_seq = np.zeros_like(f_seq)

    seqk = seq0.copy()
    fk = tsp_fun(D, adjacency(seq0))

    f_seq[0] = fk
    time_seq[0] = 0.
    _start = time.time()
    # warnflag = 0

    _rng = np.random.default_rng(random_state)
    k = 0

    while _stopping(maxiter, k):

        seqt = seqk.copy()  # attempt guess

        # draw two random indices, exclude start and end points
        i = _rng.choice(np.arange(1, (N-1)//2))
        j = _rng.choice(np.arange((N-1)//2, N-1))

        if solver == "swap":

            # split two cities
            seqt[i], seqt[j] = seqt[j], seqt[i]

        elif solver == "swap-rev":

            # split two cities
            seqt[i], seqt[j] = seqt[j], seqt[i]
            # reverse the cities between
            seqt[i+1:j] = np.flip(seqt[i+1:j])

        # compute objective function
        ft = tsp_fun(D, adjacency(seqt))

        if ft < fk:
            # if the objective function decreases update sequence
            seqk = seqt
            fk = ft

        k += 1

        f_seq[k] = fk
        time_seq[k] = time.time() - _start

    result = OptimizeResult(fun=fk, x=seqk, nit=k, solver=solver,
                            runtime=time_seq[k], fun_seq=f_seq)

    return result

