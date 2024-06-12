# -*- coding: utf-8 -*-

import time
import numpy as np
from scipy.optimize import OptimizeResult

from tsp_solvers.heuristics.local_search import _two_exchange
from tsp_solvers.solvers_utils import rand_init_guess


## Multi-start metaheuristic
def solve_multi_start(fun, D, nsim=500, local_search="swap", ls_maxiter=100,
                      random_state=42):

    ncity = D.shape[0]  # number of cities in the path

    _rng = np.random.default_rng(random_state)  # initial guess Generator
    _rng_ls = np.random.default_rng()           # local search Generator

    f_seq = np.empty(nsim)  # f(x) for each simulation
    best_f = np.Inf         # starting f(x)
    best_seq = None         # starting path

    _start = time.time()

    # consider using joblib
    for i in range(nsim):

        # generate random initial guess
        seqt = rand_init_guess(ncity, _rng)

        # perform local search with current initial guess
        # return OptimizeResult object
        rest = _local_search(fun, D, seqt, local_search, ls_maxiter, _rng_ls)

        # check current f(x) value for improvement
        if rest.fun < best_f:

            best_f = rest.fun  # new best objective function value
            best_seq = rest.x  # sequence related to best_f

        f_seq[i] = best_f

    _end = time.time()

    result = OptimizeResult(fun=best_f, x=best_seq, nit=nsim,
                            solver=("multi-start with " + local_search),
                            runtime=(_end - _start), fun_seq=f_seq)

    return result


# %% Utils

def _local_search(fun, D, seq0, solver, maxiter, generator):
    """
    Simplified local search for multi-start algorithm.

    Parameters
    ----------
    fun : callable
        DESCRIPTION.
    D : array_like
        DESCRIPTION.
    seq0 : array_like
        DESCRIPTION.
    solver : string
        DESCRIPTION.
    maxiter : int
        DESCRIPTION.
    generator : numpy.random.Generator object
        DESCRIPTION.

    Returns
    -------
    result : OptimizeResult
        DESCRIPTION.
    """

    best_seq = seq0.copy()  # initial guess
    best_f = fun(seq0, D)   # initial f(x) value

    _start = time.time()

    k = 0

    while k < maxiter:

        # perform a 2-exchange step, swap 2 cities and check f(x)
        best_seq, best_f = _two_exchange(fun, D, solver, best_seq, best_f, generator)
        # _two_exchange(fun, D, local_search, best_seq, best_f, generator)

        k += 1

    _end = time.time()

    result = OptimizeResult(fun=best_f, x=best_seq, nit=k, solver=solver,
                            runtime=(_end - _start))

    return result
