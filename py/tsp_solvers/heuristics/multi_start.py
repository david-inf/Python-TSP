# -*- coding: utf-8 -*-
"""
Multi start metaheuristics module

"""

import time
import numpy as np
from scipy.optimize import OptimizeResult
from joblib import Parallel, delayed

from tsp_solvers.heuristics.local_search import solve_local_search
from tsp_solvers.heuristics.simulated_annealing import solve_simulated_annealing

from tsp_solvers.solvers_utils import rand_init_guess


def solve_multi_start(fun, cost, base_alg, nsim=1000, random_state=42,
                      n_jobs=4, base_options=None):
    """
    Multi-start meta-heuristic.

    Parameters
    ----------
    fun : callable
        Objective function to be minimized.
    cost : array_like
        Distance matrix.
    nsim : int, optional
        Number of simulation for initial guess to perform. The default is 1000.
    base_alg : string
        Base algorithm to improve performance.
    random_state : int
        Random numbers generator, will be used for generating each initial guess.
    base_options : dict
        Base algorithm options. The default il None.

    Returns
    -------
    result : OptimizeResult

    """

    if base_options is None:

        base_options = {}

    _base_algs = ("local-search", "sim-annealing")

    if base_alg not in _base_algs:
        # check base algorithm for multi-start
        raise RuntimeError("Unknown base algorithm.")

    # initial guess Generator, guarantees that each x0 will be unique
    _rng = np.random.default_rng(random_state)

    _start = time.time()

    ## run parallel simulations
    with Parallel(n_jobs=n_jobs, backend="loky") as parallel:

        results = parallel(
            delayed(_inner_algorithm)(fun, cost, base_alg, _rng, base_options)
            for _ in range(nsim))

    _end = time.time()

    ## sort list of OptimizeResult for each nsim
    results.sort(reverse=True, key=lambda x: x.fun)

    ## metrics sequences
    f_seq = np.array([opt.fun for opt in results])
    x_seq = np.row_stack([opt.x for opt in results])

    res = OptimizeResult(
        fun=results[-1].fun, x=results[-1].x, nit=nsim,
        solver="multi-start on " + base_alg,
        solver_obj=results[-1],
        x_seq=x_seq, runtime=(_end - _start), fun_seq=f_seq)

    ## ratio unique local minima / nsim
    res.ratio = np.unique(x_seq, axis=0).shape[0] / nsim
    if base_alg == "local-search":
        res.nit_ls = results[-1].nit

    return res


# function to be parallelized
def _inner_algorithm(fun, cost, base_alg, generator, base_options):

    ## generate random initial guess
    seqt = rand_init_guess(cost.shape[0], generator)

    res_sim = None

    if base_alg == "local-search":

        ## local search
        res_sim = solve_local_search(fun, cost, seqt, **base_options)

    elif base_alg == "sim-annealing":

        ## simulated annealing
        res_sim = solve_simulated_annealing(fun, cost, seqt, **base_options)

    # simulated annealing for improving a local search

    return res_sim
