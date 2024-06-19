# -*- coding: utf-8 -*-
"""
Multi start metaheuristics module

"""

import time
import numpy as np
from scipy.optimize import OptimizeResult
from joblib import Parallel, delayed

from tsp_solvers.heuristics.local_search import solve_swap
from tsp_solvers.heuristics.simulated_annealing import solve_simulated_annealing

from tsp_solvers.solvers_utils import rand_init_guess


## Multi-start metaheuristic
# def solve_multi_start(fun, D, nsim=500, base_alg="local-search",
#                       local_search="swap", random_state=42, n_jobs=4,
#                       local_search_options=None, annealing_options=None):
#     """
#     Multi-start meta-heuristic with local search.

#     Parameters
#     ----------
#     fun : callable
#         TSP objective function.
#     D : array_like
#         Distance (cost) matrix.
#     nsim : int, optional
#         Number of simulation for initial guess to perform. The default is 500.
#     base_alg : string
#         Base algorithm to improve performance. The default is "local-search"
#     local_search : string, optional
#         Local search algorithm. The default is "swap".
#     ls_maxiter : int, optional
#         Maximum number of iterations for local search. The default is 100.
#     random_state : int, optional
#         Seed for numpy.random.Generator. The default is 42.

#     Returns
#     -------
#     result : OptimizeResult
#     """

#     _base_algs = ("local-search", "single-local-search", "sim-annealing",
#                   "local-search+annealing")

#     if base_alg not in _base_algs:
#         # check base algorithm for multi-start
#         raise RuntimeError("Unknown base algorithm.")

#     _rng = np.random.default_rng(random_state)  # initial guess Generator

#     if base_alg == "single-local-search":
#         seq0 = rand_init_guess(D.shape[0], _rng)
#     else:
#         seq0 = None

#     _start = time.time()

#     ## run parallel simulations
#     with Parallel(n_jobs=n_jobs, backend="loky") as parallel:

#         results = parallel(
#             delayed(_inner_algorithm)(
#                 fun, D, base_alg, local_search, _rng,
#                 local_search_options, annealing_options, seq0)
#             for _ in range(nsim))

#     _end = time.time()

#     ## sort list of OptimizeResult for each nsim
#     results.sort(reverse=True, key=lambda x: x.fun)

#     ## metrics sequences
#     f_seq = np.array([opt.fun for opt in results])
#     x_seq = np.row_stack([opt.x for opt in results])

#     _solver = "multi-start on " + base_alg + " with " + local_search

#     res = OptimizeResult(fun=results[-1].fun, x=results[-1].x, nit=nsim,
#                          solver=_solver, x_seq=x_seq,
#                          local_search=local_search_options,
#                          annealing=annealing_options,
#                          runtime=(_end - _start), fun_seq=f_seq)

#     return res


# def solve_multi_start(fun, D, nsim=500, base_alg="local-search",
#                       local_search="swap", random_state=42, n_jobs=4,
#                       local_search_options=None, annealing_options=None):
def solve_multi_start(fun, cost, base_alg="local-search", nsim=1000, random_state=42,
                      n_jobs=4, base_options=None):
    """
    Multi-start meta-heuristic with local search.

    Parameters
    ----------
    nsim : int, optional
        Number of simulation for initial guess to perform. The default is 1000.
    base_alg : string
        Base algorithm to improve performance. The default is "local-search"

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

    _rng = np.random.default_rng(random_state)  # initial guess Generator

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

    _solver = "multi-start on " + base_alg

    res = OptimizeResult(fun=results[-1].fun, x=results[-1].x, nit=nsim,
                         solver=_solver, x_seq=x_seq,
                         # local_search=local_search_options,
                         # annealing=annealing_options,
                         runtime=(_end - _start), fun_seq=f_seq)

    return res


# function to be parallelized
def _inner_algorithm(fun, cost, base_alg, generator, base_options):

    ## generate random initial guess
    seqt = rand_init_guess(cost.shape[0], generator)

    res_sim = None

    if base_alg == "local-search":

        ## local search
        # local_search_options["generator"] = generator
        res_sim = solve_swap(fun, cost, seqt, **base_options)

    elif base_alg == "sim-annealing":

        ## simulated annealing
        res_sim = solve_simulated_annealing(fun, cost, seqt, **base_options)

    # simulated annealing for improving a local search

    return res_sim
