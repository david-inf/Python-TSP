# -*- coding: utf-8 -*-

import time
import numpy as np
from scipy.optimize import OptimizeResult

# from tsp_solvers.heuristics.local_search import _two_exchange
# from tsp_solvers.heuristics.simulated_annealing import _annealing

from tsp_solvers.heuristics.local_search import solve_swap
from tsp_solvers.heuristics.simulated_annealing import solve_simulated_annealing

from tsp_solvers.solvers_utils import rand_init_guess


## Multi-start metaheuristic
def solve_multi_start(fun, D, nsim=500, base_alg="local-search",
                      local_search="swap", random_state=42,
                      local_search_options={}, annealing_options={}):
    """
    Multi-start meta-heuristic with local search.

    Parameters
    ----------
    fun : callable
        TSP objective function.
    D : array_like
        Distance (cost) matrix.
    nsim : int, optional
        Number of simulation for initial guess to perform. The default is 500.
    base_alg : string
        Base algorithm to improve performance. The default is "local-search"
    local_search : string, optional
        Local search algorithm. The default is "swap".
    ls_maxiter : int, optional
        Maximum number of iterations for local search. The default is 100.
    random_state : int, optional
        Seed for numpy.random.Generator. The default is 42.

    Returns
    -------
    result : OptimizeResult
    """

    if base_alg not in ("local-search", "sim-annealing", "local-search+annealing"):

        raise RuntimeError("Unknown base algorithm.")

    ncity = D.shape[0]  # number of cities in the path

    _rng = np.random.default_rng(random_state)  # initial guess Generator

    f_seq = np.empty(nsim)  # f(x) for each simulation, quite nonsense
    best_f = np.Inf         # starting f(x)
    best_seq = None         # starting path

    _start = time.time()

    # TODO: refactor with joblib
    # use sorting for best solution
    for i in range(nsim):

        ## generate random initial guess
        seqt = rand_init_guess(ncity, _rng)

        if base_alg == "local-search":

            ## local search
            res_sim = solve_swap(
                fun, D, seqt, local_search, **local_search_options)

        elif base_alg == "sim-annealing":

            ## simulated annealing
            res_sim = solve_simulated_annealing(
                fun, D, seqt, perturbation=local_search, **annealing_options)

        elif base_alg == "local-search+annealing":

            ## local search
            res_local_search = solve_swap(
                fun, D, seqt, local_search, **local_search_options)

            ## from local search solution starts sim annealing
            res_sim = solve_simulated_annealing(
                fun, D, res_local_search.x, perturbation=local_search,
                **annealing_options)

        ## check current f(x) value for improvement
        if res_sim.fun < best_f:

            best_f = res_sim.fun  # new best objective function value
            best_seq = res_sim.x  # sequence related to best_f

        f_seq[i] = best_f

    _end = time.time()

    if local_search_options == {}:
        local_search_options = None
    if annealing_options == {}:
        annealing_options = None

    result = OptimizeResult(fun=best_f, x=best_seq, nit=nsim,
                            local_search=local_search_options,
                            annealing=annealing_options,
                            solver=("multi-start on " + base_alg + " with " + local_search),
                            runtime=(_end - _start), fun_seq=f_seq)

    return result


def _inner_algorithm():

    return None
