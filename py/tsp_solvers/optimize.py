# -*- coding: utf-8 -*-

from tsp_solvers.exact.brute_force import solve_brute_force
from tsp_solvers.heuristics.local_search import solve_swap
from tsp_solvers.heuristics.multi_start import solve_multi_start
from tsp_solvers.heuristics.simulated_annealing import solve_simulated_annealing


solvers_list = ["brute-force", "swap", "swap-rev", "multi-start", "simulated-annealing"]

multi_start_ls = ["swap", "swap-rev", "sim-annealing"]


def solve_tsp(fun, D, solver="swap", seq0=None, options=None):
    """
    Wrapper for TSP solvers.

    Parameters
    ----------
    fun : TYPE
        DESCRIPTION.
    D : TYPE
        DESCRIPTION.
    solver : TYPE, optional
        DESCRIPTION. The default is "swap".
    seq0 : TYPE, optional
        DESCRIPTION. The default is None.
    options : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    RuntimeError
        DESCRIPTION.
    RuntimeWarning
        DESCRIPTION.

    Returns
    -------
    res : OptimizeResult
    """

    if solver not in solvers_list:

        raise RuntimeError("Unknown solver.")

    if options is None:

        options = {}

    ## choose solver and solve TSP
    res = None
    if solver == "brute-force":
        res = solve_brute_force(fun, D, seq0, **options)
        # raise RuntimeWarning("Brute force is computationally expensive.")
    elif solver in ("swap", "swap-rev"):
        res = solve_swap(fun, D, seq0, solver, **options)
    elif solver == "multi-start":
        res = solve_multi_start(fun, D, **options)
    elif solver == "simulated-annealing":
        res = solve_simulated_annealing(fun, D, **options)

    return res
