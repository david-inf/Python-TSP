# -*- coding: utf-8 -*-
"""
Main function for calling each solver

"""

from tsp_solvers.exact.brute_force import solve_brute_force
from tsp_solvers.heuristics.local_search import solve_local_search
from tsp_solvers.heuristics.multi_start import solve_multi_start
from tsp_solvers.heuristics.simulated_annealing import solve_simulated_annealing


_solvers_list = ["brute-force", "local-search", "multi-start", "simulated-annealing"]

_solvers_dict = {"exact": ["brute-force"],
                 "local-search": ["swap", "reverse"],
                 "meta-heuristics": ["multi-start", "simulated-annealing"]}


def solve_tsp(fun, cost, solver, x0=None, options=None):
    """
    Wrapper for TSP solvers.

    Parameters
    ----------
    fun : callable
        Objective function to minimize.
    cost : array_like
        Cost (distance) matrix.
    solver : string, optional
        Algorithm to use.
    x0 : array_like, optional
        Initial guess. The default is None.
    options : dict, optional
        Solver internal options. The default is None.

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

    if solver not in _solvers_list:

        raise RuntimeError("Unknown solver.")

    if options is None:

        options = {}

    ## choose solver and solve TSP
    res = None

    if solver == "brute-force":
        # raise RuntimeWarning("Brute force is computationally expensive.")
        res = solve_brute_force(fun, cost, x0, **options)

    elif solver == "local-search":
        res = solve_local_search(fun, cost, x0, **options)

    elif solver == "multi-start":
        res = solve_multi_start(fun, cost, **options)

    elif solver == "simulated-annealing":
        res = solve_simulated_annealing(fun, cost, x0, **options)

    if res is None:

        return "Optimization process went wrong"

    return res
