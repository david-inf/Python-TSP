# -*- coding: utf-8 -*-

from tsp_solvers import solve_brute_force, solve_swap, solve_multi_start


solvers_list = ["brute-force", "swap", "swap-rev", "multi-start"]

multi_start_ls = ["swap", "swap-rev"]


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
    res : TYPE
        DESCRIPTION.

    """

    if not solver in solvers_list:
        raise RuntimeError("Unknown solver.")

    if options is None:
        options = {}

    ## choose solver and solve TSP
    if solver == "brute-force":
        res = solve_brute_force(fun, D, seq0, **options)
        # raise RuntimeWarning("Brute force is computationally expensive.")
    elif solver in ("swap", "swap-rev"):
        res = solve_swap(fun, D, seq0, solver, **options)
    elif solver == "multi-start":
        res = solve_multi_start(fun, D, **options)

    return res
