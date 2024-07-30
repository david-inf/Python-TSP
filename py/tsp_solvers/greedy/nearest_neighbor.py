# -*- coding: utf-8 -*-
"""
Nearest Neighbor greedy algorithm module

For the circular layout obviously gets the best solution, which is a circular path

"""

import time
import numpy as np
from scipy.optimize import OptimizeResult


def solve_nn(fun, cost, method="exact", random_state=42):
    """
    Greedy algorithm.

    Parameters
    ----------
    fun : callable
        Objective function to be minimized.
    cost : array_like
        Distance (cost) matrix.
    method : string, optional
        Method for selecting the next node. The default is "exact".
        `exact`: the next node is the nearest one (lower cost)
        `random`: the next node is randomly chosen
        `weighted`: the next node is randomly chosen cost-weighted
    random_state : int, optional
        Seed for numpy.random.Generator. The default is 42.
        Used when method is `random` or `weighted`.
        If None a random choice is done every time, so not reproducible.

    Returns
    -------
    res : OptimizeResult

    """

    xk = [0]  # solution that will be build up
    maxiter = cost.shape[0] - 1

    time_seq = np.empty(cost.shape[0] + 1)
    time_seq[0] = 0.

    _start = time.time()
    k = 0

    while k < maxiter:

        ## add new node in solution
        current_x = xk.copy()
        xk = _add_node(current_x, cost, method)

        ## update sequences
        k += 1
        time_seq[k] = time.time() - _start

    xk += [0]
    x_final = np.array(xk)
    res = OptimizeResult(fun=fun(x_final, cost), x=x_final, solver="Nearest Neighbor",
                         runtime=time_seq[-1])

    return res


def _add_node(x_current, cost, method):
    """
    Procedure for adding a new node in the solution

    Parameters
    ----------
    x_current : list
        Current non-complete solution.
    cost : array_like
        Cost matrix.

    Returns
    -------
    x_new : list
        Solution with a new node added

    """

    start_node = x_current[-1]  # last node added in the solution

    ## select the nodes to draw from
    nodes_left = []

    for i in np.arange(cost.shape[0]):

        if i not in x_current:
            nodes_left.append(i)

    ## choose the next node
    next_node = None

    if method == "exact":

        # get the nearest node
        next_node = nodes_left[np.argmin(cost[start_node, nodes_left])]

    # elif method == "random":

        # get a random node
        

    # elif method == "weighted":

        # get a random node based on cost
        

    return x_current + [next_node]
