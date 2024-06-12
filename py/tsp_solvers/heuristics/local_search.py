# -*- coding: utf-8 -*-

import time
import numpy as np
from scipy.optimize import OptimizeResult

from tsp_solvers import rand_init_guess


## local search swap nodes
def solve_swap(fun, D, seq0=None, solver="swap", maxiter=100, random_state=42):
    """
    2-exchange local search.

    Parameters
    ----------
    fun : callable
        TSP objective function.
    D : array_like
        Distance (cost) matrix.
    seq0 : array_like, optional
        Initial guess. The default is None.
    solver : string, optional
        Local search algorithm. The default is "swap".
    maxiter : int, optional
        Maximum number of iteraions. The default is 100.
    random_state : int, optional
        Seed for numpy.random.Generator. The default is 42.

    Returns
    -------
    result : OptimizeResult
    """

    ncity = D.shape[0]               # number of cities in the path
    f_seq = np.empty(maxiter + 1)    # objective function sequence
    time_seq = np.zeros_like(f_seq)  # runtime for each iteration

    _rng = np.random.default_rng(random_state)  # generation seed

    if seq0 is None:
        # generate random hamiltonian cycle [0,...,0]
        seq0 = rand_init_guess(ncity, _rng)

    best_seq = seq0.copy()  # starting solution, assume City0 in seq0[0]
    best_f = fun(seq0, D)   # starting objective function

    f_seq[0] = best_f
    time_seq[0] = 0.
    _start = time.time()
    # warnflag = 0

    k = 0

    while k < maxiter:

        # 2-exchange step
        # TODO: refactor
        best_seq, best_f = _two_exchange(fun, D, solver, best_seq, best_f, _rng)

        k += 1

        f_seq[k] = best_f
        time_seq[k] = time.time() - _start

    result = OptimizeResult(fun=best_f, x=best_seq, nit=k, solver=solver,
                            runtime=time_seq[k], fun_seq=f_seq)

    return result


# %% Utils

# smooth the hard constraint
# 2-exchange routine
def _two_exchange(fun, D, local_search, best_seq, best_f, generator):
    """
    Swap two cities and check for objective function improvement.
    Single step in local search algorithm.

    Parameters
    ----------
    fun : callable
        DESCRIPTION.
    D : array_like
        DESCRIPTION.
    local_search : string
        DESCRIPTION.
    best_seq : array_like
        DESCRIPTION.
    best_f : float
        DESCRIPTION.
    generator : numpy.random.Generator
        DESCRIPTION.

    Returns
    -------
    best_seq : array_like
        DESCRIPTION.
    best_f : float
        DESCRIPTION.
    """

    # ncity = best_seq.size - 1      # number of nodes
    # current_seq = best_seq.copy()  # current sequence

    # draw 2 non-consecutive random city indices
    # i, j = _rand_city_idx(ncity, generator)

    # if local_search == "swap":

    #     # split the two selected cities
    #     current_seq[i], current_seq[j] = current_seq[j], current_seq[i]

    # elif local_search == "swap-rev":

    #     # split the two selected cities
    #     current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
    #     # access the elements between the two selected cities
    #     # reverse the cities between
    #     current_seq[i+1:j] = current_seq[i+1:j][::-1]

    current_seq = _swap_city_idx(local_search, best_seq, generator)

    # compute current objective function
    current_f = fun(current_seq, D)

    if current_f < best_f:
        # if the objective function decreases update the sequence
        best_seq = current_seq  # new sequence
        best_f = current_f      # new best objective function value

    return best_seq, best_f


def _swap_city_idx(method, current_seq, generator):
    """
    Sequence perturbation with a specified method.

    Parameters
    ----------
    method : string
        DESCRIPTION.
    current_seq : array_like
        Sequence to perturbate.
    generator : numpy.random.Generator

    Returns
    -------
    current_seq : array_like
        DESCRIPTION.
    """

    ncity = current_seq.size - 1
    i, j = _rand_city_idx(ncity, generator)

    if method == "swap":

        # split the two selected cities
        current_seq[i], current_seq[j] = current_seq[j], current_seq[i]

    elif method == "swap-rev":

        # split the two selected cities
        current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
        # access the elements between the two selected cities
        # reverse the cities between
        current_seq[i+1:j] = current_seq[i+1:j][::-1]

    return current_seq


def _rand_city_idx(ncity, generator):
    """
    Draw two different random indices among all cities indices excluding start
    and end cities, assuming seq=[0,...,0].
    Utils routine for local search algorithm

    Parameters
    ----------
    ncity : int
        Number of cities in the problem.
    generator : numpy.random.Generator

    Returns
    -------
    i : int
        First city.
    j : int
        Second city.
    """

    # cities = np.arange(1, ncity)  # numerically unstable when start>>step
    cities = np.linspace(1, ncity, ncity-1, endpoint=False, dtype=np.int32)

    # generate i and j among all city indices
    # draw two random and different indices, w/out replacement
    i, j = generator.choice(cities, 2, replace=False)

    return i, j


