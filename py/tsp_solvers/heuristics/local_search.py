# -*- coding: utf-8 -*-
"""
Local search  module

- `swap`
- `reverse`

"""

import time
import numpy as np
from scipy.optimize import OptimizeResult

from tsp_solvers import rand_init_guess


def solve_local_search(fun, cost, x0=None, solver="swap", maxiter=100, random_state=42):
    """
    Local search methods.

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
        If None a random perturbation is done every time, so not reproducible.

    Returns
    -------
    result : OptimizeResult

    """

    ## seed for initial guess and neighborhood operations
    _rng = np.random.default_rng(random_state)

    if x0 is None:
        # generate random hamiltonian cycle [0,...,0]
        x0 = rand_init_guess(cost.shape[0], _rng)

    ## allocate and initialize sequences for metrics
    f_seq = np.empty(maxiter + 1)  # objective function sequence
    f_seq[0] = fun(x0, cost)
    time_seq = np.zeros_like(f_seq)  # runtime for each iteration
    time_seq[0] = 0.
    x_seq = np.empty((maxiter+1, x0.size), dtype=np.int32)  # best solution sequence
    x_seq[0] = x0  # (N+1) x (maxiter+1)

    ## best values
    best_x = x0.copy()      # starting solution, assume City0 in seq0[0]
    best_f = fun(x0, cost)  # starting objective function

    _start = time.time()
    k = 0

    while k < maxiter:

        ## explore the neighborhood of the current best solution
        # when a new best solution is found, explore this next neighborhood
        xk = best_x.copy()

        ## 2-exchange procedure, smooth the hard constraint
        xk = _perturbation(solver, xk, _rng)
        ## compute current objective function
        fk = fun(xk, cost)

        ## check current solution
        if fk < best_f:
            # if the objective function decreases update best values
            best_x = xk.copy()  # new sequence
            best_f = fk         # new best objective function value

        ## update (next) iteration number
        k += 1

        ## update sequences with values from current iteration
        f_seq[k] = best_f
        x_seq[k] = best_x.copy()
        time_seq[k] = time.time() - _start

    res = OptimizeResult(fun=best_f, x=best_x, nit=k, solver=solver,
                         runtime=time_seq[k], x_seq=x_seq, fun_seq=f_seq)

    return res


# %% Utils

def _perturbation(method, current_seq, generator):
    """
    Sequence perturbation with a specified method. Neighborhood operator.

    Parameters
    ----------
    method : string
        Perturbation method. Can be: `swap` or `reverse`
    current_seq : array_like
        Sequence to perturbate.
    generator : numpy.random.Generator

    Returns
    -------
    current_seq : array_like
        Perturbed sequence.
    """

    # total number of cities in the sequence
    ncity = current_seq.size - 1

    # ************************#
    # canonical ensemble

    if method == "swap":

        # get two indices, i < j or i > j
        i, j = _rand_idx(ncity, generator)  # np.array([i, j])
        # split the two selected cities
        current_seq[i], current_seq[j] = current_seq[j], current_seq[i]

    elif method == "reverse":

        # get two indices s.t. i < j
        indices = _rand_idx(ncity, generator)
        i, j = np.sort(indices)
        # reverse indices between the two previously selected
        current_seq[i:j+1] = np.flip(current_seq[i:j+1])

    # ************************#

    elif method == "insert":

        # get one random index
        i = _rand_idx(ncity, generator, n_idx=1)[0]  # np.array([i])
        # remove a city from the sequence
        
    # TODO: add constraints check?

    return current_seq


def _rand_idx(ncity, generator, n_idx=2):
    """
    Draw random indices among all cities indices excluding start
    and end cities, assuming seq=[0,...,0].
    Utils routine for local search algorithm

    Parameters
    ----------
    ncity : int
        Number of cities in the problem.
    generator : numpy.random.Generator
    n_idx : int
        Number of indices to draw. The default is 2.

    Returns
    -------
    idx = np.array([i, j])
    """

    ## all available cities
    # cities = np.arange(1, ncity)  # numerically unstable when start>>step
    cities = np.linspace(1, ncity, ncity-1, endpoint=False, dtype=np.int32)

    ## generate random indices, draw w/out replacement
    idx = generator.choice(cities, n_idx, replace=False)

    return idx
