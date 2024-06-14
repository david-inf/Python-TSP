# -*- coding: utf-8 -*-
"""

"""

import time
import random
import numpy as np
from scipy.optimize import OptimizeResult, fsolve

# from scipy.optimize import dual_annealing

from tsp_solvers.solvers_utils import rand_init_guess
from tsp_solvers.heuristics.local_search import _perturbation


def solve_simulated_annealing(fun, D, seq0=None, maxiter_outer=100,
                              maxiter_inner=50, init_temp=1.,
                              perturbation="swap-rev", cooling_rate=0.99,
                              random_state=42):

    _rng = np.random.default_rng(random_state)

    ## setup initial guess
    if seq0 is None:
        # generate random starting sequence
        seq0 = rand_init_guess(D.shape[0], _rng)

    ## ****************************************** ##

    ## procedure for selecting a large enough initial temperature
    # starting with a small value
    # init_temp will be progressively multiplied by beta>1
    # until the acceptance rate is close to 1.
    q = 0

    # TODO: consider using joblib
    while q < 30:

        temp_iter = 10  # iterations for init_temp tuning

        # run sim annealing with current init_temp
        res_temp = _annealing(fun, D, seq0, maxiter_inner, perturbation,
                              cooling_rate, temp_iter, init_temp, None)

        # compute acceptance rate
        chi = np.mean(res_temp.chi_seq)

        if chi >= 0.95:
            # acceptance rate satisfied
            break

        # run again sim annealing with increased temperature
        init_temp *= 1.25

        q += 1

    ## ****************************************** ##

    ## solve TSP
    res = _annealing(fun, D, seq0, maxiter_inner, perturbation, cooling_rate,
                     maxiter_outer, init_temp, random_state)

    res.q = q

    return res


# %% Base annealing algorithm

# temperature parameter tuning
def _annealing(fun, cost, x0, maxiter_inner, neighbor_meth, cooling_rate,
               maxiter_outer, init_temp, random_state=None):
    """
    Choose an optimal initial temperature s.t. nearly all transitions are
    accepted at the first iterations. Averaged on maxiter_outer iterations.

    random_state : int
        Generation seed, used for local search perturbation

    Returns
    -------
    None.

    """

    ## random numbers generator
    _rng = np.random.default_rng(random_state)

    ## starting values
    temp = init_temp            # starting temperature
    best_x = x0.copy()          # starting sequence
    best_f = fun(best_x, cost)  # starting objective value

    ## allocate performance sequences
    chi_seq = np.empty(maxiter_outer)       # acceptance rate
    temp_seq = np.empty(maxiter_outer + 1)  # temperature after each epoch
    temp_seq[0] = init_temp
    f_seq = np.empty_like(temp_seq)         # best f(x), local minima sequence
    f_seq[0] = best_f
    f_acc_seq = np.empty_like(f_seq)        # f(x) for accepted steps
    f_acc_seq[0] = best_f

    _start = time.time()
    # _need_to_stop = False
    k = 0

    # outer loop
    while k < maxiter_outer:

        ## save solution to improve
        xk = best_x.copy()
        fk = best_f.copy()

        accepted = 0

        ## inner loop, improve fk with transitions
        # consider using joblib
        for _ in range(maxiter_inner):

            ## generate new sequence by perturbing the current one
            current_x = _perturbation(neighbor_meth, xk, _rng)
            current_f = fun(current_x, cost)

            ## check for global improvement
            if current_f < best_f:
                # update best variables, best_f can only decrease
                best_x = current_x
                best_f = current_f

            ## Metropolis acceptance rule
            if _metropolis(current_f, fk, temp):
                # fk can increase according to Metropolis
                xk, fk = current_x, current_f
                accepted += 1

        ## decrease temperature
        temp *= cooling_rate

        ## update acceptance rate: accepted transitions / proposed
        chi_seq[k] = accepted / maxiter_inner

        k += 1

        ## update sequences
        f_seq[k] = best_f   # update best objective function value
        f_acc_seq[k] = fk   # final accepted f(x)
        temp_seq[k] = temp  # update temperature value

        if temp < init_temp / 1000.:

            # _need_to_stop = True
            break

    _end = time.time()

    res = OptimizeResult(fun=best_f, x=best_x, nit=k, runtime=(_end - _start),
                         solver=("simulated-annealing with " + neighbor_meth),
                         temp=temp, fun_acc_seq=f_acc_seq,
                         chi_seq=chi_seq, temp_seq=temp_seq, fun_seq=f_seq)

    return res


# %% Utils

def _metropolis(ft, fk, temp):
    """
    Metropolis acceptance rule.

    Parameters
    ----------
    ft : float
        Current solution.
    fk : float
        Previous solution.
    temp : float
        Temperature value.

    Returns
    -------
    criterion : bool
    """

    criterion = False

    if ft < fk:
        # down-hill
        criterion = True

    else:
        # up-hill
        rand_unif = random.random()  # in (0,1)
        criterion = rand_unif < np.exp(-(ft - fk) / temp)
        
    return criterion


def _opt_iter(k, temp, cooling_rate, tol=1e-5):
    """ Choose maximum number of outer iterations """

    def fun(k, temp, cooling_rate, tol=1e-5):
    
        return (1 - cooling_rate)**k - tol

    root = fsolve(fun, 100, (temp, cooling_rate, tol))

    return int(root[0])