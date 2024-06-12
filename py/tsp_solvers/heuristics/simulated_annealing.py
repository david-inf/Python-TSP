# -*- coding: utf-8 -*-
"""

"""

import time
import numpy as np
from scipy.optimize import OptimizeResult, fsolve

# from scipy.optimize import dual_annealing

from tsp_solvers.solvers_utils import rand_init_guess
from tsp_solvers.heuristics.local_search import _swap_city_idx


def solve_simulated_annealing(fun, D, seq0=None, maxiter_outer=100,
                              maxiter_inner=100, init_temp=1.,
                              perturbation="swap", cooling_rate=0.05,
                              random_state=42):

    _rng = np.random.default_rng(random_state)
    _rng_inner = np.random.default_rng()  # seed for perturbation
    _min_temp = 1e-5

    if seq0 is None:
        # generate starting sequence
        seq0 = rand_init_guess(D.shape[0], _rng)
        best_seq = seq0.copy()
    else:
        best_seq = seq0.copy()

    # starting objective value
    best_f = fun(best_seq, D)
    # starting temperature
    temp = init_temp

    temp_seq = np.empty(maxiter_outer + 1)  # temperature after each epoch
    temp_seq[0] = init_temp
    f_seq = np.empty_like(temp_seq)         # best_f sequence
    f_seq[0] = best_f

    _start = time.time()
    _need_to_stop = False
    k = 0

    # outer loop
    while not _need_to_stop:

        ## save solution to improve
        seqk = best_seq.copy()
        fk = best_f.copy()

        # inner loop
        # consider using joblib
        for _ in range(maxiter_inner):

            ## generate new sequence by perturbing the current one
            current_seq = _swap_city_idx(perturbation, seqk, _rng_inner)
            current_f = fun(current_seq, D)

            ## check for global improvement
            if current_f < best_f:
                # update best variables
                best_seq = current_seq
                best_f = current_f

            ## Metropolis acceptance rule
            if _acceptance_rule(current_f, fk, temp, _rng.random()):
                seqk, fk = current_seq, current_f

        ## decrease temperature
        temp *= (1 - cooling_rate)

        k += 1

        f_seq[k] = best_f
        temp_seq[k] = temp

        if temp < _min_temp or k >= maxiter_outer:
            _need_to_stop = True

    _end = time.time()

    res = OptimizeResult(fun=best_f, x=best_seq, nit=k,
                         solver=("simulated-annealing with " + perturbation),
                         runtime=(_end - _start), fun_seq=f_seq,
                         temp_seq=temp_seq)

    return res


# %% Utils

def _acceptance_rule(ft, fk, temp, rand_unif):

    criterion = False

    if ft < fk:
        # down-hill
        criterion = True

    else:
        # up-hill
        criterion = rand_unif < np.exp(-(ft - fk) / temp)
        
    return criterion


def _opt_iter(k, temp, cooling_rate, tol=1e-5):

    def fun(k, temp, cooling_rate, tol=1e-5):
    
        return (1 - cooling_rate)**k - tol

    root = fsolve(fun, 100, (temp, cooling_rate, tol))

    return int(root[0])