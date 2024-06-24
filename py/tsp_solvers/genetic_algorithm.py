# -*- coding: utf-8 -*-
"""
Genetic Algorithm module

"""

import time
import numpy as np
from scipy.optimize import OptimizeResult, fsolve

from tsp_solvers.solvers_utils import rand_init_guess
from tsp_solvers.heuristics.local_search import _perturbation


def solve_genetic_algorithm(fun, cost, maxiter, npop, crossover, mutation="reverse",
                            random_state=42):

    ## generate initial population
    _rng = np.random.default_rng(random_state)

    _start = time.time()
    k = 0

    # while k < maxgen:

        ## select parent solutions

        ## generate new individuals

        ## mutate individuals

        ## select ngen individuals based on fitness

    _end = time.time()

    res = OptimizeResult(
        runtime=(_end - _start))

    return res
