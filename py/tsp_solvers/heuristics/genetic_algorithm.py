# -*- coding: utf-8 -*-
"""
Genetic Algorithm module

"""

import time
import numpy as np
from scipy.optimize import OptimizeResult

from tsp_solvers.solvers_utils import rand_init_guess
from tsp_solvers.heuristics.local_search import _perturbation


def solve_genetic_algorithm(fun, cost, maxiter, npop=10, crossover=0.8, mutation="reverse",
                            random_state=42):

    _rng = np.random.default_rng(random_state)

    _start = time.time()

    ## generate initial population that will be updated
    population = _init_population(fun, cost, npop, _rng)  # list

    k = 0

    while k < maxiter:

        ## select parent solutions
        parents = _select_parents(population, "weighted", _rng)

        ## generate new individuals

        ## mutate individuals

        ## select ngen individuals based on fitness
        pass

    _end = time.time()

    res = OptimizeResult(
        runtime=(_end - _start))

    return res


def _eval_fitness(f_list):
    """ Compute each fitness for a given list of f(x) """

    f_max = max(f_list)  # worst objective function value

    fitness = []

    # compute fitness for each solution
    for f in f_list:

        # compute fit
        fit = f_max - f  # >= 0.
        fitness.append(fit)

    return fitness / sum(fitness)


def _init_population(fun, cost, size, generator):

    ncity = cost.shape[0]

    solutions = []  # population elements (feasible solutions)
    funs = []  # their f(x)

    for i in range(size):

        # generate a random solution
        xi = rand_init_guess(ncity, generator)
        solutions.append(xi)

        # compute its objective function
        fi = fun(xi, cost)
        funs.append(fi)

    ## set fitness for each solution
    fit = _eval_fitness(funs)

    ## form population as list of tuples (x, fit)
    population = list(zip(solutions, fit))

    return population


def _select_parents(population, method="weighted", generator=None):
    """

    method : string
        exact: choose the two solutions with lowest f(x)
        random: draw two random solutions
        weighted: draw two random solutions with weighted uniform distribution

    """

    parents = None

    if method == "exact":

        # select those with higher fitness
        population.sort(reverse=True, key=lambda x: x[1])
        parents = [population[:2][0][0], population[:2][1][0]]

    elif method == "random":

        # draw two random indices
        idx = generator.choice(np.arange(len(population)), 2, replace=False)
        # get the parents
        parents = [population[idx[0]][0], population[idx[1]][0]]

    elif method == "weighted":

        # draw two random parents with weights according to fitness
        weights = [tup[1] for tup in population]
        idx = generator.choice(np.arange(len(population)), 2, replace=False, p=weights)
        parents = [population[idx[0]][0], population[idx[1]][0]]

    return parents
