# -*- coding: utf-8 -*-
"""
Genetic Algorithm module

"""

import time
import numpy as np
from scipy.optimize import OptimizeResult

from tsp_solvers.solvers_utils import rand_init_guess
from tsp_solvers.heuristics.local_search import _perturbation


def solve_genetic_algorithm(fun, cost, maxiter, individuals=20, nson=5,
                            mutation="reverse", selection="weighted",
                            crossover="common-nodes", random_state=42):

    _rng = np.random.default_rng(random_state)

    _start = time.time()

    ## generate initial population that will be updated
    # list of arrays
    population = _init_population(cost.shape[0], individuals, _rng)

    k = 0

    while k < maxiter:

        ## selection: select parent solutions
        parents = _select_parents(fun, cost, population, selection, _rng)

        ## cross-over: generate new individuals from parents
        offspring = _crossover(*parents, n=1)

        ## mutation: mutate generated individuals

        ## select individuals based on fitness to keep population constant

    _end = time.time()

    res = OptimizeResult(
        runtime=(_end - _start))

    return res


def _eval_fitness(fun, cost, x_list):
    """ Compute each fitness for the given population """

    f_list = [fun(x, cost) for x in x_list]
    f_max = max(f_list)  # worst objective function value

    fitness = []

    # compute fitness for each solution
    for f in f_list:

        # compute fit
        fit = f_max - f  # >= 0.
        fitness.append(fit)

    return fitness / sum(fitness)


def _init_population(ncity, size, generator):

    population = []  # population elements (feasible solutions)

    for i in range(size):

        # generate a random solution
        xi = rand_init_guess(ncity, generator)
        population.append(xi)

    return population


def _select_parents(fun, cost, population, method="weighted", generator=None):
    """
    Select two parents based on some criteria

    method : string
        exact: choose the two solutions with lowest f(x)
        random: draw two random solutions
        weighted: draw two random solutions with weighted uniform distribution

    """

    fitness = _eval_fitness(fun, cost, population)
    # fit_and_x = list(zip(fitness, population))

    parents = None

    if method == "exact":

        # sort based on fitness
        fit_and_x_sorted = sorted(zip(fitness, population), reverse=True)
        # get the two best parents
        parents = [x for _, x in fit_and_x_sorted[:2]]

    elif method == "random":

        # draw two random indices
        idx = generator.choice(np.arange(len(population)), 2, replace=False)
        # get the two random parents
        parents = [population[i] for i in idx]

    elif method == "weighted":

        # draw two random indices with fitness as weight
        idx = generator.choice(np.arange(len(population)), 2, replace=False, p=fitness)
        # get the parents
        parents = [population[i] for i in idx]

    return parents


# def _build_solution(x_start)


def _crossover(x1, x2, method="weighted", n=1):
    """
    Create n solutions starting with x1 and x2 common edges

    method : string
        exact: choose next node based on distance
        random: choose next node randomly
        weighted: choose next node with weighted uniform distribution

    """

    # nodes_left = np.delete(x1, x1 == x2)  # numpy array of size <=ncity

    offspring = []  # list of array_like

    for i in range(n):

        x_new = x1.copy()
        nodes_left = np.delete(x1, x1 == x2)

        for j in x1 == x2:

            if j:

                x_new[j]

    return offspring
