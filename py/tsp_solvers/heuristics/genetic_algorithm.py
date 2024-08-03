# -*- coding: utf-8 -*-
"""
Genetic Algorithm module

"""

import time
import numpy as np
from scipy.optimize import OptimizeResult
from joblib import Parallel, delayed

from tsp_solvers.solvers_utils import rand_init_guess

from tsp_solvers.heuristics.local_search import solve_local_search

from tsp_solvers.greedy.nearest_neighbor import _add_node

# RFE: other metrics could be the mean f(x) over the current population
def solve_genetic_algorithm(fun, cost, maxiter, individuals=20, nson=5,
                            method="exact", mutation_iters=10,
                            random_state=42):

    ## random generator
    _rng = np.random.default_rng(random_state)

    ## generate initial population that will be updated
    population = _init_population(cost.shape[0], individuals, _rng)
    ## keep track of the best solution and objective function value
    best_x = _select_x(fun, cost, population, "exact", 1)[0]
    best_f = fun(best_x, cost)

    ## allocate and initialize sequences for metrics
    f_seq = np.empty(maxiter+1)  # objective function sequence
    f_seq[0] = best_f
    time_seq = np.zeros_like(f_seq)  # runtime for each iteration
    time_seq[0] = 0.
    x_seq = np.empty((maxiter+1, cost.shape[0]+1), dtype=np.int32)  # best solution sequence
    x_seq[0] = best_x.copy()

    _start = time.time()
    k = 0

    while k < maxiter:

        ## selection: select parent solutions
        # TODO: generate solution from more than one couple of parents
        # draw a certain number of parents, then create offspring
        parents = _select_x(fun, cost, population, method, 2, _rng)

        ## cross-over: generate new individuals from parents
        offspring = _crossover(parents, cost, method, nson, _rng)

        ## mutation: mutate generated individuals
        offspring = _mutation(fun, cost, offspring, mutation_iters, _rng)

        ## select individuals based on fitness to keep population constant
        population = _survival(fun, cost, population, offspring, method, _rng)

        ## update best result
        best_x = _select_x(fun, cost, population, "exact", 1)[0]
        best_f = fun(best_x, cost)

        ## update (next) iteration number
        k += 1

        ## update sequences with values from current iteration
        f_seq[k] = best_f
        x_seq[k] = best_x.copy()
        time_seq[k] = time.time() - _start

    res = OptimizeResult(fun=best_f, x=best_x, nit=k, solver="Genetic Algorithm",
                         runtime=time_seq[k], x_seq=x_seq, fun_seq=f_seq)

    return res


def _eval_fitness(fun, cost, x_list):
    """ Compute each fitness for the given population """

    # cost for each individual
    f_list = np.array([fun(x, cost) for x in x_list])

    fit = np.max(f_list)*1.1 - f_list

    return fit / np.sum(fit)


def _init_population(ncity, size, generator):
    """
    Generate an intial pool of solutions.

    Parameters
    ----------
    ncity : int
        Number of cities in the map.
    size : int
        Size of the population which is len(population).
    generator : numpy.random.Generator

    Returns
    -------
    population : list of numpy arrays

    """

    population = []  # population elements (feasible solutions)

    for _ in range(size):

        # generate a random solution
        xi = rand_init_guess(ncity, generator)
        # add to population
        population.append(xi)

    return population


def _select_x(fun, cost, population, method="weighted", n=2, generator=None):
    """
    Select n solutions based on some criteria (method).

    Parameters
    ----------
    fun : callable
        Objective function.
    cost : array_like
        Cost matrix.
    population : list of array_like
        Current population.
    method : string, optional
        `exact`: choose the two solutions with highest fitness
        `random`: draw two random solutions
        `weighted`: draw two random solutions based of weights
    n : int, optional
        Number of parents to be selected. The default is 2.
    generator : numpy.random.Generator, optional

    Returns
    -------
    parents : list of array_like

    """

    # get the fitness for the given population
    fitness = _eval_fitness(fun, cost, population)

    parents = None

    if method == "exact":

        # sort based on fitness
        fit_and_x_sorted = sorted(
            zip(fitness, population), reverse=True, key=lambda pair: pair[0])
        # get the two best parents
        parents = [x for _, x in fit_and_x_sorted[:n]]

    elif method == "random":

        # draw two random indices
        idx = generator.choice(np.arange(len(population)), n, replace=False)
        # get the two random parents
        parents = [population[i] for i in idx]

    elif method == "weighted":

        # draw two random indices with fitness as weight
        idx = generator.choice(np.arange(len(population)),
                               n, replace=False, p=fitness)
        # get the parents
        parents = [population[i] for i in idx]

    return parents


def _build_solution(x1, x2, cost, method, generator):
    """ Fill missing nodes in an offspring """

    if np.sum(x1==x2) == x1.size:
        # both parents are the same
        return x1

    # there are different edges
    x_new = x1.copy().tolist()

    nodes_left = np.delete(x1, x1 == x2).tolist()

    i = 0

    for j in x1 == x2:

        if i >= x1.size-1:

            break

        if not j:  # when non-common edge

            # add new node
            x_new[:i+1] = _add_node(x_new[:i], cost,
                                    method, generator, nodes_left)
            # remove added node from those left
            nodes_left.remove(x_new[i])

        i += 1

    return np.array(x_new)


def _crossover(parents, cost, method="weighted", n=1, generator=None, n_jobs=2):
    """
    Create n solutions starting with x1 and x2 common edges.

    x1, x2 : array_like
        Parents from which the solutions are generated.
    method : string
        `exact`: choose next node based on distance
        `random`: choose next node randomly
        `weighted`: choose next node with weighted uniform distribution
    n : int, optional
        Population size. The default is 1.

    """

    x1, x2 = parents

    # offspring = []  # list of array_like

    # # TODO: can be parallelized
    # for _ in range(n):

    #     # create a descendant
    #     x_desc = _build_solution(x1, x2, cost, method, generator)
    #     # add to the offspring list
    #     offspring.append(x_desc)

    def _crossover_fun():

        x_desc = _build_solution(x1, x2, cost, method, generator)

        return x_desc

    with Parallel(n_jobs=n_jobs, backend="loky") as parallel:

        offspring = parallel(
            delayed(_crossover_fun)()
            for _ in range(n))

    return offspring


def _mutation(fun, cost, offspring, iters, generator, n_jobs=4):
    """
    Mutate randomly the offspring using local search.

    Parameters
    ----------
    fun : callable
        Objective function.
    cost : array_like
        Cost matrix.
    offspring : list of array_like
        List of descendants to mutate.
    iters : int
        Local search iterations to be performed.
    generator : numpy.random.Generator
        Random generator for local search.

    Returns
    -------
    new_offspring : list of array_like
        Offspring with random mutations.

    """

    def _mutation_fun(x):

        res = solve_local_search(fun, cost, x, "reverse", iters, generator)

        return res.x

    with Parallel(n_jobs=n_jobs, backend="loky") as parallel:

        new_offspring = parallel(
            delayed(_mutation_fun)(x)
            for x in offspring)

    return new_offspring


def _survival(fun, cost, current_pop, offspring, method, generator):
    """ Choose individuals that will survive """

    # population with more individuals than capacity
    extended_pop = current_pop + offspring

    # selected individuals based on some criteria
    new_pop = _select_x(fun, cost, extended_pop, method,
                              len(current_pop), generator)

    return new_pop
