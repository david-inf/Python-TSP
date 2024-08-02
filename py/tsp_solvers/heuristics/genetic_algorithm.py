# -*- coding: utf-8 -*-
"""
Genetic Algorithm module

"""

import time
import numpy as np
from scipy.optimize import OptimizeResult

from tsp_solvers.solvers_utils import rand_init_guess

from tsp_solvers.heuristics.local_search import _perturbation

from tsp_solvers.greedy.nearest_neighbor import _add_node


def solve_genetic_algorithm(fun, cost, maxiter, individuals=20, nson=5,
                            mutation="reverse", method="exact",
                            crossover="common-nodes", random_state=42):

    ## random generator
    _rng = np.random.default_rng(random_state)

    ## generate initial population that will be updated
    population = _init_population(cost.shape[0], individuals, _rng)
    ## keep track of the best solution and objective function value
    best_x = _select_parents(fun, cost, population, "exact", 1)[0]
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
        # TODO: more parents?
        parents = _select_parents(fun, cost, population, method, 2, _rng)

        ## cross-over: generate new individuals from parents
        offspring = _crossover(*parents, cost=cost,
                               method=method, n=nson, generator=_rng)

        ## mutation: mutate generated individuals
        # offspring_muted = _mutation()

        ## select individuals based on fitness to keep population constant
        population = _survival(fun, cost, population, offspring, _rng)

        ## update best result
        best_x = _select_parents(fun, cost, population, "exact", 1)[0]
        best_f = fun(best_x, cost)

        ## update (next) iteration number
        k += 1

        ## update sequences with values from current iteration
        f_seq[k] = best_f
        x_seq[k] = best_x.copy()
        time_seq[k] = time.time() - _start

    _end = time.time()

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

# TODO: update name on other lines
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

# si potrebbe applicare _add_node per ogni nodo mancante
def _crossover(x1, x2, cost, method="exact", n=1, generator=None):
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

    # nodes_left = np.delete(x1, x1 == x2).tolist()  # list of length <=ncity

    offspring = []  # list of array_like

    # TODO: can be parallelized
    for _ in range(n):

        offspring.append(_build_solution(x1, x2, cost, method, generator))

    return offspring


def _mutation():

    return None


def _survival(fun, cost, current_pop, offspring, generator):
    """ Choose individuals that will survive """

    # population with more individuals than capacity
    extended_pop = current_pop + offspring

    # selected individuals based on some criteria
    new_pop = _select_parents(fun, cost, extended_pop, "weighted",
                              len(current_pop), generator)

    return new_pop
