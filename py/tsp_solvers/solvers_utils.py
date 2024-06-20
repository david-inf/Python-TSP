# -*- coding: utf-8 -*-

import numpy as np

from tsp_solvers.tsp import adjacency


def rand_init_guess(ncity, generator):
    """
    Generate a random initial guess.

    Parameters
    ----------
    ncity : int
        Number of cities in the map.
    generator : numpy.random.Generator
        Generator object.

    Returns
    -------
    seq : array_like
        A random Hamiltionian cycle, starting from City0 to City0.

    """

    # initialize sequence with 0 as starting and ending city
    seq = np.zeros(ncity + 1, dtype=np.int32)

    # random middle cities sequence
    middle_cities = generator.permutation(np.arange(1, ncity))
    # create the final sequence
    seq[1:ncity] = middle_cities

    # seq: order in which the cities are visited
    # seq = [0,...,0]
    return seq


def check_constraint(x):
    """
    Check TSP optimization problem constraints
        - Cycle covering constraint
        - Connection constraint

    Parameters
    ----------
    x : array_like
        Check this solution to be in the feasible set.

    Returns
    -------
    constraint1 : bool
        The constraint are satisfied or not.

    """

    N = x.size - 1  # number of nodes in the sequence

    A = adjacency(x)  # adjacence matrix

    ## cycle covering
    # check that each node has one incoming and one outgoing edge
    # sum the adjacency matrix over rows and columns, check to be equal to 2N
    sum_rows_cols = np.sum(np.sum(A, axis=0) + np.sum(A, axis=1))
    cycle_cover = sum_rows_cols == 2 * N

    ## connection, subtour elimination
    # automatically satisfied for the algorithms implemented

    return cycle_cover
