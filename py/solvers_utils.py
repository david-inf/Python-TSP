# -*- coding: utf-8 -*-

import numpy as np


def rand_city_idx(ncity, generator):
    # random indices for swap local search
    # draw two random cities, excluding City0

    # cities = np.arange(1, ncity)  # numerically unstable when start>>step
    cities = np.linspace(1, ncity, ncity-1, endpoint=False, dtype=np.int32)

    # generate i and j among all city indices
    # draw two random and different indices
    i, j = generator.choice(cities, 2, replace=False)

    return i, j


def rand_init_guess(ncity, generator):
    # ncity: number of cities to generate
    # generate a random hamiltonian cycle
    # starts from City0

    # initialize sequence with 0 as starting and ending city
    seq = np.zeros(ncity + 1, dtype=np.int32)

    # random middle cities sequence
    middle = generator.permutation(np.arange(1, ncity))
    # create the final sequence
    seq[1:ncity] = middle

    # seq: order in which the cities are visited
    # seq = [0,...,0]
    return seq

