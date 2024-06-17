# -*- coding: utf-8 -*-

import numpy as np


def rand_init_guess(ncity, generator):
    # ncity: number of cities to generate
    # generate a random hamiltonian cycle
    # starts from City0

    # initialize sequence with 0 as starting and ending city
    seq = np.zeros(ncity + 1, dtype=np.int32)

    # random middle cities sequence
    middle_cities = generator.permutation(np.arange(1, ncity))
    # create the final sequence
    seq[1:ncity] = middle_cities

    # seq: order in which the cities are visited
    # seq = [0,...,0]
    return seq
