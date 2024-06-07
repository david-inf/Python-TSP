# -*- coding: utf-8 -*-

import numpy as np


def stopping(maxiter, nit):

    stop = nit < maxiter

    return stop


def rand_idx(ncity, generator):
    # random indices for swap local search
    # draw two random cities, excluding City0

    # _rng = np.random.default_rng(seed)  # generation seed

    ### first solution
    # draw city from the first half of the current sequence
    # i = _rng.choice(np.arange(1, ncity // 2))  # first city index
    # draw city from the second half of the current sequence
    # j = _rng.choice(np.arange(ncity // 2, ncity))  # second city index

    ### second solution
    i, j = 0, 0

    q = 0  # generation counter
    while i == j:
        # cities = np.arange(1, ncity)  # numerically unstable when start>>step
        cities = np.linspace(1, ncity, ncity-1, endpoint=False, dtype=np.int32)
        # generate i and j among all city indices
        i = generator.choice(cities)
        j = generator.choice(cities)

        q += 1

    return i, j

