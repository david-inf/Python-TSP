# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la


def generate_cities(ncity, l=10, seed=42):

    rng = np.random.default_rng(seed)

    # cities coordinates c_i=(x,y)
    C = rng.integers(0, l, ncity * 2).reshape((-1, 2))

    # empty distance matrix
    D = np.empty((ncity, ncity))
    ## check 0 diagonal ##

    # compute distances
    for i in range(ncity):  # take c_i
        for j in range(ncity):  # compute d_ij
            D[i,j] = la.norm(C[i,:] - C[j,:])

    return D

