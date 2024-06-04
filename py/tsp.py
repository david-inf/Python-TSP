# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la


def generate_cities(ncity, l=10, seed=42):

    rng = np.random.default_rng(seed)

    # cities coordinates c_i=(x,y)
    C = rng.integers(0, l, ncity * 2).reshape((-1, 2))

    # empty distance matrix D=(d_ij)
    D = np.empty((ncity, ncity))

    # compute distances
    for i in range(ncity):  # take c_i
        for j in range(ncity):  # compute d_ij
            D[i,j] = la.norm(C[i,:] - C[j,:])

    return D, C


def circle_cities(ncity, r=5):

    # delta between each city in polar coordinates
    theta = 2 * np.pi / ncity

    # cities polar coordinates (theta, rho)
    Cpol = np.zeros((ncity, 2))
    Cpol[0,:] = np.array([0, r])  # first city

    for i in range(ncity-1):
        Cpol[i+1,:] = np.array([Cpol[i,0] + theta, r])

    # cities cartesian coordinates (x, y)
    Ccart = np.zeros((ncity, 2))
    for i in range(ncity):
        Ccart[i,:] = r * np.array([np.cos(Cpol[i,0]), np.sin(Cpol[i,0])])

    # empty distance matrix D=(d_ij)
    D = np.empty((ncity, ncity))
    
    # compute distances
    for i in range(ncity):
        for j in range(ncity):
            D[i,j] = la.norm(Ccart[i,:] - Ccart[j,:])

    return D, Ccart, Cpol
