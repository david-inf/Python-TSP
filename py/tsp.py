# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la


def tsp_fun(D, A):

    return np.sum(np.multiply(A, D))


def create_city(coords):
    # coords: coordinates matrix Nx2

    ncity = coords.shape[0]

    cities = []
    for i in range(ncity):
        cities.append(City(coords[i, 0], coords[i, 1], "City" + str(i)))

    # cities: list of City, does not consider the path
    return cities


def adjacency(seq):
    # seq: index sequence of cities

    # number of cities in sequence
    ncity = len(seq)

    # compute incidence matrix
    A = np.zeros((ncity, ncity))
    for i in range(ncity-1):
        A[seq[i], seq[i+1]] = 1

    return A


def random_seq(ncity, lim=None, seed=42):
    # ncity: number of cities to generate
    # lim: start-end of path, 0 to last if not provided
    # otherwise list of two indices [start, end]
    # generate a sequence that starts from a city indexed with 0
    # and ends in a random one

    # set random numbers generator
    rng = np.random.default_rng(seed)

    # seq = None

    if lim is None:
        lim = [0, ncity-1]
        other = np.delete(np.arange(ncity), lim)
        # random permutation
        seq = np.concatenate((np.array([lim[0]]), rng.permutation(other), np.array([lim[-1]])))
    else:
        # remove start and end cities
        other = np.delete(np.arange(ncity), lim)
        # permute the other indices
        seq = np.concatenate((np.array([lim[0]]), rng.permutation(other), np.array([lim[-1]])))

    # seq: order in which each city is visited
    return seq


def generate_cities(ncity, l=10, seed=42):

    rng = np.random.default_rng(seed)

    # cities coordinates c_i=(x,y)
    # C = rng.integers(0, l, ncity * 2).reshape((-1, 2))
    C = rng.uniform(0, l, ncity * 2).reshape((-1, 2))

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


class City:
    def __init__(self, x, y, name="City"):
        self.x = x
        self.y = y
        self.name = name

    def set_xy(self, newx, newy):
        self.x = newx
        self.y = newy

        return self

    def get_xy(self):
        return self.x, self.y

    def get_polar(self):
        theta = np.arctan(self.y / self.x)
        rho = np.sqrt(self.x**2 + self.y**2)

        return theta, rho

    def __str__(self):
        return f"{self.name} @ ({self.x}, {self.y})"
