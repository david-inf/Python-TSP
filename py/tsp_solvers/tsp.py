# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la


def tsp_fun(seq, D):
    # seq: cities sequence
    # D: distance matrix

    # compute adjacency matrix
    A = adjacency(seq)

    # with possible nan values
    # due to np.Inf for unavailable edges
    cost_matrix_nan = np.multiply(A, D)
    cost_matrix = np.nan_to_num(cost_matrix_nan)

    return np.sum(cost_matrix)


def adjacency(seq):
    # seq: index sequence of cities
    # seq = [0,...,0]

    # number of cities in sequence
    ncity = seq.size - 1

    # compute incidence matrix
    A = np.zeros((ncity, ncity))
    for i in range(ncity):
        # get subsequent cities
        from_city = seq[i]  # starting city
        to_city = seq[i+1]  # ending city
        A[from_city, to_city] = 1

    return A


def random_seq(ncity, seed=42):
    # ncity: number of cities to generate
    # generate a sequence that starts from a city indexed with 0
    # and ends in a random one

    # set random numbers generator
    rng = np.random.default_rng(seed)

    # initialize sequence with 0 as starting and ending city
    seq = np.zeros(ncity + 1, dtype=np.int32)

    # random middle cities sequence
    middle = rng.permutation(np.arange(1, ncity))

    # create the final sequence
    seq[1:ncity] = middle

    # seq: order in which each city is visited
    # seq = [0,...,0]
    return seq


def create_city(coords):
    # create a list of City objects that represent the TSP nodes
    # coords: coordinates matrix Nx2
    # from City0 to CityN

    ncity = coords.shape[0]

    cities = []
    for i in range(ncity):
        cities.append(City(coords[i, 0], coords[i, 1], "City" + str(i)))

    # cities: list of City, does not consider the path
    return cities


def generate_cities(ncity, l=10, seed=42):
    # generate random coordinates
    # ncity: total number of cities

    rng = np.random.default_rng(seed)

    # cities coordinates c_i=(x,y)
    # C = rng.uniform(0, l, ncity * 2).reshape((-1, 2))

    C = np.empty((ncity, 2))
    C[:,0] = rng.uniform(-l//2, l//2, ncity)  # x-coordinate
    C[:,1] = rng.uniform(-l//2, l//2, ncity)  # y-coordinate

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

    # assign position to city
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
