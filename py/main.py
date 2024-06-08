# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from tsp import tsp_fun, generate_cities, circle_cities, create_city, random_seq, adjacency
from utils import diagnostic, plot_points, plot_fun
from solvers import relax, brute_force


# %% circle cities

# TODO: try with turtle

n1 = 10

D1, C1, _ = circle_cities(n1)
cities1 = create_city(C1)

seq1 = random_seq(n1, seed=43)
# seq2 = random_seq(n1, seed=43)
# seq3 = random_seq(n1, [3, 2])

adj1 = adjacency(seq1)
# adj2 = adjacency(seq2)

dist1 = tsp_fun(seq1, D1)
# dist2 = tsp_fun(seq2, D2)

# %%% try the algorithm

res1 = relax(D1, seq1, "swap", 1000)
res2 = relax(D1, seq1, "swap-rev", 1000)
res3 = brute_force(D1, seq1)

# %%% diagnose the solution

## plot initial guess
plot_points(C1, seq1)

# tsp_fun(D1, adjacency(np.arange(n1)))

## plot solution and objective function performance
diagnostic(C1, res1)
diagnostic(C1, res2)
diagnostic(C1, res2)

print(res1)
print("% ---- %")
print(res2)
print("% ---- %")
print(res3)


# %% random cities

n_city = 25
D_city, C_city = generate_cities(n_city)
seq_city = random_seq(n_city)

# %%% solve random cities problem

res1_city = relax(seq_city, D_city, "swap", 1000)
res2_city = relax(seq_city, D_city, "swap-rev", 1000)

# %%% diagnose random cities solution

## plot initial guess
plot_points(C_city, seq_city)

# tsp_fun(D1, adjacency(np.arange(n1)))

## plot solution and objective function performance
diagnostic(C_city, res1_city)
diagnostic(C_city, res2_city)

print(f"Init cost: {tsp_fun(seq_city, D_city):.2f}")
print("% ---- %")
print(res1_city)
print("% ---- %")
print(res2_city)


# %% es. schoen

D_schoen = np.array([[0,1,np.Inf,2,2],
                    [1,0,6,5,1],
                    [np.Inf,6,0,4,3],
                    [2,5,4,0,1],
                    [2,1,3,1,0]])

seq_schoen = np.array([0,1,2,3,4,0])
cost_schoen = tsp_fun(seq_schoen, D_schoen)
print(f"Schoen cost: {cost_schoen:.2f}")

C_schoen = np.array([[0,2],[2,3],[4,2],[1,0],[3,0]])

# %%% solve schoen

res_schoen = relax(D_schoen, seq_schoen, "swap", 1000)
res_schoen2 = relax(D_schoen, seq_schoen, "swap-rev", 1000)
res_schoen3 = brute_force(D_schoen, seq_schoen)

# %%% plot schoen

plot_points(C_schoen, seq_schoen)

diagnostic(C_schoen, res_schoen)
diagnostic(C_schoen, res_schoen2)


# %% another es

D_es = np.array([
    [0,  5, 4, 10],
    [5,  0, 8,  5],
    [4,  8, 0,  3],
    [10, 5, 3,  0]
])

# %%% solve

res1_es = relax(D_es)
res2_es = relax(D_es, solver="swap-rev")
res3_es = brute_force(D_es)

# %%% diagnose

plot_fun(res1_es.fun_seq)
plot_fun(res2_es.fun_seq)




