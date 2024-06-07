# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from tsp import tsp_fun, generate_cities, circle_cities, create_city, random_seq, adjacency
from utils import diagnostic, plot_path
from solvers import relax

# %% first try

# TODO: try with turtle

n1 = 15

# D1, C1 = generate_cities(n1)
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

res1 = relax(seq1, D1, "swap", 1000)

res2 = relax(seq1, D1, "swap-rev", 1000)

# %%% diagnose the solution

## plot initial guess
plot_path(C1, seq1)

# tsp_fun(D1, adjacency(np.arange(n1)))

## plot solution and objective function performance
diagnostic(C1, res1.x, res1.fun_seq)
diagnostic(C1, res2.x, res2.fun_seq)

print(res1)
print("% ---- %")
print(res2)


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

res_schoen = relax(seq_schoen, D_schoen, "swap", 1000)
res_schoen2 = relax(seq_schoen, D_schoen, "swap-rev", 1000)

# %%% plot schoen

plot_path(C_schoen, seq_schoen)

diagnostic(C_schoen, res_schoen.x, res_schoen.fun_seq)
diagnostic(C_schoen, res_schoen2.x, res_schoen2.fun_seq)


