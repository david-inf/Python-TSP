# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from tsp_solvers.tsp import (
    tsp_fun, circle_cities, random_seq, create_city, generate_cities, adjacency)
from tsp_solvers.diagnostic_utils import (
    diagnostic, plot_points, plot_fun, annealing_diagnostic, local_search_animation,
    annealing_animation, _plot_nodes, energy_landscape)

from tsp_solvers.optimize import solve_tsp

plots_dir = "./plots/"
N_grid = [10, 15, 20, 30]
N = 40

# %% TSP circular

D_circle, C_circle = zip(*[circle_cities(n) for n in N_grid])
D_circle = list(D_circle)
C_circle = list(C_circle)

D1, C1 = circle_cities(N)

# %%%

_plot_nodes(C1)
plt.savefig(plots_dir + "circle-nodes.pdf")

# %%% swap local search

res_circle_swap = solve_tsp(tsp_fun, D1, solver="swap",
    options=dict(maxiter=1000, random_state=42))

# %%%%

diagnostic(C1, res_circle_swap)
plt.savefig(plots_dir + "circle-swap.pdf")

# %%%%

local_search_animation(res_circle_swap, C1, "circle-swap.mp4")

# %%% reverse local search

res_circle_reverse = solve_tsp(tsp_fun, D1, solver="reverse",
    options=dict(maxiter=1000, random_state=42))

# %%%%

diagnostic(C1, res_circle_reverse)
plt.savefig(plots_dir + "circle-reverse.pdf")

# %%%%

local_search_animation(res_circle_reverse, C1, "circle-reverse.mp4")

# %%% simulated annealing

res_circle_ann = solve_tsp(tsp_fun, D1, solver="simulated-annealing",
    options=dict(perturbation="reverse", maxiter_outer=1000, maxiter_inner=800,
                 cooling_rate=0.995, random_state=42))

# %%%%

annealing_diagnostic(C1, res_circle_ann)
plt.savefig(plots_dir + "circle-annealing-quad.pdf")

# %%%%

annealing_animation(res_circle_ann, C1, "circle-annealing-quad.mp4")


# %%% energy landscape

def solve_multiple_ls(cost, size_grid, nsim, ls_iters=[30, 50, 100, 150]):

    reverse_solvers = []
    # annealing_solvers = []

    for i in range(len(size_grid)):
    
        ## local search
        res_reverse = solve_tsp(tsp_fun, cost[i], "multi-start",
            options=dict(base_alg="local-search", nsim=nsim, random_state=42, n_jobs=6,
                base_options=dict(solver="reverse", maxiter=ls_iters[i], random_state=None)))

        ## different local minima / nsim
        # ratio = 

        reverse_solvers.append((res_reverse, ls_iters[i]))
    
        ## sim annealing
        # res_annealing = solve_tsp(tsp_fun, cost[i], "multi-start",
        #     options=dict(base_alg="sim-annealing", nsim=nsim, random_state=42, n_jobs=6,
        #         base_options=dict(maxiter_outer=300, maxiter_inner=ls_iters[i],
        #             random_state=None)))

        # annealing_solvers.append(res_annealing)

    return reverse_solvers#, annealing_solvers

# %%%%

circle_energies_ls = solve_multiple_ls(D_circle, N_grid, 300, [100,500,800,1000])

# %%%%

energy_landscape(circle_energies_ls, N_grid)
plt.savefig(plots_dir + "circle-reverse-energy.pdf")

## get unique solution
# np.unique(solvers_list[0][0].x_seq, return_counts=True, axis=0)
## get theirs f(x)

## get unique f(x) from previous



# energy_landscape(energies[1], N_grid)
# plt.savefig(plots_dir + "circle-annealing-energy.pdf")


# %%% others

# multi-start but one solution and different perturbations each time
# res_multi_seed = solve_tsp(tsp_fun, D1, solver="multi-start",
#     options=dict(nsim=1400, base_alg="single-local-search", local_search="swap-rev",
#         local_search_options=dict(maxiter=500, random_state=None)))

# res_multi_ann = solve_tsp(tsp_fun, D1, solver="multi-start",
#     options=dict(nsim=5, base_alg="local-search+annealing", local_search="swap-rev", n_jobs=6,
#     local_search_options=dict(maxiter=100),
#     annealing_options=dict(maxiter_outer=500, maxiter_inner=200)))

# res_multi_swap_rev = solve_tsp(tsp_fun, D1, solver="multi-start",
#     options=dict(nsim=1000, local_search="swap-rev",
#                  local_search_options=dict(maxiter=1000, random_state=None)))

# brute = solve_tsp(tsp_fun, D1, "brute-force", options=dict(n_jobs=4))
# plot_points(C1, brute.x)


# %% TSP random

D_rand, C_rand = zip(*[generate_cities(n) for n in N_grid])
D_rand = list(D_rand)
C_rand = list(C_rand)

D2, C2 = generate_cities(N)

# %%%

_plot_nodes(C2)
plt.savefig(plots_dir + "rand-nodes.pdf")

# %%% swap local search

res_rand_swap = solve_tsp(tsp_fun, D2, solver="swap",
    options=dict(maxiter=1000, random_state=42))

# %%%%

diagnostic(C2, res_rand_swap)
plt.savefig(plots_dir + "rand-swap.pdf")

# %%%%

local_search_animation(res_rand_swap, C2, "rand-swap.mp4")

# %%% reverse local search

res_rand_reverse = solve_tsp(tsp_fun, D2, solver="reverse",
    options=dict(maxiter=1000, random_state=42))

# %%%%

diagnostic(C2, res_rand_reverse)
plt.savefig(plots_dir + "rand-reverse.pdf")

# %%%%

local_search_animation(res_rand_reverse, C2, "rand-reverse.mp4")

# %%% simulated annealing

res_rand_ann = solve_tsp(tsp_fun, D2, solver="simulated-annealing",
    options=dict(perturbation="reverse", maxiter_outer=1000, maxiter_inner=800,
                 cooling_rate=0.995, random_state=42))

# %%%%

annealing_diagnostic(C2, res_rand_ann)
plt.savefig(plots_dir + "rand-annealing-quad.pdf")

# %%%%

annealing_animation(res_rand_ann, C2, "rand-annealing-quad.mp4")


# %%% Energy landscape

energies_rand = solve_multiple(D_rand, N_grid)

# %%%%

energy_landscape(energies_rand[0], N_grid)
plt.savefig(plots_dir + "rand-reverse-energy.pdf")

energy_landscape(energies_rand[1], N_grid)
plt.savefig(plots_dir + "rand-annealing-energy.pdf")



# %% First try

fig, ax1 = plt.subplots()

# Plot data with primary x-axis
ax1.plot(1 / np.log(np.arange(2, 801)), label='1/log(x)')
ax1.grid(True, which="both", axis="both")
# ax1.set_ylabel('Y data')
# ax1.set_xlabel('sin(y)', color='g')
# ax1.tick_params(axis='x', labelcolor='g')

# Create secondary x-axis
ax2 = ax1.twinx()
ax2.plot(50 * 0.99**np.arange(2, 801), label='c_k', color="tab:orange")
# ax2.plot(res_ann_swap_rev.temp_seq[:342])
# ax2.set_xlabel('exp(y/3)', color='b')
# ax2.tick_params(axis='x', labelcolor='b')


# %%% try the algorithm

# res1 = relax(D1, seq1, "swap", 1000)
# res2 = relax(D1, seq1, "swap-rev", 1000)
# res3 = brute_force(D1, seq1)

# %%% diagnose the solution

## plot initial guess
# plot_points(C1, seq1)

# # tsp_fun(D1, adjacency(np.arange(n1)))

# ## plot solution and objective function performance
# diagnostic(C1, res1)
# diagnostic(C1, res2)
# diagnostic(C1, res2)

# print(res1)
# print("% ---- %")
# print(res2)
# print("% ---- %")
# print(res3)


# %% random cities

# n_city = 25
# D_city, C_city = generate_cities(n_city)
# seq_city = random_seq(n_city)

# # %%% solve random cities problem

# res1_city = relax(seq_city, D_city, "swap", 1000)
# res2_city = relax(seq_city, D_city, "swap-rev", 1000)

# %%% diagnose random cities solution

## plot initial guess
# plot_points(C_city, seq_city)

# # tsp_fun(D1, adjacency(np.arange(n1)))

# ## plot solution and objective function performance
# diagnostic(C_city, res1_city)
# diagnostic(C_city, res2_city)

# print(f"Init cost: {tsp_fun(seq_city, D_city):.2f}")
# print("% ---- %")
# print(res1_city)
# print("% ---- %")
# print(res2_city)


# %% es. schoen

# D_schoen = np.array([[0,1,np.Inf,2,2],
#                     [1,0,6,5,1],
#                     [np.Inf,6,0,4,3],
#                     [2,5,4,0,1],
#                     [2,1,3,1,0]])

# seq_schoen = np.array([0,1,2,3,4,0])
# cost_schoen = tsp_fun(seq_schoen, D_schoen)
# print(f"Schoen cost: {cost_schoen:.2f}")

# C_schoen = np.array([[0,2],[2,3],[4,2],[1,0],[3,0]])

# # %%% solve schoen

# res_schoen = relax(D_schoen, seq_schoen, "swap", 1000)
# res_schoen2 = relax(D_schoen, seq_schoen, "swap-rev", 1000)
# res_schoen3 = brute_force(D_schoen, seq_schoen)

# # %%% plot schoen

# plot_points(C_schoen, seq_schoen)

# diagnostic(C_schoen, res_schoen)
# diagnostic(C_schoen, res_schoen2)


# # %% another es

# D_es = np.array([
#     [0,  5, 4, 10],
#     [5,  0, 8,  5],
#     [4,  8, 0,  3],
#     [10, 5, 3,  0]
# ])

# # %%% solve

# res1_es = relax(D_es)
# res2_es = relax(D_es, solver="swap-rev")
# res3_es = brute_force(D_es)

# # %%% diagnose

# plot_fun(res1_es.fun_seq)
# plot_fun(res2_es.fun_seq)




