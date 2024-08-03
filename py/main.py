# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from tsp_solvers.tsp import (
    tsp_fun, circle_cities, random_seq, create_city, generate_cities, adjacency)
from tsp_solvers.diagnostic_utils import (
    diagnostic, plot_points, plot_fun, annealing_diagnostic, local_search_animation,
    annealing_animation, _plot_nodes, energy_landscape, energy_landscape_ann)

from tsp_solvers.optimize import solve_tsp

plots_dir = "./plots/"
N_grid = [30, 50]
N1 = 30
N2 = 50

# %% TSP circular

D_circle, C_circle = zip(*[circle_cities(n) for n in N_grid])
D_circle = list(D_circle)
C_circle = list(C_circle)

D1_circle, C1_circle = circle_cities(N1)
D2_circle, C2_circle = circle_cities(N2)

# %%% plot graph

_plot_nodes(C1_circle, text=True)
# plt.savefig(plots_dir + "circle-nodes.pdf")

# %%% greedy nearest neighbor

res_circle_greedy1 = solve_tsp(tsp_fun, D1_circle, "nearest-neighbor",
                               options=dict(method="exact"))
plot_points(C1_circle, res_circle_greedy1)
plt.savefig(plots_dir + "circle-greedy1.pdf")

res_circle_greedy2 = solve_tsp(tsp_fun, D2_circle, "nearest-neighbor",
                               options=dict(method="weighted"))
plot_points(C2_circle, res_circle_greedy2)
plt.savefig(plots_dir + "circle-greedy2.pdf")
plt.savefig(plots_dir + "circle-greedy2.png")

# %%% swap local search

# res_circle_swap = solve_tsp(tsp_fun, D1, solver="local-search",
    # options=dict(method="swap", maxiter=1000, random_state=42))

res_circle_swap1 = solve_tsp(tsp_fun, D1_circle, solver="multi-start",
    options=dict(base_alg="local-search", nsim=1000, random_state=42, n_jobs=6,
        base_options=dict(method="swap", maxiter=1000, random_state=None))).solver_obj

res_circle_swap2 = solve_tsp(tsp_fun, D2_circle, solver="multi-start",
    options=dict(base_alg="local-search", nsim=1000, random_state=42, n_jobs=6,
        base_options=dict(method="swap", maxiter=1000, random_state=None))).solver_obj

# %%%%

diagnostic(C1_circle, res_circle_swap1)
plt.savefig(plots_dir + "circle-swap1.pdf")

diagnostic(C2_circle, res_circle_swap2)
plt.savefig(plots_dir + "circle-swap2.pdf")

# %%%%

local_search_animation(res_circle_swap2, C2_circle, "circle-swap.mp4")

# %%% reverse local search

# res_circle_reverse = solve_tsp(tsp_fun, D1, solver="local-search",
#     options=dict(method="reverse", maxiter=1000, random_state=42))

res_circle_reverse1 = solve_tsp(tsp_fun, D1_circle, solver="multi-start",
    options=dict(base_alg="local-search", nsim=1000, random_state=42, n_jobs=6,
        base_options=dict(method="reverse", maxiter=1000, random_state=None))).solver_obj

res_circle_reverse2 = solve_tsp(tsp_fun, D2_circle, solver="multi-start",
    options=dict(base_alg="local-search", nsim=1000, random_state=42, n_jobs=6,
        base_options=dict(method="reverse", maxiter=1000, random_state=None))).solver_obj

# %%%%

diagnostic(C1_circle, res_circle_reverse1)
plt.savefig(plots_dir + "circle-reverse1.pdf")

diagnostic(C2_circle, res_circle_reverse2)
plt.savefig(plots_dir + "circle-reverse2.pdf")

# %%%%

local_search_animation(res_circle_reverse2, C2_circle, "circle-reverse.mp4")

# %%% simulated annealing

# res_circle_ann1 = solve_tsp(tsp_fun, D1_circle, solver="simulated-annealing",
#     options=dict(perturbation="reverse", maxiter_outer=1000, maxiter_inner=1000,
#                  cooling_rate=0.995, random_state=None))

# res_circle_ann2 = solve_tsp(tsp_fun, D2_circle, solver="simulated-annealing",
#     options=dict(perturbation="reverse", maxiter_outer=1000, maxiter_inner=1000,
#                  cooling_rate=0.995, random_state=None))

res_circle_ann1 = solve_tsp(tsp_fun, D1_circle, solver="multi-start",
    options=dict(base_alg="sim-annealing", nsim=10, random_state=42, n_jobs=8,
        base_options=dict(perturbation="reverse", maxiter_outer=1000, maxiter_inner=500,
            cooling_rate=0.995, random_state=None))).solver_obj

res_circle_ann2 = solve_tsp(tsp_fun, D2_circle, solver="multi-start",
    options=dict(base_alg="sim-annealing", nsim=10, random_state=42, n_jobs=8,
        base_options=dict(perturbation="reverse", maxiter_outer=1000, maxiter_inner=500,
            cooling_rate=0.995, random_state=None))).solver_obj

# %%%%

annealing_diagnostic(C1_circle, res_circle_ann1)
plt.savefig(plots_dir + "circle-annealing-quad1.pdf")

annealing_diagnostic(C2_circle, res_circle_ann2)
plt.savefig(plots_dir + "circle-annealing-quad2.pdf")

# %%%%

annealing_animation(res_circle_ann2, C2_circle, "circle-annealing-quad.mp4")

# %%% energy landscape

def solve_multiple_ls(cost_list, nsim, ls_iters):

    reverse_solvers = []

    for i in range(len(cost_list)):
    
        ## local search
        res_ls = []
        for j in range(len(ls_iters)):
            res_reverse = solve_tsp(tsp_fun, cost_list[i], "multi-start",
                options=dict(base_alg="local-search", nsim=nsim, random_state=42, n_jobs=8,
                    base_options=dict(method="reverse", maxiter=ls_iters[j], random_state=None)))

            res_ls.append(res_reverse)

        reverse_solvers.append(res_ls)

    return reverse_solvers


def solve_multiple_ann(cost_list, nsim):

    reverse_solvers = []

    for i in range(len(cost_list)):

        res_ann = solve_tsp(tsp_fun, cost_list[i], "multi-start",
            options=dict(base_alg="sim-annealing", nsim=nsim, random_state=42, n_jobs=8,
                base_options=dict(perturbation="reverse", maxiter_outer=500,
                                  maxiter_inner=200, random_state=None)))

        reverse_solvers.append(res_ann)

    return reverse_solvers


# def solve_multiple_ann(cost_list):

#     def _inner(cost):
#         res_annealing = solve_tsp(tsp_fun, cost, "simulated-annealing",
#             options=dict(perturbation="reverse", maxiter_outer=1000, maxiter_inner=1000,
#                          cooling_rate=0.995, random_state=None))
#         return res_annealing

#     with Parallel(n_jobs=6, backend="loky") as parallel:
#         results = parallel(
#             delayed(_inner)(cost)
#             for cost in cost_list)

#     results.sort(key=lambda x: x.x.size)

#     return results

# %%%%

energies_circle_ls = solve_multiple_ls(D_circle, 1000, [1000,5000])
energies_circle_ann = solve_multiple_ann(D_circle, 200)

# circle_ann = solve_multiple_ann(D_circle)

# %%%%

for i in range(2):
    # energy_landscape(energies_circle_ls[i], [opt.fun for opt in circle_ann])
    energy_landscape(energies_circle_ls[i])
    plt.savefig(plots_dir + f"circle-reverse-energy{i}.pdf")


energy_landscape_ann(energies_circle_ann)
plt.savefig(plots_dir + "circle-annealing-energy.pdf")




# %% TSP random

D_rand, C_rand = zip(*[generate_cities(n) for n in N_grid])
D_rand = list(D_rand)
C_rand = list(C_rand)

D1_rand, C1_rand = generate_cities(N1)
D2_rand, C2_rand = generate_cities(N2)

# %%%

_plot_nodes(C1_rand, text=True)
# plt.savefig(plots_dir + "rand-nodes.pdf")

# %%% greed nearest neighbor

res_rand_greedy1 = solve_tsp(tsp_fun, D1_rand, "nearest-neighbor", options=dict(method="weighted"))
plot_points(C1_rand, res_rand_greedy1)
plt.savefig(plots_dir + "rand-greedy1.pdf")

res_rand_greedy2 = solve_tsp(tsp_fun, D2_rand, "nearest-neighbor", options=dict(method="weighted"))
plot_points(C2_rand, res_rand_greedy2)
plt.savefig(plots_dir + "rand-greedy2.pdf")
plt.savefig(plots_dir + "rand-greedy2.png")

# %%% swap local search

# res_rand_swap = solve_tsp(tsp_fun, D2, solver="local-search",
    # options=dict(method="swap", maxiter=1000, random_state=42))

res_rand_swap1 = solve_tsp(tsp_fun, D1_rand, solver="multi-start",
    options=dict(base_alg="local-search", nsim=1000, random_state=42, n_jobs=6,
        base_options=dict(method="swap", maxiter=1000, random_state=None))).solver_obj

res_rand_swap2 = solve_tsp(tsp_fun, D2_rand, solver="multi-start",
    options=dict(base_alg="local-search", nsim=1000, random_state=42, n_jobs=6,
        base_options=dict(method="swap", maxiter=1000, random_state=None))).solver_obj

# %%%%

diagnostic(C1_rand, res_rand_swap1)
plt.savefig(plots_dir + "rand-swap1.pdf")

diagnostic(C2_rand, res_rand_swap2)
plt.savefig(plots_dir + "rand-swap2.pdf")

# %%%%

local_search_animation(res_rand_swap2, C2_rand, "rand-swap.mp4")

# %%% reverse local search

# res_rand_reverse = solve_tsp(tsp_fun, D2, solver="local-search",
    # options=dict(method="reverse", maxiter=1000, random_state=42))

res_rand_reverse1 = solve_tsp(tsp_fun, D1_rand, solver="multi-start",
    options=dict(base_alg="local-search", nsim=1000, random_state=42, n_jobs=6,
        base_options=dict(method="reverse", maxiter=1000, random_state=None))).solver_obj

res_rand_reverse2 = solve_tsp(tsp_fun, D2_rand, solver="multi-start",
    options=dict(base_alg="local-search", nsim=1000, random_state=42, n_jobs=6,
        base_options=dict(method="reverse", maxiter=1000, random_state=None))).solver_obj

# %%%%

diagnostic(C1_rand, res_rand_reverse1)
plt.savefig(plots_dir + "rand-reverse1.pdf")

diagnostic(C2_rand, res_rand_reverse2)
plt.savefig(plots_dir + "rand-reverse2.pdf")

# %%%%

local_search_animation(res_rand_reverse2, C2_rand, "rand-reverse.mp4")

# %%% simulated annealing

# res_rand_ann1 = solve_tsp(tsp_fun, D1_rand, solver="simulated-annealing",
#     options=dict(perturbation="reverse", maxiter_outer=1000, maxiter_inner=1000,
#                  cooling_rate=0.995, random_state=42))

# res_rand_ann2 = solve_tsp(tsp_fun, D2_rand, solver="simulated-annealing",
#     options=dict(perturbation="reverse", maxiter_outer=1000, maxiter_inner=1000,
#                  cooling_rate=0.995, random_state=42))

res_rand_ann1 = solve_tsp(tsp_fun, D1_rand, solver="multi-start",
    options=dict(base_alg="sim-annealing", nsim=10, random_state=42, n_jobs=8,
        base_options=dict(perturbation="reverse", maxiter_outer=1000, maxiter_inner=500,
            cooling_rate=0.995, random_state=None))).solver_obj

res_rand_ann2 = solve_tsp(tsp_fun, D2_rand, solver="multi-start",
    options=dict(base_alg="sim-annealing", nsim=10, random_state=42, n_jobs=8,
        base_options=dict(perturbation="reverse", maxiter_outer=1000, maxiter_inner=500,
            cooling_rate=0.995, random_state=None))).solver_obj

# %%%%

annealing_diagnostic(C1_rand, res_rand_ann1)
plt.savefig(plots_dir + "rand-annealing-quad1.pdf")

annealing_diagnostic(C2_rand, res_rand_ann2)
plt.savefig(plots_dir + "rand-annealing-quad2.pdf")

# %%%%

annealing_animation(res_rand_ann2, C2_rand, "rand-annealing-quad.mp4")

# %%% Energy landscape

energies_rand_ls = solve_multiple_ls(D_rand[:2], 1000, [1000,5000])
energies_rand_ann = solve_multiple_ann(D_rand[:2], 200)

# %%%%

for i in range(2):
    # energy_landscape(energies_rand_ls[i], [opt.fun for opt in rand_ann])
    energy_landscape(energies_rand_ls[i])
    plt.savefig(plots_dir + f"rand-reverse-energy{i}.pdf")

energy_landscape_ann(energies_rand_ann)
plt.savefig(plots_dir + "rand-annealing-energy.pdf")



# %% Playground

# x0 = random_seq(30)
# D1_circle[np.ix_(x0[[1,4,8]], x0[[1,4,8]])]

res_gen = solve_tsp(tsp_fun, D1_circle, "genetic-alg",
    options=dict(maxiter=300, individuals=200, nson=200, method="weighted",
                 mutation_iters=100))

diagnostic(C1_circle, res_gen)

