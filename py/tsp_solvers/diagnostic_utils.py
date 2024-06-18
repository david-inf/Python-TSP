# -*- coding: utf-8 -*-

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot_fun(opt_res, ax=None):

    if ax is None:
        fig, ax = plt.subplots()

    ## plot f(x) sequence
    ax.plot(opt_res.fun_seq)

    ax.set_xscale("log")  # logarithmic scale
    ax.set_xlim(0.9)  # for the k=0 iteration, starting solution

    ax.grid(True, which="both", axis="both")

    ax.set_xlabel("iterations")
    ax.set_ylabel(r"$f(x)$")
    ax.set_title(opt_res.solver)


def plot_points(coords, opt_res, ax=None):
    # coords: coordinates matrix
    # seq: array_like

    # coordinates according to given sequence
    x_seq = coords[opt_res.x, 0]
    y_seq = coords[opt_res.x, 1]

    # sequence colors indicating starting and end points
    colors = ["blue"] * (opt_res.x.size - 1) + ["red"]

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(x_seq, y_seq)
    ax.scatter(x_seq, y_seq, marker="o", color=colors, s=10)

    for i in range(coords.shape[0]):
        ax.text(coords[i,0], coords[i,1], f"{i}")

    ax.grid(True, which="both", axis="both")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(f"f(x): {opt_res.fun:.2f}, N: {opt_res.x.size-1}")


def diagnostic(coords, opt_res):
    # plot cities map a objective function performance
    # coords: coordinates matrix
    # opt_res: OptimizeResult

    fig, axs = plt.subplots(1, 2, layout="constrained")
    # plt.title(opt_res.solver)

    ## plot graph
    plot_points(coords, opt_res, axs[0])

    ## plot function performance
    plot_fun(opt_res, ax=axs[1])


def _plot_chi(opt_res, ax=None):
    # fun_seq: array_like

    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(np.arange(opt_res.chi_seq.size), opt_res.chi_seq, s=3)

    ax.grid(True, which="both", axis="both")

    ax.set_xlabel("iterations")
    ax.set_ylabel(r"$\chi(x)$")
    ax.set_title("Acceptance rate")


def _plot_temp(opt_res, ax=None):

    if ax is None:
        fig, ax = plt.subplots()
    
    ## plot f(x) sequence
    ax.plot(opt_res.temp_seq)
    
    ax.grid(True, which="both", axis="both")
    
    ax.set_xlabel("iterations")
    ax.set_ylabel(r"$T_k$")
    ax.set_title("Temperature cooling")
            

def annealing_diagnostic(coords, opt_res):

    fig, axs = plt.subplots(2, 2, layout="constrained")
    # plt.title(res.solver)

    ## plot graph with path
    plot_points(coords, opt_res, axs[0, 0])

    ## plot objective function against iteration (Markov chain)
    plot_fun(opt_res, ax=axs[0, 1])

    opt_res2 = copy.deepcopy(opt_res)
    opt_res2.fun_seq = opt_res2.fun_acc_seq
    plot_fun(opt_res2, ax=axs[0, 1])

    axs[0, 1].legend([r"$f(x^\ast)$", r"$f(x^k)$"])

    ## plot chi against iterations
    _plot_chi(opt_res, axs[1, 0])

    ## plot temperature against iterations
    _plot_temp(opt_res, ax=axs[1, 1])

    return None


def energy_view(opt_res, ax=None):

    if ax is None:
        fig, ax = plt.subplots()

    ax.hist(opt_res.fun_seq)

    return None


# ************************************************** #
## Animation ##

def _plot_nodes(coords, ax=None):
    # coords: coordinates matrix

    # coordinates according to given sequence
    x_coord = coords[:, 0]
    y_coord = coords[:, 1]

    # sequence colors indicating starting and end points
    colors = ["blue"] * (coords.shape[0] - 1) + ["red"]

    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(x_coord, y_coord, marker="o", color=colors, s=15)

    for i in range(coords.shape[0]):
        ax.text(coords[i,0], coords[i,1], f"{i}")

    ax.grid(True, which="both", axis="both")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_aspect('equal', adjustable='box')


def local_search_animation(opt_res, coords, filename, delay=100):

    seq_history = opt_res.x_seq  # (N+1) x (maxiter+1)
    fun_history = opt_res.fun_seq  # (maxiter+1)

    fig, axs = plt.subplots(1, 2, layout="constrained")

    ## draw nodes
    nodes = _plot_nodes(coords, axs[0])
    axs[0].set_title(f"Current best path, N: {opt_res.x.size-1}")
    line_path, = axs[0].plot([], [], lw=2)  # null path
    annotation = axs[0].text(0.025, 0.98, '', transform=axs[0].transAxes, va='top')

    ## objective function
    axs[1].set_xlim(0.9, fun_history.size)
    axs[1].set_ylim(np.min(fun_history)*0.95, np.max(fun_history)*1.05)
    axs[1].grid(True, which="both", axis="both")
    axs[1].set_xlabel("iterations")
    axs[1].set_ylabel(r"$f(x)$")
    line_fun, = axs[1].plot([], [], lw=2)  # null objective function
    axs[1].set_title(opt_res.solver)
    axs[1].set_xscale("log")


    def update(frame):

        path = seq_history[frame]
        f_val = fun_history[frame]
        f_current_seq = fun_history[:frame+1]

        ## path
        x = coords[path, 0]
        y = coords[path, 1]
        line_path.set_data(x, y)
        annotation.set_text(f"f: {f_val:.2f}")

        ## objective function
        seq = np.insert(np.arange(1, frame+1), 0, 0)
        line_fun.set_data(seq, f_current_seq)

        return line_path, annotation, line_fun


    ani = animation.FuncAnimation(
        fig, update, frames=seq_history.shape[0], interval=delay, repeat=False, blit=True)

    ani.save(filename, writer="ffmpeg", fps=30)


def annealing_animation(opt_res, coords, filename, delay=100):

    x_history = opt_res.x_seq  # (N+1) x (maxiter+1)
    f_history = opt_res.fun_seq  # (maxiter+1)
    f_acc_history = opt_res.fun_acc_seq  # (maxiter+1)
    chi_history = opt_res.chi_seq  # (maxiter+1)
    temp_history = opt_res.temp_seq  # (maxiter+1)

    fig, axs = plt.subplots(2, 2, layout="constrained")

    ## draw nodes
    nodes = _plot_nodes(coords, axs[0, 0])
    line_path, = axs[0, 0].plot([], [], lw=2)  # null path
    annotation = axs[0, 0].text(0.025, 0.98, '', transform=axs[0, 0].transAxes, va='top')
    axs[0, 0].set_title(f"Current best path, N: {opt_res.x.size-1}")

    ## draw best and accepted objective function
    axs[0, 1].set_xlim(0.9, f_history.size)
    axs[0, 1].set_ylim(np.min(f_history)*0.95, np.max(f_history)*1.05)
    axs[0, 1].grid(True, which="both", axis="both")
    axs[0, 1].set_xlabel("iterations")
    axs[0, 1].set_ylabel(r"$f(x)$")
    line_fun, = axs[0, 1].plot([], [], lw=2)  # null objective function
    line_fun_acc, = axs[0, 1].plot([], [], lw=2)  # null accepted obj fun
    axs[0, 1].set_title(opt_res.solver)
    axs[0, 1].set_xscale("log")

    ## acceptance rate
    axs[1, 0].set_xlim(0, chi_history.size)
    axs[1, 0].set_ylim(0, 1)
    axs[1, 0].grid(True, which="both", axis="both")
    axs[1, 0].set_xlabel("iterations")
    axs[1, 0].set_ylabel(r"$\chi(x)$")
    line_chi, = axs[1, 0].plot([], [], lw=2, marker="o", linestyle="None", markersize=2)
    axs[1, 0].set_title("Acceptance rate")

    ## temperature
    axs[1, 1].set_xlim(0, temp_history.size)
    axs[1, 1].set_ylim(0, np.max(temp_history)*1.05)
    axs[1, 1].grid(True, which="both", axis="both")
    axs[1, 1].set_xlabel("iterations")
    axs[1, 1].set_ylabel(r"$T_k$")
    line_temp, = axs[1, 1].plot([], [], lw=2)
    axs[1, 1].set_title("Temperature cooling")


    def update(frame):

        path = x_history[frame]
        f_val = f_history[frame]
        f_current_seq = f_history[:frame+1]
        f_acc_current_seq = f_acc_history[:frame+1]
        chi_current_seq = chi_history[:frame+1]
        temp_current_seq = temp_history[:frame+1]

        ## path
        x = coords[path, 0]
        y = coords[path, 1]
        line_path.set_data(x, y)
        annotation.set_text(f"f: {f_val:.2f}")

        ## objective function
        seq = np.insert(np.arange(1, frame+1), 0, 0)
        line_fun.set_data(seq, f_current_seq)
        line_fun_acc.set_data(seq, f_acc_current_seq)

        ## acceptance rate
        line_chi.set_data(np.arange(frame+1), chi_current_seq)

        ## temperature
        line_temp.set_data(np.arange(frame+1), temp_current_seq)

        return line_path, annotation, line_fun, line_fun_acc, line_chi, line_temp


    ani = animation.FuncAnimation(
        fig, update, frames=x_history.shape[0], interval=delay, repeat=False, blit=True)

    ani.save(filename, writer="ffmpeg", fps=30)
