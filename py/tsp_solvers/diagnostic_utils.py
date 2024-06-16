# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot_fun(fun_seq, scalex="log", ylab=r"$f(x)$", ax=None):
    # fun_seq: array_like

    if ax is None:
        fig, ax = plt.subplots()


    if scalex == "log":
        iterations = np.arange(1, fun_seq.size + 1)
        ax.plot(iterations, fun_seq)

        ax.set_xscale(scalex)

        ax.set_xlim(1)
    
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels([0, 10, 100])

    else:
        iterations = np.arange(fun_seq.size)
        ax.plot(iterations, fun_seq)

        ax.set_xscale(scalex)

        ax.set_xlim(0)

    ax.grid(True, which="both", axis="both")

    ax.set_xlabel("iterations")
    ax.set_ylabel(ylab)


def plot_points(coords, seq, ax=None):
    # coords: coordinates matrix
    # seq: array_like

    # coordinates according to given sequence
    x_seq = coords[seq, 0]
    y_seq = coords[seq, 1]

    # sequence colors indicating starting and end points
    colors = ["blue"] * (seq.size - 1) + ["red"]

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


def diagnostic(coords, opt_res):
    # plot cities map a objective function performance
    # coords: coordinates matrix
    # opt_res: OptimizeResult

    fig, axs = plt.subplots(1, 2, layout="constrained")
    # plt.title(opt_res.solver)

    ## plot graph
    plot_points(coords, opt_res.x, axs[0])
    axs[0].set_title(f"Solution path\nf(x): {opt_res.fun:.4f}")

    ## plot function performance
    fun_seq = opt_res.fun_seq[:opt_res.nit+1]
    plot_fun(fun_seq, ax=axs[1])
    axs[1].set_title(opt_res.solver)


def plot_chi(chi_seq, ax=None):
    # fun_seq: array_like

    if ax is None:
        fig, ax = plt.subplots()

    iterations = np.arange(1, chi_seq.size + 1)
    ax.scatter(iterations, chi_seq, s=5)

    ax.grid(True, which="both", axis="both")

    ax.set_xlabel("iterations")
    ax.set_ylabel(r"$\chi(x)$")
            

def annealing_diagnostic(coords, res):

    fig, axs = plt.subplots(2, 2, layout="constrained")
    # plt.title(res.solver)

    ## plot graph with path
    plot_points(coords, res.x, axs[0, 0])

    ## plot objective function against iteration (Markov chain)
    fun_seq = res.fun_seq[:res.nit+1]
    plot_fun(fun_seq, ax=axs[0, 1])

    fun_acc = res.fun_acc_seq[:res.nit+1]
    plot_fun(fun_acc, ax=axs[0, 1])

    ## plot chi against iterations
    plot_chi(res.chi_seq[:res.nit+1], axs[1, 0])

    ## plot temperature against iterations
    plot_fun(res.temp_seq[:res.nit+1], "linear", ylab=r"$T_k$", ax=axs[1, 1])

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
    fun_history = opt_res.fun_seq

    fig, axs = plt.subplots(1, 2, layout="constrained")

    ## draw nodes
    nodes = _plot_nodes(coords, axs[0])
    line_path, = axs[0].plot([], [], lw=2)  # null path
    annotation = axs[0].text(0.025, 0.98, '', transform=axs[0].transAxes, va='top')
    axs[0].set_title("Current best path")

    ## draw objective function
    axs[1].set_xlim(0, fun_history.size)
    axs[1].set_ylim(np.min(fun_history)*0.95, np.max(fun_history)*1.05)
    axs[1].grid(True, which="both", axis="both")
    axs[1].set_xlabel("iterations")
    axs[1].set_ylabel(r"$f(x)$")
    line_fun, = axs[1].plot([], [], lw=2)  # null objective function
    axs[1].set_title("Objective function performance")


    def update(frame):

        path = seq_history[frame]
        f_val = fun_history[frame]
        f_current_seq = fun_history[:frame+1]

        ## path
        x = coords[path, 0]
        y = coords[path, 1]
        line_path.set_data(x, y)
        annotation.set_text(f'f(x): {f_val:.4f}')

        ## objective function
        line_fun.set_data(range(frame+1), f_current_seq)

        return line_path, annotation, line_fun


    ani = animation.FuncAnimation(
        fig, update, frames=seq_history.shape[0], interval=delay, repeat=False, blit=True)

    ani.save(filename, writer="ffmpeg", fps=30)


def annealing_animation(opt_res, coords, filename, delay=100):

    x_history = opt_res.x_seq
    f_history = opt_res.fun_seq
    f_acc_history = opt_res.fun_acc_seq
    chi_history = opt_res.chi_seq
    temp_history = opt_res.temp_seq

    fig, axs = plt.subplots(2, 2, layout="constrained")

    ## draw nodes
    nodes = _plot_nodes(coords, axs[0, 0])
    line_path, = axs[0, 0].plot([], [], lw=2)  # null path
    annotation = axs[0, 0].text(0.025, 0.98, '', transform=axs[0, 0].transAxes, va='top')
    axs[0, 0].set_title("Current best path")

    ## draw best and accepted objective function
    axs[0, 1].set_xlim(0, f_history.size)
    axs[0, 1].set_ylim(np.min(f_history)*0.95, np.max(f_history)*1.05)
    axs[0, 1].grid(True, which="both", axis="both")
    axs[0, 1].set_xlabel("iterations")
    axs[0, 1].set_ylabel(r"$f(x)$")
    line_fun, = axs[0, 1].plot([], [], lw=2)  # null objective function
    line_fun_acc, = axs[0, 1].plot([], [], lw=2)
    axs[0, 1].set_title("Objective function performance")
    # axs[0].set_xscale("log")

    # iterations = np.arange(1, fun_seq.size + 1)
    # ax.plot(iterations, fun_seq)

    # ax.set_xlim(1)

    # ax.set_xticks([1, 10, 100])
    # ax.set_xticklabels([0, 10, 100])

    ## acceptance rate
    axs[1, 0].set_xlim(0, chi_history.size)
    axs[1, 0].set_ylim(0, 1)
    axs[1, 0].grid(True, which="both", axis="both")
    axs[1, 0].set_xlabel("iterations")
    axs[1, 0].set_ylabel(r"$\chi(x)$")
    line_chi, = axs[1, 0].plot([], [], lw=2, marker="o", linestyle="None", markersize=2)
    axs[1, 0].set_title("Acceptance rate performance")

    ## temperature
    axs[1, 1].set_xlim(0, temp_history.size)
    axs[1, 1].set_ylim(0, np.max(temp_history)*1.05)
    axs[1, 1].grid(True, which="both", axis="both")
    axs[1, 1].set_xlabel("iterations")
    axs[1, 1].set_ylabel(r"$T_k$")
    line_temp, = axs[1, 1].plot([], [], lw=2)
    axs[1, 1].set_title("Temperature performance")


    def update(frame):

        path = x_history[:, frame]
        f_val = f_history[frame]
        f_current_seq = f_history[:frame+1]
        f_acc_current_seq = f_acc_history[:frame+1]
        chi_current_seq = chi_history[:frame+1]
        temp_current_seq = temp_history[:frame+1]

        ## path
        x = coords[path, 0]
        y = coords[path, 1]
        line_path.set_data(x, y)
        annotation.set_text(f'f(x): {f_val:.4f}')

        ## objective function
        line_fun.set_data(range(frame+1), f_current_seq)
        line_fun_acc.set_data(range(frame+1), f_acc_current_seq)

        ## acceptance rate
        line_chi.set_data(range(frame+1), chi_current_seq)

        ## temperature
        line_temp.set_data(range(frame+1), temp_current_seq)

        return line_path, annotation, line_fun, line_fun_acc, line_chi, line_temp


    ani = animation.FuncAnimation(
        fig, update, frames=x_history.shape[1], interval=delay, repeat=False, blit=True)

    ani.save(filename, writer="ffmpeg", fps=30)
