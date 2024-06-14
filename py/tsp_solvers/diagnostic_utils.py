# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


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
    plt.title(opt_res.solver)

    ## plot graph
    plot_points(coords, opt_res.x, axs[0])

    ## plot function performance
    fun_seq = opt_res.fun_seq[:opt_res.nit+1]
    plot_fun(fun_seq, ax=axs[1])


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
    plot_chi(res.chi_seq[:res.nit], axs[1, 0])

    ## plot temperature against iterations
    plot_fun(res.temp_seq[:res.nit+1], "linear", ylab=r"$T_k$", ax=axs[1, 1])

    return None
