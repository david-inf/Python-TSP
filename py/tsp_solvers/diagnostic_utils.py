# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def plot_fun(fun_seq, scalex="log", ax=None):
    # fun_seq: array_like

    if ax is None:
        fig, ax = plt.subplots()

    iterations = np.arange(1, fun_seq.size + 1)
    ax.plot(iterations, fun_seq)

    ax.set_xscale(scalex)

    ax.set_xlim(1)

    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels([0, 10, 100])

    ax.grid(True, which="both", axis="both")

    ax.set_xlabel("iterations")
    ax.set_ylabel(r"$f(x)$")


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
    ax.scatter(x_seq, y_seq, marker="o", color=colors)

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

    # plot graph
    plot_points(coords, opt_res.x, axs[0])
    # plot function performance
    plot_fun(opt_res.fun_seq, ax=axs[1])
            
