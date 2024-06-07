# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def plot_path(coords, seq):

    # coordinates according to given sequence
    x_seq = coords[seq, 0]
    y_seq = coords[seq, 1]

    # sequence colors indicating starting and end points
    colors = ["red"] + ["blue"] * (len(seq) - 1)

    fig, ax = plt.subplots()

    ax.plot(x_seq, y_seq)
    ax.scatter(x_seq, y_seq, marker="o", color=colors)

    for i in range(coords.shape[0]):
        ax.text(coords[i,0], coords[i,1], f"{i}")

    ax.grid(True, which="both", axis="both")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_aspect('equal', adjustable='box')


def plot_fun(ax, fun_seq):

    ax.plot(np.arange(1, fun_seq.size+1), fun_seq)

    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels([0, 10, 100])
    ax.grid(True, which="both", axis="both")
    ax.set_xlabel("iterations")
    ax.set_ylabel(r"$f$")


def plot_points(coords, seq, ax=None):

    # coordinates according to given sequence
    x_seq = coords[seq, 0]
    y_seq = coords[seq, 1]

    # sequence colors indicating starting and end points
    colors = ["blue"] * (len(seq) - 1) + ["red"]

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


def diagnostic(coords, seq, fun_seq):

    fig, axs = plt.subplots(1, 2, layout="constrained")

    # plot graph
    plot_points(coords, seq, axs[0])
    # plot function performance
    plot_fun(axs[1], fun_seq)
            
