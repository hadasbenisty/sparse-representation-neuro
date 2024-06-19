import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def vis_code(x, title="Visualization"):
    x = x.clone().detach().cpu().numpy()
    colors = ["b", "r"]
    plt.figure(figsize=(20, 3))
    for c in range(x.shape[0]):
        xc = x[c, :]
        nz_x = np.where(xc > 0)[0]
        plt.plot(np.zeros(xc.shape), "black")
        plt.stem(
            nz_x,
            xc[nz_x],
            label=f"x{c+1}",
            linefmt=colors[c],
            markerfmt=f"o{colors[c]}",
            basefmt="black",
        )
    plt.legend()
    plt.xlabel("Time [ms]")
    plt.title(title)
    plt.show()


def vis_code_grid(x, title="Grid Visualization"):
    x = x.clone().detach().cpu().numpy()
    N, C, Ne = x.shape
    fig, axes = plt.subplots(N, C, figsize=(C * 5, N * 3))

    for i in range(N):
        for j in range(C):
            ax = axes[i, j] if N > 1 and C > 1 else axes[i] if N > 1 else axes[j]
            xc = x[i, j, :]
            nz_x = np.where(xc > 0)[0]
            ax.plot(np.zeros(xc.shape), "black")
            ax.stem(
                nz_x,
                xc[nz_x],
                linefmt="b",
                markerfmt="ob",
                basefmt="black",
            )
            ax.set_title(f"x[{i + 1},{j + 1}]")
            ax.set_xlabel("Time [ms]")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def vis_filters(h, use_subplots=False, title="Filters"):
    h = h.clone().detach().cpu().numpy()
    N = h.shape[0]
    colors = ["b", "r", "y", "g"]

    if use_subplots:
        fig, axes = plt.subplots(N, 1, figsize=(5, 2 * N), sharex=True)
        for c in range(N):
            ax = axes[c] if N > 1 else axes
            ax.plot(h[c, 0, :], label=f"h{c + 1}", color=colors[c % len(colors)])
            ax.set_title(f"h{c + 1}")
            ax.set_xlabel("Time [ms]")
            ax.legend()
    else:
        plt.figure(figsize=(10, 5))
        for c in range(N):
            plt.plot(h[c, 0, :], label=f"h{c + 1}", color=colors[c % len(colors)])
        plt.title(title)
        plt.xlabel("Time [ms]")
        plt.legend()

    plt.tight_layout()
    plt.show()


def vis_data(y, title="Data Visualization"):
    y = y.clone().detach().cpu().numpy()
    N, M = y.shape
    fig, axes = plt.subplots(N, 1, figsize=(20, 3 * N), sharex=True)

    for i in range(N):
        ax = axes[i] if N > 1 else axes
        ax.plot(y[i, :], label=f"y[{i + 1}]", color="black")
        ax.set_title(f"y[{i + 1}]")
        ax.set_xlabel("Time [ms]")
        ax.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
