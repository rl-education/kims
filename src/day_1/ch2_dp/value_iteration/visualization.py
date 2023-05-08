"""Visualization.

Reference: https://github.com/sjchoi86/rl_tutorial/blob/main/notebooks/04_policy_iteration.ipynb
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def visualize_results(
    policy: np.ndarray,
    value: np.ndarray,
    title: str = "Value Iteration",
    fontsize: int = 15,
    title_fs: int = 15,
) -> None:
    """Visualize the gridworld of environment, policy, and value."""
    matrix = np.zeros(shape=(8, 8))
    matrix[0, 0] = 1  # Start
    matrix[2, 3] = matrix[3, 5] = matrix[4, 3] = matrix[5, 1] = matrix[5, 2] = matrix[5, 6] = matrix[
        6,
        1,
    ] = matrix[6, 4] = matrix[6, 6] = matrix[
        7,
        3,
    ] = 2  # Hole
    matrix[7, 7] = 3  # Goal

    strs = [
        "S",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "H",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "H",
        "F",
        "F",
        "F",
        "F",
        "F",
        "H",
        "F",
        "F",
        "F",
        "F",
        "F",
        "H",
        "H",
        "F",
        "F",
        "F",
        "H",
        "F",
        "F",
        "H",
        "F",
        "F",
        "H",
        "F",
        "H",
        "F",
        "F",
        "F",
        "F",
        "H",
        "F",
        "F",
        "F",
        "G",
    ]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    # Visualize matrix
    n_row, n_col = matrix.shape[0], matrix.shape[1]
    divider = make_axes_locatable(ax[0])

    im = ax[0].imshow(
        matrix,
        cmap=plt.get_cmap("Set3"),
        extent=(0, n_col, n_row, 0),
        interpolation="nearest",
        aspect="equal",
    )
    ax[0].set_xticks(np.arange(0, n_col, 1))
    ax[0].set_yticks(np.arange(0, n_row, 1))
    ax[0].grid(color="w", linewidth=2)
    ax[0].set_frame_on(False)

    x, y = np.meshgrid(np.arange(0, n_col, 1.0), np.arange(0, n_row, 1.0))
    if len(strs) == n_row * n_col:
        idx = 0
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = strs[idx]
            idx = idx + 1
            ax[0].text(x_val + 0.5, y_val + 0.5, c, va="center", ha="center", size=fontsize)

    # Visualize policy and value
    n_row, n_col = value.shape[0], value.shape[1]
    divider = make_axes_locatable(ax[1])

    im = ax[1].imshow(value, cmap=plt.get_cmap("binary"), extent=(0, n_col, n_row, 0))
    ax[1].set_xticks(np.arange(0, n_col, 1))
    ax[1].set_yticks(np.arange(0, n_row, 1))
    ax[1].grid(color="w", linewidth=2)

    arr_len = 0.2
    for i in range(8):
        for j in range(8):
            s = i * 8 + j
            if policy[s][0] > 0:
                plt.arrow(
                    j + 0.5,
                    i + 0.5,
                    -arr_len,
                    0,
                    color="r",
                    alpha=policy[s][0],
                    width=0.01,
                    head_width=0.5,
                    head_length=0.2,
                    overhang=1,
                )
            if policy[s][1] > 0:
                plt.arrow(
                    j + 0.5,
                    i + 0.5,
                    0,
                    arr_len,
                    color="r",
                    alpha=policy[s][1],
                    width=0.01,
                    head_width=0.5,
                    head_length=0.2,
                    overhang=1,
                )
            if policy[s][2] > 0:
                plt.arrow(
                    j + 0.5,
                    i + 0.5,
                    arr_len,
                    0,
                    color="r",
                    alpha=policy[s][2],
                    width=0.01,
                    head_width=0.5,
                    head_length=0.2,
                    overhang=1,
                )
            if policy[s][3] > 0:
                plt.arrow(
                    j + 0.5,
                    i + 0.5,
                    0,
                    -arr_len,
                    color="r",
                    alpha=policy[s][3],
                    width=0.01,
                    head_width=0.5,
                    head_length=0.2,
                    overhang=1,
                )

    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")
    fig.suptitle(title, size=title_fs)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    plt.setp(ax[0].get_xticklabels(), visible=False)
    plt.setp(ax[0].get_yticklabels(), visible=False)
    plt.setp(ax[1].get_yticklabels(), visible=False)
    plt.setp(ax[1].get_yticklabels(), visible=False)
    plt.show()
