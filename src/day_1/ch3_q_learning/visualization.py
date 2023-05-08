"""Visualization.

Reference: https://github.com/sjchoi86/rl_tutorial/blob/main/notebooks/07_q_learning.ipynb
"""

import matplotlib.pyplot as plt
import numpy as np


def visualize_results(
    q_table: np.ndarray,
    title: str = "Q-Learning",
    fontsize: int = 15,
    title_fs: int = 15,
    text_fs: int = 9,
) -> None:
    """Visualize the gridworld of environment and q table."""
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

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

    # Visualize matrix
    n_row, n_col = matrix.shape[0], matrix.shape[1]

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

    # Display Q table
    _, n_action = q_table.shape
    nRow = 8

    # Triangle patches for each action
    lft_tri = np.array([[0, 0], [-0.5, -0.5], [-0.5, 0.5]])
    dw_tri = np.array([[0, 0], [-0.5, 0.5], [0.5, 0.5]])
    up_tri = np.array([[0, 0], [0.5, -0.5], [-0.5, -0.5]])
    rgh_tri = np.array([[0, 0], [0.5, 0.5], [0.5, -0.5]])

    # Color
    high_color = np.array([1.0, 0.0, 0.0, 0.8])
    low_color = np.array([1.0, 1.0, 1.0, 0.8])

    for i in range(nRow):
        for j in range(nRow):
            s = i * nRow + j
            min_q = np.min(q_table[s])
            max_q = np.max(q_table[s])
            for a in range(n_action):
                q_value = q_table[s, a]
                ratio = (q_value - min_q) / (max_q - min_q + 1e-10)
                if ratio > 1:
                    clr = high_color
                elif ratio < 0:
                    clr = low_color
                else:
                    clr = high_color * ratio + low_color * (1 - ratio)
                if a == 0:  # Left arrow
                    plt.gca().add_patch(plt.Polygon([j, i] + lft_tri, color=clr, ec="k"))
                    plt.text(
                        j - 0.25,
                        i + 0.0,
                        "%.2f" % (q_value),
                        fontsize=text_fs,
                        va="center",
                        ha="center",
                    )
                if a == 1:  # Down arrow
                    plt.gca().add_patch(plt.Polygon([j, i] + dw_tri, color=clr, ec="k"))
                    plt.text(
                        j - 0.0,
                        i + 0.25,
                        "%.2f" % (q_value),
                        fontsize=text_fs,
                        va="center",
                        ha="center",
                    )
                if a == 2:  # Right arrow
                    plt.gca().add_patch(plt.Polygon([j, i] + rgh_tri, color=clr, ec="k"))
                    plt.text(
                        j + 0.25,
                        i + 0.0,
                        "%.2f" % (q_value),
                        fontsize=text_fs,
                        va="center",
                        ha="center",
                    )
                if a == 3:  # Up arrow
                    plt.gca().add_patch(plt.Polygon([j, i] + up_tri, color=clr, ec="k"))
                    plt.text(
                        j - 0.0,
                        i - 0.25,
                        "%.2f" % (q_value),
                        fontsize=text_fs,
                        va="center",
                        ha="center",
                    )

    ax[1].set_xlim([-0.5, nRow - 0.5])
    ax[1].set_xticks(range(nRow))
    ax[1].set_ylim([-0.5, nRow - 0.5])
    ax[1].set_yticks(range(nRow))
    plt.gca().invert_yaxis()

    fig.suptitle(title, size=title_fs)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9, right=0.96)

    plt.setp(ax[0].get_xticklabels(), visible=False)
    plt.setp(ax[0].get_yticklabels(), visible=False)
    plt.setp(ax[1].get_xticklabels(), visible=False)
    plt.setp(ax[1].get_yticklabels(), visible=False)
    plt.show()
