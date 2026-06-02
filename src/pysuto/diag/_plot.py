"""
_plot.py
========
Shared plotting palette + figure-save helper used by every diagnostic
in this subpackage. Solver-specific palette + styling lives next to
:class:`SolverDiag` in :mod:`.solver`.
"""

from __future__ import annotations

import os

import numpy as np


# ---------------------------------------------------------------------------
# Shared colour palette
# ---------------------------------------------------------------------------

_C = {
    "green":  "#2ECC71",
    "red":    "#E74C3C",
    "orange": "#E67E22",
    "blue":   "#3498DB",
    "purple": "#9B59B6",
    "grey":   "#95A5A6",
    "navy":   "#2C3E50",
}


def _savefig(fig, path: str, dpi: int = 150) -> str:
    """Save figure to *path*, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def _grouped_box_grid(
    *,
    cells: dict,
    row_keys: list,
    col_keys: list,
    group_label: str,
    y_label: str,
    suptitle: str,
    highlight: tuple = (),
    cell_title=lambda r, c: f"{r} · {c}",
    cats_order=None,
):
    """Render a faceted grid of category boxplots.

    ``cells`` maps ``(row_key, col_key) -> (labels, values)`` where ``labels``
    is a 1-D array of category strings and ``values`` the matching 1-D array of
    (already finite) numbers to box. Rows/cols share a single global category
    x-order; sharey is per column.

    - ``cats_order`` None -> derived from the pooled median across all cells,
      sorted descending.
    - ``highlight`` non-empty -> those categories keep a distinct tab10 colour,
      every other category renders grey at low alpha.
    - all cells empty -> a single-axes placeholder figure carrying ``suptitle``.
    """
    import matplotlib.pyplot as plt

    # Pool everything to build the global category order.
    pool_lbl = (
        np.concatenate([cells[k][0] for k in cells])
        if any(cells[k][0].size for k in cells)
        else np.array([], dtype=str)
    )
    pool_val = (
        np.concatenate([cells[k][1] for k in cells])
        if any(cells[k][1].size for k in cells)
        else np.array([], dtype=float)
    )

    if pool_val.size == 0:
        fig = plt.figure(figsize=(8, 3))
        fig.suptitle(suptitle, fontsize=12)
        return fig

    if cats_order is not None:
        cats_sorted = np.asarray(cats_order)
    else:
        cats_pool, inv_pool = np.unique(pool_lbl, return_inverse=True)
        medians_pool = np.array([
            np.median(pool_val[inv_pool == i]) if np.any(inv_pool == i) else -np.inf
            for i in range(len(cats_pool))
        ])
        cats_sorted = cats_pool[np.argsort(-medians_pool)]

    # Colour assignment per category (highlight or grey).
    tab10 = plt.get_cmap("tab10").colors
    if highlight:
        highlight_set = set(highlight)
        highlight_color = {h: tab10[k % 10] for k, h in enumerate(highlight)}
        face_colors, edge_colors, alphas = [], [], []
        for cat in cats_sorted:
            if cat in highlight_set:
                face_colors.append(highlight_color[cat])
                edge_colors.append(highlight_color[cat])
                alphas.append(0.7)
            else:
                face_colors.append(_C["grey"])
                edge_colors.append(_C["grey"])
                alphas.append(0.25)
    else:
        face_colors = [tab10[i % 10] for i in range(len(cats_sorted))]
        edge_colors = face_colors
        alphas = [0.7] * len(cats_sorted)

    # Figure layout — sharex (categorical) across all cells; sharey per column.
    n_rows = len(row_keys)
    n_cols = len(col_keys)
    fig_w = max(10.0, 0.3 * len(cats_sorted) + 4.0)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(min(fig_w, 22), 3.2 * n_rows + 1.0),
        sharex=True, sharey="col", squeeze=False,
    )
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    for r, rk in enumerate(row_keys):
        for c, ck in enumerate(col_keys):
            ax = axes[r, c]
            lbl_cell, val_cell = cells[(rk, ck)]
            if val_cell.size == 0:
                ax.text(0.5, 0.5, "n=0", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color=_C["grey"])
            else:
                per_cat = [val_cell[lbl_cell == cat] for cat in cats_sorted]
                positions = [i for i, v in enumerate(per_cat) if v.size]
                data = [v for v in per_cat if v.size]
                bp = ax.boxplot(
                    data, positions=positions,
                    widths=0.6, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color="black", lw=1.2),
                )
                face_used = [face_colors[i] for i in positions]
                edge_used = [edge_colors[i] for i in positions]
                alpha_used = [alphas[i] for i in positions]
                for patch, face, edge, a in zip(
                    bp["boxes"], face_used, edge_used, alpha_used
                ):
                    patch.set_facecolor(face)
                    patch.set_edgecolor(edge)
                    patch.set_alpha(a)
                for whisker_pair in zip(bp["whiskers"][0::2], bp["whiskers"][1::2]):
                    for w, edge in zip(whisker_pair, edge_used):
                        w.set_color(edge)
                for cap_pair in zip(bp["caps"][0::2], bp["caps"][1::2]):
                    for cp, edge in zip(cap_pair, edge_used):
                        cp.set_color(edge)

            ax.set_xlim(-0.6, len(cats_sorted) - 0.4)
            ax.set_title(cell_title(rk, ck), loc="left", fontsize=10)
            if c == 0:
                ax.set_ylabel(y_label)
            # Modern style (no top/right spines, light grid). Inlined here so
            # _plot.py needs no back-import of solver.
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(alpha=0.25)

    # Only the bottom row gets x-tick labels.
    for c in range(n_cols):
        axes[-1, c].set_xticks(range(len(cats_sorted)))
        axes[-1, c].set_xticklabels(cats_sorted, rotation=60, ha="right", fontsize=8)
        axes[-1, c].set_xlabel(group_label)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
