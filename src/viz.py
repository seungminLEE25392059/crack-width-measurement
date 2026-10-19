"""Figure helpers shared by every script, so that all report figures look alike.

Colours come from a colour-blind-checked palette: one fixed colour per method
(A-D, for any per-method chart) and a separate set for overlay outcomes (TP,
FP, FN). Text stays in neutral ink; figures are saved at 300 dpi.

Inputs : arrays and tables from the scripts
Outputs: matplotlib figures saved as PNG
Author : Seungmin Lee
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only; no display window needed

import matplotlib.pyplot as plt  # noqa: E402  (after the backend is chosen)
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

METHOD_COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
OUTCOME_COLORS = {"TP": "#008300", "FP": "#e87ba4", "FN": "#6250d6"}
OUTCOME_LABELS = {"TP": "Correct crack (TP)", "FP": "False crack (FP)", "FN": "Missed crack (FN)"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}
BLUES = LinearSegmentedColormap.from_list(
    "report_blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def apply_style() -> None:
    """Quiet axes and grid, readable text; call once before plotting."""
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK["primary"],
        "axes.labelcolor": INK["secondary"],
        "axes.edgecolor": INK["axis"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": INK["grid"],
        "grid.linewidth": 0.6,
        "xtick.color": INK["axis"],
        "ytick.color": INK["axis"],
        "xtick.labelcolor": INK["secondary"],
        "ytick.labelcolor": INK["secondary"],
        "legend.frameon": False,
    })


def crop_around(mask: np.ndarray, size: int = 512) -> tuple[slice, slice]:
    """Rows and columns of a size x size window centred on a crack pixel.

    Full photographs are far larger than a figure panel, so thin cracks vanish
    when the whole image is shown; a crop keeps them visible.
    """
    height, width = mask.shape
    rows, cols = np.nonzero(mask)
    if rows.size:
        middle = rows.size // 2
        centre_r, centre_c = int(rows[middle]), int(cols[middle])
    else:
        centre_r, centre_c = height // 2, width // 2
    r0 = int(np.clip(centre_r - size // 2, 0, max(height - size, 0)))
    c0 = int(np.clip(centre_c - size // 2, 0, max(width - size, 0)))
    return slice(r0, r0 + size), slice(c0, c0 + size)


def overlay_outcomes(image: np.ndarray, pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Grey copy of the image with TP, FP and FN pixels coloured (RGB floats in [0, 1])."""
    grey = image.mean(axis=2) / 255.0 if image.ndim == 3 else image / 255.0
    out = np.repeat(grey[..., None], 3, axis=2)
    for name, region in (("TP", pred & ref), ("FP", pred & ~ref), ("FN", ~pred & ref)):
        out[region] = to_rgb(OUTCOME_COLORS[name])
    return out


def outcome_legend(fig) -> None:
    """TP / FP / FN legend centred below the panels."""
    handles = [Patch(color=OUTCOME_COLORS[key], label=OUTCOME_LABELS[key]) for key in OUTCOME_COLORS]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))


def plot_confusion(ax, matrix: np.ndarray, row_labels: list[str], col_labels: list[str], title: str,
                   show_row_labels: bool = True) -> None:
    """Count matrix as a sequential-blue heatmap with the count printed in every cell."""
    peak = max(int(matrix.max()), 1)
    ax.imshow(matrix, cmap=BLUES, vmin=0, vmax=peak)
    ax.grid(False)
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(row_labels)), row_labels if show_row_labels else [])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title)
    for (r, c), value in np.ndenumerate(matrix):
        ax.text(c, r, str(int(value)), ha="center", va="center", fontsize=9,
                color="white" if value > peak / 2 else INK["primary"])


def hide_axes(ax) -> None:
    """Remove ticks, grid and frame (for image panels) but keep axis labels."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_figure(fig, path: Path) -> Path:
    """Save at 300 dpi with a tight bounding box, then close the figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
