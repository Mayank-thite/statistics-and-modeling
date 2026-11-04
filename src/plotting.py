"""One consistent, quiet matplotlib style for every chart in the repo."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

IMG_DIR = Path(__file__).resolve().parents[1] / "docs" / "img"


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "axes.prop_cycle": plt.cycler(color=PALETTE),
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def save(fig, name: str) -> Path:
    """Save a figure into docs/img/ for the README."""
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    out = IMG_DIR / name
    fig.savefig(out)
    return out
