"""One consistent, quiet matplotlib style for every chart in the repo.

Colours follow my site: warm amber for the series I want you to look at, cool blue
for the second series, greys for everything else. The site is dark, but README charts
sit on GitHub's white page, so these are darker versions of the site's amber (#FFC77A)
and blue (#9CCBFF) that still read on white.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

COOL = "#3D7CC4"
ACCENT = "#D08A1F"
# COOL first because most charts have one main series; ACCENT is for the highlight.
PALETTE = [COOL, ACCENT, "#5E9E7E", "#8C6BB8", "#B85C5C"]
INK = "#04050A"
INK_2 = "#5B6172"
GRID = "#E3E6EE"
SURFACE = "#FFFFFF"

IMG_DIR = Path(__file__).resolve().parents[1] / "docs" / "img"


def set_style() -> None:
    """Apply the shared chart style. Call once at the top of each notebook."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
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
