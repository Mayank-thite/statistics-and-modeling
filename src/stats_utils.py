"""Small statistics helpers: effect sizes, bootstrap, outliers, sample size."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d with a pooled standard deviation (a minus b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled)


def rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation from a Mann-Whitney U (for the first sample).

    Ranges from -1 to 1. It equals 2 * P(X > Y) - 1, ignoring ties.
    """
    return float(2 * u_stat / (n1 * n2) - 1)


def cramers_v(table: pd.DataFrame | np.ndarray) -> float:
    """Cramer's V for a contingency table (no bias correction)."""
    table = np.asarray(table, float)
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k)))


def eta_squared(groups: Sequence[Sequence[float]]) -> float:
    """Eta squared for a one-way layout: between-group SS over total SS."""
    arrays = [np.asarray(g, float) for g in groups]
    allv = np.concatenate(arrays)
    grand = allv.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in arrays)
    ss_total = ((allv - grand) ** 2).sum()
    return float(ss_between / ss_total)


def bootstrap_ci(
    data: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 5000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """Percentile bootstrap confidence interval.

    Returns (low, high, bootstrap_distribution).
    """
    x = np.asarray(data, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.apply_along_axis(stat, 1, x[idx])
    alpha = (1 - ci) / 2
    low, high = np.quantile(boots, [alpha, 1 - alpha])
    return float(low), float(high), boots


def iqr_outliers(x: Sequence[float], k: float = 1.5) -> np.ndarray:
    """Boolean mask of points outside [Q1 - k*IQR, Q3 + k*IQR]."""
    x = np.asarray(x, float)
    q1, q3 = np.quantile(x, [0.25, 0.75])
    iqr = q3 - q1
    return (x < q1 - k * iqr) | (x > q3 + k * iqr)


def sample_size_two_proportions(
    p1: float, p2: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Per-group n for a two-sided z-test of two proportions (normal approximation).

    Uses the classic formula with the pooled variance under H0 and the
    unpooled variance under H1, rounded up.
    """
    if not (0 < p1 < 1 and 0 < p2 < 1) or p1 == p2:
        raise ValueError("need 0 < p1, p2 < 1 and p1 != p2")
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2
    num = z_a * np.sqrt(2 * p_bar * (1 - p_bar)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return int(np.ceil(num**2 / (p1 - p2) ** 2))
