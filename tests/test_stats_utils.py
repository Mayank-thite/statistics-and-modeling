import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.stats_utils import (
    bootstrap_ci,
    cohens_d,
    cramers_v,
    eta_squared,
    iqr_outliers,
    rank_biserial,
    sample_size_two_proportions,
)


def test_cohens_d_known_value():
    a = [2, 4, 6]  # mean 4, var 4
    b = [1, 3, 5]  # mean 3, var 4
    assert cohens_d(a, b) == pytest.approx(0.5)
    assert cohens_d(b, a) == pytest.approx(-0.5)


def test_rank_biserial_bounds():
    a, b = [10, 11, 12], [1, 2, 3]
    u = stats.mannwhitneyu(a, b).statistic
    assert rank_biserial(u, 3, 3) == pytest.approx(1.0)
    u = stats.mannwhitneyu(b, a).statistic
    assert rank_biserial(u, 3, 3) == pytest.approx(-1.0)


def test_cramers_v_extremes():
    assert cramers_v(np.array([[50, 0], [0, 50]])) == pytest.approx(1.0)
    assert cramers_v(np.array([[25, 25], [25, 25]])) == pytest.approx(0.0)


def test_eta_squared_matches_anova_ss():
    rng = np.random.default_rng(0)
    groups = [rng.normal(m, 1, 30) for m in (0, 0.5, 1)]
    eta = eta_squared(groups)
    f, _ = stats.f_oneway(*groups)
    k, n = 3, 90
    # eta^2 = F*(k-1) / (F*(k-1) + (n-k))
    assert eta == pytest.approx(f * (k - 1) / (f * (k - 1) + (n - k)))


def test_bootstrap_ci_is_reproducible_and_brackets_mean():
    x = np.random.default_rng(1).exponential(2, 400)
    lo1, hi1, _ = bootstrap_ci(x, n_boot=2000, seed=7)
    lo2, hi2, _ = bootstrap_ci(x, n_boot=2000, seed=7)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 < x.mean() < hi1


def test_iqr_outliers_flags_extreme_point():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
    mask = iqr_outliers(x)
    assert mask.tolist() == [False] * 9 + [True]


def test_sample_size_matches_textbook_and_statsmodels():
    # Fleiss (no continuity correction): 10% vs 15% at alpha 0.05, power 0.8 -> 686 per group
    assert sample_size_two_proportions(0.10, 0.15) == 686
    with pytest.raises(ValueError):
        sample_size_two_proportions(0.1, 0.1)
