import numpy as np
import pandas as pd
import pytest

from src.modeling import (
    best_threshold,
    chronological_split,
    cost_curve,
    expected_cost,
    regression_metrics,
    vif_table,
)


def test_vif_independent_vs_collinear():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    X = pd.DataFrame({"a": a, "b": rng.normal(size=500), "c": a + rng.normal(scale=0.05, size=500)})
    v = vif_table(X).set_index("feature")["vif"]
    assert v["b"] < 1.1
    assert v["a"] > 100 and v["c"] > 100


def test_chronological_split_has_no_overlap():
    df = pd.DataFrame({"t": pd.date_range("2020-01-01", periods=10, freq="D"), "y": range(10)})
    train, test = chronological_split(df, "t", "2020-01-08")
    assert len(train) == 7 and len(test) == 3
    assert train["t"].max() < test["t"].min()


def test_regression_metrics_known_values():
    m = regression_metrics([1, 2, 3], [1, 2, 3])
    assert m == {"rmse": 0.0, "mae": 0.0, "r2": 1.0}
    m = regression_metrics([0, 0, 4], [1, 1, 1])
    assert m["mae"] == pytest.approx(5 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(11 / 3))


def test_expected_cost_by_hand():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.3])
    # threshold 0.5: predicted positive = idx 0, 2 -> 1 FP (idx 2), 1 FN (idx 1)
    assert expected_cost(y, p, 0.5, cost_fp=1, cost_fn=10) == 11
    # threshold 0.15: positives idx 0,1,2,4 -> 2 FP, 0 FN
    assert expected_cost(y, p, 0.15, cost_fp=1, cost_fn=10) == 2


def test_best_threshold_picks_cheapest():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.3])
    curve = cost_curve(y, p, 1, 10)
    t = best_threshold(y, p, 1, 10)
    assert curve["cost"].min() == expected_cost(y, p, t, 1, 10)
    assert t <= 0.2
