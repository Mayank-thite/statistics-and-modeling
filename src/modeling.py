"""Modeling helpers: VIF, time-aware splits, metrics and cost-based thresholds."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation factor for each column of a numeric design matrix."""
    Xc = add_constant(X.astype(float), has_constant="add")
    rows = [
        (col, variance_inflation_factor(Xc.values, i))
        for i, col in enumerate(Xc.columns)
        if col != "const"
    ]
    return pd.DataFrame(rows, columns=["feature", "vif"]).sort_values("vif", ascending=False, ignore_index=True)


def chronological_split(df: pd.DataFrame, time_col: str, cutoff) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows strictly before ``cutoff`` go to train, the rest to test."""
    cutoff = pd.Timestamp(cutoff)
    train = df[df[time_col] < cutoff]
    test = df[df[time_col] >= cutoff]
    return train, test


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """RMSE, MAE and R squared."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_true - y_pred
    ss_res = (err**2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return {
        "rmse": float(np.sqrt((err**2).mean())),
        "mae": float(np.abs(err).mean()),
        "r2": float(1 - ss_res / ss_tot),
    }


def expected_cost(y_true, proba, threshold: float, cost_fp: float, cost_fn: float) -> float:
    """Total cost of acting on ``proba >= threshold``.

    A false positive is a call that does not convert (cost_fp each).
    A false negative is a subscriber we never called (cost_fn each).
    """
    y_true = np.asarray(y_true)
    pred = np.asarray(proba) >= threshold
    fp = np.sum(pred & (y_true == 0))
    fn = np.sum(~pred & (y_true == 1))
    return float(fp * cost_fp + fn * cost_fn)


def cost_curve(y_true, proba, cost_fp: float, cost_fn: float, thresholds=None) -> pd.DataFrame:
    """Expected cost at each threshold in a grid."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2)
    costs = [expected_cost(y_true, proba, t, cost_fp, cost_fn) for t in thresholds]
    return pd.DataFrame({"threshold": thresholds, "cost": costs})


def best_threshold(y_true, proba, cost_fp: float, cost_fn: float, thresholds=None) -> float:
    """Threshold from the grid with the lowest expected cost."""
    curve = cost_curve(y_true, proba, cost_fp, cost_fn, thresholds)
    return float(curve.loc[curve["cost"].idxmin(), "threshold"])
