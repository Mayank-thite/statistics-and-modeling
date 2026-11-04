"""Modeling helpers: VIF, time-aware splits, metrics, cost-based thresholds."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
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


def time_ordered_split(
    X: pd.DataFrame, y: pd.Series, train_frac: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split rows that are already in time order: the first ``train_frac`` train, the rest test."""
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1")
    cut = int(len(X) * train_frac)
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


def within_group_auc(y_true, proba, groups) -> float:
    """ROC-AUC computed inside each group, averaged with weights equal to group size.

    Groups where only one class appears are skipped, since AUC is undefined there.
    This removes any signal that only comes from groups having different base rates.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba), "g": np.asarray(groups)})
    aucs, weights = [], []
    for _, part in df.groupby("g"):
        if part["y"].nunique() < 2:
            continue
        aucs.append(roc_auc_score(part["y"], part["p"]))
        weights.append(len(part))
    if not aucs:
        raise ValueError("no group has both classes")
    return float(np.average(aucs, weights=weights))


def cost_sensitivity(
    y_val, p_val, y_test, p_test, cost_pairs: list[tuple[float, float]]
) -> pd.DataFrame:
    """For each (cost_fp, cost_fn) pair, pick the threshold on validation data and score it on test.

    Also reports what calling everyone and the default 0.5 cut-off would cost on test,
    so it is clear whether the model saves anything under that cost assumption.
    """
    rows = []
    for cost_fp, cost_fn in cost_pairs:
        t = best_threshold(y_val, p_val, cost_fp, cost_fn)
        rows.append({
            "cost_fp": cost_fp,
            "cost_fn": cost_fn,
            "threshold": t,
            "calls": int(np.sum(np.asarray(p_test) >= t)),
            "cost_model": expected_cost(y_test, p_test, t, cost_fp, cost_fn),
            "cost_default_0.5": expected_cost(y_test, p_test, 0.5, cost_fp, cost_fn),
            "cost_call_everyone": expected_cost(y_test, p_test, 0.0, cost_fp, cost_fn),
        })
    out = pd.DataFrame(rows)
    out["saving_vs_call_everyone"] = 1 - out["cost_model"] / out["cost_call_everyone"]
    return out
