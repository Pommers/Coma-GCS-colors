import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union
from sklearn.linear_model import HuberRegressor

from src.config import (
    Columns,
    BCG,
    AnalysisConfig,
    ClusterCenter,
)

def background_correct_scalar(v_in, v_bg, n_in, n_bg, area_inner, area_annulus):
    if not np.isfinite(area_inner) or not np.isfinite(area_annulus):
        return np.nan
    if area_inner <= 0 or area_annulus <= 0:
        return np.nan
    if n_in <= 0 or n_bg < 0:
        return np.nan

    bg_scaled = v_bg * n_bg * (area_inner / area_annulus)
    return v_in - bg_scaled / max(n_in, 1)


def bootstrap_Pblue_mean(Pblue: np.ndarray, n_boot: int, seed: int = 0) -> Tuple[float, float]:
    """
    Bootstrap median and std of mean(Pblue).
    Returns (median, std) across bootstrap resamples.
    """
    rng = np.random.default_rng(seed)
    Pblue = np.asarray(Pblue, dtype=float)
    Pblue = Pblue[np.isfinite(Pblue)]

    n = Pblue.size
    if n < 3:
        return (np.nan, np.nan)

    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(Pblue, size=n, replace=True)
        means[i] = np.mean(sample)

    return float(np.median(means)), float(np.std(means, ddof=1))

# ----------------------------
# 5) Baseline vs MV and residuals (per metric)
# ----------------------------

def fit_baseline_huber(MV: np.ndarray, y: np.ndarray, eps: float = 1.35):
    """
    Robust linear baseline y(MV) using Huber regression.
    Returns (model_or_None, yhat).
    If insufficient finite data, returns (None, all-NaN yhat).
    """
    ok = np.isfinite(MV) & np.isfinite(y)
    yhat = np.full_like(y, np.nan, dtype=float)

    n_ok = int(np.sum(ok))
    if n_ok < 3:
        return None, yhat  # not enough points to fit anything meaningful

    X = MV[ok].reshape(-1, 1)
    yy = y[ok]

    model = HuberRegressor(epsilon=eps)
    model.fit(X, yy)

    yhat[ok] = model.predict(X)
    return model, yhat


def add_residuals(df: pd.DataFrame, cols: Columns, cfg: AnalysisConfig,
                  metrics: List[str]) -> pd.DataFrame:
    """
    For each metric in metrics, fit baseline vs MV and add:
      metric_exp, Delta_metric
    """
    logger.debug("Adding residuals")
    out = df.copy()
    MV = out[cols.gal_MV].to_numpy()

    for m in metrics:
        y = out[m].to_numpy()
        _, yhat = fit_baseline_huber(MV, y, eps=cfg.huber_eps)
        out[m + "_exp"] = yhat
        out["Delta_" + m] = y - yhat

    return out

