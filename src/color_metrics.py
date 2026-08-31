import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union
from sklearn.linear_model import HuberRegressor
import logging

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
                  metrics: List[str], logger=None) -> pd.DataFrame:
    """
    For each metric in metrics, fit baseline vs MV and add:
      metric_exp, Delta_metric
    """
    if logger:
        logger.debug("Adding residuals")
    out = df.copy()
    MV = out[cols.gal_MV].to_numpy()

    for m in metrics:
        y = out[m].to_numpy()
        _, yhat = fit_baseline_huber(MV, y, eps=cfg.huber_eps)
        out[m + "_exp"] = yhat
        out["Delta_" + m] = y - yhat

    return out

# -------------------------------- #

def _normal_pdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Vectorized Normal PDF."""
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * sigma)

# These were the submitted parameters for paper three, but I found an indexing error (see. Coma_Gal_Color_CMD.ipynb)
#
    # gmm_params: Tuple[Tuple[float, float, float, float, float], ...] = (
    #     (1.58, 0.30, 1.76, 0.19, 0.54),  # 23-24
    #     (1.51, 0.15, 1.68, 0.23, 0.59),  # 24-25
    #     (1.46, 0.18, 1.67, 0.25, 0.55),  # 25-26
    #     (1.40, 0.22, 1.64, 0.28, 0.57),  # 26-27
    # ),

def compute_Pblue(
    mag814: np.ndarray,
    color: np.ndarray,
    *,
    bright_mag_cut: float = 23.0,
    # mag-binned mixture model for >= bright_mag_cut
    mag_edges: Tuple[float, ...] = (23.0, 24.0, 25.0, 26.0, 27.0),
    # These are the 'recalibrated values'
    gmm_params: Tuple[Tuple[float, float, float, float, float], ...] = (
        (1.521, 0.162, 1.707, 0.229, 0.56),  # 23-24
        (1.457, 0.176, 1.668, 0.246, 0.58),  # 24-25
        (1.399, 0.224, 1.641, 0.282, 0.57),  # 25-26
        (1.217, 0.183, 1.596, 0.291, 0.45),  # 26-27
    ),
    # faint handling
    faint_mode: str = "extend",   # "extend" or "clamp"
    faint_max_edge: float = 28.0, # used only if extend
    # optional prior modifier (later): fb(R) model
    # If provided, pass an array fb of shape (N,) OR provide fb_func(r) separately.
    fb: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute P(blue | color, mag) under a mag-binned 2-Gaussian mixture.

    Returns:
      Pblue (N,) float array
      mag_bin_id (N,) int array (bright regime -> -1)
    """
    mag814 = np.asarray(mag814, dtype=float)
    color = np.asarray(color, dtype=float)
    if mag814.shape != color.shape:
        raise ValueError("mag814 and color must have the same shape.")

    N = mag814.size
    Pblue = np.full(N, np.nan, dtype=float)
    mag_bin_id = np.full(N, -1, dtype=int)

    # Bright regime: undefined for bimodality by construction
    bright = mag814 < bright_mag_cut
    mid_or_faint = ~bright

    # Prepare edges/params with faint handling
    edges = np.asarray(mag_edges, dtype=float)
    if faint_mode not in ("extend", "clamp"):
        raise ValueError('faint_mode must be either "extend" or "clamp".')
    params = np.array(gmm_params, dtype=float)
    nbins = edges.size - 1
    if params.shape[0] != nbins:
        raise ValueError("gmm_params length must match len(mag_edges)-1.")

    if faint_mode == "extend":
        if faint_max_edge <= edges[-1]:
            raise ValueError("faint_max_edge must be > mag_edges[-1] when extend.")
        edges = np.concatenate([edges, [float(faint_max_edge)]])
        params = np.vstack([params, params[-1:]])
        nbins = edges.size - 1

    if np.any(mid_or_faint):
        m = mag814[mid_or_faint]
        c = color[mid_or_faint]

        bin_id = np.digitize(m, edges) - 1
        bin_id = np.clip(bin_id, 0, nbins - 1)

        mu_b = params[bin_id, 0]
        sg_b = params[bin_id, 1]
        mu_r = params[bin_id, 2]
        sg_r = params[bin_id, 3]
        w_b  = params[bin_id, 4]

        # Likelihood terms
        pb = w_b * _normal_pdf(c, mu_b, sg_b)
        pr = (1.0 - w_b) * _normal_pdf(c, mu_r, sg_r)

        # Optional prior modifier (later): replace w_b -> fb*w_b and (1-w_b)->(1-fb)*(1-w_b)?
        # For now, if fb is provided, treat it as a multiplicative reweighting of the mixture prior:
        if fb is not None:
            fb_arr = np.asarray(fb, dtype=float)
            if fb_arr.shape != mag814.shape:
                raise ValueError("fb must have the same shape as mag814/color if provided.")
            fb_sub = fb_arr[mid_or_faint]
            pb = fb_sub * pb
            pr = (1.0 - fb_sub) * pr

        denom = pb + pr
        # Safe division
        p = np.divide(pb, denom, out=np.full_like(pb, np.nan), where=denom > 0)

        Pblue[mid_or_faint] = p
        mag_bin_id[mid_or_faint] = bin_id

    return Pblue, mag_bin_id

    