import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union
from sklearn.linear_model import HuberRegressor
import logging
from astropy.coordinates import SkyCoord
import astropy.units as u

from src.config import (
    Columns,
    BCG,
    AnalysisConfig,
    ClusterCenter,
    AsymmetryConfig,
)

# -----------------------
# Utilities
# -----------------------

def _select_css_within_radius(gal_row: pd.Series,
                              css_df: pd.DataFrame,
                              r_arcsec: float) -> pd.DataFrame:
    gal_ra, gal_dec = float(gal_row['ra']), float(gal_row['dec'])

    # require coordinates
    m = np.isfinite(css_df['x_wcs']) & np.isfinite(css_df['y_wcs'])
    css = css_df.loc[m].copy()

    gal_c = SkyCoord(ra=gal_ra * u.deg, dec=gal_dec * u.deg)
    css_c = SkyCoord(ra=css['x_wcs'].values * u.deg,
                     dec=css['y_wcs'].values * u.deg)

    sep = gal_c.separation(css_c).arcsec
    sel = sep <= float(r_arcsec)
    if not np.any(sel):
        # include Pblue column if it exists to keep downstream code simple
        cols = {'sep_arcsec': [], 'pa_deg': []}
        if 'Pblue' in css_df.columns:
            cols['Pblue'] = []
        return css.iloc[0:0].assign(**cols)

    pa_deg = gal_c.position_angle(css_c[sel]).to(u.deg).value  # 0=N, 90=E
    out = (css.loc[sel]
            .assign(sep_arcsec=sep[sel],
                    pa_deg=pa_deg))

    # if Pblue exists, keep it (and ensure float)
    if 'Pblue' in out.columns:
        out['Pblue'] = pd.to_numeric(out['Pblue'], errors='coerce')

    return out


def _mean_direction_and_Rbar(pa_deg: np.ndarray) -> Tuple[float, float]:
    """
    Circular mean (deg) and mean resultant length Rbar in [0,1].
    """
    theta = np.deg2rad(pa_deg)
    C = np.mean(np.cos(theta))
    S = np.mean(np.sin(theta))
    mu = (np.rad2deg(np.arctan2(S, C)) + 360.0) % 360.0
    Rbar = np.hypot(C, S)
    return mu, Rbar


def _mean_direction_and_Rbar_weighted(pa_deg: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    """
    For non-integer (weighted) counts.
    Returns the Pblue dipole direction and dipole strength as output.
    """
    pa_deg = np.asarray(pa_deg, dtype=float)
    w = np.asarray(w, dtype=float)

    ok = np.isfinite(pa_deg) & np.isfinite(w) & (w > 0)
    if ok.sum() < 3:
        return np.nan, np.nan

    theta = np.deg2rad(pa_deg[ok])
    ww = w[ok]

    C = np.sum(ww * np.cos(theta)) / np.sum(ww)
    S = np.sum(ww * np.sin(theta)) / np.sum(ww)

    mu = (np.rad2deg(np.arctan2(S, C)) + 360.0) % 360.0
    Rbar = np.hypot(C, S)
    return mu, Rbar

def _perm_pvalue_from_shuffle(weights, pa_deg, stat_fn, n_perm=999, rng=None):
    """
    Permute weights among angles to build a null distribution for a statistic.
    Two-sided in the sense of comparing absolute values if stat is signed.
    """
    if rng is None:
        rng = np.random.default_rng()

    w = np.asarray(weights, dtype=float)
    pa = np.asarray(pa_deg, dtype=float)
    m = np.isfinite(w) & np.isfinite(pa)
    w, pa = w[m], pa[m]
    if w.size < 3:
        return np.nan

    obs = stat_fn(pa, w)
    if not np.isfinite(obs):
        return np.nan

    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        wp = rng.permutation(w)
        null[i] = stat_fn(pa, wp)

    null = null[np.isfinite(null)]
    if null.size == 0:
        return np.nan

    # If the statistic is signed, compare abs; if nonnegative, abs is harmless.
    p = (np.sum(np.abs(null) >= np.abs(obs)) + 1.0) / (null.size + 1.0)
    return float(p)

def _hemi_masks(pa_deg: np.ndarray, center_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hemispheres centered on 'center_deg' (±90°) and its opposite.
    Returns boolean masks A, B of equal angular widths.
    """
    # bring angles to [-180, +180] around center_deg
    delta = ((pa_deg - center_deg + 180.0) % 360.0) - 180.0
    A = (delta >= -90.0) & (delta < 90.0)
    B = ~A
    return A, B

# following added for color asymmetry analysis
def permute_weighted_hemi_p_refit_axis(pa_deg, w, axis_deg=None, nperm=2000, seed=0):
    """
    binomial/chi² assume integer counts, so for color weights (Pblue), need to use this
    """    
    rng = np.random.default_rng(seed)
    pa_deg = np.asarray(pa_deg, float)
    w = np.asarray(w, float)

    ok = np.isfinite(pa_deg) & np.isfinite(w) & (w > 0)
    pa_deg, w = pa_deg[ok], w[ok]
    if pa_deg.size < 8:
        return np.nan

    # observed axis
    if axis_deg is None or not np.isfinite(axis_deg):
        axis_obs, _ = _mean_direction_and_Rbar_weighted(pa_deg, w)
    else:
        axis_obs = float(axis_deg)

    A, B = _hemi_masks(pa_deg, axis_obs)
    SA, SB = np.sum(w[A]), np.sum(w[B])
    obs = np.abs((SA - SB) / (SA + SB)) if (SA + SB) > 0 else np.nan
    if not np.isfinite(obs):
        return np.nan

    more = 0
    for _ in range(nperm):
        wp = rng.permutation(w)

        axis_p, _ = _mean_direction_and_Rbar_weighted(pa_deg, wp)
        if not np.isfinite(axis_p):
            continue

        A_p, B_p = _hemi_masks(pa_deg, axis_p)
        SA_p, SB_p = np.sum(wp[A_p]), np.sum(wp[B_p])
        stat = np.abs((SA_p - SB_p) / (SA_p + SB_p)) if (SA_p + SB_p) > 0 else np.nan
        if not np.isfinite(stat):
            continue

        more += (stat >= obs)

    return (more + 1) / (nperm + 1)

def _quadrant_counts(pa_deg: np.ndarray, nbins: int = 4) -> np.ndarray:
    """
    Simple equal-angle binning (e.g., 4 quadrants). Returns counts per bin.
    """
    bins = np.linspace(0, 360, nbins + 1)
    counts, _ = np.histogram(pa_deg % 360.0, bins=bins)
    return counts

# following added for color asymmetry analysis
def permute_weighted_quad_p_gtest(pa_deg, w, nbins=4, nperm=2000, seed=0):
    """
    Quadrants assume integer counts, so for color weights (Pblue), need to use this. 
    Also, permutation p-value for weighted quadrant non-uniformity using a G-test style statistic.
    """
    rng = np.random.default_rng(seed)
    pa_deg = np.asarray(pa_deg, float)
    w = np.asarray(w, float)

    ok = np.isfinite(pa_deg) & np.isfinite(w) & (w > 0)
    pa_deg, w = pa_deg[ok], w[ok]
    if pa_deg.size < 8:
        return np.nan

    bins = np.linspace(0, 360, nbins + 1)
    eps = 1e-12

    def g_like(angles_deg, ww):
        sums, _ = np.histogram(angles_deg % 360.0, bins=bins, weights=ww)
        tot = np.sum(sums)
        if not np.isfinite(tot) or tot <= 0:
            return np.nan
        exp = tot / nbins
        # G = 2 * sum O * ln(O/E); define 0*ln(0)=0
        O = np.maximum(sums, 0.0)
        term = np.where(O > 0, O * np.log((O + eps) / (exp + eps)), 0.0)
        return float(2.0 * np.sum(term))

    obs = g_like(pa_deg, w)
    if not np.isfinite(obs):
        return np.nan

    more = 0
    for _ in range(nperm):
        angp = rng.permutation(pa_deg)
        stat = g_like(angp, w)
        more += (stat >= obs)

    return (more + 1) / (nperm + 1)


# ----------------------------
# helpers for circular handling
# ----------------------------
    
def _wrap_deg(x):
    x = np.asarray(x)
    return (x % 360.0 + 360.0) % 360.0

def _circ_delta_to(mu_deg, angles_deg):
    """
    Minimal signed angular diff in [-180, +180] from mu to each angle.
    """
    return ((np.asarray(angles_deg) - mu_deg + 180.0) % 360.0) - 180.0

def _ci_from_samples(arr, lo=16, hi=84):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    return (np.nanpercentile(arr, lo), np.nanmedian(arr), np.nanpercentile(arr, hi))

def _ci_for_circular_angle(mu_deg, bootstrap_means_deg, lo=16, hi=84):
    """
    Give CI around a reference mean angle mu_deg by taking percentiles of
    minimal signed deltas, then re-wrapping back around mu_deg.
    """
    if not np.isfinite(mu_deg):
        return (np.nan, np.nan, np.nan)
    bootstrap_means_deg = np.asarray(bootstrap_means_deg, dtype=float)
    bootstrap_means_deg = bootstrap_means_deg[np.isfinite(bootstrap_means_deg)]
    if bootstrap_means_deg.size == 0:
        return (np.nan, np.nan, np.nan)

    deltas = _circ_delta_to(mu_deg, bootstrap_means_deg)  # in [-180, 180]
    lo_d, med_d, hi_d = _ci_from_samples(deltas, lo, hi)
    return (_wrap_deg(mu_deg + lo_d), _wrap_deg(mu_deg + med_d), _wrap_deg(mu_deg + hi_d))

# -------- CI attachment helper ---------
def _attach_cis(out: dict, draws: dict, ci=(16, 84)):
    lo, hi = ci
    for k, vals in draws.items():
        arr = np.asarray(vals, dtype=float)

        if k in ANGLE_KEYS:
            mu0 = out.get(k, np.nan)
            loA, medA, hiA = _ci_for_circular_angle(mu0, arr, lo=lo, hi=hi)
            out[f"{k}_lo"]  = loA
            out[f"{k}_med"] = medA
            out[f"{k}_hi"]  = hiA
        else:
            lo_v, med_v, hi_v = _ci_from_samples(arr, lo=lo, hi=hi)
            out[f"{k}_lo"]  = lo_v
            out[f"{k}_med"] = med_v
            out[f"{k}_hi"]  = hi_v
