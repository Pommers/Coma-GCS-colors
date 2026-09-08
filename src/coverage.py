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

# ----------------------------
# compute per-galaxy support from footprint / point polygons
# ----------------------------

def coverage_fraction_elliptical(
    gal_ra, gal_dec,
    Re_arcsec, q_axis, pa_deg,
    rin_kRe, rout_kRe,
    polys, nsamp=5000, rng=None
):
    """
    Monte Carlo coverage fraction for an elliptical annulus aligned with the galaxy.

    rin_kRe, rout_kRe are in units of Re along the semi-major axis.
    q_axis is b/a.
    pa_deg is PA east of north, matching elliptical_radius_kRe().
    """
    if rng is None:
        rng = np.random.default_rng(0)

    vals = [gal_ra, gal_dec, Re_arcsec, q_axis, pa_deg, rin_kRe, rout_kRe]
    if not np.all(np.isfinite(vals)):
        return np.nan
    if Re_arcsec <= 0 or q_axis <= 0 or rout_kRe <= rin_kRe:
        return np.nan

    a_in = float(rin_kRe * Re_arcsec)
    a_out = float(rout_kRe * Re_arcsec)

    if not np.isfinite(a_in) or not np.isfinite(a_out):
        return np.nan
    if a_in < 0 or a_out <= 0 or a_out <= a_in:
        return np.nan
    if a_out > 1e5:
        return np.nan

    q = float(q_axis)

    u = rng.uniform(a_in**2, a_out**2, nsamp)
    r = np.sqrt(u)
    theta = rng.uniform(0, 2*np.pi, nsamp)

    x_ell = r * np.cos(theta)
    y_ell = q * r * np.sin(theta)

    pa = np.deg2rad(pa_deg)
    dra_arcsec  = x_ell * np.cos(pa) - y_ell * np.sin(pa)
    ddec_arcsec = x_ell * np.sin(pa) + y_ell * np.cos(pa)

    ra = gal_ra + dra_arcsec / 3600.0 / np.cos(np.deg2rad(gal_dec))
    dec = gal_dec + ddec_arcsec / 3600.0

    inside = 0
    for x, y in zip(ra, dec):
        p = Point(x, y)
        for poly in polys:
            if poly.contains(p):
                inside += 1
                break

    return inside / nsamp


def weighted_coverage_fraction_elliptical(
    gal_ra, gal_dec,
    Re_arcsec, q_axis, pa_deg,
    rin_kRe, rout_kRe,
    footprint_df,
    nsamp=5000, rng=None,
    weight_col=None,   # None -> geometric coverage
):
    if rng is None:
        rng = np.random.default_rng(0)

    vals = [gal_ra, gal_dec, Re_arcsec, q_axis, pa_deg, rin_kRe, rout_kRe]
    if not np.all(np.isfinite(vals)):
        return np.nan
    if Re_arcsec <= 0 or q_axis <= 0 or rout_kRe <= rin_kRe:
        return np.nan

    a_in = float(rin_kRe * Re_arcsec)
    a_out = float(rout_kRe * Re_arcsec)
    if not np.isfinite(a_in) or not np.isfinite(a_out):
        return np.nan
    if a_in < 0 or a_out <= 0 or a_out <= a_in:
        return np.nan

    u = rng.uniform(a_in**2, a_out**2, nsamp)
    r = np.sqrt(u)
    theta = rng.uniform(0, 2*np.pi, nsamp)

    x_ell = r * np.cos(theta)
    y_ell = q_axis * r * np.sin(theta)

    pa = np.deg2rad(pa_deg)
    dra_arcsec  = x_ell * np.cos(pa) - y_ell * np.sin(pa)
    ddec_arcsec = x_ell * np.sin(pa) + y_ell * np.cos(pa)

    ra = gal_ra + dra_arcsec / 3600.0 / np.cos(np.deg2rad(gal_dec))
    dec = gal_dec + ddec_arcsec / 3600.0

    weights = np.zeros(nsamp, dtype=float)

    for i, (x, y) in enumerate(zip(ra, dec)):
        p = Point(x, y)
        w = 0.0
        for _, fp in footprint_df.iterrows():
            if fp["polygon"].contains(p):
                this_w = 1.0 if weight_col is None else float(fp[weight_col])
                w = max(w, this_w)
        weights[i] = w

    return float(np.mean(weights))
    