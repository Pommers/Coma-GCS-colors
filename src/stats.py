import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.linear_model import HuberRegressor

# ---------------------------------------------
# --- Quantify trends and add uncertainties ---
# ---------------------------------------------

def bootstrap_slope(x, y, nboot=2000, seed=0):
    rng = np.random.default_rng(seed)
    slopes = []

    x = np.asarray(x)
    y = np.asarray(y)

    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    n = len(x)
    if n < 10:
        return np.nan, (np.nan, np.nan)

    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        xs, ys = x[idx], y[idx]

        try:
            model = HuberRegressor(epsilon=1.35).fit(xs.reshape(-1,1), ys)
            slopes.append(model.coef_[0])
        except ValueError:
            continue   # skip failed fits

    slopes = np.array(slopes)

    if len(slopes) < 0.5 * nboot:
        # too many failures -> unreliable
        return np.nan, (np.nan, np.nan)

    return np.median(slopes), np.percentile(slopes, [16, 84])

def quantify_trend(df, xcol, ycol, mask, rmax=None, split_value=None, test_name="", sample_def="", label="", logger=None):

    m = mask
    bcg_names = ["NGC 4874", "NGC 4889"]
    
    # Exclude the BCGs only for the tidal-proxy panel
    if xcol == "log_tidal_proxy":
        m = mask & ~df["name"].isin(bcg_names)
        
    d = df.loc[m, [xcol, ycol]].dropna()
    
    if rmax is not None:
        d = d[d[xcol] <= rmax]

    x = d[xcol].values
    y = d[ycol].values

    logger.info(f"\n--- {label} ---")
    logger.info(f"N = {len(x)}")

    if len(x) < 10:
        logger.warning("Too few points.")
        return

    rho, p = spearmanr(x, y)
    slope, (lo, hi) = bootstrap_slope(x, y)

    logger.info(f"Spearman ρ = {rho:.3f}, p = {p:.3e}")
    logger.info(f"Slope = {slope:.3e} [{lo:.3e}, {hi:.3e}]")

    # Inner vs outer comparison
    if split_value is None:
        split = np.median(x)
        split_label = "median x"
    else:
        split = split_value
        split_label = f"{split_value:.3f}"

    # Inner vs outer comparison
    medx = split
    yin = y[x <= medx]
    yout = y[x > medx]
    delta_inner_outer = np.median(yin) - np.median(yout)
    logger.info(f"Δ(inner−outer) = {delta_inner_outer:.3f}")

    if sample_def=="":
        logger.warning("Warning! - Sample definition not set.")
    
    return {
        "test_name": test_name,
        "sample_definition": sample_def,
        "N": len(x),
        "rho": rho,
        "p": p,
        "slope": slope,
        "slope_lo": lo,
        "slope_hi": hi,
        "delta_inner_outer": delta_inner_outer,
    }

# ---------------------------------------------
# ---------------------------------------------
# ---------------------------------------------

    