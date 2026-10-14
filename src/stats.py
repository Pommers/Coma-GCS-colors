import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from scipy.stats import rankdata, pearsonr, t as student_t

import statsmodels.api as sm

from sklearn.linear_model import HuberRegressor
from src.color_metrics import fit_baseline_huber

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

    logger.info(f"--- {label} ---")
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
    
    logger.info(f"\n")

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

def spearman_matrix(df, cols):
    rho = pd.DataFrame(
        np.nan,
        index=cols,
        columns=cols,
        dtype=float,
    )

    pval = rho.copy()
    nmat = rho.copy()

    for x in cols:
        for y in cols:

            valid = (np.isfinite(df[x]) & np.isfinite(df[y]))

            n = valid.sum()

            if n >= 3:
                res = spearmanr(df.loc[valid, x], df.loc[valid, y],)

                rho.loc[x, y] = res.statistic
                pval.loc[x, y] = res.pvalue
                nmat.loc[x, y] = n

    return rho, pval, nmat


# --- partial Spearman correlation function --- #

def partial_spearman(
    df,
    x,
    y,
    controls,
):
    """
    Partial Spearman correlation between x and y,
    controlling for one or more variables.

    All variables are rank transformed first.
    """

    cols = [x, y] + list(controls)

    d = df[cols].dropna().copy()

    n = len(d)
    k = len(controls)

    if n <= k + 2:
        return {
            "N": n,
            "rho_partial": np.nan,
            "p": np.nan,
        }

    # Rank transform
    ranks = pd.DataFrame({col: rankdata(d[col]) for col in cols}, index=d.index,)

    # Controls
    X = sm.add_constant(ranks[list(controls)], has_constant="add",)

    # Residualize x and y against controls
    resid_x = sm.OLS(ranks[x], X, ).fit().resid
    resid_y = sm.OLS( ranks[y], X, ).fit().resid

    # Pearson correlation of rank residuals
    r = pearsonr(resid_x, resid_y, ).statistic

    # Approximate significance
    dof = n - k - 2

    tval = (r * np.sqrt(dof / (1.0 - r**2)))

    p = 2.0 * student_t.sf(np.abs(tval), df=dof,)

    return {
        "N": n,
        "rho_partial": r,
        "p": p,
    }

# ------------------------------------------------------------------------- #

def bootstrap_mean_draws(
    values,
    n_boot,
    rng,
    batch_size=250,
):
    """
    Bootstrap draws of the sample mean.

    Resamples the supplied 1-D values with replacement while
    keeping the original sample size fixed.
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n = len(values)

    if n == 0:
        return np.full(n_boot, np.nan)

    draws = np.full(n_boot, np.nan)

    for start in range(0, n_boot, batch_size):

        stop = min(start + batch_size, n_boot)
        m = stop - start

        idx = rng.integers(
            0,
            n,
            size=(m, n),
        )

        draws[start:stop] = values[idx].mean(axis=1)

    return draws


def bootstrap_huber_residuals(
    MV,
    boot_metric,
    eps=1.35,
):
    """
    Refit the Huber luminosity baseline for every bootstrap realization.

    Parameters
    ----------
    MV : array, shape (N_gal,)
        Host absolute magnitudes.

    boot_metric : array, shape (N_boot, N_gal)
        Bootstrap realizations of one galaxy-level metric.

    eps : float
        Huber epsilon, matching the nominal analysis.

    Returns
    -------
    boot_delta : array, shape (N_boot, N_gal)
        Residuals from the realization-specific Huber baseline.
    """
    MV = np.asarray(MV, dtype=float)
    boot_metric = np.asarray(boot_metric, dtype=float)

    boot_delta = np.full_like(
        boot_metric,
        np.nan,
        dtype=float,
    )

    for b in range(boot_metric.shape[0]):

        y = boot_metric[b]

        _, yhat = fit_baseline_huber(
            MV,
            y,
            eps=eps,
        )

        boot_delta[b] = y - yhat

    return boot_delta

# ------------------------------------------------------------------ #

def bootstrap_median_ci(vals, n_boot=5000, ci=(16, 84), random_state=None):
    """
    Bootstrap uncertainty on the median.

    Returns:
        median, lo, hi, err_lo, err_hi
    """
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    rng = np.random.default_rng(random_state)

    boot_meds = np.empty(n_boot)
    n = len(vals)

    for i in range(n_boot):
        sample = rng.choice(vals, size=n, replace=True)
        boot_meds[i] = np.median(sample)

    med = np.median(vals)
    lo, hi = np.percentile(boot_meds, ci)

    return med, lo, hi, med - lo, hi - med

# ------------------------------------------------------------------ #

def bootstrap_median_contrast_ci(
    inner_colors,
    annulus_colors,
    n_boot=5000,
    random_state=42,
):
    """
    Bootstrap inner median, annular median, and inner-annulus
    median-color contrast simultaneously.

    Returns a dictionary containing median, 16th/84th percentiles,
    and asymmetric errors for all three diagnostics.
    """

    inner_colors = np.asarray(inner_colors)
    annulus_colors = np.asarray(annulus_colors)

    inner_colors = inner_colors[np.isfinite(inner_colors)]
    annulus_colors = annulus_colors[np.isfinite(annulus_colors)]

    if len(inner_colors) == 0 or len(annulus_colors) == 0:
        return None

    # Observed statistics
    obs_inner = np.median(inner_colors)
    obs_annulus = np.median(annulus_colors)
    obs_contrast = obs_inner - obs_annulus

    rng = np.random.default_rng(random_state)

    inner_boot = np.empty(n_boot)
    ann_boot = np.empty(n_boot)
    contrast_boot = np.empty(n_boot)

    for i in range(n_boot):

        inner_sample = rng.choice(
            inner_colors,
            size=len(inner_colors),
            replace=True,
        )

        ann_sample = rng.choice(
            annulus_colors,
            size=len(annulus_colors),
            replace=True,
        )

        inner_boot[i] = np.median(inner_sample)
        ann_boot[i] = np.median(ann_sample)
        contrast_boot[i] = inner_boot[i] - ann_boot[i]

    def summarize(boot, observed):
        lo, hi = np.percentile(boot, [16, 84])

        return {
            "median": observed,
            "lo": lo,
            "hi": hi,
            "err_lo": observed - lo,
            "err_hi": hi - observed,
            "draws": boot,
        }

    return {
        "inner": summarize(inner_boot, obs_inner),
        "annulus": summarize(ann_boot, obs_annulus),
        "contrast": summarize(contrast_boot, obs_contrast),
    }

# ------------------------------------------------------------------ #

def bootstrap_spearman(
    df,
    x_col,
    y_col,
    bootstrap_draws,
    name_col="name",
    n_iter=10_000,
    seed=42,
):
    """
    Propagate per-galaxy bootstrap uncertainty in y into a Spearman
    correlation with x.

    Parameters
    ----------
    df : pandas.DataFrame
        Galaxy-level science sample.

    x_col : str
        Environmental predictor column.

    y_col : str
        Observed response column.

    bootstrap_draws : dict
        Mapping:
            galaxy_name -> array-like bootstrap draws for y

    name_col : str
        Galaxy identifier used to access bootstrap_draws.

    n_iter : int
        Number of Monte Carlo realizations.

    seed : int
        Random seed.

    Returns
    -------
    dict
        Observed rho/p plus bootstrap distribution summary.
    """

    rng = np.random.default_rng(seed)

    cols = [name_col, x_col, y_col]
    d = df[cols].dropna().copy()

    # Require bootstrap draws for every retained galaxy
    has_boot = d[name_col].map(
        lambda name: (
            name in bootstrap_draws
            and len(bootstrap_draws[name]) > 0
        )
    )

    d = d.loc[has_boot].copy()

    # Observed Spearman
    rho_obs, p_obs = spearmanr(
        d[x_col].to_numpy(dtype=float),
        d[y_col].to_numpy(dtype=float),
    )

    names = d[name_col].to_numpy()
    x = d[x_col].to_numpy(dtype=float)

    rho_draws = np.empty(n_iter)

    for i in range(n_iter):

        y_draw = np.array([
            rng.choice(bootstrap_draws[name])
            for name in names
        ])

        rho_draws[i] = spearmanr(x, y_draw).statistic

    q16, q50, q84 = np.percentile(
        rho_draws,
        [16, 50, 84]
    )

    return {
        "N": len(d),
        "rho_obs": rho_obs,
        "p_obs": p_obs,
        "rho_boot_median": q50,
        "rho_boot_lo": q16,
        "rho_boot_hi": q84,
        "rho_boot_std": np.std(rho_draws, ddof=1),
        "frac_positive": np.mean(rho_draws > 0),
        "frac_negative": np.mean(rho_draws < 0),
        "rho_draws": rho_draws,
    }

# ------------------------------------------------------------------ #

def bootstrap_spearman_from_realizations(
    x,
    boot_y,
):
    """
    Propagate per-galaxy bootstrap uncertainty in a response variable
    into the Spearman rank correlation.

    Parameters
    ----------
    x : array, shape (N_gal,)
        Environmental predictor.

    boot_y : array, shape (N_boot, N_gal)
        Bootstrap realizations of the galaxy-level response.

    Returns
    -------
    dict
        Median and percentile interval of the bootstrap rho
        distribution.
    """

    x = np.asarray(x, dtype=float)
    boot_y = np.asarray(boot_y, dtype=float)

    rho_boot = np.full(
        boot_y.shape[0],
        np.nan,
        dtype=float,
    )

    for b in range(boot_y.shape[0]):

        y = boot_y[b]

        mask = (
            np.isfinite(x)
            & np.isfinite(y)
        )

        if mask.sum() < 3:
            continue

        rho_boot[b] = spearmanr(
            x[mask],
            y[mask],
        ).statistic

    good = np.isfinite(rho_boot)

    q16, q50, q84 = np.percentile(
        rho_boot[good],
        [16, 50, 84],
    )

    return {
        "N_boot": good.sum(),
        "rho_q16": q16,
        "rho_q50": q50,
        "rho_q84": q84,
        "frac_negative": np.mean(
            rho_boot[good] < 0
        ),
        "frac_positive": np.mean(
            rho_boot[good] > 0
        ),
    }


# ------------------------------------------------------------------ #

def hierarchical_bootstrap_spearman(
    x,
    boot_y,
    n_iter=10_000,
    seed=42,
):
    """
    Hierarchical bootstrap for Spearman rho.

    Combines:
      1. per-galaxy response uncertainty from bootstrap realizations
      2. galaxy-level resampling with replacement

    Parameters
    ----------
    x : array, shape (N_gal,)
        Environmental predictor.

    boot_y : array, shape (N_boot, N_gal)
        Bootstrap realizations of the luminosity-residualized response.

    n_iter : int
        Number of hierarchical bootstrap iterations.

    seed : int
        RNG seed.

    Returns
    -------
    dict
        Bootstrap distribution summary for Spearman rho.
    """

    x = np.asarray(x, dtype=float)
    boot_y = np.asarray(boot_y, dtype=float)

    rng = np.random.default_rng(seed)

    n_boot_metric, n_gal = boot_y.shape

    rho_draws = np.full(n_iter, np.nan)

    for i in range(n_iter):

        # choose one response-bootstrap realization
        b = rng.integers(0, n_boot_metric)
        y = boot_y[b]

        # finite galaxies for this realization
        mask = np.isfinite(x) & np.isfinite(y)

        x_valid = x[mask]
        y_valid = y[mask]

        n = len(x_valid)

        if n < 3:
            continue

        # galaxy-level bootstrap
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        rho_draws[i] = spearmanr(
            x_valid[idx],
            y_valid[idx],
        ).statistic

    good = np.isfinite(rho_draws)

    q16, q50, q84 = np.percentile(
        rho_draws[good],
        [16, 50, 84],
    )

    q025, q975 = np.percentile(
        rho_draws[good],
        [2.5, 97.5],
    )

    return {
        "N_boot": good.sum(),
        "rho_q16": q16,
        "rho_q50": q50,
        "rho_q84": q84,
        "rho_q025": q025,
        "rho_q975": q975,
        "frac_negative": np.mean(rho_draws[good] < 0),
        "frac_positive": np.mean(rho_draws[good] > 0),
    } 

# ------------------------------------------------------------------ #

