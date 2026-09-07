# ----------------------------
# 2) GMM metrics (P_blue, f_b, mu_blue/red) with bootstrap (not currently used)
# ----------------------------

def fit_2gmm_and_label_blue(color: np.ndarray, random_state: int = 0) -> Tuple[GaussianMixture, int, int]:
    """
    Fit 2-component GMM in 1D color. Returns model and (blue_idx, red_idx).
    Blue = component with smaller mean color.
    """
    x = color.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=random_state)
    gmm.fit(x)
    means = gmm.means_.flatten()
    blue_idx = int(np.argmin(means))
    red_idx = int(np.argmax(means))
    return gmm, blue_idx, red_idx

def gmm_metrics_for_colors(color: np.ndarray, random_state: int = 0) -> Dict[str, float]:
    gmm, b, r = fit_2gmm_and_label_blue(color, random_state=random_state)
    x = color.reshape(-1, 1)
    probs = gmm.predict_proba(x)  # Nx2
    p_blue = probs[:, b]

    # fraction of blue in mixture = weight of blue component
    f_b = float(gmm.weights_[b])
    mu_b = float(gmm.means_[b, 0])
    mu_r = float(gmm.means_[r, 0])

    return {
        "Pblue_mean": float(np.mean(p_blue)),
        "fb": f_b,
        "mu_blue": mu_b,
        "mu_red": mu_r,
    }


def metrics_from_precomputed_Pblue(Pblue: np.ndarray) -> Dict[str, float]:
    """
    Compute galaxy-level mixture diagnostics using *fixed* Coma Pblue values.

    Parameters
    ----------
    Pblue : (N,) array
        Per-GC blue membership probabilities from compute_Pblue().

    Returns
    -------
    dict with:
      Pblue_mean : mean(Pblue)
      fb         : estimate of blue fraction; by default same as mean(Pblue)
    """
    Pblue = np.asarray(Pblue, dtype=float)
    Pblue = Pblue[np.isfinite(Pblue)]

    if Pblue.size == 0:
        return {"Pblue_mean": np.nan, "fb": np.nan}

    # For a fixed external model, the clean estimator of the blue fraction in the sample
    # is simply the mean blue membership probability (soft counts).
    pmean = float(np.mean(Pblue))

    return {
        "Pblue_mean": pmean,
        "fb": pmean,   # <- see note below
    }

def bootstrap_gmm_metrics(color: np.ndarray, n_boot: int, seed: int = 0) -> Dict[str, Tuple[float, float]]:
    """
    Returns (median, std) for each metric across bootstrap resamples.
    """
    rng = np.random.default_rng(seed)
    metrics = {"Pblue_mean": [], "fb": [], "mu_blue": [], "mu_red": []}

    n = len(color)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = color[idx]
        m = gmm_metrics_for_colors(sample, random_state=int(rng.integers(0, 2**31 - 1)))
        for k in metrics:
            metrics[k].append(m[k])

    out = {}
    for k, arr in metrics.items():
        arr = np.asarray(arr)
        out[k] = (float(np.median(arr)), float(np.std(arr, ddof=1)))
    return out

