# This file contains python routines and helpers for creating latex content, generally tables, from dataframes in the analysis
# While there are routines to do this automatically, I've not found any that create AAS deluxetable format

import numpy as np


def choose_exponent(values):
    """
    Choose a common exponent for scientific notation based on the
    largest absolute finite value in the set.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = np.abs(arr[arr != 0])

    if arr.size == 0:
        return 0

    vmax = np.max(arr)
    exp = int(np.floor(np.log10(vmax)))

    # For readability, use plain decimals for moderate values
    if -0 <= exp <= 0:
        return 0

    return exp

def format_signed(x, ndp=2):
    return f"{x:+.{ndp}f}"

def format_p(x, ndp=2):
    if x < 0.001:
        return f"{x:.2e}"
    return f"{x:.{ndp}f}"

def rescale_slope_bounds(val, lo, hi, factor=1.0):
    return val * factor, lo * factor, hi * factor


def rescale_slope_bounds_for_test(val, lo, hi, test_name=None, config=None):
    """
    Rescale slope/bounds using defaults defined for a named test.
    """
    if config is None:
        config = get_test_format_config(test_name)

    factor = config.get("rescale_factor", 1.0)

    return rescale_slope_bounds(val, lo, hi, factor=factor)

def get_unit_label_for_test(test_name=None, config=None):
    if config is None:
        config = get_test_format_config(test_name)
    return config.get("unit_label", "")

def format_pm_latex(val, lo, hi, exponent=0, ndp=2, unit_latex="", nodata=r"\nodata"):
    if not (np.isfinite(val) and np.isfinite(lo) and np.isfinite(hi)):
        return nodata

    scale = 10.0 ** (-exponent)
    v = val * scale
    dm = (val - lo) * scale
    dp = (hi - val) * scale

    if exponent == 0:
        core = rf"{v:.{ndp}f}_{{-{dm:.{ndp}f}}}^{{+{dp:.{ndp}f}}}"
    else:
        core = rf"{v:.{ndp}f}_{{-{dm:.{ndp}f}}}^{{+{dp:.{ndp}f}}}\times10^{{{exponent}}}"

    if unit_latex:
        return rf"${core}\,{unit_latex}$"
    else:
        return rf"${core}$"

def format_pm_latex_for_test(val, lo, hi, test_name=None, config=None, nodata=r"\nodata"):
    """
    Format slope/bounds using defaults defined for a named test.
    """
    if config is None:
        config = get_test_format_config(test_name)

    ndp = config.get("ndp", 2)
    exponent = config.get("exponent", 0)
    unit_str = config.get("unit_label", "")

    return format_pm_latex(
        val, lo, hi,
        exponent=exponent,
        ndp=ndp,
        unit_latex=unit_str,
        nodata=nodata
    )


STYLE_CONFIG = {
    "per_100kpc": {
        "rescale_factor": 100.0,
        "ndp": 3,
        "exponent": 0,
        "unit_label": r"(100\,\mathrm{kpc})^{-1}",
    },
    "per_dex": {
        "rescale_factor": 1.0,
        "ndp": 2,
        "exponent": 0,
        "unit_label": r"\mathrm{dex}^{-1}",
    },
    "plain": {
        "rescale_factor": 1.0,
        "ndp": 2,
        "exponent": 0,
        "unit_label": "",
    },
}

# --- paper 3 specific ---#


TEST_STYLE_MAP = {
    "cluster_radius": "per_100kpc",
    "bcg_mid": "per_100kpc",
    "xray_peak": "per_100kpc",
    "sz_peak": "per_100kpc",
    "geom": "per_100kpc",
    "NGC 4874": "per_100kpc",
    "NGC 4889": "per_100kpc",
    "nearBCG": "per_100kpc",
    "NearBCG_NGC 4874": "per_100kpc",
    "NearBCG_NGC 4889": "per_100kpc",
    "log_gc_voronoi_density": "per_dex",
    "log_tidal_proxy": "plain",
}

def get_test_format_config(test_name, style_map=None, style_config=None, default_style="plain"):
    if style_map is None:
        style_map = TEST_STYLE_MAP
    if style_config is None:
        style_config = STYLE_CONFIG

    style_name = style_map.get(test_name, default_style)
    return style_config[style_name]

def format_test_name(x):
    mapping = {
        "xray_peak": "X-ray peak",
        "sz_peak": "SZ peak",
        "geom": "Geometric",
        "R_bcg_mid_kpc": "BCG midpoint",
        "bcg_mid": "BCG midpoint",
        "nearBCG": "BCG proximity",
        "NGC 4874": "NGC 4874",
        "NGC 4889": "NGC 4889",
        "gc_voronoi_density_arcsec2_inv": "Local GC density",
        "log_gc_voronoi_density": "Local GC density",
        "log_tidal_proxy": "BCG tidal proxy",
        "R_nearBCG_kpc": "BCG proximity",
        "NearBCG_NGC 4874": "Near NGC 4874",
        "NearBCG_NGC 4889": "Near NGC 4889",
    }
    return mapping.get(x, str(x))

def split_test_name(test_name):
    """
    Split e.g. 'bcg_mid_annulus' into ('bcg_mid', 'annulus').

    Handles center names containing underscores by matching
    only the known final suffix.
    """
    components = ("inner", "annulus", "contrast")

    for component in components:
        suffix = f"_{component}"

        if test_name.endswith(suffix):
            center = test_name[:-len(suffix)]
            return center, component

    raise ValueError(
        f"Could not identify population suffix in test_name={test_name!r}"
    )

def format_component_name(x):
    mapping = {
        "inner": "Inner",
        "annulus": "Annulus",
        "contrast": "Contrast",
    }
    return mapping.get(x, str(x))
    