import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union
from astropy.coordinates import SkyCoord
import astropy.units as u
from sklearn.linear_model import HuberRegressor
import logging

from src.config import (
    Columns,
    BCG,
    AnalysisConfig,
    ClusterCenter,
    ColorConfig,
)

CenterSpec = Union[ClusterCenter, Tuple[float, float], SkyCoord]

def center_to_skycoord(center: CenterSpec) -> SkyCoord:
    if isinstance(center, SkyCoord):
        return center.icrs
    if isinstance(center, ClusterCenter):
        return SkyCoord(center.ra_deg * u.deg, center.dec_deg * u.deg, frame="icrs")
    # assume (ra_deg, dec_deg)
    ra_deg, dec_deg = center
    return SkyCoord(float(ra_deg) * u.deg, float(dec_deg) * u.deg, frame="icrs")
    
def skycoord_from_df(df: pd.DataFrame, ra_col: str, dec_col: str) -> SkyCoord:
    return SkyCoord(ra=df[ra_col].to_numpy() * u.deg,
                    dec=df[dec_col].to_numpy() * u.deg,
                    frame="icrs")

def angsep_arcsec(ra1_deg, dec1_deg, ra2_deg, dec2_deg) -> np.ndarray:
    c1 = SkyCoord(ra=ra1_deg * u.deg, dec=dec1_deg * u.deg)
    c2 = SkyCoord(ra=ra2_deg * u.deg, dec=dec2_deg * u.deg)
    return c1.separation(c2).to(u.arcsec).value

def tangent_plane_xy_arcsec(coords: SkyCoord, ref: SkyCoord) -> np.ndarray:
    """
    Project ICRS coords onto a tangent plane around ref.
    Returns Nx2 array [x_arcsec, y_arcsec].
    """
    # small-angle approximation using spherical offsets
    dlon = (coords.ra - ref.ra).to(u.rad).value
    dlat = (coords.dec - ref.dec).to(u.rad).value
    # cos(dec) factor for RA
    x = dlon * np.cos(ref.dec.to(u.rad).value) * (u.rad.to(u.arcsec))
    y = dlat * (u.rad.to(u.arcsec))
    return np.column_stack([x, y])

def elliptical_radius_kRe(
    gc_ra, gc_dec,
    gal_ra, gal_dec,
    Re_arcsec,
    q_axis,
    pa_deg,
):
    """
    Compute elliptical radius in units of Re for each GC relative to a galaxy.
    PA assumed East of North (SIMBAD convention).
    """
    # SkyCoord objects
    gc = SkyCoord(gc_ra * u.deg, gc_dec * u.deg)
    gal = SkyCoord(gal_ra * u.deg, gal_dec * u.deg)

    # Tangent-plane offsets (arcsec)
    dra = (gc.ra - gal.ra).to(u.arcsec).value * np.cos(gal.dec.to(u.rad).value)
    ddec = (gc.dec - gal.dec).to(u.arcsec).value

    # Rotate into galaxy frame
    pa = np.deg2rad(pa_deg)
    x =  dra * np.cos(pa) + ddec * np.sin(pa)
    y = -dra * np.sin(pa) + ddec * np.cos(pa)

    # Elliptical radius (arcsec)
    r_ell = np.sqrt(x**2 + (y / q_axis)**2)

    # Return in units of Re
    return r_ell / Re_arcsec

def circular_area(r_arcsec):
    return np.pi * r_arcsec**2

def annulus_area(rin_arcsec, rout_arcsec):
    return np.pi * (rout_arcsec**2 - rin_arcsec**2)
    
# ----------------------------
# 1) Select GCs within aperture (requires host association)
# ----------------------------

def select_gcs_within_kRe(
    data: pd.DataFrame,
    gals_df: pd.DataFrame,
    cols: Columns,
    cfg: AnalysisConfig,
) -> pd.DataFrame:
    """
    Returns a copy of data with a boolean column 'in_aperture' (<= cfg.aperture_Re * Re)
    Assumes data already has host galaxy ID in cols.gc_host.
    """
    if cols.gc_host is None or cols.gc_host not in data.columns:
        raise ValueError("GC host galaxy assignment column missing. "
                         "Add data[cols.gc_host] with host IDs first.")

    gsub = gals_df[[cols.gal_id, cols.gal_ra, cols.gal_dec, cols.gal_Re_arcsec]].copy()
    gsub = gsub.set_index(cols.gal_id)

    # Map galaxy center + Re onto each GC
    host_ids = data[cols.gc_host].values
    valid = pd.notnull(host_ids)

    ra_g = np.full(len(data), np.nan)
    dec_g = np.full(len(data), np.nan)
    Re_g = np.full(len(data), np.nan)

    # vectorized mapping via reindex
    mapped = gsub.reindex(host_ids[valid])
    ra_g[valid] = mapped[cols.gal_ra].to_numpy()
    dec_g[valid] = mapped[cols.gal_dec].to_numpy()
    Re_g[valid] = mapped[cols.gal_Re_arcsec].to_numpy()

    sep = angsep_arcsec(data[cols.gc_ra].to_numpy(), data[cols.gc_dec].to_numpy(), ra_g, dec_g)
    kRe = sep / Re_g

    out = data.copy()
    out["kRe"] = kRe
    out["in_aperture"] = (kRe <= cfg.aperture_Re)
    return out

def select_gc_samples_for_galaxy(
    data,
    gal_row,
    cols,
    kRe_inner=8.0,
    kRe_outer=(12.0, 20.0),
):
    """
    Returns inner and annulus GC samples for one galaxy.
    """
    kRe = elliptical_radius_kRe(
        data[cols.gc_ra].values,
        data[cols.gc_dec].values,
        gal_row[cols.gal_ra],
        gal_row[cols.gal_dec],
        gal_row[cols.gal_Re_arcsec],
        gal_row["q_axis"],
        gal_row["pa_deg"],
    )

    data = data.copy()
    data["kRe"] = kRe

    inner = data[kRe <= kRe_inner]
    ann = data[(kRe > kRe_outer[0]) & (kRe <= kRe_outer[1])]

    return inner, ann

def _select_css_color_within_radius(gal_row: pd.Series,
                                    css_df: pd.DataFrame,
                                    r_arcsec: float) -> pd.DataFrame:
    """
    Select CSS candidates within r_arcsec of a galaxy center.
    Adds sep_arcsec, pa_deg, color columns for the selected subset of css.
    """
    # guard: drop rows lacking coordinates
    gal_ra, gal_dec = float(gal_row['ra']), float(gal_row['dec'])
    m = np.isfinite(css_df['x_wcs']) & np.isfinite(css_df['y_wcs'])
    css = css_df.loc[m].copy()

    gal_c = SkyCoord(ra=gal_ra * u.deg, dec=gal_dec * u.deg)
    css_c = SkyCoord(ra=css['x_wcs'].values * u.deg,
                     dec=css['y_wcs'].values * u.deg)

    sep = gal_c.separation(css_c).arcsec
    sel = sep <= float(r_arcsec)
    if not np.any(sel):
        return css.iloc[0:0].assign(sep_arcsec=[], pa_deg=[], color=[])

    pa_deg = gal_c.position_angle(css_c[sel]).to(u.deg).value  # 0=N, 90=E
    color = css[sel]['color']
    return (css.loc[sel]
            .assign(sep_arcsec=sep[sel],
                    pa_deg=pa_deg,
                    color=color))


def get_selected_css_for_galaxy(
    gal_name: str,
    gals_df: pd.DataFrame,
    css_df: pd.DataFrame,
    cfg: ColorConfig,
    name_col: str = "name",
) -> pd.DataFrame:
    """
    Return the CSS objects selected within cfg.re * Re for a named galaxy.
    This is intended for plotting/diagnostics, not for the scalar summary table.
    """
    matches = gals_df.loc[gals_df[name_col] == gal_name]

    if len(matches) == 0:
        raise ValueError(f"No galaxy found with {name_col} == {gal_name!r}")

    if len(matches) > 1:
        raise ValueError(f"Multiple galaxies found with {name_col} == {gal_name!r}")

    gal_row = matches.iloc[0]
    r_arcsec = cfg.re * gal_row.get("Re_best_arcsec", 5)

    css = _select_css_color_within_radius(gal_row, css_df, r_arcsec).copy()

    # Useful metadata for plotting/debugging
    css["host_galaxy"] = gal_name
    css["host_radius_arcsec"] = float(r_arcsec)

    return css