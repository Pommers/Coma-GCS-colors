from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Union


# ----------------------------
# Color Config / assumptions
# ----------------------------

@dataclass
class ColorConfig:
    re: float = 8.0                 # Re scale search radius for all galaxies
    min_css: int = 8                # minimum N to attempt tests
    n_boot: int = 5000
    random_state: int = 42

# ----------------------------
# Pblue Config / assumptions
# ----------------------------

@dataclass
class Columns:
    # galaxy table columns
    gal_id: str = "name"
    gal_ra: str = "ra"
    gal_dec: str = "dec"
    gal_Re_arcsec: str = "Re_best_arcsec"
    gal_MV: str = "MV_abs"

    # GC table columns
    gc_ra: str = "x_wcs"
    gc_dec: str = "y_wcs"
    gc_color: str = "color"           # e.g., F475W-F814W
    gc_mag: Optional[str] = "mag_814"  # optional
    gc_host: Optional[str] = "gal_id"  # if we already have assignment

@dataclass
class BCG:
    name: str
    ra_deg: float
    dec_deg: float
    MV: Optional[float] = None  # for tidal proxy via luminosity, optional

@dataclass
class AnalysisConfig:
    aperture_Re: float = 8.0
    # optional background annulus for "local GC density"
    bg_annulus_inner_Re: float = 12.0
    bg_annulus_outer_Re: float = 20.0

    # bootstrap
    n_boot: int = 300
    min_gc_for_gmm: int = 30  # below this, we can fall back to median-only
    min_gc_for_Pblue: int = 10
    # baseline fit
    huber_eps: float = 1.35   # robust regression tuning

@dataclass(frozen=True)
class ClusterCenter:
    name: str
    ra_deg: float
    dec_deg: float