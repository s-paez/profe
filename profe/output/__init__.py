"""
PROFE output plotting module.

This package generates and saves key plots and data products from the
PROFE pipeline, including:

    - Alt-azimuth trajectory and centroid movement plots.
    - Multi-band binned light curves with RMS (PDF, PNG, CSV).
    - Time-averaging noise characterization (red vs. white noise).
    - Aperture visualization (field of view) plots.
    - Radial (seeing) profile plots.
    - Comparison star light curves.
    - Automated WCS solving via Astrometry.net.
    - Transit timing retrieval from TESS Transit Finder (TTF).
    - Consolidated ExoFOP notes report generation.

Classes:
    AltAzGuidingPlotter: Generates alt-azimuth and centroid plots.
    LightCurvePlotter: Creates light curve plots and corresponding CSV files.
    TimeAveragingPlotter: Produces time-averaging noise curves.
    FieldViewPlotter: Generates aperture visualization plots.
    SeeingProfilePlotter: Generates radial brightness profile plots.
    ComparisonStarsPlotter: Generates comparison star light curve plots.
    AstrometrySolver: Solves astrometry via Astrometry.net API.
    TransitDataManager: Retrieves transit data from TTF.
    ReportGenerator: Generates consolidated ExoFOP notes reports.
"""

from __future__ import annotations

from .alt_az_centroid import AltAzGuidingPlotter
from .astrometry_out import AstrometrySolver
from .comparison_stars import ComparisonStarsPlotter
from .correlated_noise import TimeAveragingPlotter
from .field_view import FieldViewPlotter
from .light_curves import LightCurvePlotter
from .report_generator import ReportGenerator
from .seeing_profile import SeeingProfilePlotter
from .transit_info import TransitDataManager

__all__: list[str] = [
    "AltAzGuidingPlotter",
    "AstrometrySolver",
    "ComparisonStarsPlotter",
    "FieldViewPlotter",
    "LightCurvePlotter",
    "ReportGenerator",
    "SeeingProfilePlotter",
    "TimeAveragingPlotter",
    "TransitDataManager",
]
