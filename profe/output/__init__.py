"""
PROFE output plotting module.

This package generates and saves key plots and data products from the
PROFE pipeline, including:

    - Alt-azimuth trajectory of the target star for each observation night.
    - Star centroid motion plots.
    - Time-averaging curves for noise characterization.
    - Light curves, including plots and CSV exports.

Classes:
    AltAzGuidingPlotter: Generates alt-azimuth and centroid plots.
    TimeAveragingPlotter: Produces time-averaging noise curves.
    LightCurvePlotter: Creates light curve plots and corresponding CSV files.
"""

from .alt_az_centroid import AltAzGuidingPlotter
from .correlated_noise import TimeAveragingPlotter
from .light_curves import LightCurvePlotter

__all__: list[str] = [
    "AltAzGuidingPlotter",
    "LightCurvePlotter",
    "TimeAveragingPlotter",
]
