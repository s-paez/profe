"""
Outputs module

This PROFE's module create and save the following plots:
    - Alt azimutal trajectory of the target star in each observation ngiht
    - Star centroid movement plot
    - Time-averaging curves
    - Light curves: plots and csv files
"""

from .alt_az_centroid import AltAzGuidingPlotter
from .correlated_noise import TimeAveragingPlotter
from .light_curves import LightCurvePlotter

__all__: list[str] = [
    "AltAzGuidingPlotter",
    "LightCurvePlotter",
    "TimeAveragingPlotter",
]
