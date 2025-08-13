"""
Comman Line Interface to execute the output modules

This PROFE's module implement the AltAzGuidingPlotter, TimeAveragingPlotter, and
LightCurvePlotter classes to crate all the PROFE's outputs

TO DO: Outputs for the Exofop submission
"""

import logging
import os

# Import the processing classes
from .alt_az_centroid import AltAzGuidingPlotter
from .correlated_noise import TimeAveragingPlotter
from .exofop_out import ExofopPlotter
from .light_curves import LightCurvePlotter

# Make the logs dir
os.makedirs("logs", exist_ok=True)
# Config thr logging
logging.basicConfig(
    filename="logs/profe_out.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)

logger.info("Running PROFE-outputs")


def main() -> None:
    """The main fucntion that apply each output method"""
    ag_plotter: AltAzGuidingPlotter = AltAzGuidingPlotter()  # AltAz and Guiding plotter
    ag_plotter.run()

    lc_plotter: LightCurvePlotter = LightCurvePlotter()  # Light Curve plotter
    lc_plotter.run()

    avg_plotter: TimeAveragingPlotter = TimeAveragingPlotter()
    avg_plotter.run()

    exofop_plotter: ExofopPlotter = ExofopPlotter()
    exofop_plotter.run()
