"""
Command-line interface for generating PROFE output products.

This module coordinates the execution of all PROFE output-generating classes,
including:

    - AltAzGuidingPlotter: Generates alt-azimuth and guiding (centroid) plots.
    - LightCurvePlotter: Produces light curves and corresponding CSV files.
    - TimeAveragingPlotter: Creates time-averaging plots for correlated noise
      analysis.
    - ExofopPlotter: Generates plots and products for EXOFOP submission.

Logs are saved to `logs/profe_out.log`.

TODO:
    Implement additional outputs for EXOFOP submission.
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
    """
    Run all PROFE output modules sequentially.

    Executes each output-generating class in order to produce all standard
    plots and data products for the PROFE pipeline.

    Steps:
        1. Run AltAzGuidingPlotter to produce alt-azimuth and centroid plots.
        2. Run LightCurvePlotter to generate light curve plots and CSV files.
        3. Run TimeAveragingPlotter to create time-averaging noise plots.
        4. Run ExofopPlotter to prepare EXOFOP submission plots.

    Returns:
        None
    """
    ag_plotter: AltAzGuidingPlotter = AltAzGuidingPlotter()
    ag_plotter.run()

    lc_plotter: LightCurvePlotter = LightCurvePlotter()
    lc_plotter.run()

    avg_plotter: TimeAveragingPlotter = TimeAveragingPlotter()
    avg_plotter.run()

    exofop_plotter: ExofopPlotter = ExofopPlotter()
    exofop_plotter.run()


if __name__ == "__main__":
    main()
