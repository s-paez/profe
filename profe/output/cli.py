"""
Command-line interface for generating PROFE output products.

This module coordinates the execution of all PROFE output-generating classes,
including:

    - AltAzGuidingPlotter: Generates alt-azimuth and guiding (centroid) plots.
    - LightCurvePlotter: Produces light curves and corresponding CSV files.
    - TimeAveragingPlotter: Creates time-averaging plots for correlated noise
      analysis.
    - ExofopPlotter: Generates plots and products for EXOFOP submission.

Each module skips already-processed (object, date) pairs by checking whether
the expected output files exist on disk.

Logs are saved to `logs/profe_out.log`.
"""

import logging
import os

# Import the processing classes
from .alt_az_centroid import AltAzGuidingPlotter
from .astrometry_out import AstrometrySolver
from .comparison_stars import ComparisonStarsPlotter
from .correlated_noise import TimeAveragingPlotter
from .field_view import FieldViewPlotter
from .light_curves import LightCurvePlotter
from .seeing_profile import SeeingProfilePlotter

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


def run_output() -> None:
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

    ex_plotter: FieldViewPlotter = FieldViewPlotter()
    ex_plotter.run()

    sp_plotter: SeeingProfilePlotter = SeeingProfilePlotter()
    sp_plotter.run()

    comp_plotter: ComparisonStarsPlotter = ComparisonStarsPlotter()
    comp_plotter.run()

    astrometry_solver: AstrometrySolver = AstrometrySolver()
    astrometry_solver.run()


if __name__ == "__main__":
    run_output()
