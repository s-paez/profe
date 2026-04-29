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
from .report_generator import ReportGenerator
from .seeing_profile import SeeingProfilePlotter
from .transit_info import TransitDataManager


logger = logging.getLogger(__name__)

logger.info("Running PROFE-outputs")


def run_output(target: str | None = None) -> None:
    """
    Run all PROFE output modules sequentially.

    Executes each output-generating class in order to produce all standard
    plots and data products for the PROFE pipeline.

    Args:
        target (str | None): If specified, only process outputs for this
            target name (e.g. ``"TOI-1234"``). When ``None``, all targets
            in ``organized_data/`` are processed.

    Steps:
        1. Run AltAzGuidingPlotter to produce alt-azimuth and centroid plots.
        2. Run LightCurvePlotter to generate light curve plots and CSV files.
        3. Run TimeAveragingPlotter to create time-averaging noise plots.
        4. Run ExofopPlotter to prepare EXOFOP submission plots.

    Returns:
        None
    """
    if target:
        logger.info(f"Target filter active: processing only '{target}'")

    ag_plotter: AltAzGuidingPlotter = AltAzGuidingPlotter()
    ag_plotter.run(target=target)

    lc_plotter: LightCurvePlotter = LightCurvePlotter()
    lc_plotter.run(target=target)

    avg_plotter: TimeAveragingPlotter = TimeAveragingPlotter()
    avg_plotter.run(target=target)

    ex_plotter: FieldViewPlotter = FieldViewPlotter()
    ex_plotter.run(target=target)

    sp_plotter: SeeingProfilePlotter = SeeingProfilePlotter()
    sp_plotter.run(target=target)

    comp_plotter: ComparisonStarsPlotter = ComparisonStarsPlotter()
    comp_plotter.run(target=target)

    astrometry_solver: AstrometrySolver = AstrometrySolver()
    astrometry_solver.run(target=target)

    transit_manager: TransitDataManager = TransitDataManager()
    transit_manager.run(target=target)

    report_generator: ReportGenerator = ReportGenerator()
    report_generator.run(target=target)


if __name__ == "__main__":
    run_output()
