"""
Command-line interface for generating PROFE output products.

This module coordinates the sequential execution of all PROFE
output-generating classes:

    1. AltAzGuidingPlotter: Altitude–Azimuth trajectory and centroid plots.
    2. LightCurvePlotter: Multi-band light curves (PDF/PNG/CSV).
    3. TimeAveragingPlotter: Red vs. white noise characterization.
    4. FieldViewPlotter: Aperture visualization (field of view) plots.
    5. SeeingProfilePlotter: Radial brightness profile plots.
    6. ComparisonStarsPlotter: Comparison star light curves.
    7. AstrometrySolver: WCS solving via Astrometry.net.
    8. TransitDataManager: Transit timing retrieval from TTF.
    9. ReportGenerator: Consolidated ExoFOP notes report.

Each module skips already-processed (object, date, band) triples by checking
whether the expected output files exist on disk.

Logs are saved to ``logs/profe_output_<timestamp>.log``.
"""

import logging

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
        4. Run FieldViewPlotter to generate aperture visualization plots.
        5. Run SeeingProfilePlotter to generate radial profile plots.
        6. Run ComparisonStarsPlotter to generate comparison star plots.
        7. Run AstrometrySolver to solve WCS via Astrometry.net.
        8. Run TransitDataManager to retrieve transit times from TTF.
        9. Run ReportGenerator to create consolidated ExoFOP notes.

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
