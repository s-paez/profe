"""
Alt-Az trajectory and star-centroid movement plotting.

This module generates two diagnostic plots for OPTICAM time-series photometry:

    1. A polar plot of the target star's trajectory in the sky during an observation
       night, using the altitude-azimuth coordinate system.
    2. A star-centroid movement plot showing the change in pixel position of the
       centroid throughout the time series.

These plots are useful for evaluating tracking performance and guiding stability.
Already-processed (target, date) pairs are detected by checking whether the
expected output PNG exists on disk, so no external state file is needed.
"""

import logging
from pathlib import Path
from typing import Any

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
from matplotlib.pyplot import Figure
from matplotlib.ticker import FixedLocator
from pandas import DataFrame, Series


class AltAzGuidingPlotter:
    """
    Generate Alt-Az and centroid movement plots for OPTICAM time-series photometry.

    This class scans the `organized_data/OBJECT/DATE-OBS/` directory structure,
    computes the telescope pointing (altitude, azimuth) and centroid shifts for
    each sequence of FITS images, and saves two plots per observation night:

        1. Altitude vs. time and Azimuth vs. time in polar projection.
        2. XY centroid displacement (in pixels) vs. time.

    Already-processed pairs are detected by the presence of the output PNG.
    """

    def __init__(
        self,
        site_lat: float = 31.0439,  # OAN SPM latitude HARD-SCRIPT
        site_lon: float = -115.4637,  # OAN SPM longitude
    ) -> None:
        """
        Initialize the AltAzGuidingPlotter instance.

        Sets up directory paths and configures the observatory location for
        altitude–azimuth calculations.

        Args:
            site_lat (float, optional): Observatory latitude in decimal degrees.
                Defaults to OAN-SPM latitude (31.0439°).
            site_lon (float, optional): Observatory longitude in decimal degrees.
                Defaults to OAN-SPM longitude (-115.4637°).
        """
        self.base_dir = Path.cwd()
        self.data_dir = self.base_dir / "organized_data"
        self.location = EarthLocation(lat=site_lat * u.deg, lon=site_lon * u.deg)
        self.logger = logging.getLogger(__name__)

    def _is_processed(self, obj_dir: Path, date_name: str) -> bool:
        """
        Check whether Alt-Az outputs already exist for (object, date).

        Args:
            obj_dir (Path): Path to the object directory.
            date_name (str): Observation date folder name.

        Returns:
            bool: True if the AltAz plot PNG already exists.
        """
        expected = (
            obj_dir
            / "plots"
            / date_name
            / "AltAz_and_guiding"
            / f"{date_name}_AltAz_movement.png"
        )
        return expected.exists()

    def _generate_plots(
        self, obj_dir: Path, date_folder: Path, RA: float, DEC: float, target_name: str
    ) -> None:
        """
        Generate and save Alt-Az and centroid movement plots for one observation.

        Reads centroid and timing data from the `.tbl` file in `date_folder`,
        computes the target's altitude–azimuth coordinates for each timestamp,
        and produces:

            1. A polar plot of azimuth vs. altitude.
            2. A plot of centroid displacement in X and Y pixels vs. time.

        Args:
            obj_dir (Path): Path to the object directory.
            date_folder (Path): Path to the specific date directory.
            RA (float): Right ascension of the target in hours.
            DEC (float): Declination of the target in degrees.
            target_name (str): Identifier for plot titles and filenames.
        """
        plot_dir: Path = obj_dir / "plots" / date_folder.name / "AltAz_and_guiding"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Read measurements
        meas_files: list[Path] = [
            f
            for f in date_folder.iterdir()
            if f.is_file()
            and not f.name.startswith(".")
            and f.suffix in (".tbl", ".csv")
        ]
        if not meas_files:
            self.logger.info(f"No measurements in {date_folder}. Skipping date.")
            return

        file_to_read = meas_files[0]
        data: DataFrame
        if file_to_read.suffix == ".tbl":
            data = pd.read_csv(
                file_to_read, sep=r"\t+", engine="python", encoding="latin1"
            )
        else:
            data = pd.read_csv(file_to_read, encoding="latin1")

        # Alt-Az computation
        jd: Series
        if "JD_UTC" in data.columns:
            jd = data["JD_UTC"]
        elif "BJD_TDB" in data.columns:
            jd = data["BJD_TDB"]
        else:
            self.logger.warning(
                f"No JD_UTC or BJD_TDB in {file_to_read.name}. Skipping AltAz."
            )
            return

        times: Time | Any = Time(jd, format="jd", scale="utc")
        sky: SkyCoord = SkyCoord(ra=RA, dec=DEC, unit=(u.hourangle, u.deg))
        altaz: Any = sky.transform_to(AltAz(obstime=times, location=self.location))
        az_rad: Any = np.deg2rad(altaz.az.deg)
        alt_rad: Any = np.deg2rad(altaz.alt.deg)

        # Alt-Az plot
        fig: Figure = plt.figure()
        ax: Any = fig.add_subplot(111, projection="polar")
        ax.plot(az_rad, alt_rad, lw=2)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(np.pi / 2, 0)
        ticks: Any = ax.get_yticks()
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.set_yticklabels([f"{int(round(np.degrees(t)))}°" for t in ticks])
        altaz_path: Path = plot_dir / f"{date_folder.name}_AltAz_movement.png"
        plt.savefig(altaz_path, dpi=300)
        plt.close()
        msg: str = f"Saved Alt_Az plot for {target_name} in {date_folder.name}"
        self.logger.info(msg)

    def process_object(self, obj_dir: Path) -> None:
        """
        Generate plots for all observation dates of a single object.

        Finds an example FITS file to extract RA, DEC, and target name, then
        iterates through all date folders in `measurements/` to produce plots.

        Steps:
            1. Verify FITS files exist in `obj_dir`.
            2. Extract RA, DEC, and OBJECT from FITS header.
            3. For each date folder in `measurements/`:
                - Skip if output already exists.
                - Generate plots using `_generate_plots()`.

        Args:
            obj_dir (Path): Path to the object folder.
        """
        fits_files: list[Path] = list(obj_dir.rglob("*.fit"))

        if not fits_files:
            self.logger.warning(f"No FITS file in {obj_dir.name}. Skipping.")
            return
        example: Path = fits_files[0]
        with fits.open(example) as hdul:
            hdr = hdul[0].header  # type: ignore[attr-defined]
            RA = hdr.get("RA")
            DEC = hdr.get("DEC")
            target = hdr.get("OBJECT", obj_dir.name)

        measurements_root: Path = obj_dir / "measurements"
        if not measurements_root.exists():
            self.logger.warning(f"No measurements for {obj_dir.name}. Skipping.")
            return

        for date_folder in sorted(measurements_root.iterdir()):
            if not date_folder.is_dir():
                continue
            if self._is_processed(obj_dir, date_folder.name):
                msg: str = f"({obj_dir.name}, {date_folder.name}) already processed"
                self.logger.info(msg)
                continue

            self._generate_plots(obj_dir, date_folder, RA, DEC, target)

    def run(self) -> None:
        """
        Process all objects in the organized data directory.

        Scans each subdirectory under `self.data_dir`. If it contains a
        `measurements/` subfolder and its output does not yet exist, calls
        `process_object()` to generate plots.

        Returns:
            None
        """
        for obj_dir in self.data_dir.iterdir():
            if not obj_dir.is_dir():
                continue
            if not (obj_dir / "measurements").exists():
                continue
            self.process_object(obj_dir)
