"""
Alt-Az trajectory and star-centroid movement plotting.

This module generates two diagnostic plots for OPTICAM time-series photometry:

    1. A polar plot of the target star's trajectory in the sky during an observation
       night, using the altitude-azimuth coordinate system.
    2. A star-centroid movement plot showing the change in pixel position of the
       centroid throughout the time series.

These plots are useful for evaluating tracking performance and guiding stability.
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

    A hidden control file (`.Alt-Az_and_guiding.dat`) tracks processed
    (target, date) combinations to avoid regenerating existing plots.
    """

    def __init__(
        self,
        site_lat: float = 31.0439,  # OAN SPM latitude HARD-SCRIPT
        site_lon: float = -115.4637,  # OAN SPM longitude
    ) -> None:
        """
        Initialize the AltAzGuidingPlotter instance.

        Sets up directory paths, loads the list of already processed (object, date)
        entries, and configures the observatory location for altitude–azimuth
        calculations.

        Args:
            site_lat (float, optional): Observatory latitude in decimal degrees.
                Defaults to OAN-SPM latitude (31.0439°).
            site_lon (float, optional): Observatory longitude in decimal degrees.
                Defaults to OAN-SPM longitude (-115.4637°).
        """
        self.base_dir = Path.cwd()
        self.data_dir = self.base_dir / "organized_data"
        self.log_dir = self.base_dir / "logs"
        self.log_file = self.log_dir / ".Alt-Az_and_guiding.dat"
        self.location = EarthLocation(lat=site_lat * u.deg, lon=site_lon * u.deg)
        self.processed = self._load_processed()
        self.logger = logging.getLogger(__name__)

    def _load_processed(self) -> set:
        """
        Load processed object/date entries from the control log.

        Reads the hidden log file (`self.log_file`) and parses lines containing
        an object name and an observation date separated by a comma.

        Returns:
            set[tuple[str, str]]: A set of (object, date) pairs already processed.
        """
        processed: set = set()
        if self.log_file.exists():
            for line in self.log_file.read_text().splitlines():
                parts: list = [p.strip() for p in line.split(",")]
                if len(parts) == 2:
                    processed.add((parts[0], parts[1]))
        return processed

    def _record_processed(self, obj_name: str, date: str) -> None:
        """
        Record a processed object/date pair in the control log.

        Appends the given (object, date) pair to the log file and updates the
        `self.processed` set.

        Args:
            obj_name (str): Target object name.
            date (str): Observation date in YYYY-MM-DD format.
        """
        with open(self.log_file, "a") as f:
            f.write(f"{obj_name},{date}\n")
        self.processed.add((obj_name, date))

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

        # Read measurements TBL
        tbl_files: list[Path] = list(date_folder.glob("*.tbl"))
        if not tbl_files:
            self.logger.info(f"No TBL in {date_folder}. Skipping date.")
            return
        data: DataFrame = pd.read_table(tbl_files[0])

        # Alt-Az computation
        jd: Series = data["JD_UTC"]
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

        # Star-centroid movement
        x0 = data["X(FITS)_T1"][0]
        y0 = data["Y(FITS)_T1"][0]
        x_mov = (data["X(FITS)_T1"] - x0).tolist()
        y_mov = (data["Y(FITS)_T1"] - y0).tolist()

        # Star-centroid movement plot
        plt.figure()
        plt.plot(data["BJD_TDB"], x_mov, ".", label="X mov.", alpha=0.5)
        plt.plot(data["BJD_TDB"], y_mov, ".", label="Y mov.", alpha=0.5)
        plt.xlabel("BJD_TDB")
        plt.ylabel("Movement (pixels)")
        plt.title(f"{date_folder.name}: {target_name}-centroid movement")
        plt.legend()
        plt.grid(alpha=0.3)
        cent_path: Path = plot_dir / f"{date_folder.name}_T1_centroid_mov.png"
        plt.savefig(cent_path, dpi=300)
        plt.close()
        msg2: str = f"Saved {target_name} in {date_folder.name} centroid movement plot"
        self.logger.info(msg2)

    def process_object(self, obj_dir: Path) -> None:
        """
        Generate plots for all observation dates of a single object.

        Finds an example FITS file to extract RA, DEC, and target name, then
        iterates through all date folders in `measurements/` to produce plots.

        Steps:
            1. Verify FITS files exist in `obj_dir`.
            2. Extract RA, DEC, and OBJECT from FITS header.
            3. For each date folder in `measurements/`:
                - Skip if already processed.
                - Generate plots using `_generate_plots()`.
                - Record the processed (object, date) in the log.

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
            key: tuple[str, str] = (obj_dir.name, date_folder.name)
            if key in self.processed:
                msg: str = f"{key} already processed"
                self.logger.info(msg)
                continue

            self._generate_plots(obj_dir, date_folder, RA, DEC, target)
            self._record_processed(obj_dir.name, date_folder.name)

    def run(self) -> None:
        """
        Process all objects in the organized data directory.

        Scans each subdirectory under `self.data_dir`. If it contains a
        `measurements/` subfolder, calls `process_object()` to generate plots.

        Returns:
            None
        """
        for obj_dir in self.data_dir.iterdir():
            if not obj_dir.is_dir():
                continue
            if not (obj_dir / "measurements").exists():
                continue
            self.process_object(obj_dir)
