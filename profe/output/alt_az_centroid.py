"""
AltAz Trajectory and star-centroid movement

This module Creates a polar plot with the trajectory of the target star in the sky
during the observation in alt azimutal coordinate system.

It also creates a star-centroid movement plot that show how many pixels the
star-centroid moved during the time series.
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
    Generate Alt-Az and centroid movement plots for OPTICAM time series photometry.

    This class scans the `organized_data/OBJECT/DATE-OBS/` directory structure, computes
    the telescope pointing (altitude, azimuth) and star-centroid shifts for each
    sequence of FITS images, and saves two plots per observation:

        1. Altitude vs. time and Azimuth vs. time in polar projection
        2. XY centroid movement (in pixels) vs. time.

    A hidden control file at `.Alt-Az_and_guiding.dat` tracks which target/date
    combinations have already been plotted, so rerunning the pipeline does not
    regenerate existing plots.
    """

    def __init__(
        self,
        site_lat: float = 31.0439,  # OAN SPM latitude
        site_lon: float = -115.4637,  # OAN SPM longitude
    ) -> None:
        """
        Initialize the AltAzGudingPlotter for generating pointing and guiding plots.

        Set up directory paths, load the list of already processed target/dates, and
        configures the observatory location for Alt-Az computation.

        Args:
            site_lat (float): Observatory latitude in decimal degrees (OAN-SPM default)
            site_lon (float): Observatory longitude in decimal degrees (OAN-SPM default)
        """
        # Use the working directory. It will be the same dir where the profe_pre runs
        self.base_dir = Path.cwd()
        # Dir of organized data
        self.data_dir = self.base_dir / "organized_data"
        # Dir of logs
        self.log_dir = self.base_dir / "logs"

        # Log file path
        self.log_file = self.log_dir / ".Alt-Az_and_guiding.dat"

        # OAN-SPM coordinate to Alt-Az computation
        self.location = EarthLocation(lat=site_lat * u.deg, lon=site_lon * u.deg)

        # Objects and data aready processed
        self.processed = self._load_processed()

        # Logger para mensajes informativos
        self.logger = logging.getLogger(__name__)

    def _load_processed(self) -> set:
        """
        Load processed object/date entries from the log file.

        This hidden method reads `self.log_file`, where each line is expected to contain
        an object name and an observation date separated by a comma. It returns a set of
        (object, date) tuples for all valid entries.

        Returns:
            set[tuple[str, str]]: A set of (object, date) pairs already processed.
        """
        # Iterable object set()
        processed: set = set()
        if self.log_file.exists():
            for line in self.log_file.read_text().splitlines():
                parts: list = [p.strip() for p in line.split(",")]
                if len(parts) == 2:
                    processed.add((parts[0], parts[1]))  # [0] object and [1] date
        return processed

    def _record_processed(self, obj_name: str, date: str) -> None:
        """Write in the logs file processed objects and dates"""
        with open(self.log_file, "a") as f:  # Open the log file in `appending` mode
            f.write(f"{obj_name},{date}\n")
        self.processed.add((obj_name, date))

    def _generate_plots(
        self, obj_dir: Path, date_folder: Path, RA: float, DEC: float, target_name: str
    ) -> None:
        """
        Generate and save Alt-Az and centroid movement plots for a given target/date.

        This hidden method:
            1. Reads the first TBL in `date_folder` containing time, JD, and centroid
                data
            2. Computes altitude and azimuth from the provided RA/DEC at each
                observation time.
            3. Creates a polar plot of azimuth vs. altitude.
            4. Computes XY centroid offsets relative to the first frame.
            5. Creates a plot of centroid movement vs. BJD_TDB.
            6. Saves both figures under
                `obj_dir/"plots"/date_folder.name/"AltAz_and_guiding"`.

            Args:
                obj_dir (Path): Path to the object directory
                date_folder (Path): Path to de specific date directory
                RA (float): Right ascension of the target in hours.
                DEC (flotat): Declination of the target in degrees.
                target_name (str): Identifier for plot titles and filenames.

            Returns:
                None
        """
        # Output dir
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
        Process a single object directory by generating plots for each observation date.

        This method:
            1. Finds an example FITS file under `obj_dir` to extract RA, DEC, and target
                name.
            2. Verifies that a ``measurements/`` subfolder exists.
            3. Iterate through each date subdolder under ``measurements/``:
            
                - Skips non-directory entries
                - Checks in (object, date) is already processed; if so, logs and
                    continues
                - Calls ``_generate_plot()`` to create Alt-Az and centroid movement
                    figures.
                -Records the processed (object, date) pair via ``_record_processed()``.
                
        Args:
            obj_dir (Path): Path to the object folder

        Returns:
            None
        """
        # Look a .fit file to take RA/DEC header keywords
        fits_files: list[Path] = list(obj_dir.rglob("*.fit"))

        # Waring for no FITS files
        if not fits_files:
            self.logger.warning(f"No FITS file in {obj_dir.name}. Skipping.")
            return
        example: Path = fits_files[0]  # Just one FITS to take the RA DEC
        with fits.open(example) as hdul:
            hdr = hdul[0].header  # type: ignore[attr-defined]
            RA = hdr.get("RA")
            DEC = hdr.get("DEC")
            target = hdr.get("OBJECT", obj_dir.name)

        # Measurements path
        measurements_root: Path = obj_dir / "measurements"
        if not measurements_root.exists():  # Verify measurements dir exists
            self.logger.warning(f"No measurements for {obj_dir.name}. Skipping.")
            return

        for date_folder in sorted(measurements_root.iterdir()):
            if not date_folder.is_dir():
                continue
            key: tuple[str, str] = (obj_dir.name, date_folder.name)
            if key in self.processed:  # Verify processed object and dates
                msg: str = f"{key} already processed"
                self.logger.info(msg)
                continue

            # Generate the plos
            self._generate_plots(obj_dir, date_folder, RA, DEC, target)
            self._record_processed(obj_dir.name, date_folder.name)

    def run(self) -> None:
        """
        Iterate over all objects in the organized data directory and generate plots.

        This method scans each subdirectory under `self.data_dir`. For every
        directory that contains a `measurements/` subfolder, it calls
        `self.process_object()` to produce the Alt-Az and centroid movement plots.

        Returns:
            None
        """
        for obj_dir in self.data_dir.iterdir():
            if not obj_dir.is_dir():
                continue
            if not (obj_dir / "measurements").exists():
                continue
            self.process_object(obj_dir)
