"""
calibration.py

Script to perform dark and flat calibration on median-filtered FITS data.

Directory structure assumed under working directory:

 corrected_3x3/                <- median-filtered data base
   ├── darks/                  <- median-filtered dark frames
   │     ├── <DATE>/
   │     └── ...
   ├── flats/                  <- median-filtered flat frames
   │     ├── <DATE>/
   │     └── ...
   └── <OBJECT>/               <- median-filtered science images
         ├── <DATE>/
         └── ...

Masters are saved alongside inputs:
 corrected_3x3/darks/<DATE>/master_dark_<filter>_<exp>s.fits
 corrected_3x3/flats/<DATE>/master_flat_<filter>.fits

Calibrated science images are stored under:
 corrected_3x3/<OBJECT>/<DATE>/calibrated/<filename>_out.fits

Logs and control files under:
 logs/calibration.log
 logs/calibrated_files.dat
 logs/flat_selection.csv  # CSV mapping science_date->flat_date
"""

import logging
import multiprocessing
from collections import defaultdict
from functools import partial
from logging import Logger
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import Angle, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time, TimeDelta
from tqdm import tqdm

logger: Logger = logging.getLogger(__name__)


class Calibration:
    """
    Class to handle dark and flat calibration of FITS images.

    It reads median-filtered data, creates master darks and flats,
    and calibrates science images using these masters.

    The results are saved in a structured directory under `corrected_3x3/`.
    The calibration process is logged, and a record of processed files is maintained.
    The flat selection mapping is stored in a CSV file for reference.
    The script is designed to run in parallel using multiple CPU cores for efficiency.
    It assumes a specific directory structure for input data and outputs.
    """

    def __init__(self) -> None:
        """
        Initialize the Calibration class.

        Sets up the base directory for data, logging configuration,and paths for
        control files. Initializes logging to capture calibration process details.
        Sets up paths for the base directory, logs directory, and control files for
        calibrated files and flat selection.
        The base directory is set to `corrected_3x3/`, and the logs directory is set to
        `logs/`.

        The control files include:
        - `calibrated_files.dat`: Records the files that have been calibrated.
        - `flat_selection.csv`: Maps science dates to flat dates for calibration.

        """
        # --- Configuration ---
        self.base_dir = Path.cwd() / "corrected_3x3"
        self.logs_dir = Path.cwd() / "logs"
        self.cal_file = self.logs_dir / "calibrated_files.dat"
        self.flat_file = self.logs_dir / "flat_selection.csv"
        # --- Setup Logging ---

    # --- Read Processed Science Files ---
    def _read_processed(self) -> set:
        if self.cal_file.exists():
            with open(self.cal_file) as f:
                return set(line.strip() for line in f if line.strip())
        return set()

    # --- Read Flat Selection CSV ---
    def read_flat_selection(self) -> dict:
        """
        Read the flat selection CSV file.

        Returns a dictionary mapping science dates to flat dates.
        If the file does not exist, returns an empty dictionary.
        """
        mapping: dict = {}
        df: pd.DataFrame = pd.read_csv(self.flat_file)
        for _, row in df.iterrows():
            mapping[row["science_date"]] = row["flat_date"]
        return mapping

    # Group Files by Filter (and Exposure for darks)
    def group_files(
        self,
        files: list,
        by_exp: bool = False,
    ) -> dict:
        """
        Group FITS files by filter and optionally by exposure time.

        Args:
            files (list): List of file paths to FITS files.
            by_exp (bool): If True, group by filter and exposure time;
                            if False, group only by filter.
        Returns:
            dict: A dictionary where keys are tuples of (filter, exposure) or just
                filter, and values are lists of file paths that match those criteria.
        """
        groups: dict = defaultdict(list)
        for fpath in files:
            try:
                hdr = fits.getheader(fpath)
                filt = hdr.get("FILTER", "").strip()
                if by_exp:
                    exp = hdr.get("EXPOSURE", None)
                    if exp is None:
                        logger.info(f"WARNING: EXPOSURE missing in {fpath}")
                        continue
                    exp_val: float = float(exp)
                    if exp_val.is_integer():
                        exp_str: str = str(int(exp_val))
                    else:
                        exp_str = str(exp_val)
                    key: tuple = (filt, exp_str)
                else:
                    key = filt
                groups[key].append(fpath)
            except Exception as e:
                logger.error(f"ERROR reading header {fpath}: {e}")
        return groups

    # Create Master Dark for a Given Date
    def make_master_dark(self, date: str) -> dict:
        """
        Create master dark frames for a given date.

        Args:
            date (str): The date for which to create master darks, in 'YYYY-MM-DD'
                format.
        Returns:
            dict: A dictionary mapping (filter, exposure) tuples to the paths of the
                created master dark files.

        If no dark frames are found for the date, returns an empty dictionary.
        If an error occurs during processing, logs the error and continues.
        The master darks are created by stacking the median of all dark frames for each
            filter and exposure time, and saved in the `dark` directory under the base
            directory.

        The output files are named `master_dark_<filter>_<exp>s.fits`, where
        `<filter>` is the filter name and `<exp>` is the exposure time in seconds.
        The header of the master dark includes the number of frames combined and a
        comment indicating that it was created by the PROFE pipeline.

        If the `dark` directory for the specified date does not exist, a warning is
            logged
        and an empty dictionary is returned.
        """
        dark_dir: Path = self.base_dir / "dark" / date
        masters: dict = {}
        if not dark_dir.exists():
            logger.warning(f"No darks for date {date}")
            return masters

        files: list = list(dark_dir.glob("*.fit*"))
        groups: dict = self.group_files(files, by_exp=True)
        for (filt, exp_s), flist in groups.items():
            try:
                # Stack data
                # type: ignore[no-matching-overload]
                stack = np.stack([fits.getdata(f) for f in flist])
                median_img: Any = np.median(stack, axis=0).astype(np.float32)

                # Copiar header del primer frame
                hdr0 = fits.getheader(flist[0])
                # Opcional: añadir info extra
                hdr0["NCOMB"] = (len(flist), "Number of frames combined")
                hdr0.add_comment("Master dark created by PROFE pipeline")

                # Crear HDU con header copiado
                hdu: fits.PrimaryHDU = fits.PrimaryHDU(data=median_img, header=hdr0)
                out: Path = dark_dir / f"master_dark_{filt}_{exp_s}s.fits"
                hdu.writeto(out, overwrite=True)

                logger.info(f"Master dark created: {out}")
                masters[(filt, exp_s)] = out
            except Exception as e:
                logger.error(f"Failed master dark {filt} {exp_s}s: {e}")
        return masters

    # Create Master Flat for a Given Date
    def make_master_flat(self, date: str) -> dict:
        """
        Create master flat frames for a given date.

        Args:
            date (str): The date for which to create master flats, in 'YYYY-MM-DD
                format.
        Returns:
            dict: A dictionary mapping filter names to the paths of the created master
                flat files.
        """
        flat_dir: Any = self.base_dir / "flat" / date
        masters: dict = {}
        if not flat_dir.exists():
            logger.warning(f"No flats for date {date}")
            return masters

        files: list = list(flat_dir.glob("*.fit*"))
        groups: dict = self.group_files(files, by_exp=False)
        for filt, flist in groups.items():
            try:
                # type: ignore[no-matching-overload]
                stack = np.stack([fits.getdata(f) for f in flist])
                median_img: Any = np.median(stack, axis=0).astype(np.float32)
                norm: Any = np.median(median_img)
                norm_img: Any = (median_img / norm).astype(np.float32)

                # Copiar header del primer frame
                hdr0 = fits.getheader(flist[0])
                hdr0["NCOMB"] = (len(flist), "Number of frames combined")
                hdr0["NORM"] = (norm, "Normalization factor")
                hdr0.add_comment("Master flat created by PROFE pipeline")

                # Crear HDU con header copiado
                hdu = fits.PrimaryHDU(data=norm_img, header=hdr0)
                out = flat_dir / f"master_flat_{filt}.fits"
                hdu.writeto(out, overwrite=True)

                logger.info(f"Master flat created: {out}")
                masters[filt] = out
            except Exception as e:
                logger.error(f"Failed master flat {filt}: {e}")
        return masters

    #  Parse DMS String to Angle
    def parse_angle(self, s: str) -> Angle:
        """
        Parse a DMS (Degrees, Minutes, Seconds) string to an Angle object.

        Args:
            s (str): The DMS string to parse, e.g., "12:34
            56.78" or "12 34 56.78".
        Returns:
            Angle: An Angle object representing the parsed DMS string in degrees.
        """
        return Angle(s, unit=u.deg)

    # Calibrate a Single Science Image
    def calibrate_image(
        self, fpath: Path, masters_dark: dict, masters_flat: dict, processed: set
    ) -> bool:
        """

        Calibrate a single science image using master dark and flat frames.

        Args:
            fpath (Path): Path to the FITS file to be calibrated.
            masters_dark (dict): Dictionary mapping (filter, exposure) tuples to master
                dark file paths.
            masters_flat (dict): Dictionary mapping filter names to master flat file
                paths.
            processed (set): Set of already processed file keys to avoid re-calibration.
        Returns:
            bool: True if the image was successfully calibrated, False otherwise.

        If the file has already been processed, it is skipped.
        If the master dark or flat for the image's filter and exposure is not found,
        a warning is logged and the image is skipped.
        The calibration formula applied is:
            calibrated_data = (data - dark_img) / flat_img
        The calibrated image is saved in a subdirectory named `calibrated` under the
        original file's directory, with the suffix `_out.fits`.
        The header of the calibrated image is updated with:
            - JD_SOBS: Julian Date at start of exposure
            - JD_UTC: Julian Date (UTC) at mid-exposure
            - HJD_UTC: Heliocentric Julian Date (UTC) at mid-exposure
            - BJD_TDB: Barycentric Julian Date (TDB) at mid-exposure
            - RAOBJ2K, DECOBJ2K: J2000 coordinates of the target
            - RA_OBJ, DEC_OBJ: EOD coordinates of the target
            - SITELAT, SITELONG: Observatory latitude and longitude
        The calibration process is logged, including the master dark and flat used.
        If an error occurs during calibration, it is logged, and the function returns
            False.
        If the file is successfully calibrated, it is recorded in the
            `calibrated_files.dat`
        file for future reference.
        """
        rel: Path = fpath.relative_to(self.base_dir)
        key: str = str(rel)
        if key in processed:
            logger.info(f"Already calibrated: {key}")
            return False
        try:
            hdr = fits.getheader(fpath)
            # type: ignore[missing-attribute]
            data = fits.getdata(fpath).astype(np.float32)
            filt = hdr.get("FILTER", "").strip()
            exp: float = float(hdr.get("EXPOSURE", 0))
            exp_s: str = str(int(exp)) if exp.is_integer() else str(exp)
            mdark: Any = masters_dark.get((filt, exp_s))
            if mdark is None:
                logger.warning(f"No master dark for {filt},{exp_s}s -> skip {rel}")
                return False
            mflat: Any = masters_flat.get(filt)
            if mflat is None:
                logger.warning(f"No master flat for {filt} -> skip {rel}")
                return False
            # type: ignore[missing-attribute]
            dark_img = fits.getdata(mdark).astype(np.float32)
            # type: ignore[missing-attribute]
            flat_img = fits.getdata(mflat).astype(np.float32)
            calib = (data - dark_img) / flat_img  # Calibration formula
            # Update headers
            hdr["JD_SOBS"] = (hdr.get("JD"), "Julian Date at start of exposure")
            # Mid-exposure times
            ut_mid = hdr.get("UTMIDDLE")
            if ut_mid:
                tmid: Time | Any = Time(f"{ut_mid}", format="iso", scale="utc")
                hdr["JD_UTC"] = (tmid.jd, "Julian Date (UTC) at mid-exposure")
                # Heliocentric
                lat: Angle = self.parse_angle(hdr.get("LATITUDE"))
                lon: Angle = self.parse_angle(hdr.get("LONGITUD"))
                alt = hdr.get("ALTITUDE", 0)
                loc = EarthLocation.from_geodetic(lat=lat, lon=lon, height=alt * u.m)
                sky: SkyCoord = SkyCoord(
                    ra=hdr.get("RA"), dec=hdr.get("DEC"), unit=(u.hourangle, u.deg)
                )
                ltt: TimeDelta | Any = tmid.light_travel_time(sky, location=loc)
                hdr["HJD_UTC"] = ((tmid.utc + ltt).jd, "Heliocentric JD (UTC)")
                # Barycentric
                hdr["BJD_TDB"] = ((tmid.tdb + ltt).jd, "Barycentric JD (TDB)")
            # Copy coords
            hdr["RAOBJ2K"] = (hdr.get("RA"), "J2000 RA of target (h)")
            hdr["DECOBJ2K"] = (hdr.get("DEC"), "J2000 DEC of target (deg)")
            hdr["RA_OBJ"] = (hdr.get("RA"), "EOD RA of target (h)")
            hdr["DEC_OBJ"] = (hdr.get("DEC"), "EOD DEC of target (deg)")
            hdr["SITELAT"] = (hdr.get("LATITUDE"), "Obs. latitude")
            hdr["SITELONG"] = (hdr.get("LONGITUD"), "Obs. longitude")
            hdr.add_history(f"Dark corrected with {Path(mdark).name}")
            hdr.add_history(f"Flat corrected with {Path(mflat).name}")
            # Write
            outdir: Path = fpath.parent / "calibrated"
            outdir.mkdir(exist_ok=True)
            out_file: Path = outdir / f"{fpath.stem}_out.fits"
            fits.PrimaryHDU(data=calib, header=hdr).writeto(out_file, overwrite=True)
            # Record
            with open(self.cal_file, "a") as cf:
                cf.write(key + "\n")
            logger.info(f"Calibrated {key} -> {out_file}")
            return True
        except Exception as e:
            logger.error(f"Error calibrating {rel}: {e}")
            return False

    # Main Routine
    def run(self) -> None:
        """
        Main routine to perform calibration of all science images.

        It reads the processed files, flat selection mapping, and creates master darks
        and flats. Then, it calibrates each science image in parallel using the master
        darks and flats.

        The calibration process is logged, and progress is displayed using tqdm.
        The function iterates through each science image directory, retrieves the
        corresponding master dark and flat for the date, and applies the calibration
        function in parallel using a multiprocessing pool.
        If a science image does not have a corresponding flat, it is skipped with a
        warning message.
        """
        processed: set = self._read_processed()
        flat_map: dict = self.read_flat_selection()

        # Create master darks and flats
        dark_masters: dict = {
            d.name: self.make_master_dark(d.name)
            for d in (self.base_dir / "dark").iterdir()
            if d.is_dir()
        }
        flat_masters: dict = {
            d.name: self.make_master_flat(d.name)
            for d in (self.base_dir / "flat").iterdir()
            if d.is_dir()
        }

        # Calibrate science images in parallel
        for obj in self.base_dir.iterdir():
            if obj.name in ("dark", "flat") or not obj.is_dir():
                continue

            for date_dir in obj.iterdir():
                if not date_dir.is_dir():
                    continue

                sci_date: str = date_dir.name
                flat_date: Any | None = flat_map.get(sci_date)
                if not flat_date:
                    logger.warning(f"No flat in date {sci_date}, Skipping {date_dir}")
                    continue

                masters_d: dict = dark_masters.get(sci_date, {})
                masters_f: dict = flat_masters.get(flat_date, {})
                files: list = list(date_dir.glob("*.fit*"))
                if not files:
                    continue

                calibrate_fn: partial[bool] = partial(
                    self.calibrate_image,
                    masters_dark=masters_d,
                    masters_flat=masters_f,
                    processed=processed,
                )

                nproc: int = multiprocessing.cpu_count()
                with multiprocessing.Pool(processes=nproc) as pool:
                    for _ in tqdm(
                        pool.imap(calibrate_fn, files),
                        total=len(files),
                        desc=f"Calibrating {obj.name}/{sci_date}",
                    ):
                        pass
