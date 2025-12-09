"""
Organize and update headers of FITS files.

This module processes raw FITS files by moving them from the `data/` directory to the
`organized_data/` directory. It updates relevant FITS headers, skipping files that have
already been processed to avoid redundant work. Multiprocessing is used to maximize
performance, utilizing 100% of available CPU cores.
"""

import logging
import os
import shutil
from multiprocessing import Pool, cpu_count

# type: ignore[import-untyped]
from astropy.io import fits
from astropy.time import Time
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FitsProcessor:
    """
    Preprocess FITS by updating headers, organizing files, and generating summaries.

    This class provides methods to:

        1. Time updating:
            - Compute and insert the 'JD' and 'UTMIDDLE' based on existing 'UT' and 'EXPOSURE' header values.

        2. File organization:
            - Sort FITS files into subdirectories by 'OBJECT' and 'DATE-OBS' keywords.

        3. Summary report:
            - Create a `.dat` file with the number of images per target and date folder.

    Multiprocessing is used to maximize performance, and previously processed files
    are skipped to avoid redundant work.

    """

    def __init__(self, n_processes: int | None = None) -> None:
        """
        Initialize directory paths and load the set of already processed files.

        Attributes:
            base_dir (str): Root directory for input and output files.
            data_dir (str): Directory containing raw FITS files.
            output_dir (str): Directory for saving organized files.
            logs (str): Directory for log files, summaries, and processed file records.
            processed_list_path (str): Path to the file recording processed files.
            counts_file (str): Path to the summary file with image counts.
            extensions (tuple[str, ...]): Accepted FITS file extensions.
            processed (set[str]): Filenames of already processed files.
            n_processes (int): Number of processes to use.
        """
        self.base_dir = os.getcwd()
        self.data_dir = os.path.join(self.base_dir, "data")
        self.output_dir = os.path.join(self.base_dir, "organized_data")
        self.logs = os.path.join(self.base_dir, "logs")

        self.processed_list_path = os.path.join(self.logs, ".organized_files.dat")
        self.counts_file = os.path.join(self.logs, "images_summary.dat")
        self.extensions = (".fit", ".fits", ".FIT", ".FITS")
        
        if n_processes is None:
            self.n_processes = cpu_count()
        else:
            self.n_processes = n_processes

        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("Starting to Organize and update time")

        self.processed = self._load_processed_files()

    def _load_processed_files(self) -> set:
        """
        Load the set of filenames for already processed FITS files.

        Reads `self.processed_list_path` if it exists, stripping whitespace from each
        line and returning the result as a set. Returns an empty set if the file does
        not exist.

        Returns:
            set[str]: Filenames of processed FITS files.
        """
        if os.path.exists(self.processed_list_path):
            with open(self.processed_list_path, "r") as pf:
                return set(line.strip() for line in pf if line.strip())
        return set()

    def _gather_fits_files(self) -> list:
        """
        Recursively collect all FITS files under the data directory.

        Traverses `self.data_dir` and returns a list of file paths whose extensions
        match any in `self.extensions`.

        Returns:
            list[str]: Full paths to all found FITS files.
        """
        file_list: list = []
        for root, _, files in os.walk(self.data_dir):
            for name in files:
                if name.endswith(self.extensions):
                    file_list.append(os.path.join(root, name))
        return file_list

    def _add_jd_to_files(self, file: str) -> None:
        """
        Compute and add Julian Date and mid-exposure UT to a FITS header.

        Opens a FITS file in update mode, reads the 'UT' and 'EXPOSURE' keywords, and:

            1. Converts 'UT' (ISO format) to an astropy Time object.
            2. Calculates exposure duration in days.
            3. Computes start and mid-exposure Julian Dates.
            4. Inserts 'JD' (Julian Date at start) and 'UTMIDDLE' (ISO mid-exposure)
            into the header.
            5. Adds a HISTORY entry noting the update.
            6. Logs success or warnings.

        Args:
            file (str): Path to the FITS file.
        """
        try:
            with fits.open(file, mode="update") as image:
                header = image[0].header  # type: ignore[attr-defined]

                if "UT" not in header or "EXPOSURE" not in header:
                    logger.warning(f"{file}: Missing UT or EXPOSURE in header")
                    return

                date_ut = header["UT"]

                try:
                    t: Time = Time(date_ut, format="iso")
                except ValueError:
                    logger.warning(f'{file}: UT format is not format="iso"')
                    return

                texp = header["EXPOSURE"]
                texp_day = texp / 86400.0

                jdmid = t.jd + texp_day / 2.0
                time_jdmid: Time = Time(jdmid, format="jd")

                header["JD"] = (t.jd, "Julian Date at start of observation")
                header["UTMIDDLE"] = (time_jdmid.iso, "UT at middle of exposure")

                header.add_history("YGMC/SPA added JD and UTMIDDLE at mid-exposure")

            logger.info(f"{file}: JD updated")
        except Exception as e:
            logger.error(f"Error updating JD in {file}: {e}")

    def update_jd_headers(self) -> None:
        """
        Update 'JD' and 'UTMIDDLE' headers for all FITS files in parallel.

        Steps:

            1. Collect all FITS file paths.
            2. Skip if no files are found.
            3. Log file count and number of CPU cores used.
            4. Use multiprocessing to apply `_add_jd_to_files` to each file.
            5. Display progress with a tqdm progress bar.

        """
        file_list: list = self._gather_fits_files()
        if len(file_list) == 0:
            logger.error("No FITS files found for JD updating")
            return

        msg: str = f"{len(file_list)} FITS files. Updating with {self.n_processes} cores"
        logger.info(msg)

        with Pool(processes=self.n_processes) as pool:
            for _ in tqdm(
                pool.imap_unordered(self._add_jd_to_files, file_list),
                total=len(file_list),
                desc="Upating headers",
            ):
                pass

    def organize_files(self) -> None:
        """
        Organize FITS files into object/date folders and create subdirectories.

        Steps:
            1. Gather all FITS file paths.
            2. Skip already processed files.
            3. For each new file:
                a. Read 'OBJECT' and 'DATE-OBS' (fallback to 'IMGTYPE' if no 'OBJECT' ).
                b. Create `OBJECT/DATE` directory in `output_dir`.
                c. For non-calibration FITS, create `measurements/DATE`, `lcs/DATE`, and `exofop` subdirectories.
                d. Move the file to the target directory.
                e. Record it as processed.

            4. Log the total number of moved files.

        """
        file_list: list = self._gather_fits_files()
        new_count = 0

        for fits_file in file_list:
            if fits_file in self.processed:
                continue

            try:
                with fits.open(fits_file, mode="readonly") as hdul:
                    header = hdul[0].header  # type: ignore[attr-defined]

                    if header.get("OBJECT") == "":
                        obj = header.get("IMGTYPE")
                    else:
                        obj = header.get("OBJECT")

                    date_obs = header.get("DATE-OBS", "")

                date_folder = date_obs

                target_dir: str = os.path.join(self.output_dir, obj, date_folder)

                calibration_frames: set = {"flat", "flats", "dark", "darks"}

                os.makedirs(target_dir, exist_ok=True)

                if obj.lower() not in calibration_frames:
                    measurements: str = os.path.join(
                        self.output_dir, obj, "measurements"
                    )
                    lcs: str = os.path.join(self.output_dir, obj, "lcs")
                    exofop: str = os.path.join(self.output_dir, obj, "exofop")
                    os.makedirs(measurements, exist_ok=True)
                    os.makedirs(os.path.join(measurements, date_folder), exist_ok=True)

                    os.makedirs(exofop, exist_ok=True)
                    os.makedirs(lcs, exist_ok=True)
                    os.makedirs(os.path.join(lcs, date_folder), exist_ok=True)

                shutil.move(fits_file, target_dir)
                logger.info(f"Moved {fits_file} --> {target_dir}")

                with open(self.processed_list_path, "a") as pf:
                    pf.write(fits_file + "\n")

                self.processed.add(fits_file)
                new_count += 1

            except Exception as e:
                logger.error(f"Error organizing {fits_file}: {e}")

        msg: str = f"Moved {new_count} new files."
        logger.info(msg)

    def generate_counts(self) -> None:
        """
        Generate a summary of image counts per object and date.

        Scans `self.output_dir` for object/date folders and counts the number of
        files in each date directory. Writes the results to `self.counts_file` in
        columns:

            OBJECT   DATE    COUNT
        """
        with open(self.counts_file, "w") as cf:
            cf.write("OBJECT         DATE      COUNT\n")

            for obj_name in sorted(os.listdir(self.output_dir)):
                obj_dir: str = os.path.join(self.output_dir, obj_name)

                if not os.path.isdir(obj_dir):
                    continue

                for date_folder in sorted(os.listdir(obj_dir)):
                    date_dir: str = os.path.join(obj_dir, date_folder)

                    if not os.path.isdir(date_dir):
                        continue

                    count: int = len(
                        [
                            f
                            for f in os.listdir(date_dir)
                            if os.path.isfile(os.path.join(date_dir, f))
                        ]
                    )
                    cf.write(f"{obj_name: >15}{date_folder:>10}{count: >5}\n")

        msg: str = f"Image summary written in {self.counts_file}"
        logger.info(msg)
