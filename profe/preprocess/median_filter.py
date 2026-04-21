"""
Apply a 3x3-pixel median filter to raw science images.

As proposed by Paez et al. (2026), OPTICAM science images with exposure times
longer than 10 seconds require hot-pixel correction. This module implements
that correction using a median filter with a 3x3-pixel window.
"""

import datetime
import logging
import os
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import IMapIterator
from typing import Any

from astropy.io import fits
from scipy.ndimage import median_filter  # type: ignore[attr-defined]
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MedianFilter:
    """
    Apply a median filter to OPTICAM FITS images.

    This class implements a pixel-wise median filter with a configurable
    square window (default 3x3) as described by Paez et al. (in prep.). The
    filter removes hot pixels and mitigates correlated noise in OPTICAM
    science images.
    """

    def __init__(self, ws: int = 3, n_processes: int | None = None) -> None:
        """
        Initialize the median filter with directory paths and filter settings.

        Args:
            ws (int, optional): Size of the square median filter window (ws x ws).
            Defaults to 3.

        Attributes:
            base_dir (str): Base directory for input and output.
            data_dir (str): Directory containing organized FITS files to be
            processed.
            window_size (int): Dimension of the median filter window.
            extensions (tuple[str, ...]): Accepted FITS file extensions.
            logs (str): Directory for log and control files.
            n_processes (int): Number of processes to use.
        """
        self.base_dir = os.getcwd()
        self.data_dir = os.path.join(self.base_dir, "organized_data")
        self.window_size = ws
        self.extensions = (".fit", ".fits", ".FIT", ".FITS")
        self.logs = os.path.join(self.base_dir, "logs")

        if n_processes is None:
            self.n_processes = cpu_count()
        else:
            self.n_processes = n_processes

        logger.info("Starting to apply median filter")

    def _process_image(self, args: list) -> Any | None:
        """
        Apply a median filter to a single FITS image and save the result.

        Opens the input FITS file, applies a SciPy median filter of size
        `window_size` x `window_size` with reflective edges, and writes a new
        FITS file with the same header to the output path.

        Args:
            args (tuple):
                image_path (str): Path to the input FITS file.
                output_path (str): Path where the filtered FITS file will be saved.
                window_size (int): Side length (in pixels) of the square filter
                    window.

        Returns:
            Optional[str]: The original `image_path` if processing succeeded,
            otherwise None.
        """
        image_path, output_path, window_size = args

        try:
            already_filtered = False
            raw_history_base = (
                f"PROFE: raw image, {window_size}x{window_size} median copy exists"
            )

            with fits.open(image_path, mode="update") as hdul:
                header = hdul[0].header  # type: ignore[attr-defined]

                # Verify if history exists
                if "HISTORY" in header:
                    for hist in header["HISTORY"]:
                        if raw_history_base in str(hist):
                            already_filtered = True
                            break

                if already_filtered:
                    logger.info(f"{image_path}: Already filtered. Skipping...")
                    return image_path

                data = hdul[0].data  # type: ignore[attr-defined]

                # Sciypy median filter
                corrected_data = median_filter(data, size=window_size, mode="reflect")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                filtered_header = header.copy()
                current_time = datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S UT"
                )
                filtered_header.add_history(
                    f"{window_size}x{window_size} median filter applied by PROFE at {current_time}"
                )

                hdu: fits.PrimaryHDU = fits.PrimaryHDU(
                    data=corrected_data, header=filtered_header
                )
                hdu.writeto(output_path, overwrite=True)

                # Add history to the raw image (opened in mode='update')
                header.add_history(f"{raw_history_base} at {current_time}")

                logger.info(f"{image_path}: Corrected with median filter")
            return image_path
        except Exception as e:
            logger.warning(f"{image_path} skipped due to {e}")
            return None

    def apply_filter(self) -> None:
        """
        Apply the median filter to all unprocessed FITS files in parallel.

        Steps:
            1. Walk through `data_dir` to find FITS files matching `extensions`.
            2. For each file, construct the output path under
            `corrected_{window_size}x{window_size}`, preserving the subdirectory
            structure.
            3. Use multiprocessing with all available CPU cores to run `_process_image`
            in parallel, showing progress with `tqdm`.
            4. Log a summary of the operation.
        """
        args: list = []
        skipped_targets: set = set()

        for dirpath, _, filenames in os.walk(self.data_dir):
            parts = os.path.relpath(dirpath, self.data_dir).split(os.sep)
            if len(parts) < 1 or parts[0] == ".":
                continue

            target = parts[0]
            if target in skipped_targets:
                continue

            # Check if corrected folder exists for this target
            corrected_dir_name = f"corrected_{self.window_size}x{self.window_size}"
            corrected_path = os.path.join(self.data_dir, target, corrected_dir_name)

            if os.path.exists(corrected_path):
                logger.info(
                    f"Target '{target}' already has a {corrected_dir_name} folder. "
                    "Skipping median filter for this target."
                )
                skipped_targets.add(target)
                continue

            if len(parts) < 3 or parts[1] != "raw":
                continue

            for filename in filenames:
                if filename.startswith("._") or filename.startswith("."):
                    continue
                if filename.endswith(self.extensions):
                    image_path: str = os.path.join(dirpath, filename)

                    date_sub_path = os.path.join(*parts[2:])

                    output_folder: str = os.path.join(
                        self.data_dir,
                        target,
                        corrected_dir_name,
                        date_sub_path,
                    )
                    output_path: str = os.path.join(output_folder, filename)
                    args.append((image_path, output_path, self.window_size))

        with Pool(processes=self.n_processes) as pool:
            imap: IMapIterator = pool.imap_unordered(self._process_image, args)
            results: list = [
                r
                for r in tqdm(
                    imap,
                    total=len(args),
                    desc="Applying median filter",
                )
                if r is not None
            ]

        msg: str = f"{len(results)} files processed."
        logger.info(msg)
