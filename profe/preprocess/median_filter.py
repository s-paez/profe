"""
Apply a 3x3-pixel median filter to raw science images.

As proposed by Paez et al. (2025), OPTICAM science images with exposure times
longer than 10 seconds require hot-pixel correction. This module implements
that correction using a median filter with a 3x3-pixel window.
"""

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

    def __init__(self, ws: int = 3) -> None:
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
            processed_list_path (str): Path to the hidden file that tracks which
            FITS files have already been median-filtered.
        """
        self.base_dir = os.getcwd()
        self.data_dir = os.path.join(self.base_dir, "organized_data")
        self.window_size = ws
        self.extensions = (".fit", ".fits", ".FIT", ".FITS")
        self.logs = os.path.join(self.base_dir, "logs")
        self.processed_list_path = os.path.join(
            self.logs, f".corrected_files_{ws}x{ws}.dat"
        )
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
            with fits.open(image_path) as hdul:
                data = hdul[0].data  # type: ignore[attr-defined]
                header = hdul[0].header  # type: ignore[attr-defined]

                # Sciypy median filter
                corrected_data = median_filter(data, size=window_size, mode="reflect")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                hdu: fits.PrimaryHDU = fits.PrimaryHDU(
                    data=corrected_data, header=header
                )
                hdu.writeto(output_path, overwrite=True)
                logger.info(f"{image_path}: Corredted with median filter")
            return image_path
        except Exception as e:
            logger.warning(f"{image_path} skipped due to {e}")
            return None

    def apply_filter(self) -> None:
        """
        Apply the median filter to all unprocessed FITS files in parallel.

        Steps:
            1. Load the list of already processed files from `processed_list_path`.
            2. Walk through `data_dir` to find FITS files matching `extensions`.
            3. Skip files already processed.
            4. For each new file, construct the output path under `corrected_{window_size}x{window_size}`, preserving the subdirectory structure.
            5. Use multiprocessing with all available CPU cores to run `_process_image` in parallel, showing progress with `tqdm`.
            6. Append successfully processed file paths to `processed_list_path`.
            7. Log a summary of the operation.
        """
        processed_files: set
        if os.path.exists(self.processed_list_path):
            with open(self.processed_list_path, "r") as f:
                processed_files = set(line.strip() for line in f)
        else:
            processed_files = set()

        args: list = []

        for dirpath, _, filenames in os.walk(self.data_dir):
            for filename in filenames:
                if filename.endswith(self.extensions):
                    image_path: str = os.path.join(dirpath, filename)
                    if image_path in processed_files:
                        continue

                    relative_path: str = os.path.relpath(dirpath, self.data_dir)
                    output_folder: str = os.path.join(
                        self.base_dir,
                        "corrected_" + f"{self.window_size}x{self.window_size}",
                        relative_path,
                    )
                    output_path: str = os.path.join(output_folder, filename)
                    args.append((image_path, output_path, self.window_size))

        with Pool(processes=cpu_count()) as pool:
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

        with open(self.processed_list_path, "a") as f:
            for file_path in results:
                if file_path is not None:
                    f.write(file_path + "\n")

        msg: str = f"{len(results)} processed files. List: {self.processed_list_path}"
        logger.info(msg)
