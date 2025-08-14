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
    square window (default 3×3) as described by Paez et al. (in prep.). The
    filter removes hot pixels and mitigates correlated noise in OPTICAM
    science images.

    Methods:
        apply_filter():
            Iterate over all eligible FITS files, apply the median filter,
            save the corrected images, and track processed files to avoid
            duplicate work.
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
        # Working dir to execute in the folder with the data
        self.base_dir = os.getcwd()
        # Dir with the organized data, sama name that used in the organizing script
        self.data_dir = os.path.join(self.base_dir, "organized_data")
        # The window size. It is in 3x3 as default but easily changed with `ws` param
        self.window_size = ws
        # Different FITS extensions
        self.extensions = (".fit", ".fits", ".FIT", ".FITS")
        # Logs dir to save al the control files
        self.logs = os.path.join(self.base_dir, "logs")
        # Path to the hiden corrected list of files, just for control
        self.processed_list_path = os.path.join(
            self.logs, f".corrected_files_{ws}x{ws}.dat"
        )
        # Log the starting point
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
        # Args
        image_path, output_path, window_size = args

        # Try environment to manage error without interrupting the process
        try:
            with fits.open(image_path) as hdul:  # Open the fits
                data = hdul[0].data  # type: ignore[attr-defined]
                header = hdul[0].header  # type: ignore[attr-defined]

                # Sciypy median filter
                corrected_data = median_filter(data, size=window_size, mode="reflect")
                # Create the output directory
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Create a new FITS with the same header but with corrected data
                hdu: fits.PrimaryHDU = fits.PrimaryHDU(
                    data=corrected_data, header=header
                )
                # Save the new FITS in the output data
                # Overwrite in case older corrected version of the frame
                hdu.writeto(output_path, overwrite=True)
                # Log the corrected file name
                logger.info(f"{image_path}: Corredted with median filter")
            return image_path  # So we know which one was processed

        except Exception as e:
            # Log the possible errora
            logger.warning(f"{image_path} skipped due to {e}")
            return None

    def apply_filter(self) -> None:
        """
        Apply the median filter to all unprocessed FITS files in parallel.

        Steps:
            1. Load the list of already processed files from `processed_list_path`.
            2. Walk through `data_dir` to find FITS files matching `extensions`.
            3. Skip files already processed.
            4. For each new file, construct the output path under
            `corrected_{window_size}x{window_size}`, preserving the subdirectory
            structure.
            5. Use multiprocessing with all available CPU cores to run
            `_process_image` in parallel, showing progress with `tqdm`.
            6. Append successfully processed file paths to `processed_list_path`.
            7. Log a summary of the operation.
        """
        # Verify if the processed_list exist
        processed_files: set
        if os.path.exists(self.processed_list_path):
            # If yes it open the file as a reader
            with open(self.processed_list_path, "r") as f:
                # Put each line a set() object to iterate
                processed_files = set(line.strip() for line in f)
        else:
            processed_files = set()

        args: list = []  # Arguments for the _process_image() fuction

        # Walk all the data_dir
        for dirpath, _, filenames in os.walk(self.data_dir):
            for filename in filenames:
                # Verify FITS extensions to avoid errors
                if filename.endswith(self.extensions):
                    # The complete path to each file
                    image_path: str = os.path.join(dirpath, filename)
                    if image_path in processed_files:
                        continue  # Verify if tt is already processed

                    # Relative path of the data folder
                    relative_path: str = os.path.relpath(dirpath, self.data_dir)
                    # The complete output folfer with the NxN window size
                    output_folder: str = os.path.join(
                        self.base_dir,
                        "corrected_" + f"{self.window_size}x{self.window_size}",
                        relative_path,
                    )
                    # Output path for each file
                    output_path: str = os.path.join(output_folder, filename)
                    # Arguments for the _process_image() function
                    args.append((image_path, output_path, self.window_size))

        # Apply median filter in parallel (with all available) CPU cores and show the
        # progress with tqdm
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

        # Save the results
        with open(self.processed_list_path, "a") as f:  # `appending mode` of the list
            for file_path in results:
                if file_path is not None:
                    f.write(file_path + "\n")

        # Log the final of the process saving the processed_files
        msg: str = f"{len(results)} processed files. List: {self.processed_list_path}"
        logger.info(msg)
