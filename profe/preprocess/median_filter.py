"""
This module implements the 3x3-pixel window size median filter to each raw image

As proposed by Paez et al.(2025) OPTICAM science images with exposures longer than
10s need a hot-pixel correction. This correction is performed by a 3x3-box size
median filter.
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
    Appy a 3x3-pixel window median filfer to OPTICAM FITS images.

    Implements the pixel-wise median filter with a 3x3 window as describred by
    Paez et al. (2025). This filter removes hot pixels and their induced correlated
    noise.

    Methods:
        apply_filter():
        Iterate over each FITS in `data_dir`, appy the 3x3-pixel window median
        filter, write the corrected image to `output_dir`, and record the filename in
        `processed` to prevent duplicated work.
    """

    def __init__(self, ws: int = 3) -> None:
        """
        Initizalize the median filter with directory paths and filter settings.

        Args:
            base_dir (str, optional): Root directory for data and logs. If None, uses
                the current working directory. None by default.
            ws (int): Size of the square median filter window (ws x ws). 3 by default.

        Attributes:
        base_dir (str): Base directory for input and output
        data_dir (str): Directory containing organized FITS files to apply the median
            filter.
        window_size (int): Dimension of the median filter window.
        extensions (tuple[str, ...]): Allowed FITS file extensions.
        logs (str): Directory where log and control files are stored.
        processed_list_path (str): Path to the hidden file tracking which FITS files
            have already been corrected with the median filter.
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
        Process one FITS by applying a median filter and saving the corrected file.

        This method unpacks a tuple of (image_path, output_path, window_size), opens the
        FITS file at `image_path`, applies a SciPy median filter of size
        `window_size` x `window_size` with reflective edges, and writes out a new FITS
        file preserbing the original header to `output_path`. Any errors are printed and
        logged without interrupting the pipeline.
        Function to processes each image with the median filter

        Args:
            args (tuple):
            image_path (str): Path to the input FITS file
            output_path (str): Path where the filtered FITS file will be saved.
            window_size (int): Side length (in pixels) of the square filter window

        Retunrns:
            Optional[str]:
                The original `image_path` if processing succeeded;  `None` if an error
                ocurred.
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
        Apply a median filter to all new FITS images in parallel and track the progress.

        This method performs the following steps:
            1. Loads the set of already processed file paths in
                `self.processed_list_path`.
            2. Walks through `self.data_dir` to find FITS files matching
                `self.extensions`.
            3. Skips files already recorded as processed
            4. For each new file, constructs an outputh path under
                `corrected_{window_size}x{window_size}` while preserving subdirectory
            structure.
            5. Uses a multiprocessing Pool with all available CPU cores to apply
                `_process_image` to each file, displaying progress via `tqdm`.
            6. Appends succesfully processed file paths to `self.processed_list_path`.
            7. Logs and prints a summary of the operation.

        Returns:
            None
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
