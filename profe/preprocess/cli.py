"""
Main function for the preprocess module.

This module also sets the log file.

The logical order of execution is required.
"""

import logging
import os

# Import the processing classes
from .fits_processor import FitsProcessor
from .median_filter import MedianFilter

# Make the logs dir
os.makedirs("logs", exist_ok=True)
# Config thr logging
logging.basicConfig(
    filename="logs/profe_pre.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)

logger.info("Running PROFE-prepocess")


def main() -> None:
    """
    Executes the preprocessing pipeline.

    The preprocessing involves updating FITS headers, organizing files,
    generating summary files, and applying a median filter.

    Steps:

    1. Initialize a FitsProcessor object.
    2. Update the Julian Date headers.
    3. Organize the files into the correct directory structure.
    4. Generate the `summary.dat` file with counts.
    5. Initialize a MedianFilter object.
    6. Apply the median filter with a 3x3-pixel window size.

    """
    org: FitsProcessor = FitsProcessor()
    org.update_jd_headers()
    org.organize_files()
    org.generate_counts()

    mf: MedianFilter = MedianFilter()
    mf.apply_filter()
