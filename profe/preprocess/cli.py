"""
Main() function for the preprocess module

This module also set the log file.

The logical order is needed
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
    Main() function for the preprocessing

    Steps:
        1. Initialize FitsProcessor() object
        2. Apply the update_jd_headers
        3. Organize the files
        4. Genernate the summary.dat file
        5. Initialize MedianFilter() object
        6. Apply the median filter with the 3x3-pixel window size
    """
    org: FitsProcessor = FitsProcessor()  # Load the Organize_and_JD class
    org.update_jd_headers()  # Update the headers
    org.organize_files()  # Organize the files
    org.generate_counts()  # Create the summary files

    mf: MedianFilter = MedianFilter()  # Load thr MedianFilter class
    mf.apply_filter()  # Apply the filter
