"""
Main() function for the calibration module

This module also set the log file.

The logical order is needed
"""

import logging
import os

# Import the processing classes
from .calibration import Calibration

# Make the logs dir
os.makedirs("logs", exist_ok=True)
# Config thr logging
logging.basicConfig(
    filename="logs/profe_cal.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)

logger.info("Running PROFE-calibration")


def main() -> None:
    """
    Main() function for the preprocessing

    Steps:
        1. Initialize Calibrartion() object
        2. Apply the run() method
            This method will:
            2.1. Load the images
            2.2. Apply the calibration
            2.3. Save the results
            2.6. Save the calibration log
    """
    red: Calibration = Calibration()  # Load the Calibration class
    red.run()  # Run the calibration process
