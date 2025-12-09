"""
Main function for the preprocess module.

This module also sets the log file.

The logical order of execution is required.
"""

import argparse
import logging
import os

# Import the processing classes
from .fits_processor import FitsProcessor
from .median_filter import MedianFilter


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
    parser = argparse.ArgumentParser(description="PROFE Preprocessing Pipeline")
    parser.add_argument(
        "--cores",
        "-n",
        type=int,
        default=None,
        help="Number of CPU cores to use. Defaults to all available.",
    )
    args = parser.parse_args()

    # Prevent thread oversubscription by numpy/scipy in workers
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # Make the logs dir
    os.makedirs("logs", exist_ok=True)
    # Config thr logging
    logging.basicConfig(
        filename="logs/profe_pre.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    logger = logging.getLogger(__name__)

    logger.info(
        f"Running PROFE-prepocess with {args.cores if args.cores else 'all'} cores"
    )

    org: FitsProcessor = FitsProcessor(n_processes=args.cores)
    org.update_jd_headers()
    org.organize_files()
    org.generate_counts()

    mf = MedianFilter(
        n_processes=args.processes if hasattr(args, "processes") else args.cores
    )
    mf.apply_filter()


if __name__ == "__main__":
    main()
