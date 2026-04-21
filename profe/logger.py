"""
Centralized logging configuration for the PROFE pipeline.
"""

import logging
import os
from datetime import datetime


def setup_logging(command_name: str) -> None:
    """
    Configures logging for the PROFE pipeline.

    Creates a new log file in the 'logs' directory with a timestamp in the filename.

    Args:
        command_name: The name of the command being executed (e.g., 'preprocess', 'output').
    """
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/profe_{command_name}_{timestamp}.log"

    # Configure logging
    # Use force=True to ensure this configuration is applied even if logging
    # was already configured elsewhere (though we aim to remove those).
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(),
        ],
        force=True,
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_filename}")
