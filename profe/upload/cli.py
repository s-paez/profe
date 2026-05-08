import logging
from typing import List
from .packager import ExoFOPPackager

logger = logging.getLogger(__name__)

def run_prepare_upload(targets: List[str]) -> None:
    """
    Entry point for the --prepare-upload command.
    
    Args:
        targets: List of specific targets to process, or empty to process all.
    """
    logger.info("Starting ExoFOP upload preparation...")
    if targets:
        logger.info(f"Targets specified: {', '.join(targets)}")
    else:
        logger.info("No specific targets provided. Processing all pending targets.")
        
    packager = ExoFOPPackager(targets)
    packager.run_prepare()

def run_upload(targets: List[str]) -> None:
    """
    Entry point for the --upload command.
    
    Args:
        targets: List of specific targets to process, or empty to process all.
    """
    logger.info("Starting ExoFOP upload process...")
    if targets:
        logger.info(f"Targets specified: {', '.join(targets)}")
    else:
        logger.info("No specific targets provided. Uploading all prepared targets.")
        
    # TODO: Implement the upload logic (Step 7, 8)
    pass
