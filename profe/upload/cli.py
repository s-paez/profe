import logging
from typing import List
from pathlib import Path

from .packager import ExoFOPPackager
from .utils import UploadTracker
from .api import ExoFOPClient

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
        
    tracker = UploadTracker(Path.cwd() / "logs")
    client = ExoFOPClient()
    data = tracker.read_tracking()
    
    targets_to_process = [t.lower() for t in targets] if targets else list(data.keys())
    
    found_any = False
    for target in data:
        if targets and target.lower() not in targets_to_process:
            continue
            
        for date, info in data[target].items():
            if info.get("status") == "prepared":
                found_any = True
                tar_name = info.get("tar_file")
                if not tar_name:
                    logger.error(f"Missing tar_file name for {target} on {date}")
                    continue
                    
                tar_path = Path.cwd() / "tmp" / "exofop_uploads" / tar_name
                if not tar_path.exists():
                    logger.error(f"Prepared tarball not found: {tar_path}")
                    continue
                    
                success = client.upload_tarball(tar_path)
                if success:
                    tracker.update_status(target, date, "uploaded", tar_file=tar_name)
                    logger.info(f"Marked {target} on {date} as uploaded.")
                else:
                    logger.error(f"Failed to upload {tar_name}")

    if not found_any:
        logger.info("No prepared targets found to upload.")
