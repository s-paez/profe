import logging
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class ExoFOPClient:
    """Handles communication with the ExoFOP-TESS website."""

    def __init__(self):
        self.base_dir = Path.cwd()
        self.creds_file = self.base_dir / ".exofop_credentials"

    def load_credentials(self) -> Optional[Tuple[str, str]]:
        """
        Loads username and password from .exofop_credentials file.
        Format inside the file must be 'user:password'
        """
        if not self.creds_file.exists():
            logger.error(
                f"Credentials file {self.creds_file} not found. "
                "Please create it with format 'user:password'."
            )
            return None

        try:
            with open(self.creds_file, "r") as f:
                line = f.readline().strip()
                if ":" in line:
                    user, password = line.split(":", 1)
                    return user, password
                else:
                    logger.error(
                        f"Invalid format in {self.creds_file}. Expected 'user:password'."
                    )
                    return None
        except Exception as e:
            logger.error(f"Error reading credentials file: {e}")
            return None
