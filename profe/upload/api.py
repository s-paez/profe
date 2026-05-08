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

    def upload_tarball(self, tar_path: Path) -> bool:
        """
        Logs into ExoFOP and uploads the specified tarball.
        Returns True if successful or already exists, False otherwise.
        """
        import requests
        from bs4 import BeautifulSoup
        
        creds = self.load_credentials()
        if not creds:
            return False
            
        user, password = creds
        
        login_url = "https://exofop.ipac.caltech.edu/tess/password_check.php"
        upload_url = "https://exofop.ipac.caltech.edu/tess/newbulk_upload.php"
        
        with requests.Session() as session:
            # Login payload
            login_data = {
                "user": user,
                "password": password,
                "submit": "Log In"
            }
            logger.info("Authenticating with ExoFOP...")
            try:
                res = session.post(login_url, data=login_data, timeout=30)
                res.raise_for_status()
            except Exception as e:
                logger.error(f"Network error during ExoFOP login: {e}")
                return False
                
            logger.info(f"Uploading {tar_path.name} to ExoFOP...")
            try:
                with open(tar_path, "rb") as f:
                    # Using 'fileToUpload' as expected by ExoFOP bulk upload forms
                    files = {"fileToUpload": (tar_path.name, f, "application/x-tar")}
                    upload_data = {"submit": "Upload File"}
                    res = session.post(upload_url, files=files, data=upload_data, timeout=120)
                    res.raise_for_status()
            except Exception as e:
                logger.error(f"Network error during file upload: {e}")
                return False
                
            # Parse HTML response
            soup = BeautifulSoup(res.text, "html.parser")
            text_content = soup.get_text().lower()
            
            if "already exists" in text_content:
                logger.warning(f"ExoFOP reported that {tar_path.name} or its contents already exist.")
                return True
            elif "error" in text_content or "failed" in text_content or "not authorized" in text_content:
                logger.error("ExoFOP reported an error during upload. Please check the credentials and website.")
                # We can print the text for debugging
                logger.debug(f"Response text: {soup.get_text(strip=True)[:500]}")
                return False
            else:
                logger.info(f"ExoFOP accepted {tar_path.name} successfully.")
                return True
