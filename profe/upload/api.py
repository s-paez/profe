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

    def upload_package(self, package_path: Path) -> bool:
        """
        Logs into ExoFOP and uploads each file listed in the package JSON individually
        via the Single File Upload endpoint (insert_file.php) to preserve original names.
        """
        import requests
        import json
        from bs4 import BeautifulSoup

        creds = self.load_credentials()
        if not creds:
            return False

        user, password = creds

        login_url = "https://exofop.ipac.caltech.edu/tess/password_check.php"
        upload_url = "https://exofop.ipac.caltech.edu/tess/insert_file.php"

        with requests.Session() as session:
            # Login payload
            login_data = {"username": user, "password": password, "ref": "login_user"}
            logger.info("Authenticating with ExoFOP...")
            try:
                res = session.post(login_url, data=login_data, timeout=30)
                res.raise_for_status()
            except Exception as e:
                logger.error(f"Network error during ExoFOP login: {e}")
                return False

            # Read package metadata
            if not package_path.exists():
                logger.error(f"Package file {package_path} not found.")
                return False

            with open(package_path, "r") as f:
                try:
                    metadata = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to read package {package_path}: {e}")
                    return False

            target_name = metadata.get("target_name")
            data_tag = metadata.get("data_tag")
            file_paths = metadata.get("files", [])

            # Derive TID and Planet for the form
            from profe.output.naming import get_tic_from_toi

            clean_target = target_name.upper().replace(".01", "").replace("-01", "")
            if clean_target.startswith("TOI"):
                tid = (
                    get_tic_from_toi(clean_target)
                    .replace("TIC", "")
                    .replace("-", "")
                    .strip()
                )
                planet_val = (
                    f"{clean_target.replace('-', ' ')}.01"  # e.g. 'TOI 7393.01'
                )
            else:
                tid = clean_target.replace("TIC", "").replace("-", "").strip()
                planet_val = "0"  # '0' means no planet assigned, just the star

            files_to_upload = [Path(p) for p in file_paths if Path(p).is_file()]
            
            logger.info(
                f"Found {len(files_to_upload)} files to upload individually."
            )

            all_success = True

            for file_path in files_to_upload:
                # Determine File Type and Description
                file_type = (
                    "Light_Curve"  # Per user request, always use Light_Curve
                )
                lname = file_path.name.lower()
                if "notes.txt" in lname:
                    file_desc = "Observing Notes"
                elif "field" in lname and lname.endswith((".png", ".pdf")):
                    file_desc = "Field of View"
                elif "lightcurve" in lname and lname.endswith((".png", ".pdf")):
                    file_desc = "Light Curve Plot"
                elif "profile" in lname and lname.endswith((".png", ".pdf")):
                    file_desc = "Seeing Profile"
                elif "comparison" in lname and lname.endswith((".png", ".pdf")):
                    file_desc = "Comparison Stars Light Curve Plot"
                elif lname.endswith(".fits"):
                    file_desc = "WCS FITS Image"
                elif lname.endswith((".csv", ".tbl")):
                    file_desc = "AstroImageJ full measurement table"
                else:
                    file_desc = "Data Product"

                upload_data = {
                    "tid": tid,
                    "planet": planet_val,
                    "file_type": file_type,
                    "file_desc": file_desc,
                    "file_tag": data_tag,
                    "groupname": "tfopwg",
                    "propflag": "on",  # Checkbox for 12 months proprietary
                }

                logger.info(f"Uploading {file_path.name} ({file_type})...")
                try:
                    with open(file_path, "rb") as f:
                        files_dict = {"file_name": (file_path.name, f)}
                        res = session.post(
                            upload_url,
                            data=upload_data,
                            files=files_dict,
                            timeout=60,
                        )
                        res.raise_for_status()
                except Exception as e:
                    logger.error(f"Network error uploading {file_path.name}: {e}")
                    all_success = False
                    continue

                soup = BeautifulSoup(res.text, "html.parser")
                text_content = soup.get_text().lower()

                if "already exists" in text_content:
                    logger.warning(
                        f"File {file_path.name} already exists on ExoFOP."
                    )
                elif (
                    "invalid" in text_content
                    or "error" in text_content
                    or "not authorized" in text_content
                ):
                    logger.error(f"ExoFOP rejected {file_path.name}.")
                    clean_text = soup.get_text(separator=" | ", strip=True)
                    logger.error(f"Response: {clean_text[:500]}...")
                    all_success = False
                else:
                    logger.info(f"Successfully uploaded {file_path.name}.")

            return all_success
