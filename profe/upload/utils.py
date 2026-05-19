import json
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class UploadTracker:
    """Manages the tracking of ExoFOP uploads to prevent duplicates."""

    def __init__(self, logs_dir: Path):
        self.tracker_file = logs_dir / "exofop_uploads.json"
        self._ensure_file()

    def _ensure_file(self) -> None:
        """Create the tracking file if it doesn't exist."""
        if not self.tracker_file.exists():
            self.tracker_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.tracker_file, "w") as f:
                json.dump({}, f, indent=4)

    def read_tracking(self) -> Dict[str, Any]:
        """Read the current tracking state."""
        try:
            with open(self.tracker_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading tracking file {self.tracker_file}: {e}")
            return {}

    def write_tracking(self, data: Dict[str, Any]) -> None:
        """Write the tracking state to the JSON file."""
        try:
            with open(self.tracker_file, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Error writing tracking file {self.tracker_file}: {e}")

    def update_status(
        self, target: str, date: str, status: str, package_file: str = None
    ) -> None:
        """
        Update the status of an upload for a specific target and date.

        Args:
            target: The target name or TIC ID (e.g., 'TOI-3884').
            date: The local observation date (e.g., '2025-02-23').
            status: The status string ('prepared', 'uploaded', etc.).
            package_file: The name of the prepared package JSON file.
        """
        data = self.read_tracking()

        if target not in data:
            data[target] = {}

        if date not in data[target]:
            data[target][date] = {}

        data[target][date]["status"] = status
        data[target][date]["timestamp"] = datetime.utcnow().isoformat() + "Z"

        if package_file is not None:
            data[target][date]["package_file"] = package_file

        self.write_tracking(data)

    def get_status(self, target: str, date: str) -> str:
        """Get the current upload status for a target and date."""
        data = self.read_tracking()
        return data.get(target, {}).get(date, {}).get("status", "pending")

    def get_package_file(self, target: str, date: str) -> str:
        """Get the associated package file name for a prepared target and date."""
        data = self.read_tracking()
        return data.get(target, {}).get(date, {}).get("package_file")
