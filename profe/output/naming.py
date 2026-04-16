import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Cache for TOI -> TIC mappings to avoid redundant API calls
_TIC_CACHE: dict[str, str] = {}


def normalize_band(band: str) -> str:
    """Normalize band names to gp/rp/ip."""
    return {"g": "gp", "r": "rp", "i": "ip"}.get(band, band)


def get_tic_from_toi(toi_name: str) -> str:
    """
    Fetch TIC ID from ExoFOP API for a given TOI name.

    Logic:
    1. Parse "TOI-3884.01" -> "3884.01"
    2. Fetch CSV from ExoFOP
    3. Extract TIC ID

    Args:
        toi_name: TOI identifier (e.g., "TOI-3884.01", "TOI 3884.01")

    Returns:
        str: "TIC <id>" if found, otherwise the original toi_name.
    """
    if toi_name in _TIC_CACHE:
        return _TIC_CACHE[toi_name]

    # Parse numeric part
    # Support TOI-xxxx.yy, TOI xxxx.yy, or just xxxx.yy
    clean_toi = toi_name.upper().replace("TOI-", "").replace("TOI ", "").strip()

    try:
        url = f"https://exofop.ipac.caltech.edu/tess/download_toi.php?toi={clean_toi}&output=csv"
        df = pd.read_csv(url)

        if df.empty or "TIC ID" not in df.columns:
            logger.warning(f"No TIC ID found for {toi_name} in ExoFOP.")
            _TIC_CACHE[toi_name] = toi_name
            return toi_name

        tic_id = df["TIC ID"].iloc[0]
        result = f"TIC{tic_id}"
        _TIC_CACHE[toi_name] = result
        return result

    except Exception as e:
        logger.error(f"Error fetching TIC for {toi_name} from ExoFOP: {e}")
        _TIC_CACHE[toi_name] = toi_name
        return toi_name


def get_exofop_id(target: str) -> str:
    """
    Get the standardized ID for ExoFOP products.
    Converts TOIs to TIC IDs if possible.
    """
    if target.upper().startswith("TOI"):
        return get_tic_from_toi(target)
    return target


def exofop_title(target: str, date: str, band: str) -> str:
    """
    Build standardized plot title for ExoFOP products.

    Convention:
    target-01_yyyymmdd_OAN-SPM-2m1-OPTICAM_band

    Args:
        target: Target name (TIC or TOI).
        date: Date string in "YYYY-MM-DD" format.
        band: Filter band.

    Returns:
        str: Standardized title.
    """
    band_norm = normalize_band(band)
    date_compact = date.replace("-", "")
    return f"{target}-01_{date_compact}_OAN-SPM-2m1-OPTICAM_{band_norm}"


def exofop_path(
    obj_folder: Path,
    date: str,
    target: str,
    band: str,
    filetype: str,
    ext: str,
) -> Path:
    """
    Build standardized exofop output path.

    Convention:
    target-01_yyyymmdd_OAN-SPM-2m1-OPTICAM_band_filetype.ext

    Args:
        obj_folder: Base object folder path.
        date: Date string in "YYYY-MM-DD" format.
        target: Target name (e.g., "TOI-1234").
        band: Filter band (e.g., "g", "gp", "rp").
        filetype: Type of file (e.g., "_lightcurve", "_field").
        ext: File extension including dot (e.g., ".png").

    Returns:
        Path: Standardized path.
    """
    band_norm = normalize_band(band)
    date_compact = date.replace("-", "")
    name = f"{target}-01_{date_compact}_OAN-SPM-2m1-OPTICAM_{band_norm}{filetype}{ext}"
    return obj_folder / "exofop" / date / band_norm / name
