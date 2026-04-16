from pathlib import Path


def normalize_band(band: str) -> str:
    """Normalize band names to gp/rp/ip."""
    return {"g": "gp", "r": "rp", "i": "ip"}.get(band, band)


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
