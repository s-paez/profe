"""
Generate radial profile plots (seeing profile) for EXOFOP submission.

This module computes and plots the radial brightness profile of the target star
using the photutils RadialProfile utility.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from pandas import DataFrame
from photutils.profiles import RadialProfile

from .naming import exofop_path, exofop_title, get_exofop_id


class SeeingProfilePlotter:
    """
    Generate radial profile plots for EXOFOP.
    """

    def __init__(self) -> None:
        self.base_dir = Path.cwd()
        self.data_dir = self.base_dir / "organized_data"
        self.corrected = self.base_dir / "corrected_3x3"
        self.logger = logging.getLogger(__name__)

    def _is_processed(
        self, obj_dir: Path, date_name: str, target_name: str, band: str
    ) -> bool:
        """Check if _seeing-profile.png already exists."""
        exofop_obj = get_exofop_id(target_name)
        expected = exofop_path(
            obj_dir, date_name, exofop_obj, band, "_seeing-profile", ".png"
        )
        return expected.exists()

    def _generate_plot(
        self,
        obj_dir: Path,
        date_folder: Path,
        target_name: str,
        file_to_read: Path,
        band: str,
    ) -> None:
        """Generate the radial profile plot."""
        data: DataFrame
        if file_to_read.suffix == ".tbl":
            data = pd.read_csv(
                file_to_read, sep=r"\t+", engine="python", encoding="latin1"
            )
        else:
            data = pd.read_csv(file_to_read, encoding="latin1")

        fits_dir: Path = (
            obj_dir / "corrected_3x3" / date_folder.name / f"calibrated_{band}"
        )
        fits_cands: list[Path] = list(fits_dir.rglob("*_out.fit*"))
        if not fits_cands:
            self.logger.warning(
                f"No calibrated FITS in {fits_dir}. Skipping radial profile."
            )
            return

        with fits.open(fits_cands[0]) as hdul:
            vis_data = hdul[0].data.astype(float)  # type: ignore[missing-attribute]

        # Extract star centroids
        cent: dict = {}
        for col in data.columns:
            if col.startswith("X(FITS)") or col.startswith("Y(FITS)"):
                cent[col] = int(data[col].iloc[0])
        names: set = {c.split("_")[1] for c in cent if "_" in c}
        stars: dict = {
            name: (cent[f"X(FITS)_{name}"], cent[f"Y(FITS)_{name}"]) for name in names
        }

        # Photometry parameters
        exofop_obj = get_exofop_id(target_name)
        title_str = exofop_title(exofop_obj, date_folder.name, band)
        source: int = int(data["Source_Radius"].iloc[0])
        sky_min: float = float(data["Sky_Rad(min)"].iloc[0])
        sky_max: float = float(data["Sky_Rad(max)"].iloc[0])
        radius: np.ndarray = np.arange(int(sky_max) + 1)

        # Radial profile (choose target or first star)
        center_key: str = target_name if target_name in stars else next(iter(stars))
        rp: RadialProfile = RadialProfile(vis_data, stars[center_key], radius)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        rp.plot(ax=ax, c="m")
        rp.plot_error(ax=ax, c="m")
        rad: Any = np.asanyarray(rp.data_radius)
        d_profile: Any = np.asanyarray(rp.data_profile)
        ax.scatter(rad, d_profile, s=1, alpha=0.5, rasterized=True)
        ax.axvline(source, color="red", ls=":", label="Source Radius")
        ax.axvline(sky_min, color="red", linestyle="--", label="Sky Min")
        ax.axvline(sky_max, color="red", label="Sky Max")
        ax.set_xlabel("Radius [pixels]")
        ax.set_ylabel("Counts")
        ax.set_title(title_str, fontsize=9)
        ax.legend()

        rp_path: Path = exofop_path(
            obj_dir, date_folder.name, exofop_obj, band, "_seeing-profile", ".png"
        )
        rp_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(rp_path, dpi=300)
        plt.close(fig)
        self.logger.info(f"Saved radial profile plot at {rp_path}")

    def process_object(self, obj_dir: Path) -> None:
        """Process one object directory."""
        fits_files: list[Path] = [
            p for ext in ("*.fits", "*.fit") for p in obj_dir.rglob(ext)
        ]
        if not fits_files:
            return
        with fits.open(fits_files[0]) as hdul:
            hdr = hdul[0].header  # type: ignore[missing-attribute]
            target = hdr.get("OBJECT", obj_dir.name)

        measurements_root: Path = obj_dir / "measurements"
        if not measurements_root.exists():
            return

        for date_folder in sorted(measurements_root.iterdir()):
            if not date_folder.is_dir():
                continue

            meas_files: list[Path] = [
                f
                for f in date_folder.iterdir()
                if f.is_file()
                and not f.name.startswith(".")
                and f.suffix in (".tbl", ".csv")
            ]
            for file_to_read in meas_files:
                band: str = file_to_read.stem.split("_")[-1]
                if self._is_processed(obj_dir, date_folder.name, target, band):
                    continue
                self._generate_plot(obj_dir, date_folder, target, file_to_read, band)

    def run(self) -> None:
        """Process all objects in organized_data."""
        self.logger.info("Running Seeing Profile Plotter")
        for obj_dir in self.data_dir.iterdir():
            if not obj_dir.is_dir() or not (obj_dir / "measurements").exists():
                continue
            self.process_object(self.corrected / obj_dir)
