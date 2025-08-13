"""
Exofop product plots generator

This module generates plots for Exofop data, including aperture visualizations
and radial profiles for each measurement. It saves these plots in the
exofop/<DATE-OBS>/ directory for each object, and logs processed (object, date) pairs
to avoid reprocessing.
"""

import logging
from pathlib import Path
from typing import Any, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.patches import Circle, Wedge
from pandas import DataFrame
from photutils.profiles import RadialProfile


class ExofopPlotter:
    """
    ExofopPlotter

    Save aperture visualization and radial profile plots for each measurement
    in the organized_data structure. Plots are stored under each object's
    exofop/<DATE-OBS>/ directory, and processed (object, date) pairs are
    logged in logs/.exofop_processed.dat to avoid reprocessing.
    """

    def __init__(self) -> None:
        """
        Initialize the ExofopPlotter.

        Sets up directories, logger, and processed file. It uses the current
        working directory as the base directory and organizes data under
        "organized_data". The logs are stored in "logs", and processed data
        is tracked in ".exofop_processed.dat" within the logs directory.

        """
        # Base and logs directories
        self.base_dir = Path.cwd()
        self.logs_dir = self.base_dir / "logs"
        self.data_dir = self.base_dir / "organized_data"
        self.corrected = self.base_dir / "corrected_3x3"

        # Processed-file setup
        self.processed_file = self.logs_dir / ".exofop_processed.dat"
        if not self.processed_file.exists():
            self.processed_file.write_text("")

        # Logger
        self.logger = logging.getLogger(__name__)
        self.processed = self._load_processed()

    def _load_processed(self) -> Set:
        """Read logs/.exofop_processed.dat and return a set of (object, date)."""
        processed: set = set()
        text: str = self.processed_file.read_text()
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts: list = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                processed.add((parts[0], parts[1]))
        return processed

    def _mark_processed(self, obj: str, date: str) -> None:
        """Append (object, date) to the processed-file and update in-memory set."""
        with open(self.processed_file, "a") as f:
            f.write(f"{obj},{date}\n")
        self.processed.add((obj, date))

    def _generate_plots(
        self, obj_dir: Path, date_folder: Path, target_name: str
    ) -> None:
        exofop_dir: Path = obj_dir / "exofop" / date_folder.name
        exofop_dir.mkdir(parents=True, exist_ok=True)

        tbl_files: list = list(date_folder.glob("*.tbl"))
        if not tbl_files:
            self.logger.info(f"No TBL in {date_folder}. Skipping.")
            return
        data: DataFrame = pd.read_table(tbl_files[0])

        fits_dir: Path = obj_dir / date_folder.name
        fits_cands: list = list(fits_dir.rglob("*.fit*"))
        if not fits_cands:
            self.logger.warning(f"No FITS in {fits_dir}. Skipping aperture plot.")
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
        source: int = int(data["Source_Radius"].iloc[0])
        sky_min: Any = data["Sky_Rad(min)"].iloc[0]
        sky_max: Any = data["Sky_Rad(max)"].iloc[0]
        radius: np.ndarray = np.arange(int(sky_max) + 1)

        # Radial profile (choose target or first star)
        center_key: str = target_name if target_name in stars else next(iter(stars))
        rp: RadialProfile = RadialProfile(vis_data, stars[center_key], radius)

        # Aperture visualization
        fig, ax = plt.subplots(figsize=(6, 6))
        vmin, vmax = np.nanpercentile(vis_data, (5, 95))
        ax.imshow(vis_data, vmin=vmin, vmax=vmax)
        for name, (x, y) in stars.items():
            ax.add_patch(
                Circle(
                    (x, y),
                    radius=source,
                    edgecolor="springgreen",
                    facecolor="none",
                    lw=1,
                )
            )
            ax.add_patch(
                Wedge(
                    (x, y),
                    sky_max,
                    theta1=0,
                    theta2=360,
                    width=sky_max - sky_min,
                    facecolor="none",
                    edgecolor="red",
                    linewidth=0.7,
                )
            )
        ax.set_title(f"{target_name} Aperture Visualization")
        ap_path: Path = exofop_dir / f"{date_folder.name}_aperture.png"
        plt.savefig(ap_path, dpi=300)
        plt.close(fig)
        self.logger.info(f"Saved aperture plot at {ap_path}")

        # Radial profile plot
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        rp.plot(ax=ax2)
        rp.plot_error(ax=ax2)
        rad: Any = np.asanyarray(rp.data_radius)
        d_profile: Any = np.asanyarray(rp.data_profile)
        ax2.scatter(rad, d_profile, s=1, alpha=0.5, rasterized=True)
        ax2.axvline(source, color="limegreen", label="Source Radius")
        ax2.axvline(sky_min, color="red", linestyle="--", label="Sky Min")
        ax2.axvline(sky_max, color="red", label="Sky Max")
        ax2.set_xlabel("Radius [pixels]")
        ax2.set_ylabel("Counts")
        ax2.set_title(f"{target_name} Radial Profile")
        ax2.legend()
        rp_path: Path = exofop_dir / f"{date_folder.name}_radial_profile.png"
        plt.savefig(rp_path, dpi=300)
        plt.close(fig2)
        self.logger.info(f"Saved radial profile plot at {rp_path}")

    def process_object(self, obj_dir: Path) -> None:
        """Process one object: find target name from a FITS header and iterate dates."""
        # Find example FITS for target name
        fits_files: list[Path] = [
            p for ext in ("*.fits", "*.fit") for p in obj_dir.rglob(ext)
        ]
        if not fits_files:
            self.logger.warning(f"No FIT file in {obj_dir.name}. Skipping.")
            return
        with fits.open(fits_files[0]) as hdul:
            hdr = hdul[0].header  # type: ignore[missing-attribute]
            target = hdr.get("OBJECT", obj_dir.name)

        measurements_root: Path = obj_dir / "measurements"
        if not measurements_root.exists():
            self.logger.warning(f"No measurements for {obj_dir.name}. Skipping.")
            return

        for date_folder in sorted(measurements_root.iterdir()):
            if not date_folder.is_dir():
                continue
            key: tuple = (obj_dir.name, date_folder.name)
            if key in self.processed:
                self.logger.info(f"{key} already processed")
                continue

            self._generate_plots(obj_dir, date_folder, target)
            self._mark_processed(obj_dir.name, date_folder.name)

    def run(self) -> None:
        """Iterate all object directories and process each."""
        self.logger.info("Running Exofop Plotter")
        for obj_dir in self.data_dir.iterdir():
            if not obj_dir.is_dir():
                continue
            if not (obj_dir / "measurements").exists():
                continue
            self.process_object(self.corrected / obj_dir)
