"""
Generate EXOFOP product plots for OPTICAM observations.

This module creates two products for each target and observation date:

    1) Aperture visualization: overlays source and sky annuli on a science image.
    2) Radial profile: plots the radial brightness profile (with errors).

Outputs are saved under `exofop/<DATE-OBS>/` inside each object directory.
Already-processed (object, date) pairs are detected by checking whether the
expected output PNG exists on disk, so no external state file is needed.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.patches import Circle, Wedge
from pandas import DataFrame
from photutils.profiles import RadialProfile


class ExofopPlotter:
    """
    Create aperture-visualization and radial-profile plots for EXOFOP.

    This class scans the `organized_data` tree, builds EXOFOP-ready plots per
    target and date. Already-processed pairs are detected by the presence of
    the output PNG.
    """

    def __init__(self) -> None:
        """
        Initialize directories and logger.

        Sets the base, organized-data, and corrected-data paths.
        """
        self.base_dir = Path.cwd()
        self.data_dir = self.base_dir / "organized_data"
        self.corrected = self.base_dir / "corrected_3x3"
        self.logger = logging.getLogger(__name__)

    def _is_processed(
        self, obj_dir: Path, date_name: str, target_name: str, band: str
    ) -> bool:
        """
        Check whether EXOFOP outputs already exist for (object, date, band).

        Args:
            obj_dir (Path): Path to the object directory.
            date_name (str): Observation date folder name.
            target_name (str): The object target name.
            band (str): The measurement band.

        Returns:
            bool: True if the aperture plot PNG already exists.
        """
        expected = (
            obj_dir
            / "exofop"
            / date_name
            / f"{target_name}_{date_name}_{band}_apertures.png"
        )
        return expected.exists()

    def _generate_plots(
        self,
        obj_dir: Path,
        date_folder: Path,
        target_name: str,
        file_to_read: Path,
        band: str,
    ) -> None:
        """
        Generate aperture-visualization and radial-profile plots for one date and band.

        Reads centroid and aperture parameters from `file_to_read`, selects a
        representative FITS image for background visualization, and produces:

            - Aperture visualization (source radius and sky annulus overlaid).
            - Radial profile with uncertainties and aperture markers.

        Plots are saved to `exofop/<DATE-OBS>/`.

        Args:
            obj_dir (Path): Path to the object directory.
            date_folder (Path): Path to the date-specific `measurements/` folder.
            target_name (str): Name used for plot titles and file naming.
            file_to_read (Path): Path to the correct band measurement file.
            band (str): The corresponding physical band.
        """
        exofop_dir: Path = obj_dir / "exofop" / date_folder.name
        exofop_dir.mkdir(parents=True, exist_ok=True)

        data: DataFrame
        if file_to_read.suffix == ".tbl":
            data = pd.read_csv(
                file_to_read, sep=r"\t+", engine="python", encoding="latin1"
            )
        else:
            data = pd.read_csv(file_to_read, encoding="latin1")

        fits_dir: Path = self.corrected / obj_dir.name / f"calibrated_{band}"
        fits_cands: list[Path] = list(fits_dir.rglob("*_out.fit*"))
        if not fits_cands:
            self.logger.warning(
                f"No calibrated FITS in {fits_dir}. Skipping aperture plot."
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
            if name == "T1":
                color = "limegreen"
            elif name.startswith("C"):
                color = "red"
            elif name.startswith("T") and name != "T1":
                color = "orange"
            else:
                color = "white"

            ax.add_patch(
                Circle(
                    (x, y),
                    radius=source,
                    edgecolor=color,
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
                    edgecolor=color,
                    linewidth=0.7,
                )
            )
        ax.set_title(f"{target_name} ({band}) Aperture Visualization")
        ap_path: Path = (
            exofop_dir / f"{target_name}_{date_folder.name}_{band}_apertures.png"
        )
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
        ax2.set_title(f"{target_name} ({band}) Radial Profile")
        ax2.legend()
        rp_path: Path = (
            exofop_dir / f"{target_name}_{date_folder.name}_{band}_radial_profile.png"
        )
        plt.savefig(rp_path, dpi=300)
        plt.close(fig2)
        self.logger.info(f"Saved radial profile plot at {rp_path}")

    def process_object(self, obj_dir: Path) -> None:
        """
        Process a single object by iterating over all observation dates.

        Finds a representative FITS to determine the target name, then for each date
        subfolder under `measurements/`:

            - Skip if output already exists.
            - Generate EXOFOP plots via `_generate_plots`.

        Args:
            obj_dir (Path): Path to the object directory within `organized_data`.
        """
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

            meas_files: list[Path] = [
                f
                for f in date_folder.iterdir()
                if f.is_file()
                and not f.name.startswith(".")
                and f.suffix in (".tbl", ".csv")
            ]
            if not meas_files:
                self.logger.info(f"No measurements in {date_folder.name}. Skipping.")
                continue

            for file_to_read in meas_files:
                band: str = file_to_read.stem.split("_")[-1]
                if self._is_processed(obj_dir, date_folder.name, target, band):
                    self.logger.info(
                        f"({obj_dir.name}, {date_folder.name}, {band}) already processed"
                    )
                    continue

                self._generate_plots(obj_dir, date_folder, target, file_to_read, band)

    def run(self) -> None:
        """
        Process all objects found in `organized_data`.

        Iterates over object directories that contain a `measurements/` subfolder and
        invokes `process_object` for each. Saves outputs under `exofop/<DATE-OBS>/`.
        """
        self.logger.info("Running Exofop Plotter")
        for obj_dir in self.data_dir.iterdir():
            if not obj_dir.is_dir():
                continue
            if not (obj_dir / "measurements").exists():
                continue
            self.process_object(self.corrected / obj_dir)
