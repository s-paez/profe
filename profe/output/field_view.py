"""
Generate aperture visualization plots (field view) for EXOFOP submission.

This module overlays source and sky apertures on a scientific calibrated image
to verify successful photometry.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from matplotlib.patches import Circle, Wedge
from pandas import DataFrame

from .naming import exofop_path, exofop_title, get_exofop_id


class FieldViewPlotter:
    """
    Generate field view plots (aperture visualization) for EXOFOP.
    """

    def __init__(self) -> None:
        self.base_dir = Path.cwd()
        self.data_dir = self.base_dir / "organized_data"
        self.corrected = self.base_dir / "corrected_3x3"
        self.logger = logging.getLogger(__name__)

    def _is_processed(
        self, obj_dir: Path, date_name: str, target_name: str, band: str
    ) -> bool:
        """Check if _field.png already exists."""
        exofop_obj = get_exofop_id(target_name)
        expected = exofop_path(obj_dir, date_name, exofop_obj, band, "_field", ".png")
        return expected.exists()

    def _generate_plot(
        self,
        obj_dir: Path,
        date_folder: Path,
        target_name: str,
        file_to_read: Path,
        band: str,
    ) -> None:
        """Generate the aperture visualization plot."""
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
                f"No calibrated FITS in {fits_dir}. Skipping field plot."
            )
            return

        selected_fits = None
        target_row = None

        if "Label" in data.columns:
            for cand in fits_cands:
                mask = data["Label"].astype(str).str.contains(cand.name, regex=False)
                if mask.any():
                    selected_fits = cand
                    target_row = data[mask].iloc[0]
                    break

        if selected_fits is None:
            self.logger.warning(
                f"Could not find a match in measurements for any FITS file in {fits_dir}. Using first FITS."
            )
            selected_fits = fits_cands[0]
            target_row = data.iloc[0]

        with fits.open(selected_fits) as hdul:
            vis_data = hdul[0].data.astype(float)  # type: ignore[missing-attribute]

        # Extract star centroids
        cent: dict = {}
        for col in data.columns:
            if col.startswith("X(FITS)") or col.startswith("Y(FITS)"):
                cent[col] = int(target_row[col])
        names: set = {c.split("_")[1] for c in cent if "_" in c}
        stars: dict = {
            name: (cent[f"X(FITS)_{name}"], cent[f"Y(FITS)_{name}"]) for name in names
        }

        # Photometry parameters
        exofop_obj = get_exofop_id(target_name)
        title_str = exofop_title(exofop_obj, date_folder.name, band)
        source: int = int(target_row["Source_Radius"])
        sky_min: float = float(target_row["Sky_Rad(min)"])
        sky_max: float = float(target_row["Sky_Rad(max)"])

        # Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(vis_data)

        ax.imshow(vis_data, vmin=vmin, vmax=vmax, origin="lower")
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
        ax.set_title(title_str, fontsize=9)
        ap_path: Path = exofop_path(
            obj_dir, date_folder.name, exofop_obj, band, "_field", ".png"
        )
        ap_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(ap_path, dpi=300)
        plt.close(fig)
        self.logger.info(f"Saved aperture plot at {ap_path}")

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
        self.logger.info("Running Field View Plotter")
        for obj_dir in self.data_dir.iterdir():
            if not obj_dir.is_dir() or not (obj_dir / "measurements").exists():
                continue
            self.process_object(self.corrected / obj_dir)
