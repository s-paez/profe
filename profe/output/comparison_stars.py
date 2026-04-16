"""
Comparison star light curve plotting for AIJ measurements.

This module generates plots showing the relative flux of comparison stars
(C-labeled stars) from AstroImageJ (AIJ) measurement tables (``.tbl``/``.csv``).
For each target, observation date, and photometric band it produces a 6-panel
plot mirroring the standard light curve layout:

    - Panel 0: Comparison star light curves with vertical offsets and 10-min bins.
    - Panel 1: AIRMASS.
    - Panel 2: Width_T1 (PSF width in pixels).
    - Panel 3: Total comparison counts.
    - Panel 4: Sky/Pixel for the target.
    - Panel 5: Centroid shift (X/Y).

Outputs are saved as PNG (DPI=300) in the ``exofop/<DATE>/`` directory.
Already-processed (object, date, band) triples are detected by checking whether
the expected output PNG exists on disk.
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame, Series

from .naming import exofop_path, exofop_title, get_exofop_id, normalize_band


class ComparisonStarsPlotter:
    """
    Generate comparison star light curve plots from AIJ measurements.

    For each target, date, and band in ``organized_data``, this class:
        - Loads ``.tbl`` measurement files from ``measurements/``.
        - Identifies comparison star columns (``rel_flux_Cn``).
        - Creates 6-panel plots with offset comparison star curves.
        - Saves outputs as PNG (DPI=300) to ``exofop/<date>/``.

    Already-processed triples are detected by the presence of the output PNG.
    """

    def __init__(self, bin_minutes: int = 10) -> None:
        """
        Initialize the comparison stars plotter.

        Args:
            bin_minutes (int, optional): Time bin size for plots in minutes.
                Defaults to 10.
        """
        self.base_dir = Path.cwd()
        self.logs_dir = self.base_dir / "logs"
        self.data_dir = self.base_dir / "organized_data"
        os.makedirs(self.logs_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.bin_minutes = bin_minutes

    def _is_processed(self, obj_folder: Path, obj: str, date: str, band: str) -> bool:
        """
        Check whether the comparison stars plot already exists.

        Args:
            obj_folder (Path): Path to the object directory.
            obj (str): Target object name.
            date (str): Observation date.
            band (str): Photometric band.

        Returns:
            bool: True if the output PNG already exists.
        """
        exofop_obj = get_exofop_id(obj)
        expected = exofop_path(
            obj_folder, date, exofop_obj, band, "_compstar-lightcurves", ".png"
        )
        return expected.exists()

    def _get_comp_star_ids(self, df: DataFrame) -> list[str]:
        """
        Extract comparison star IDs from DataFrame columns.

        Scans for columns matching the ``rel_flux_C<n>`` pattern.
        Returns up to 6 IDs sorted by their numeric index.

        Args:
            df (DataFrame): Measurement table with AIJ columns.

        Returns:
            list[str]: Up to 6 comparison star IDs, e.g. ``['C5', 'C7', ...]``.
        """
        comp_ids: list[str] = []
        for col in df.columns:
            m = re.match(r"^rel_flux_(C\d+)$", col)
            if m:
                comp_ids.append(m.group(1))
        comp_ids.sort(key=lambda x: int(x[1:]))
        return comp_ids[:6]

    def _calc_rms_ppt(self, data: Series) -> float:
        """Calculate RMS in parts per thousand."""
        clean_data = data[~np.isnan(data)]
        if len(clean_data) == 0:
            return np.nan
        return float(np.std(clean_data) / np.abs(np.median(clean_data)) * 1000)

    @staticmethod
    def _sigma_clip(data: NDArray, sigma: float = 3.0) -> NDArray:
        """Replace outliers beyond +-sigma*std from the median with NaN.

        Args:
            data: Input array (may already contain NaNs).
            sigma: Number of standard deviations for the clipping threshold.

        Returns:
            A copy of *data* with outliers set to NaN.
        """
        clipped = data.copy()
        med = np.nanmedian(clipped)
        std = np.nanstd(clipped)
        mask = np.abs(clipped - med) > sigma * std
        clipped[mask] = np.nan
        return clipped

    def _calc_rms_in_intervals(
        self, time: NDArray, data: NDArray, times_df: Optional[DataFrame]
    ) -> float:
        """Calculate RMS within defined time intervals, or over all data."""
        if times_df is not None and not times_df.empty:
            mask = np.zeros(len(time), dtype=bool)
            for i, f in zip(times_df["init_time"], times_df["final_time"]):
                mask |= (time >= i) & (time <= f)
            selected = data[mask]
            if len(selected) == 0:
                selected = data
        else:
            selected = data
        return self._calc_rms_ppt(pd.Series(selected))

    def _bin_data(
        self,
        time: NDArray,
        col: NDArray,
        err_col: NDArray,
        bin_width_minutes: float,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Bin time-series data into fixed-width time bins."""
        bin_width_days: float = bin_width_minutes / (24 * 60)
        bins = np.arange(time.min(), time.max() + bin_width_days, bin_width_days)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        bin_vals = np.zeros(len(bin_centers))
        bin_errs = np.zeros(len(bin_centers))

        for j in range(len(bin_centers)):
            mask = (time >= bins[j]) & (time < bins[j + 1])
            n_pts: int = np.sum(mask)
            if n_pts > 0:
                bin_vals[j] = np.nanmean(col[mask])
                bin_errs[j] = np.nanstd(col[mask]) / np.sqrt(n_pts)
            else:
                bin_vals[j] = np.nan
                bin_errs[j] = np.nan

        return bin_centers, bin_vals, bin_errs

    def _load_times(self, folder: Path) -> DataFrame | None:
        """Load optional time-interval definitions for RMS calculation."""
        tf: Path = folder / "times" / "times.csv"
        if tf.exists():
            return pd.read_csv(tf)
        return None

    def _obtain_measurements(self, folder: Path) -> list[Path]:
        """List all AIJ measurement table files in a folder (.tbl or .csv)."""
        return [
            f
            for f in folder.iterdir()
            if f.is_file()
            and not f.name.startswith(".")
            and f.suffix in (".tbl", ".csv")
        ]

    def _create_comparison_plot(
        self,
        obj: str,
        date: str,
        band: str,
        df: DataFrame,
        times_df: Optional[DataFrame],
        obj_folder: Path,
    ) -> None:
        """
        Create a 6-panel comparison star plot for one band.

        Panel 0 shows up to 6 comparison star light curves with vertical
        offsets (auto-calculated from each curve's own dispersion after
        sigma-clipping outliers) and 10-minute binned data, colored with
        the plasma colormap (0–70%).  Panels 1–5 mirror the standard
        multipanel diagnostic layout.

        Args:
            obj (str): Target object name.
            date (str): Observation date.
            band (str): Photometric band identifier.
            df (DataFrame): Measurement table for this band.
            times_df (Optional[DataFrame]): Time intervals for RMS calculation.
            exofop_dir (Path): Output directory for the PNG.
        """
        mpl.rcParams.update({"font.family": "serif", "font.size": 14})

        comp_ids = self._get_comp_star_ids(df)
        if not comp_ids:
            self.logger.warning(f"No comparison stars found for {obj}, {date}, {band}")
            return

        n_stars = len(comp_ids)
        cmap = mpl.colormaps["plasma"]
        colors = [cmap(i / max(n_stars - 1, 1) * 0.7) for i in range(n_stars)]

        time_col = "BJD_TDB"
        t0 = df[time_col].values[0]

        fig, axs = plt.subplots(
            6,
            1,
            figsize=(10, 17),
            sharex=True,
            gridspec_kw={
                "hspace": 0.0,
                "height_ratios": [1.5, 0.8, 0.8, 0.8, 0.8, 0.8],
            },
        )

        # ── Panel 0: Comparison star curves with offsets ─────────────────
        cumulative_offset: float = 0.0
        for i, comp_id in enumerate(comp_ids):
            flux_col = f"rel_flux_{comp_id}"
            err_col = f"rel_flux_err_{comp_id}"

            t = df[time_col].values - t0
            flux_vals = np.asarray(df[flux_col].values, dtype=float)
            err_vals = np.asarray(df[err_col].values, dtype=float)
            median_flux = np.nanmedian(flux_vals)
            f_raw = flux_vals / median_flux
            e_norm = err_vals / median_flux

            # Sigma-clip outliers (3σ)
            f_norm = self._sigma_clip(f_raw, sigma=3.0)

            # Offset proportional to this curve's own dispersion
            curve_std = np.nanstd(f_norm)
            if i > 0:
                cumulative_offset += 5 * curve_std
            offset = cumulative_offset

            axs[0].errorbar(
                t,
                f_norm - offset,
                yerr=e_norm,
                fmt=".",
                alpha=0.1,
                elinewidth=0.5,
                capsize=2,
                color=colors[i],
                label="_nolegend_",
            )

            t_bin, f_bin, e_bin = self._bin_data(t, f_norm, e_norm, self.bin_minutes)

            rms_data = self._calc_rms_in_intervals(t, f_norm, times_df)
            rms_bin = self._calc_rms_in_intervals(t_bin, f_bin, times_df)

            axs[0].errorbar(
                t_bin,
                f_bin - offset,
                yerr=e_bin,
                fmt="o",
                markerfacecolor="white",
                markeredgecolor=colors[i],
                color=colors[i],
                label=(
                    f"{comp_id} (RMS: {rms_data:.2f} ppt,"
                    f" {rms_bin:.2f} ppt/{self.bin_minutes}min)"
                ),
                capsize=2,
                zorder=20,
            )

        if times_df is not None and not times_df.empty:
            for init, final in zip(times_df["init_time"], times_df["final_time"]):
                axs[0].axvline(
                    init - t0,
                    color="gray",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.7,
                )
                axs[0].axvline(
                    final - t0,
                    color="gray",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.7,
                )

        axs[0].set_ylabel("Relative Flux + Offset")
        axs[0].grid(ls=":", zorder=0, alpha=0.5)
        axs[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
        exofop_obj = get_exofop_id(obj)
        title_str = exofop_title(exofop_obj, date, band)
        axs[0].set_title(title_str)

        # ── Panels 1–4: Diagnostic variables ─────────────────────────────
        panel_vars = [
            (1, "AIRMASS", "AIRMASS"),
            (2, "Width_T1", "Width_T1 [Pixels]"),
            (3, "tot_C_cnts", "Total Comp. Counts"),
            (4, "Sky/Pixel_T1", "Sky/Pixel_T1"),
        ]

        for idx, col, ylabel in panel_vars:
            t = df[time_col].values - t0
            y = df[col]
            color = "darkslateblue" if idx == 1 else "gray"
            axs[idx].plot(t, y, ".", alpha=0.5, label="_nolegend_", ms=4, color=color)
            axs[idx].grid(ls=":", zorder=0, alpha=0.5)
            axs[idx].set_ylabel(ylabel)

        # ── Panel 5: Centroid Shift ──────────────────────────────────────
        time_xy = df[time_col].values - t0
        x_fits = df["X(FITS)_T1"]
        x_rel = x_fits - x_fits.iloc[0]
        axs[5].plot(
            time_xy,
            x_rel,
            ".",
            color="m",
            alpha=0.6,
            label=r"X",
            ms=6,
            fillstyle="none",
        )

        y_fits = df["Y(FITS)_T1"]
        y_rel = y_fits - y_fits.iloc[0]
        axs[5].plot(
            time_xy,
            y_rel,
            "^",
            color="limegreen",
            alpha=0.6,
            label=r"Y",
            ms=3,
            fillstyle="none",
        )

        axs[5].set_ylabel(r"Centroid shift [pixels]")
        axs[5].set_xlabel(r"BJD$_{TDB}$ - " + f"{t0:.2f}")
        axs[5].legend(loc="best", fontsize=10)
        axs[5].grid(ls=":", alpha=0.5)

        # ── Save ─────────────────────────────────────────────────────────
        exofop_obj = get_exofop_id(obj)
        out_file: Path = exofop_path(
            obj_folder, date, exofop_obj, band, "_compstar-lightcurves", ".png"
        )
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Plot: {out_file}, saved")

    def run(self) -> None:
        """
        Process all objects and dates to generate comparison star plots.

        For each (object, date, band) whose output PNG does not yet exist:
            - Load the ``.tbl`` / ``.csv`` measurement file.
            - Identify comparison star columns.
            - Generate the 6-panel comparison star plot.
            - Save as PNG to ``exofop/<date>/``.
        """
        for obj_folder in sorted(self.data_dir.iterdir()):
            if not obj_folder.is_dir():
                continue

            obj: str = obj_folder.name
            meas_root: Path = obj_folder / "measurements"

            if not meas_root.exists():
                continue

            for date_folder in sorted(meas_root.iterdir()):
                if not date_folder.is_dir():
                    continue
                date: str = date_folder.name

                meas_files: list[Path] = self._obtain_measurements(date_folder)
                if not meas_files:
                    self.logger.warning(f"No measurements in {date_folder} — Skipping")
                    continue

                times_df: DataFrame | None = self._load_times(date_folder)

                for f in meas_files:
                    df: DataFrame
                    if f.suffix == ".tbl":
                        df = pd.read_csv(
                            f,
                            sep=r"\t+",
                            engine="python",
                            encoding="latin1",
                        )
                    elif f.suffix == ".csv":
                        df = pd.read_csv(f, encoding="latin1")
                    else:
                        continue

                    stem = f.stem
                    band = normalize_band(stem.split("_")[-1])

                    if self._is_processed(obj_folder, obj, date, band):
                        self.logger.info(
                            f"Skipping comp stars {obj},{date},{band}."
                            " Already processed"
                        )
                        continue

                    self._create_comparison_plot(
                        obj, date, band, df, times_df, obj_folder
                    )
