"""
Light curve plotting and CSV export for AIJ measurements.

This module generates light curve plots and normalized multiband CSV files
from AstroImageJ (AIJ) measurement tables (`.tbl`). For each target and
observation date, it can optionally use a `times.csv` file to define time
intervals for normalization and RMS calculation.

Outputs include:
    - Plots binned to a fixed time resolution (default: 10 minutes).
    - Per-method plots and per-filter plots.
    - Normalized multiband CSV files containing flux and error columns.

Already-processed (object, date) pairs are detected by checking whether the
expected output PDF exists on disk, so no external state file is needed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

from .naming import (
    exofop_path,
    exofop_title,
    get_exofop_id,
    normalize_band,
    get_utc_date_from_bjd,
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame, Series


class LightCurvePlotter:
    """
    Generate light curve plots and normalized CSV files from AIJ measurements.

    For each target and date in `organized_data`, this class:
        - Loads `.tbl` measurement files from `measurements/`.
        - Optionally applies time-interval selections from `times.csv`.
        - Creates plots showing both raw and binned light curve data.
        - Computes and displays RMS values in plot legends.
        - Saves normalized multiband CSV files combining all filters.

    Already-processed pairs are detected by the presence of the output PDF.
    """

    def __init__(self, bin_minutes: int = 10) -> None:
        """
        Initialize the light curve plotter.

        Sets up working directories, logger, and binning configuration.

        Args:
            bin_minutes (int, optional): Time bin size for plots in minutes.
                Defaults to 10.
        """
        # Directories
        self.base_dir = Path.cwd()
        self.logs_dir = self.base_dir / "logs"
        self.data_dir = self.base_dir / "organized_data"
        os.makedirs(self.logs_dir, exist_ok=True)
        # Logs
        self.logger = logging.getLogger(__name__)
        # argument about mins to binning
        self.bin_minutes = bin_minutes

    def _plots_exist(self, obj_folder: Path, obj: str, date: str) -> bool:
        """
        Check whether the main plot outputs already exist for (object, date).

        Args:
            obj_folder (Path): Path to the object directory.
            obj (str): Target object name.
            date (str): Observation date in YYYY-MM-DD format.

        Returns:
            bool: True if the main output PDF already exists.
        """
        expected = obj_folder / "plots" / date / f"{obj}_{date}_PROFE_lc.pdf"
        return expected.exists()

    def _missing_exofop_bands(
        self, obj_folder: Path, obj: str, date: str, utc_date: str, bands: list[str]
    ) -> list[str]:
        """
        Return band names whose per-band exofop PNGs do not yet exist.

        Args:
            obj_folder (Path): Path to the object directory.
            obj (str): Target object name.
            date (str): Local observation date in YYYY-MM-DD format.
            utc_date (str): UTC observation date in YYYY-MM-DD format.
            bands (list[str]): Available photometric band names.

        Returns:
            list[str]: Bands whose exofop PNG is missing.
        """
        return [
            b
            for b in bands
            if not exofop_path(
                obj_folder, date, utc_date, obj, b, "_lightcurve", ".png"
            ).exists()
        ]

    def _load_times(self, folder: Path) -> DataFrame | None:
        """
        Load optional time-interval definitions for RMS calculation.

        Looks for `times/times.csv` under the given folder. If present, loads it
        into a DataFrame. If missing, creates an empty `times.csv` and returns None.

        Args:
            folder (Path): Path to the date-specific `measurements/` folder.

        Returns:
            Optional[DataFrame]: Time intervals with 'init_time' and 'final_time'
            columns, or None if not available.
        """
        tf: Path = folder / "times" / "times.csv"
        if tf.exists():
            return pd.read_csv(tf)
        else:
            self.logger.info("No current times.csv file. Creating an empty times.csv ")
            new_time: dict = {"init_time": [], "final_time": []}
            df_new_tf = pd.DataFrame(new_time)
            os.makedirs(folder / "times", exist_ok=True)
            df_new_tf.to_csv(folder / "times" / "times.csv", index=False)
            return None

    def _obtain_measurements(self, folder: Path) -> list[Path]:
        """
        List all AIJ measurement table files in a folder (.tbl or .csv).

        Args:
            folder (Path): Path to a `measurements/` directory.

        Returns:
            list[Path]: Paths to all `.tbl` and `.csv` files in the folder.
        """
        return [
            f
            for f in folder.iterdir()
            if f.is_file()
            and not f.name.startswith(".")
            and f.suffix in (".tbl", ".csv")
        ]

    def _calc_rms_ppt(self, data: Series) -> float:
        clean_data = data[~np.isnan(data)]
        if len(clean_data) == 0:
            return np.nan
        return float(np.std(clean_data) / np.abs(np.median(clean_data)) * 1000)

    def _calc_rms_ppt_abs(self, data: Series) -> float:
        """RMS in ppt for quantities already centered near zero (e.g. fit
        residuals), where normalizing by the median would blow up."""
        clean_data = data[~np.isnan(data)]
        if len(clean_data) == 0:
            return np.nan
        return float(np.std(clean_data) * 1000)

    def _calc_rms_in_intervals(
        self,
        time: NDArray,
        data: NDArray,
        times_df: Optional[DataFrame],
        absolute: bool = False,
    ) -> float:
        if times_df is not None and not times_df.empty:
            mask = np.zeros(len(time), dtype=bool)
            for i, f in zip(times_df["init_time"], times_df["final_time"]):
                mask |= (time >= i) & (time <= f)
            selected = data[mask]

            if len(selected) == 0:
                selected = data
        else:
            selected = data
        calc = self._calc_rms_ppt_abs if absolute else self._calc_rms_ppt
        return calc(pd.Series(selected))

    def _bin_data(
        self, time: NDArray, col: NDArray, err_col: NDArray, bin_width_minutes: float
    ) -> tuple[NDArray, NDArray, NDArray]:
        # Filter NaNs to avoid ValueError in np.arange and logic errors
        mask_nan = ~np.isnan(time) & ~np.isnan(col)
        time = time[mask_nan]
        col = col[mask_nan]
        err_col = err_col[mask_nan]

        if len(time) == 0:
            return np.array([]), np.array([]), np.array([])

        bin_width_days: float = bin_width_minutes / (24 * 60)
        t_min, t_max = np.min(time), np.max(time)

        # Ensure we have a valid range
        if t_min == t_max:
            return (
                np.array([t_min]),
                np.array([np.mean(col)]),
                np.array([np.mean(err_col)]),
            )

        bins = np.arange(t_min, t_max + bin_width_days, bin_width_days)
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

    def _get_band_color(self, band: str) -> Optional[str]:
        if band in ("gp", "g"):
            return "dodgerblue"
        elif band in ("rp", "r"):
            return "forestgreen"
        elif band in ("ip", "i"):
            return "red"
        return None

    def _load_transit_times(
        self, obj_folder: Path, obj: str, local_date: str, utc_date: str
    ) -> Optional[Dict[str, float]]:
        """
        Load the predicted ingress/egress BJD times from the `_transit_times.dat`
        file produced by `TransitDataManager`, if present.

        Only the first listed transit is used, matching the convention already
        used for the predicted metrics in `report_generator.py`.

        Args:
            obj_folder (Path): Path to the object directory.
            obj (str): Target object name.
            local_date (str): Local observation date folder name.
            utc_date (str): UTC observation date in YYYY-MM-DD format.

        Returns:
            Optional[dict[str, float]]: Dict with 'ingress' and 'egress' BJD
            values, or None if no transit data is available.
        """
        exofop_id = get_exofop_id(obj)
        date_compact = utc_date.replace("-", "")
        dat_file = (
            obj_folder
            / "exofop"
            / local_date
            / f"{exofop_id}-01_{date_compact}_OAN-SPM-2m1-OPTICAM_transit_times.dat"
        )
        if not dat_file.exists():
            return None
        try:
            df = pd.read_csv(dat_file, sep="\t")
            if df.empty:
                return None
            row = df.iloc[0]
            return {
                "ingress": float(row["Ingress(BJD)"]),
                "egress": float(row["Egress(BJD)"]),
            }
        except Exception as e:
            self.logger.warning(f"Could not read transit times from {dat_file}: {e}")
            return None

    def _add_transit_markers(
        self, axs, lc_ax, t0: float, transit: Optional[Dict[str, float]]
    ) -> None:
        """
        Draw dotted vertical lines for predicted ingress/egress on every panel,
        with "Expected Ingress"/"Expected Egress" annotations near the bottom
        of the light curve panel.

        Args:
            axs: Iterable of all panel axes in the figure.
            lc_ax: The light curve (flux) panel axes, where annotations go.
            t0 (float): Time origin subtracted from the BJD times in the plot.
            transit (Optional[dict[str, float]]): 'ingress'/'egress' BJD times,
                or None to skip drawing.
        """
        if not transit:
            return

        ingress_t = transit["ingress"] - t0
        egress_t = transit["egress"] - t0

        for ax in axs:
            ax.axvline(
                ingress_t,
                color="red",
                linestyle=":",
                linewidth=1.2,
                alpha=0.8,
                zorder=15,
            )
            ax.axvline(
                egress_t,
                color="red",
                linestyle=":",
                linewidth=1.2,
                alpha=0.8,
                zorder=15,
            )

        trans = lc_ax.get_xaxis_transform()
        lc_ax.text(
            ingress_t,
            0.02,
            "Expected\nIngress",
            transform=trans,
            ha="center",
            va="bottom",
            fontsize=8,
            color="red",
        )
        lc_ax.text(
            egress_t,
            0.02,
            "Expected\nEgress",
            transform=trans,
            ha="center",
            va="bottom",
            fontsize=8,
            color="red",
        )

    def _save_csv(self, date_folder: Path, filt_dict: Dict) -> None:
        """
        Save a normalized multiband light curve CSV.

        Normalizes flux and error columns for each filter by the median flux, then
        merges all filters side by side into a single CSV.

        Args:
            date_folder (Path): Output directory for the CSV.
            filt_dict (dict[str, DataFrame]): Mapping of filter name to its light
                curve DataFrame. Each DataFrame must have 'BJD_TDB', 'rel_flux_T1',
                and 'rel_flux_err_T1' columns.
        """
        frames: list = []
        for filt, df in filt_dict.items():
            # Normalize per filter
            norm: Series = df["rel_flux_T1"] / df["rel_flux_T1"].median()
            err: Series = df["rel_flux_err_T1"] / df["rel_flux_T1"].median()

            tmp = pd.DataFrame(
                {
                    f"BJD_TDB_{filt}": df["BJD_TDB"],
                    f"norm_flux_{filt}": norm,
                    f"norm_flux_err_{filt}": err,
                }
            )
            frames.append(tmp)
        df_merged: DataFrame = pd.concat(frames, axis=1)
        os.makedirs(date_folder, exist_ok=True)
        out_csv: Path = date_folder / "norm_gri_lcs.csv"
        df_merged.to_csv(out_csv, index=False)
        msg: str = f"Saved normalized CSV: {out_csv}"
        self.logger.info(msg)

    def _create_multipanel_plot(
        self,
        obj: str,
        date: str,
        utc_date: str,
        data: Dict,
        times_df: Optional[DataFrame],
        out_folder: Path,
        obj_folder: Path,
    ) -> None:
        mpl.rcParams.update({"font.family": "serif", "font.size": 14})
        if not data:
            return

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

        flux_col = "rel_flux_T1"
        err_col = "rel_flux_err_T1"
        time_col = "BJD_TDB"

        first_band = list(data.keys())[0]
        df_first = data[first_band]
        # Use nanmin for robust t0
        t0 = float(np.nanmin(df_first[time_col].to_numpy()))

        # 1. Light curve panel
        for band, df in data.items():
            t = df[time_col].to_numpy() - t0
            med_f = float(df[flux_col].median())
            f = df[flux_col].to_numpy() / med_f
            e = df[err_col].to_numpy() / med_f

            c = self._get_band_color(band)
            kwargs = {}
            if c:
                kwargs["color"] = c

            axs[0].errorbar(
                t,
                f,
                yerr=e,
                fmt=".",
                alpha=0.1,
                elinewidth=0.5,
                capsize=2,
                label="_nolegend_",
                rasterized=True,
                **kwargs,
            )

            t_bin, f_bin, e_bin = self._bin_data(t, f, e, self.bin_minutes)

            rms_data = self._calc_rms_in_intervals(t, f, times_df)
            rms_bin = self._calc_rms_in_intervals(t_bin, f_bin, times_df)

            markeredge = c if c else "black"
            axs[0].errorbar(
                t_bin,
                f_bin,
                yerr=e_bin,
                fmt="o",
                markerfacecolor="white",
                markeredgecolor=markeredge,
                label=f"{band} (RMS: {rms_data:.2f} ppt, {rms_bin:.2f} ppt/{self.bin_minutes}min)",
                capsize=2,
                zorder=20,
                rasterized=True,
                **kwargs,
            )

        if times_df is not None and not times_df.empty:
            for i, f in zip(times_df["init_time"], times_df["final_time"]):
                axs[0].axvline(
                    i - t0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7
                )
                axs[0].axvline(
                    f - t0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7
                )

        transit = self._load_transit_times(obj_folder, obj, date, utc_date)
        self._add_transit_markers(axs, axs[0], t0, transit)

        axs[0].set_ylabel("Relative Flux")
        axs[0].grid(ls=":", zorder=0, alpha=0.5)
        axs[0].legend(loc="upper right", fontsize=9)

        panel_vars = [
            (1, "AIRMASS", "AIRMASS", False),
            (2, "Width_T1", "Width_T1 [Pixels]", False),
            (3, "tot_C_cnts", "Total Comp. Counts", False),
            (4, "Sky/Pixel_T1", "Sky/Pixel_T1", False),
        ]

        for idx, col, ylabel, do_legend in panel_vars:
            for band, df in data.items():
                if (
                    idx == 1
                    and band not in ("gp", "g")
                    and ("gp" in data or "g" in data)
                ):
                    continue

                t = df[time_col].values - t0
                y = df[col]

                c = self._get_band_color(band)
                if idx == 1:
                    c = "darkslateblue"
                kwargs = {}
                if c:
                    kwargs["color"] = c

                rms = self._calc_rms_ppt(y)
                label_text = (
                    f"{band} (RMS: {rms:.2f} ppt)" if do_legend else "_nolegend_"
                )

                axs[idx].plot(
                    t, y, ".", alpha=0.5, label=label_text, ms=4, rasterized=True, **kwargs
                )
                axs[idx].grid(ls=":", zorder=0, alpha=0.5)
            axs[idx].set_ylabel(ylabel)
            if do_legend:
                axs[idx].legend(loc="upper right", fontsize=8)

        # Panel 5: Centroid Shift
        time_xy = df_first["BJD_TDB"].to_numpy() - t0
        x_fits = df_first["X(FITS)_T1"].to_numpy()
        x_rel = x_fits - x_fits[0]
        axs[5].plot(
            time_xy,
            x_rel,
            ".",
            color="m",
            alpha=0.6,
            label=r"X",
            ms=6,
            fillstyle="none",
            rasterized=True,
        )

        y_fits = df_first["Y(FITS)_T1"].to_numpy()
        y_rel = y_fits - y_fits[0]
        axs[5].plot(
            time_xy,
            y_rel,
            "^",
            color="limegreen",
            alpha=0.6,
            label=r"Y",
            ms=3,
            fillstyle="none",
            rasterized=True,
        )

        axs[5].set_ylabel(r"Centroid shift [pixels]")
        axs[5].set_xlabel(r"BJD$_{TDB}$ - " + f"{t0:.2f}")
        axs[5].legend(loc="best", fontsize=10)
        axs[5].grid(ls=":", alpha=0.5)

        out_folder.mkdir(parents=True, exist_ok=True)
        out_file: Path = out_folder / f"{obj}_{date}_PROFE_lc.pdf"
        fig.savefig(out_file, format="pdf", bbox_inches="tight")

        plt.close(fig)
        self.logger.info(f"Plot: {out_file}, saved")

        # Save individual per-band PNGs in exofop
        for band, df_band in data.items():
            self._save_single_band_multipanel(
                obj, date, utc_date, band, df_band, times_df, t0, obj_folder
            )

    def _save_single_band_multipanel(
        self,
        obj: str,
        date: str,
        utc_date: str,
        band: str,
        df: DataFrame,
        times_df: Optional[DataFrame],
        t0: float,
        obj_folder: Path,
    ) -> None:
        """Save a 6-panel multipanel PNG for a single photometric band."""
        mpl.rcParams.update({"font.family": "serif", "font.size": 14})

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

        flux_col = "rel_flux_T1"
        err_col = "rel_flux_err_T1"
        time_col = "BJD_TDB"

        df = df.sort_values(time_col).reset_index(drop=True)

        has_fn = "rel_flux_T1_fn" in df.columns
        if has_fn:
            flux_col = "rel_flux_T1_fn"
            if "rel_flux_err_T1_fn" in df.columns:
                err_col = "rel_flux_err_T1_fn"
            elif "rel_flux_err_T1" in df.columns:
                err_col = "rel_flux_err_T1"
            else:
                err_cols = [c for c in df.columns if "err_T1" in c]
                err_col = err_cols[0] if err_cols else "rel_flux_err_T1"

        has_fit = has_fn and "rel_flux_T1_fn_model" in df.columns and "rel_flux_T1_fn_residual" in df.columns

        exofop_obj = get_exofop_id(obj)

        t = df[time_col].to_numpy() - t0
        if has_fn:
            f = df[flux_col].to_numpy()
            if err_col == "rel_flux_err_T1_fn":
                e = df[err_col].to_numpy()
            else:
                raw_med = float(np.nanmedian(df["rel_flux_T1"].to_numpy()))
                if raw_med != 0 and not np.isnan(raw_med):
                    e = df[err_col].to_numpy() / raw_med
                else:
                    e = df[err_col].to_numpy()
        else:
            med_f = float(np.nanmedian(df[flux_col].to_numpy()))
            f = df[flux_col].to_numpy() / med_f
            e = df[err_col].to_numpy() / med_f

        c = self._get_band_color(band)
        kwargs: dict = {}
        if c:
            kwargs["color"] = c

        # Panel 0: Light curve
        axs[0].errorbar(
            t,
            f,
            yerr=e,
            fmt=".",
            alpha=0.1,
            elinewidth=0.5,
            capsize=2,
            label="_nolegend_",
            rasterized=True,
            **kwargs,
        )

        t_bin, f_bin, e_bin = self._bin_data(t, f, e, self.bin_minutes)

        if has_fit:
            # Normalize model by its own median to remove baseline offset
            model_raw = df["rel_flux_T1_fn_model"].to_numpy()
            model_med = float(np.nanmedian(model_raw))
            if model_med != 0 and not np.isnan(model_med):
                model = model_raw / model_med
            else:
                model = model_raw

            # Get residuals and their errors
            if has_fn:
                residual = df["rel_flux_T1_fn_residual"].to_numpy()
                if "rel_flux_err_T1_fn_residual" in df.columns:
                    residual_err = df["rel_flux_err_T1_fn_residual"].to_numpy()
                else:
                    residual_err = e
            else:
                med_f = float(np.nanmedian(df["rel_flux_T1"].to_numpy()))
                if med_f != 0 and not np.isnan(med_f):
                    residual = df["rel_flux_T1_fn_residual"].to_numpy() / med_f
                    if "rel_flux_err_T1_fn_residual" in df.columns:
                        residual_err = df["rel_flux_err_T1_fn_residual"].to_numpy() / med_f
                    else:
                        residual_err = e
                else:
                    residual = df["rel_flux_T1_fn_residual"].to_numpy()
                    residual_err = e

            # Bin the residuals using the same bins as the light curve
            _, residual_bin, residual_err_bin = self._bin_data(t, residual, residual_err, self.bin_minutes)

            # Compute RMS of residuals (unbinned and binned)
            rms_data = self._calc_rms_in_intervals(t, residual, times_df, absolute=True)
            rms_bin = self._calc_rms_in_intervals(t_bin, residual_bin, times_df, absolute=True)
        else:
            rms_data = self._calc_rms_in_intervals(t, f, times_df)
            rms_bin = self._calc_rms_in_intervals(t_bin, f_bin, times_df)

        markeredge = c if c else "black"
        axs[0].errorbar(
            t_bin,
            f_bin,
            yerr=e_bin,
            fmt="o",
            markerfacecolor="white",
            markeredgecolor=markeredge,
            label=f"{band} (RMS: {rms_data:.2f} ppt, {rms_bin:.2f} ppt/{self.bin_minutes}min)",
            capsize=2,
            zorder=20,
            rasterized=True,
            **kwargs,
        )

        if times_df is not None and not times_df.empty:
            for i_t, f_t in zip(times_df["init_time"], times_df["final_time"]):
                axs[0].axvline(
                    i_t - t0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7
                )
                axs[0].axvline(
                    f_t - t0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7
                )

        transit = self._load_transit_times(obj_folder, obj, date, utc_date)
        self._add_transit_markers(axs, axs[0], t0, transit)

        if has_fit:
            axs[0].plot(
                t,
                model,
                "-",
                color="black",
                lw=1.2,
                alpha=0.9,
                zorder=18,
                label="Model",
                rasterized=True,
            )

            # Offset residuals below the light curve by a multiple of their own
            # scatter so they never overlap with the data or model above them.
            # Use a robust minimum of f (1st percentile) to avoid outlier issues.
            f_min = float(np.nanpercentile(f, 1.0)) if len(f) > 0 else 1.0
            offset = f_min - 6 * float(np.nanstd(residual))
            axs[0].axhline(
                offset, color="gray", linestyle=":", linewidth=0.8, alpha=0.6, zorder=5
            )
            # Plot unbinned residuals
            axs[0].errorbar(
                t,
                residual + offset,
                yerr=residual_err,
                fmt=".",
                alpha=0.15,
                elinewidth=0.5,
                capsize=2,
                color="gray",
                label="_nolegend_",
                zorder=10,
                rasterized=True,
            )
            # Plot binned residuals
            axs[0].errorbar(
                t_bin,
                residual_bin + offset,
                yerr=residual_err_bin,
                fmt="o",
                markerfacecolor="white",
                markeredgecolor="gray",
                color="gray",
                label="Residuals",
                capsize=2,
                zorder=10,
                rasterized=True,
            )

        axs[0].set_ylabel("Relative Flux")
        axs[0].grid(ls=":", zorder=0, alpha=0.5)
        axs[0].legend(loc="upper right", fontsize=9)
        title_str = exofop_title(exofop_obj, utc_date, band)
        axs[0].set_title(title_str)

        # Panels 1-4: auxiliary variables
        panel_vars = [
            (1, "AIRMASS", "AIRMASS"),
            (2, "Width_T1", "Width_T1 [Pixels]"),
            (3, "tot_C_cnts", "Total Comp. Counts"),
            (4, "Sky/Pixel_T1", "Sky/Pixel_T1"),
        ]
        for idx, col, ylabel in panel_vars:
            y = df[col]
            color = "darkslateblue" if idx == 1 else (c if c else None)
            kw: dict = {}
            if color:
                kw["color"] = color
            axs[idx].plot(
                t, y, ".", alpha=0.5, label="_nolegend_", ms=4, rasterized=True, **kw
            )
            axs[idx].grid(ls=":", zorder=0, alpha=0.5)
            axs[idx].set_ylabel(ylabel)

        # Panel 5: Centroid shift
        time_xy = df["BJD_TDB"].to_numpy() - t0
        x_fits = df["X(FITS)_T1"].to_numpy()
        x_rel = x_fits - x_fits[0]
        axs[5].plot(
            time_xy,
            x_rel,
            ".",
            color="m",
            alpha=0.6,
            label=r"X",
            ms=6,
            fillstyle="none",
            rasterized=True,
        )
        y_fits = df["Y(FITS)_T1"].to_numpy()
        y_rel = y_fits - y_fits[0]
        axs[5].plot(
            time_xy,
            y_rel,
            "^",
            color="limegreen",
            alpha=0.6,
            label=r"Y",
            ms=3,
            fillstyle="none",
            rasterized=True,
        )
        axs[5].set_ylabel(r"Centroid shift [pixels]")
        axs[5].set_xlabel(r"BJD$_{TDB}$ - " + f"{t0:.2f}")
        axs[5].legend(loc="best", fontsize=10)
        axs[5].grid(ls=":", alpha=0.5)

        png_file: Path = exofop_path(
            obj_folder, date, utc_date, exofop_obj, band, "_lightcurve", ".png"
        )
        png_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_file, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Plot: {png_file}, saved")

    def _create_lightcurves_plot(
        self,
        obj: str,
        date: str,
        utc_date: str,
        data: Dict,
        times_df: Optional[DataFrame],
        out_folder: Path,
        obj_folder: Path,
    ) -> None:
        mpl.rcParams.update({"font.family": "serif", "font.size": 14})
        n_bands = len(data)
        if n_bands == 0:
            return

        fig_lc, axs_lc = plt.subplots(
            n_bands,
            1,
            figsize=(6, 3 * n_bands + 1),
            sharex=True,
            gridspec_kw={"hspace": 0.0},
        )
        if n_bands == 1:
            axs_lc = [axs_lc]

        time_col = "BJD_TDB"
        flux_col = "rel_flux_T1"
        err_col = "rel_flux_err_T1"

        first_band = list(data.keys())[0]
        t0 = np.nanmin(data[first_band][time_col].values)

        for i, (band, df) in enumerate(data.items()):
            t = df[time_col].values - t0
            f = df[flux_col].values / np.median(df[flux_col].values)
            e = df[err_col].values / np.median(df[flux_col].values)

            c = self._get_band_color(band)
            kwargs = {}
            if c:
                kwargs["color"] = c

            axs_lc[i].errorbar(
                t,
                f,
                yerr=e,
                fmt=".",
                alpha=0.07,
                elinewidth=0.5,
                label="_nolegend_",
                rasterized=True,
                **kwargs,
            )

            t_bin, f_bin, e_bin = self._bin_data(t, f, e, 6)

            markeredge = c if c else "black"
            axs_lc[i].errorbar(
                t_bin,
                f_bin,
                yerr=e_bin,
                fmt="o",
                markerfacecolor="white",
                markeredgecolor=markeredge,
                label=f"{band} ",
                capsize=2,
                ms=5,
                rasterized=True,
                **kwargs,
            )

            axs_lc[i].set_ylabel("Relative Flux")
            axs_lc[i].legend(loc="lower right", fontsize=10)
            axs_lc[i].grid(ls=":", zorder=0, alpha=0.5)

        axs_lc[-1].set_xlabel(r"BJD$_{{TDB}}$ - " + f"{t0:.2f}")

        transit = self._load_transit_times(obj_folder, obj, date, utc_date)
        self._add_transit_markers(axs_lc, axs_lc[0], t0, transit)

        out_folder.mkdir(parents=True, exist_ok=True)
        out_file: Path = out_folder / f"{obj}_{date}_lc_{n_bands}panels.pdf"
        plt.savefig(out_file, format="pdf", bbox_inches="tight")

        plt.close(fig_lc)
        self.logger.info(f"Plot: {out_file}, saved")

    def run(self, target: str | None = None) -> None:
        """
        Process all objects and dates to generate light curve outputs.

        For each (object, date) whose output PDF does not yet exist:
            - Load all `.tbl` measurement files.
            - Group by processing method and filter.
            - Load optional time intervals.
            - Generate per-method and per-filter plots.
            - Save normalized multiband CSV files.

        Args:
            target (str | None): If specified, only process this target.

        Returns:
            None
        """
        for obj_folder in sorted(self.data_dir.iterdir()):
            if not obj_folder.is_dir():
                continue
            if target and obj_folder.name.lower() != target.lower():
                continue

            obj: str = obj_folder.name
            meas_root: Path = obj_folder / "measurements"
            lcs_root: Path = obj_folder / "lcs"

            if not meas_root.exists():
                continue

            for date_folder in sorted(meas_root.iterdir()):
                if not date_folder.is_dir():
                    continue
                date: str = date_folder.name
                utc_date: str = get_utc_date_from_bjd(date_folder)

                plots_done = self._plots_exist(obj_folder, obj, date)

                # Determine which exofop bands are missing (need data to know
                # available bands, but we can do a quick pre-check with the
                # known standard bands to avoid loading data unnecessarily).
                meas_files: list = self._obtain_measurements(date_folder)
                if not meas_files:
                    self.logger.warning(f"No measurements in {date_folder} — Skipping")
                    continue

                # Load data only if plots are needed OR exofop may be missing
                data: Dict = {}
                for f in meas_files:
                    df: DataFrame
                    if f.suffix == ".tbl":
                        df = pd.read_csv(
                            f, sep=r"\t+", engine="python", encoding="latin1"
                        )
                    elif f.suffix == ".csv":
                        df = pd.read_csv(f, encoding="latin1")
                    else:
                        continue

                    stem = f.stem
                    filtr = normalize_band(stem.split("_")[-1])

                    data[filtr] = df

                missing_bands = self._missing_exofop_bands(
                    obj_folder, obj, date, utc_date, list(data.keys())
                )

                if plots_done and not missing_bands:
                    self.logger.info(f"Skipping {obj},{date}. Already processed")
                    continue

                times: DataFrame | None = self._load_times(date_folder)
                out_base: Path = obj_folder / "plots" / date
                exofop_base: Path = obj_folder / "exofop" / date

                # Generate plots (PDF) if not already present
                if not plots_done:
                    self._create_multipanel_plot(
                        obj, date, utc_date, data, times, out_base, obj_folder
                    )
                    self._create_lightcurves_plot(
                        obj, date, utc_date, data, times, out_base, obj_folder
                    )
                    # CSV saving
                    lcs_folder: Path = lcs_root / date
                    self._save_csv(lcs_folder, data)
                elif missing_bands:
                    # Plots exist but some exofop PNGs are missing
                    first_band = list(data.keys())[0]
                    t0: float = data[first_band]["BJD_TDB"].values[0]
                    exofop_base.mkdir(parents=True, exist_ok=True)
                    for band in missing_bands:
                        if band in data:
                            self._save_single_band_multipanel(
                                obj,
                                date,
                                utc_date,
                                band,
                                data[band],
                                times,
                                t0,
                                obj_folder,
                            )

                # Save measurements to exofop as .tbl (tab-separated, latin1)
                for filtr, df_band in data.items():
                    exofop_obj = get_exofop_id(obj)
                    dest = exofop_path(
                        obj_folder,
                        date,
                        utc_date,
                        exofop_obj,
                        filtr,
                        "_measurements",
                        ".tbl",
                    )
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    df_band.to_csv(dest, sep="\t", index=False, encoding="latin1")
                    self.logger.info(f"Saved {filtr} measurements to {dest} as .tbl")
