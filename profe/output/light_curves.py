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

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

from .naming import exofop_path, exofop_title, get_exofop_id, normalize_band

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
        self, obj_folder: Path, obj: str, date: str, bands: list[str]
    ) -> list[str]:
        """
        Return band names whose per-band exofop PNGs do not yet exist.

        Args:
            obj_folder (Path): Path to the object directory.
            obj (str): Target object name.
            date (str): Observation date in YYYY-MM-DD format.
            bands (list[str]): Available photometric band names.

        Returns:
            list[str]: Bands whose exofop PNG is missing.
        """
        return [
            b
            for b in bands
            if not exofop_path(obj_folder, date, obj, b, "_lightcurve", ".png").exists()
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

    def _calc_rms_in_intervals(
        self, time: NDArray, data: NDArray, times_df: Optional[DataFrame]
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
        return self._calc_rms_ppt(pd.Series(selected))

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
            return np.array([t_min]), np.array([np.mean(col)]), np.array([np.mean(err_col)])

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
        time_mid = np.nanmedian(df_first[time_col].to_numpy() - t0)

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

        axs[0].set_ylabel("Relative Flux")
        axs[0].grid(ls=":", zorder=0, alpha=0.5)

        # Legend position
        f_med = float(df_first[flux_col].median())
        f_std = np.nanstd(df_first[flux_col])
        tr_mask = df_first[flux_col].to_numpy() < (f_med - 1.5 * f_std)
        if np.sum(tr_mask) > 0:
            tr_time_center = np.nanmean((df_first[time_col].to_numpy() - t0)[tr_mask])
            loc_choice = "lower right" if tr_time_center < time_mid else "lower left"
        else:
            loc_choice = "lower left"

        axs[0].legend(loc=loc_choice, fontsize=9)

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

                axs[idx].plot(t, y, ".", alpha=0.5, label=label_text, ms=4, **kwargs)
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
                obj, date, band, df_band, times_df, t0, obj_folder
            )

    def _save_single_band_multipanel(
        self,
        obj: str,
        date: str,
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

        exofop_obj = get_exofop_id(obj)

        t = df[time_col].to_numpy() - t0
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

        axs[0].set_ylabel("Relative Flux")
        axs[0].grid(ls=":", zorder=0, alpha=0.5)
        axs[0].legend(loc="lower left", fontsize=9)
        title_str = exofop_title(exofop_obj, date, band)
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
            axs[idx].plot(t, y, ".", alpha=0.5, label="_nolegend_", ms=4, **kw)
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
        )
        axs[5].set_ylabel(r"Centroid shift [pixels]")
        axs[5].set_xlabel(r"BJD$_{TDB}$ - " + f"{t0:.2f}")
        axs[5].legend(loc="best", fontsize=10)
        axs[5].grid(ls=":", alpha=0.5)

        png_file: Path = exofop_path(
            obj_folder, date, exofop_obj, band, "_lightcurve", ".png"
        )
        png_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_file, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Plot: {png_file}, saved")

    def _create_lightcurves_plot(
        self,
        obj: str,
        date: str,
        data: Dict,
        times_df: Optional[DataFrame],
        out_folder: Path,
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
                **kwargs,
            )

            axs_lc[i].set_ylabel("Relative Flux")
            axs_lc[i].legend(loc="lower right", fontsize=10)
            axs_lc[i].grid(ls=":", zorder=0, alpha=0.5)

        axs_lc[-1].set_xlabel(r"BJD$_{{TDB}}$ - " + f"{t0:.2f}")

        out_folder.mkdir(parents=True, exist_ok=True)
        out_file: Path = out_folder / f"{obj}_{date}_lc_{n_bands}panels.pdf"
        plt.savefig(out_file, format="pdf", bbox_inches="tight")

        plt.close(fig_lc)
        self.logger.info(f"Plot: {out_file}, saved")

    def run(self) -> None:
        """
        Process all objects and dates to generate light curve outputs.

        For each (object, date) whose output PDF does not yet exist:
            - Load all `.tbl` measurement files.
            - Group by processing method and filter.
            - Load optional time intervals.
            - Generate per-method and per-filter plots.
            - Save normalized multiband CSV files.

        Returns:
            None
        """
        for obj_folder in sorted(self.data_dir.iterdir()):
            if not obj_folder.is_dir():
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
                    obj_folder, obj, date, list(data.keys())
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
                        obj, date, data, times, out_base, obj_folder
                    )
                    self._create_lightcurves_plot(obj, date, data, times, out_base)
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
                        exofop_obj,
                        filtr,
                        "_measurements",
                        ".tbl",
                    )
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    df_band.to_csv(dest, sep="\t", index=False, encoding="latin1")
                    self.logger.info(f"Saved {filtr} measurements to {dest} as .tbl")
