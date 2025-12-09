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

Processed (object, date) pairs are recorded in `logs/.lc_processed.dat` to
avoid reprocessing.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from matplotlib.pyplot import Colormap
from numpy import ndarray
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

    Processed (object, date) pairs are tracked to avoid duplicate work.
    """

    def __init__(self, bin_minutes: int = 10) -> None:
        """
        Initialize the light curve plotter.

        Sets up working directories, logger, binning configuration, and the
        processed-state tracking file.

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
        # Already processed lcs
        self.processed_file = self.logs_dir / ".lc_processed.dat"
        if not self.processed_file.exists():
            self.processed_file.write_text("")

    def _load_processed(self) -> Set:
        """
        Load the set of processed (object, date) pairs.

        Reads `logs/.lc_processed.dat` and returns its contents as a set of tuples.

        Returns:
            set[tuple[str, str]]: Processed object/date pairs.
        """
        text: str = self.processed_file.read_text()
        lines: list = text.splitlines()
        processed: set = set()  # To make an iterable object
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts: list = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                processed.add((parts[0], parts[1]))
        return processed

    def _mark_processed(self, obj: str, date: str) -> None:
        """
        Record a processed (object, date) pair.

        Appends the pair to `logs/.lc_processed.dat`.

        Args:
            obj (str): Target object name.
            date (str): Observation date in YYYY-MM-DD format.
        """
        with open(self.processed_file, "a") as f:
            f.write(f"{obj},{date}\n")

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

    def _obtain_tbls(self, folder: Path) -> list[Path]:
        """
        List all AIJ measurement table files in a folder.

        Args:
            folder (Path): Path to a `measurements/` directory.

        Returns:
            list[Path]: Paths to all `.tbl` files in the folder.
        """
        return [f for f in folder.glob("*.tbl")]

    def _calculate_rms(
        self,
        flux: Series,
        times: Optional[DataFrame],
        t: Series,
    ) -> tuple[float, str]:
        """
        Calculate RMS of the flux, optionally within specific time intervals.

        Args:
            flux (Series): Normalized flux values.
            times (Optional[DataFrame]): DataFrame containing time intervals.
            t (Series): Time values corresponding to the flux.

        Returns:
            tuple[float, str]: Calculated RMS value and formatted string for plot label.
        """
        diff: Series
        if times is None:
            median_val = flux.median()
            diff = flux - median_val
            rms = np.sqrt(np.mean(diff**2))
            rms_txt = f" (RMS:{rms:.4f})"
        else:
            mask: Any = np.zeros(len(flux), dtype=bool)
            for i, f in zip(times["init_time"], times["final_time"]):
                mask |= (t >= i) & (t <= f)
            sel = flux[mask]
            
            if len(sel):
                median_val = sel.median()
                diff = sel - median_val
                rms = np.sqrt(np.mean(diff**2))
                rms_txt = f" (RMS:{rms:.4f})"
            else:
                # Fallback to full data if mask is empty
                median_val = flux.median()
                diff = flux - median_val
                rms = np.sqrt(np.mean(diff**2))
                rms_txt = f" (RMS:{rms:.4f})"
        return rms, rms_txt

    def _save_method_csv(self, date_folder: Path, method: str, filt_dict: Dict) -> None:
        """
        Save a normalized multiband light curve CSV for a processing method.

        Normalizes flux and error columns for each filter by the median flux, then
        merges all filters side by side into a single CSV.

        Args:
            date_folder (Path): Output directory for the CSV.
            method (str): Processing method name (used in filename).
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
        out_csv: Path = date_folder / f"{method}_norm_gri_lcs.csv"
        df_merged.to_csv(out_csv, index=False)
        msg: str = f"Saved normalized CSV: {out_csv}"
        self.logger.info(msg)

    def _create_plot(
        self,
        obj: str,
        date: str,
        datasets: Dict,
        times: Optional[DataFrame],
        out_folder: Path,
        label_type: str,
        label_value: str,
    ) -> None:
        """
        Generate and save a light curve plot with raw and binned data.

        Plots individual points with transparency, binned medians at
        `self.bin_minutes`, and RMS values in the legend. Optionally marks
        time-interval boundaries from a `times.csv` file.

        Args:
            obj (str): Target object name.
            date (str): Observation date.
            datasets (dict[str, tuple[DataFrame, str]]): Mapping from dataset label
                to (DataFrame, color).
            times (Optional[DataFrame]): Optional time intervals for RMS calculation.
            out_folder (Path): Output directory for the plot.
            label_type (str): Label descriptor (e.g., 'filter', 'method').
            label_value (str): Label value for title/filename.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        bin_days: float = self.bin_minutes / (24 * 60)
        for label, (df, color) in datasets.items():
            t: Series = df["BJD_TDB"]
            flux: Series = df["rel_flux_T1"] / df["rel_flux_T1"].median()
            err: Series = df["rel_flux_err_T1"] / df["rel_flux_T1"].median()
            rms: float
            rms_txt: str
            median_val: float
            diff: Series

            if times is None:
                rms, rms_txt = self._calculate_rms(flux, None, t)
            else:
                rms, rms_txt = self._calculate_rms(flux, times, t)

            # All datapoints
            ax.errorbar(
                t, flux, yerr=err, fmt=".", color=color, alpha=0.05, markersize=4
            )

            # Binned datapoints
            bin_edges: ndarray = np.arange(t.min(), t.max() + bin_days, bin_days)
            bins: Sequence = bin_edges.tolist()

            temp = pd.DataFrame({"t": t, "f": flux, "e": err})
            grp = temp.groupby(pd.cut(temp["t"], bins), observed=False)

            bt_arr: NDArray[np.float64] = grp["t"].median().to_numpy(dtype=np.float64)
            bf_arr: NDArray[np.float64] = grp["f"].median().to_numpy(dtype=np.float64)

            sum_e2: pd.Series = grp["e"].apply(lambda x: np.sum(x**2))
            count_e: pd.Series = grp["e"].count()
            be_arr: NDArray[np.float64] = np.sqrt(
                sum_e2.to_numpy(dtype=np.float64)
            ) / count_e.to_numpy(dtype=np.float64)

            ax.errorbar(
                bt_arr.tolist(),
                bf_arr.tolist(),
                yerr=be_arr.tolist(),
                fmt="o",
                color=color,
                alpha=1.0,
                markersize=4,
                label=label + rms_txt,
            )

            # Plot RMS intervals
            if times is not None:
                for i, f in zip(times["init_time"], times["final_time"]):
                    ax.axvline(i, linestyle="-.", color="g", alpha=0.7)
                    ax.axvline(f, linestyle="-.", color="g", alpha=0.7)

        h, leg = ax.get_legend_handles_labels()
        by_label: dict = dict(zip(leg, h))
        ax.legend(by_label.values(), by_label.keys())
        ax.set_title(
            f"{obj} {date} — {self.bin_minutes}min bins - {label_type}: {label_value} "
        )
        ax.set_xlabel("BJD_TDB")
        ax.set_ylabel("Normalized flux")

        out_folder.mkdir(parents=True, exist_ok=True)
        out_file: Path = out_folder / f"{obj}_{date}_{label_value}_lightcurve.png"
        fig.savefig(out_file, dpi=300)
        plt.close(fig)
        self.logger.info(f"Plot: {out_file}, saved")

    def run(self) -> None:
        """
        Process all objects and dates to generate light curve outputs.

        For each unprocessed (object, date):
            - Load all `.tbl` measurement files.
            - Group by processing method and filter.
            - Load optional time intervals.
            - Generate per-method and per-filter plots.
            - Save normalized multiband CSV files.
            - Mark as processed.

        Returns:
            None
        """
        processed: set = self._load_processed()

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
                key: tuple = (obj, date)
                if key in processed:
                    self.logger.info(f"Skipping {obj},{date}. Already processed")
                    continue

                tbls: list = self._obtain_tbls(date_folder)
                if not tbls:
                    self.logger.warning(f"No TBLs in {date_folder} — Skipping")
                    continue

                # Load data
                data: Dict = {}
                for f in tbls:
                    parts: list = f.stem.split("_")
                    if len(parts) != 3:
                        self.logger.warning(f"Invalid name {f.name} — Ignoring")
                        continue
                    date_str, filtr, method = parts
                    df: DataFrame = pd.read_table(f)
                    data.setdefault(method, {})[filtr] = df

                times: DataFrame | None = self._load_times(date_folder)

                # Light curves plots per method
                out_base: Path = obj_folder / "plots" / date
                for method, filt_dict in data.items():
                    datasets_m: dict = {
                        f: (df, {"g": "blue", "r": "green", "i": "red"}.get(f, "black"))
                        for f, df in filt_dict.items()
                    }
                    self._create_plot(
                        obj,
                        date,
                        datasets_m,
                        times,
                        out_base / "lcs_method",
                        label_type="",
                        label_value=method,
                    )

                # Light curves plots per filter
                all_filts: set = {f for d in data.values() for f in d}
                for filtr in all_filts:
                    methods: list = [m for m, d in data.items() if filtr in d]
                    cmap: Colormap = plt.get_cmap("tab10", len(methods))
                    datasets_f: Dict = {
                        m: (data[m][filtr], to_hex(cmap(i)))
                        for i, m in enumerate(methods)
                    }

                    self._create_plot(
                        obj,
                        date,
                        datasets_f,
                        times,
                        out_base / "lcs_filter",
                        label_type="",
                        label_value=filtr,
                    )

                for method, filt_dict in data.items():
                    lcs_folder: Path = lcs_root / date
                    self._save_method_csv(lcs_folder, method, filt_dict)
                self._mark_processed(obj, date)
