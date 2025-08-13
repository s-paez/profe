"""
Light curves plots and CSV files

This PROFE's module create plots and CSV files for each measurement AIJ file in each
target and date folder. It uses times.csv file (if exists or not empty) to use that time
range to normalize the light curve and compute the time-averaging and the RMS.
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
    LightCurvePlotter

    Save light curves .csv files and binned plots for light curves with RMS values from
    AIJ measurements in TBL in each object/target folfer in `organized_data`.
    The plots are organized each object plots/ folder with subforlder for each DATE-OBS
    It records in logs/.lc_processed.txt those (object, dates) that are already
    processed to avoid reprocessing files.
    """

    def __init__(self, bin_minutes: int = 10) -> None:
        """
        Initialize the LightCurvePlotter with directory paths and binning settings.

        This constructor sets up the working directories for logs and organized data,
        configures the logger, and initializes the file used to track which light curves
        have been already processed.

        Args:
            bin_minutes (int): Time interval, in minutes, for binning light-curves data/
                Defaults to 10 minutes.
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
        """Read logs/.lc_processed.txt and return a tuples set (object, date)."""
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
        return processed  # This is a set() object over one can interate

    def _mark_processed(self, obj: str, date: str) -> None:
        """
        Add a line to `.lc_processed.dat` with the object and date processed.

        Args:
            obj (str): Object name
            date (str): Observation date

        Returns:
            None
        """
        with open(self.processed_file, "a") as f:  # Appending mode
            f.write(f"{obj},{date}\n")

    def _load_times(self, folder: Path) -> DataFrame | None:
        """
        Try to load times.csv file

        Look for a times.csv file in times folder.
        Otherwise in create the times.csv file and return none
        It returns the DataFrame of the times.csv

        Args:
            folder (Path): times folder with times.csv file.

        Returns:
            Times DataFrame.
        """
        # Path to the times.csv file in measurements/
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
        Get all measuremts TBL files in `measurements/` folder.

        Args:
            folder (Path): Path to measurements folder

        Returns:
            List with all TBL files in measurements/ folder
        """
        return [f for f in folder.glob("*.tbl")]

    def _save_method_csv(self, date_folder: Path, method: str, filt_dict: Dict) -> None:
        """
        Save a multiband light curve CSV with normalized flux and error columns.

        For each filter in `filt_dict`, this hidden method:

            1. Computes the median of `rel_flux_T1` and uses it to normalize both
                `rel_flux_T1` and `rel_flux_err_T1`.
            2. Builds a DataFrame with columns:
                - BJD_TDB_<filter>
                - norm_flux_<filter>
                - norm_flux_err_<filter>
            3. Concatenates all filter DataFrames side by side, aligning by index.
            4. Writes the merged to: `<date_folder>/<method>_norm_gri_lcs.csv`.

        Args:
            date_folder (Path): Directory where the CSV will be saved.
            method (str) Binning or processing method name (used as filename prefix).
            filt_dict (Dict[str, pd.DataFrame]): Mapping from filter name (e.g. 'g',
                'r', 'i') to its light curve DataFrame. Each DataFrame must contain
                'BJD_TDB', 'rel_flux_T1', and 'rel_flux_err_T1' columns.

        Returns:
            None
        """
        frames: list = []
        for filt, df in filt_dict.items():
            # Normalize per filter
            norm: Series = df["rel_flux_T1"] / df["rel_flux_T1"].median()
            err: Series = df["rel_flux_err_T1"] / df["rel_flux_T1"].median()

            # Build DataFrame for this filter
            tmp = pd.DataFrame(
                {
                    f"BJD_TDB_{filt}": df["BJD_TDB"],
                    f"norm_flux_{filt}": norm,
                    f"norm_flux_err_{filt}": err,
                }
            )
            frames.append(tmp)
        # Concat side by side (aligns by index; filters may have different lengths)
        df_merged: DataFrame = pd.concat(frames, axis=1)
        # Ensure output directory exists
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
        Create and save a ligth-curve plot showing raw and binned data with RMS legends.

        This hidden method generates a single figure that includes:
            - All individual datapoints plotted with transparency
            - Binned median points in intervals of `self.bin_minutes`.
            - Compute RMS in the legend, either over all data or within specific
                intervals
            - Vertical lines marking the star and end of each intervall in `times`
                (if provided)

        Args:
            obj (str): Name of the object or target
            date (str): Observation date string
            datasets(Dict[str, Tuple[pd.DataFrame, str]]): Mapping from a dataset label
                to a tuple of (DataFrame with 'BJD_TDB', 'rel_flux_T1',
                'rel_flux_err_T1', color string).
            times (Optional[pd.DataFrame]): DataFrame with 'init_time' and 'final_time'
                columns defining time intervals for interval-based RMS calculation and
                plotting. If None, RMS is global.
            out_folder (Path): Directory where the output plot will be saved.
            label_type (str): Descriptor for the kind of label.
            label_value (str): Specific label value ('g', 'r', 'i'; or 'w3', 'w5', 'w7')
                to include in title and filename.

        Returns:
            None
        """
        # Prepare figure
        fig, ax = plt.subplots(figsize=(10, 6))
        bin_days: float = self.bin_minutes / (24 * 60)
        for label, (df, color) in datasets.items():
            t: Series = df["BJD_TDB"]  # Time
            # Normalized flux
            flux: Series = df["rel_flux_T1"] / df["rel_flux_T1"].median()
            # Normalized errors
            err: Series = df["rel_flux_err_T1"] / df["rel_flux_T1"].median()
            # RMS
            rms: float
            rms_txt: str
            median_val: float
            diff: Series

            if times is None:
                median_val = flux.median()
                diff = flux - median_val
                rms = np.sqrt(np.mean(diff**2))
                rms_txt = f" (RMS:{rms:.4f})"
            else:
                mask: Any = np.zeros(len(df), dtype=bool)
                for i, f in zip(times["init_time"], times["final_time"]):
                    mask |= (t >= i) & (t <= f)
                sel = flux[mask]
                if len(sel):
                    median_val = sel.median()
                    diff = sel - median_val
                    rms = np.sqrt(np.mean(diff**2))
                    rms_txt = f" (RMS:{rms:.4f})"
                # If `times.csv` exists but with no data it will still use all data
                else:
                    median_val = flux.median()
                    diff = flux - median_val
                    rms = np.sqrt(np.mean(diff**2))
                    rms_txt = f" (RMS:{rms:.4f})"

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

            # t_arr: NDArray[np.float64] = grp["t"].to_numpy(dtype=np.float64)
            # f_arr: NDArray[np.float64] = grp["f"].to_numpy(dtype=np.float64)
            # e_arr: NDArray[np.float64] = grp["e"].to_numpy(dtype=np.float64)

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

        # legends
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
        Execute the light-curve processing and plotting stage for all objects and dates.

        This method:
            1. Reads the set of already processed (object, date) pairs.
            2. Iterates through each object folder in `self.data_dir`:
                a. Skips non-directory entries.
                b. Ensures a `measurements/` subfolder exists.
            3. For each date subfolder under `measurements/`:
                a. Skips non-directory entries.
                b. Checks if the (object, date) combination is alreadty processed.
                c. obtains all TBL files via `_obtain_tbls()`. Skips if none found.
                d. Groups TBLs by processing method and filter band.
                e. Loads any time intervals via `_load_times().
            4. Generates and saves light-curve plots by:
                a. Method: (`lcs_method` subfolder) using `_create_plot()`.
                b. Filter: (`lcs_filter` subfolder) using `_create_plot()`.
            5. Writes out normalized multiband CSVs via `_save_method_csv()`.
            6. Marks each succesfully completed (object, date) as processed.

        Returns:
            None
        """
        processed: set = self._load_processed()  # Verify processed lcs

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

                # Save the light curve CSV files
                for method, filt_dict in data.items():
                    lcs_folder: Path = lcs_root / date
                    self._save_method_csv(lcs_folder, method, filt_dict)
                # Record it as a processed date
                self._mark_processed(obj, date)
