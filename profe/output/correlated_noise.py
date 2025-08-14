"""
Time-averaging noise analysis for light curves.

This module computes the root mean square (RMS) of light-curve residuals as a
function of bin size to characterize correlated (red) noise. For each target and
observation night, it uses either the full light curve or user-defined, non-transit
time intervals provided via a `times.csv` file.

The analysis follows the time-averaging method proposed by Cubillos et al. (2017)
using the MC3 package.
"""

import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mc3.stats as ms
import numpy as np
import pandas as pd
from pandas import DataFrame, Series


class TimeAveragingPlotter:
    """
    Perform time-averaging noise analysis and generate red-noise plots.

    This class iterates over all objects and dates in the `organized_data` directory,
    computing RMS-vs.-bin-size curves for each filter and photometric method. The
    results are plotted and saved to disk, with dashed lines indicating the expected
    white-noise scaling.
    """

    def __init__(self) -> None:
        """
        Initialize the time-averaging plotter.

        Sets directory paths, determines the number of CPU cores for parallel
        processing, and configures the logger.

        Attributes:
            base_dir (Path): Working directory where the pipeline is executed.
            data_dir (Path): Directory containing organized measurement TBL files.
            n_processes (int): Number of CPU cores for parallel processing.
            log_dir (Path): Directory where log files and output plots are saved.
            logger (logging.Logger): Logger instance for status and error messages.
        """
        self.base_dir = Path.cwd()
        self.data_dir = self.base_dir / "organized_data"
        self.n_processes = cpu_count()
        self.log_dir = self.base_dir / "logs"
        self.logger = logging.getLogger(__name__)

    def load_times_file(self, date_dir: Path) -> DataFrame | None:
        """
        Load an optional CSV defining time intervals to include in the analysis.

        If a `times` subfolder exists under the given date directory, this method
        reads the first CSV file found. The file must contain `init_time` and
        `final_time` columns. If the folder or CSV is missing or empty, returns None.

        Args:
            date_dir (Path): Path to the date-specific observation folder.

        Returns:
            Optional[DataFrame]: DataFrame with time intervals, or None if no
            valid CSV is found.
        """
        times_dir: Path = date_dir / "times"
        if not times_dir.exists() or not times_dir.is_dir():
            msg: str = f"Times folder does not exist in {date_dir.name}."
            "Using all datapoints."
            self.logger.warning(msg)
            return None
        csvs: list[Path] = list(times_dir.glob("*.csv"))
        if not csvs:
            msg = f"No CSV in {times_dir}. Using all datapoints."
            self.logger.warning(msg)
            return None
        df_times: DataFrame = pd.read_csv(csvs[0])
        if df_times.empty:
            msg = f"Empty {times_dir} file. Using all datapoints."
            self.logger.warning(msg)
            return None

        return df_times

    def mask(self, data: DataFrame, times: DataFrame | None) -> Any:
        """
        Create a boolean mask for data points within the specified time intervals.

        If `times` is None, returns an array of True values for all rows.

        Args:
            data (DataFrame): Data containing a 'BJD_TDB' column.
            times (Optional[DataFrame]): Time intervals with 'init_time' and
                'final_time' columns.

        Returns:
            np.ndarray: Boolean mask where True means the row is inside at
            least one interval.
        """
        if times is None:
            mask: Any = np.ones(len(data), dtype=bool)
            return mask

        data_JD: Any = data["BJD_TDB"].values
        start: Any = times["init_time"].values
        end: Any = times["final_time"].values
        mask_array: Any = (data_JD >= start[:, None]) & (data_JD <= end[:, None])
        return np.any(mask_array, axis=0)

    def process_filter_method(self, args: tuple) -> tuple:
        """
        Compute time-averaging noise metrics for one filter–method combination.

        Reads a light-curve TBL file, optionally applies a time mask, normalizes
        the flux, calculates residuals, and uses `mc3.stats.time_avg` to compute:

            - RMS
            - Lower RMS uncertainty
            - Upper RMS uncertainty
            - Standard error
            - Bin sizes

        Args:
            args (tuple): (filter, method, tbl_path, times) where:
                filter (str): Photometric filter name.
                method (str): Photometric extraction method.
                tbl_path (Path): Path to the TBL file.
                times (Optional[DataFrame]): Time intervals for masking.

        Returns:
            tuple[str, str, DataFrame]: Filter, method, and the computed metrics
            as a DataFrame.
        """
        fil, meth, tbl_path, times = args
        df: DataFrame = pd.read_table(tbl_path, encoding="latin1", sep="\t")
        mask_idx: Series = self.mask(df, times)
        sub: DataFrame | Series = df[mask_idx]

        if sub.empty:
            msg: str = f"No data after masking for {fil}-{meth}."
            self.logger.warning(msg)
            return fil, meth, DataFrame()

        flux: Any = sub["rel_flux_T1"].values
        med: float = float(np.median(flux))
        norm: Any = flux / med
        residuals: Any = 1 - norm
        maxbins: int = len(residuals) // 2

        rms, rmslo, rmshi, stderr, bins = ms.time_avg(residuals, maxbins)
        pref: str = f"{fil}-{meth}"
        df_out: DataFrame = DataFrame(
            {
                f"{pref}-rms": rms,
                f"{pref}-rmslo": rmslo,
                f"{pref}-rmshi": rmshi,
                f"{pref}-stderr": stderr,
                f"{pref}-binszmc": bins,
            }
        )

        msg = f" Time-averaging: {fil}-{meth}"
        self.logger.info(msg)
        return fil, meth, df_out

    def run(self) -> None:
        """
        Run the time-averaging analysis for all objects and dates.

        For each target and observation date:
            1. Load optional time intervals from a CSV.
            2. Identify all TBL files and parse their filter/method names.
            3. Process each filter–method combination in parallel.
            4. Aggregate results by filter.
            5. Generate red-noise plots and save them to disk.

        Returns:
            None
        """
        for obj_dir in sorted(self.data_dir.iterdir()):
            if not obj_dir.is_dir():
                continue
            meas_root: Path = obj_dir / "measurements"
            if not meas_root.exists() or not meas_root.is_dir():
                msg: str = f"No measurements in {obj_dir.name}. Skipping."
                self.logger.warning(msg)
                continue

            for date_dir in sorted(meas_root.iterdir()):
                if not date_dir.is_dir():
                    continue
                msg = f"Processing {obj_dir.name} Date {date_dir.name}"
                self.logger.info(msg)

                times_df: DataFrame | None = self.load_times_file(date_dir)

                tbls: list = list(date_dir.glob("*.tbl"))
                if not tbls:
                    msg = f"No TBL in {date_dir}. Skipping."
                    self.logger.warning(msg)
                    continue

                filters: set = set()
                methods: set = set()
                fmap: dict = {}
                for p in tbls:
                    parts: list = p.stem.split("_")
                    if len(parts) < 3:
                        continue
                    f, m = parts[1], parts[2]
                    filters.add(f)
                    methods.add(m)
                    fmap[(f, m)] = p

                if not fmap:
                    msg = f"Invalid format in {date_dir}. Skipping."
                    self.logger.warning(msg)
                    continue

                # Build argument list for multiprocessing
                filters = set(sorted(filters))
                methods = set(sorted(methods))
                args: list = [
                    (f, m, fmap[(f, m)], times_df)
                    for f in filters
                    for m in methods
                    if (f, m) in fmap
                ]

                # Process each (filter, method) in parallel
                with Pool(processes=self.n_processes) as pool:
                    results: list = pool.map(self.process_filter_method, args)

                # Aggregate results by filter
                time_avgs: dict = {f: DataFrame() for f in filters}
                for f, m, df_r in results:
                    if not df_r.empty:
                        time_avgs[f] = pd.concat([time_avgs[f], df_r], axis=1)

                cmap: plt.Colormap = plt.get_cmap("Paired")
                colors: dict = {m: cmap(i % cmap.N) for i, m in enumerate(methods)}

                # Prepare output directory for time-avg plots
                out_root: Path = obj_dir / "plots" / date_dir.name / "time-avg"
                out_root.mkdir(parents=True, exist_ok=True)

                # Generate and save red-noise plots per filter
                for f in filters:
                    dfp: DataFrame = time_avgs[f]
                    if dfp.empty:
                        msg = f"No data in filter{f}."
                        self.logger.warning(msg)
                        continue

                    plt.figure(figsize=(8, 6))
                    for m in methods:
                        pre: str = f"{f}-{m}"
                        colb: str = f"{pre}-binszmc"
                        if colb not in dfp:
                            continue
                        b: DataFrame | Series | Any = dfp[colb]
                        r: DataFrame | Series | Any = dfp[f"{pre}-rms"]
                        lo: DataFrame | Series | Any = dfp[f"{pre}-rmslo"]
                        hi: DataFrame | Series | Any = dfp[f"{pre}-rmshi"]
                        se: DataFrame | Series | Any = dfp[f"{pre}-stderr"]

                        plt.errorbar(
                            b,
                            r,
                            yerr=[lo, hi],
                            fmt="-",
                            capsize=0,
                            color="0.8",
                            zorder=0,
                        )
                        plt.loglog(b, r, ls="-", label=m, color=colors[f"{m}"])
                        plt.loglog(b, se, ls="--", lw=2, color=colors[f"{m}"])

                    plt.legend(loc="best")
                    plt.xlabel("Bin size")
                    plt.ylabel("RMS")
                    plt.title(f"{obj_dir.name} | {date_dir.name} | filter {f}")
                    plt.tight_layout()

                    fname: Path = out_root / f"time-averaging-{f}.png"
                    plt.savefig(fname, dpi=300)
                    plt.close()
                    msg = f"{fname} Saved."
                    self.logger.info(msg)
