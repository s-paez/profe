"""
Time-averaging analysis for the light curves.

This module performs the time-averaging analysis of the light curve (or a no-transit nor
stellar-contaminated regions if times.csv file exist for a given target and night)

It uses the MC3 package prosed by Cubillos et. al (2017).
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

    This class processes all measurements TBLs for each object and observation date in
    the `organized_data` directory. For each light curve it computes the RMS of the
    residuals as a function of time-bin size, compares the results to the expectation
    for the white noise, and produces red-noise plots illustrating correlated noise at
    different time scales.
    """

    def __init__(self) -> None:
        """
        Initialize TimeAveragingPlotter with directory paths and processing settings.

        It sets up:
        - The base working directory.
        - The location of organized measurement TBLs.
        - The number of parallel processes to use.
        - The directory for logs and output plots.
        - A logger for status messages.

        Attributes:
            base_dir (Path):
                Current working directory where the pipeline is executed.
            data_dir (Path):
                Directory containing per-object, per-date `measurements/` TBL files.
            n_processes (int):
                Number of CPU cores to use for parallel processing.
            log_dir (Path):
                Directory where log files and red-noise plots will be saved.
            logger (logging.Logger):
                Logger instance for informational and error messages.
        """
        self.base_dir = Path.cwd()
        self.data_dir = self.base_dir / "organized_data"
        self.n_processes = cpu_count()
        self.log_dir = self.base_dir / "logs"
        self.logger = logging.getLogger(__name__)

    def load_times_file(self, date_dir: Path) -> DataFrame | None:
        """
        Load the first time-interval CSV for filtering data.

        This method looks for a subdirectory named 'times' under 'measurements folder',
        and attempts to read the first CSV file found there. The CSV is expected to
        contain at least two columns: 'init_time' and 'final_time'. If the 'times'
        directory does not exist, contains no CSV files, or the first CSV is empty,
        the method returns None to indicate that no time-based mask should be
        applied (i.e., use all data points).

        Args:
            date_dir (Path): Path to the directory for a specific observation date,
                            which may contain a 'times' subfolder.

        Returns:
            Optional[DataFrame]:
                A DataFrame with 'init_time' and 'final_time' columns if a valid
                CSV is found; otherwise, None.
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
        Generate a mask with rows whose 'BJD_TDB' value falls within any time interval.

        If `times` is None, returns an array of all True values.

        Args:
            data (DataFrame):
                Input DataFrame containing a 'BJD_TDB' column of Julian Dates.
            times (Optional[DataFrame]):
                DataFrame with 'init_time' and 'final_time' columns defining inclusion
                intervals.
                Each row represents one interval [init_time, final_time]. If None, no
                masking is applied.

        Returns:
            NDArray[np.bool_]:
                Boolean array of length len(data), where True indicates the
                corresponding row's 'BJD_TDB' lies within at least one interval.
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
        Compute time-averaged noise metrics for a single filter-method TBL.

        This private method processes a light-curve TBL by:
            1. Reading the file into a DataFrame.
            2. Applying a boolean mask (`self.mask`) to select data within specified
                time intervals.
            3. Normalizing the 'rel_flux_T1' values by their median.
            4. Computing residuals as 1 minus the normalized flux.
            5. Running `ms.time_avg` on the residuals to obtain:
                - RMS (`rms`)
                - Lower RMS uncertainty (`rmslo`)
                - Upper RMS uncertainty (`rmshi`)
                - Standard error (`stderr`)
                - Bin sizes (`binszmc`)
            6. Packaging the results into a summary DataFrame with columns:
                `{filter}-{method}-rms`, `{filter}-{method}-rmslo`,
                `{filter}-{method}-rmshi`, `{filter}-{method}-stderr`, and
                `{filter}-{method}-binszmc`.
            7. Logging the operation and returning `(filter, method, summary_df)`.

            If masking removes all data, returns `(filter, method, empty DataFrame)`.

        Args:
            args (tuple):
                fil (str): Filter name (e.g., 'g', 'r', 'i').
                meth (str): Method identifier.
                tbl_path (str or Path): Path to the light-curve TBL file.
                times (DataFrame or None): Optional DataFrame with 'init_time' and
                'final_time' columns defining intervals for time-based masking.

        Returns:
            tuple[str, str, DataFrame]:
                A tuple containing the filter name, method name, and a DataFrame of
                time-averaging results.
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
        Execute time-averaging analysis and plotting for all objects and dates.

        This method performs the following steps for each object in `self.data_dir`:

        1. Skip non-directory entries.
        2. Verify that a `measurements/` subfolder exists; skip otherwise.
        3. For each date subfolder under `measurements/`:

            a. Load optional time-interval definitions via `self.load_times_file()`.
            b. Collect all tbl files named `<obj>_<filter>_<method>.tbl`.
            c. Build a list of (filter, method, path, times) tuples for valid files.
            d. Use a multiprocessing Pool (size `self.n_processes`) to apply
                `self.process_filter_method()` to each tuple.
            e. Aggregate the returned DataFrames by filter into `time_avgs`.
            f. For each filter, generate a red-noise plot of RMS vs. bin size:
                - Error bars for RMS with lower/upper uncertainties.
                - Dashed lines for standard error.
                - Save figure under `plots/<date>/time-avg/`.

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
