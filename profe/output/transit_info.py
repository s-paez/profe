from __future__ import annotations

import logging
import os
from pathlib import Path

from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from .naming import get_exofop_id, get_tic_from_toi, get_utc_date_from_bjd

logger = logging.getLogger(__name__)


class TransitDataManager:
    """
    Manager to retrieve and save transit timing information from TESS Transit Finder (TTF).

    This module scrapes the Swarthmore TTF tool to find transit events corresponding
    to the observation dates in the PROFE pipeline.
    """

    def __init__(self) -> None:
        self.base_dir = Path.cwd()
        self.data_dir = self.base_dir / "organized_data"
        self.creds_file = self.base_dir / ".ttf_credentials"
        self.logs_dir = self.base_dir / "logs"
        os.makedirs(self.logs_dir, exist_ok=True)

    def _load_credentials(self) -> tuple[str, str] | None:
        """Loads username and password from .ttf_credentials file."""
        if not self.creds_file.exists():
            logger.warning(
                f"Credentials file {self.creds_file} not found. Skipping TTF data retrieval."
            )
            return None

        try:
            with open(self.creds_file, "r") as f:
                line = f.readline().strip()
                if ":" in line:
                    user, password = line.split(":", 1)
                    return user, password
                else:
                    logger.error(
                        f"Invalid format in {self.creds_file}. Expected 'user:password'."
                    )
                    return None
        except Exception as e:
            logger.error(f"Error reading credentials file: {e}")
            return None

    def _query_ttf_single(
        self,
        tic_id: str,
        start_date_str: str,
        days_to_print: int,
        user: str,
        password: str,
    ) -> requests.Response | None:
        """Send a single TTF query and return the response, or None on failure."""
        url = "https://astro.swarthmore.edu/telescope/tess-secure/print_eclipses.cgi"
        params = {
            "target_string": f"TIC {tic_id}",
            "observatory_string": "31.029167;-115.486944;America/Tijuana;OAN-SPM 2.1m",
            "use_utc": "0",  # Local Time (Evening Date)
            "start_date": start_date_str,
            "days_to_print": str(days_to_print),
            "days_in_past": "0",
            "print_html": "1",
            "maximum_priority": "5",
            "max_airmass": "2.6",
        }

        try:
            response = requests.get(
                url, params=params, auth=(user, password), timeout=15
            )
            response.raise_for_status()
            return response
        except Exception as e:
            logger.error(f"Failed to connect to TTF for TIC {tic_id}: {e}")
            return None

    def _get_ttf_data(self, tic_id: str, target_date: str) -> list[dict]:
        """
        Scrapes TTF for transits of a specific TIC ID near the target date.

        Because the observation date in the pipeline is stored as a UTC calendar
        date but TTF uses the local "Evening Date" at the observatory
        (America/Tijuana, UTC-7/UTC-8), a single-night query can miss transits
        that fall on the boundary between local and UTC dates.

        To handle this robustly the method searches a **3-day local-time
        window** centred on the observation date (obs_date − 1 → obs_date + 1).
        Duplicate transit events are filtered out by their mid-transit BJD.

        Args:
            tic_id (str): The TIC ID (numeric string).
            target_date (str): Observation date in YYYY-MM-DD format.

        Returns:
            list[dict]: List of unique transit events found.
        """
        creds = self._load_credentials()
        if not creds:
            return []

        user, password = creds

        # Search a 3-day window (obs_date - 1 through obs_date + 1) in local
        # time to guarantee we cover any UTC ↔ local-time offset.
        try:
            obs_dt = datetime.strptime(target_date, "%Y-%m-%d")
            search_start = obs_dt - timedelta(days=1)
            start_date_str = search_start.strftime("%m-%d-%Y")
        except Exception as e:
            logger.error(f"Error calculating search date for {target_date}: {e}")
            return []

        days_to_print = 3  # covers obs_date-1, obs_date, obs_date+1

        logger.info(
            f"TTF query for TIC {tic_id}: local-time window "
            f"{search_start.strftime('%Y-%m-%d')} → "
            f"{(search_start + timedelta(days=days_to_print - 1)).strftime('%Y-%m-%d')} "
            f"({days_to_print} days)"
        )

        response = self._query_ttf_single(
            tic_id, start_date_str, days_to_print, user, password
        )
        if response is None:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table")
        if not isinstance(table, Tag):
            logger.info(f"No transit table found for TIC {tic_id} at TTF.")
            return []

        rows = table.find_all("tr")
        if len(rows) < 2:
            return []

        first_row = rows[0]
        if not isinstance(first_row, Tag):
            return []

        headers = [th.get_text(strip=True) for th in first_row.find_all(["td", "th"])]

        try:
            idx_comment = headers.index("Comments andfollowup status")
            idx_times = next(i for i, h in enumerate(headers) if "BJDTDB" in h)
            # Find Depth and Duration columns using flexible matching
            idx_depth = next(
                (i for i, h in enumerate(headers) if "depth" in h.lower()), None
            )
            idx_dur = next(
                (i for i, h in enumerate(headers) if "durat" in h.lower()), None
            )
        except (ValueError, StopIteration) as e:
            logger.error(
                f"Could not find expected columns in TTF table for TIC {tic_id}: {e}"
            )
            return []

        results = []
        seen_mids: set[str] = set()  # deduplicate by mid-transit BJD

        for row in rows[1:]:
            if not isinstance(row, Tag):
                continue
            cols = row.find_all("td")
            if len(cols) <= max(
                idx_comment,
                idx_times,
                idx_depth if idx_depth is not None else 0,
                idx_dur if idx_dur is not None else 0,
            ):
                continue

            comment = cols[idx_comment].get_text(strip=True)
            times_raw = cols[idx_times].get_text()
            depth = (
                cols[idx_depth].get_text(strip=True) if idx_depth is not None else ""
            )
            duration = cols[idx_dur].get_text(strip=True) if idx_dur is not None else ""

            # Parsing logic from reference script
            parts = times_raw.replace("\u2014", " ").replace("\n", " ").split()

            if len(parts) >= 3:
                try:
                    t1_str = parts[0]
                    t2_str = parts[1]
                    t3_str = parts[2]

                    t1 = float(t1_str)
                    if t1 < 1000000:
                        t1 += 2450000

                    t1_int = int(t1)

                    def parse_part(part: str, prev_val: float) -> float:
                        if part.startswith("."):
                            val = t1_int + float(part)
                            if val < prev_val:  # Day crossing
                                val += 1.0
                            return val
                        val = float(part)
                        if val < 1000000:
                            val += 2450000
                        return val

                    t2 = parse_part(t2_str, t1)
                    t3 = parse_part(t3_str, t2)

                    mid_key = f"{t2:.4f}"
                    if mid_key in seen_mids:
                        continue  # skip duplicate transit from overlapping window
                    seen_mids.add(mid_key)

                    results.append(
                        {
                            "comment": comment,
                            "ingress": f"{t1:.4f}",
                            "mid": mid_key,
                            "egress": f"{t3:.4f}",
                            "depth": depth,
                            "duration": duration,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error parsing transit times for TIC {tic_id}: {e}")

        logger.info(
            f"TTF returned {len(results)} unique transit(s) for TIC {tic_id} "
            f"in 3-day window around {target_date}"
        )
        return results

    def run(self, target: str | None = None) -> None:
        """Processes all objects and dates to generate transit data .dat files.

        Args:
            target (str | None): If specified, only process this target.
        """
        if not self.data_dir.exists():
            logger.error(f"Data directory {self.data_dir} does not exist.")
            return

        for obj_folder in sorted(self.data_dir.iterdir()):
            if not obj_folder.is_dir():
                continue
            if target and obj_folder.name.lower() != target.lower():
                continue

            target_name = obj_folder.name  # e.g. TOI-3884.01
            meas_root = obj_folder / "measurements"
            if not meas_root.exists():
                continue

            for date_folder in sorted(meas_root.iterdir()):
                if not date_folder.is_dir():
                    continue

                obs_date = date_folder.name  # YYYY-MM-DD

                # Check if file already exists
                # Standard name: target-01_yyyymmdd_OAN-SPM-2m1-OPTICAM_transit_times.dat
                # Note: naming.py handles the compact date and target standardization.
                exofop_id = get_exofop_id(target_name)  # e.g. TIC86263325
                utc_date = get_utc_date_from_bjd(date_folder)
                date_compact = utc_date.replace("-", "")

                # Place at the date level in exofop folder
                exofop_dir = obj_folder / "exofop" / obs_date
                out_file = (
                    exofop_dir
                    / f"{exofop_id}-01_{date_compact}_OAN-SPM-2m1-OPTICAM_transit_times.dat"
                )

                if out_file.exists():
                    logger.info(
                        f"Skipping {target_name} on {obs_date}: {out_file.name} already exists."
                    )
                    continue

                # Fetch TIC ID numeric part for TTF
                tic_numeric = exofop_id.replace("TIC", "").strip()
                if not tic_numeric.isdigit():
                    # Try to fetch it if it's just TOI
                    tic_str = get_tic_from_toi(target_name)
                    tic_numeric = tic_str.replace("TIC", "").strip()

                if not tic_numeric.isdigit():
                    logger.warning(
                        f"Could not determine TIC ID for {target_name}. Skipping TTF."
                    )
                    continue

                logger.info(
                    f"Fetching transit data for {target_name} (TIC {tic_numeric}) on {obs_date}..."
                )
                transits = self._get_ttf_data(tic_numeric, obs_date)

                if not transits:
                    logger.info(
                        f"No relevant transits found for {target_name} on {obs_date}."
                    )
                    continue

                # Write results
                exofop_dir.mkdir(parents=True, exist_ok=True)
                try:
                    with open(out_file, "w") as f:
                        f.write(
                            "TOI\tTIC\tIngress(BJD)\tMid(BJD)\tEgress(BJD)\tDepth(ppt)\tDuration(hrs)\tComment\n"
                        )
                        for t in transits:
                            f.write(
                                f"{target_name}\t{tic_numeric}\t{t['ingress']}\t{t['mid']}\t{t['egress']}\t{t['depth']}\t{t['duration']}\t{t['comment']}\n"
                            )
                    logger.info(f"Saved transit data to {out_file}")
                except Exception as e:
                    logger.error(
                        f"Failed to write transit data for {target_name} on {obs_date}: {e}"
                    )
