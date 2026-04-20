import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .naming import get_exofop_id, normalize_band

logger = logging.getLogger(__name__)

# Pixel scales for each band as defined in tmp/generate_report.py
PIXEL_SCALES = {"gp": 0.139, "rp": 0.140, "ip": 0.166}

OBSERVERS = "A. Khandelwal, Y. Gómez Maqueo Chew, M. Pichardo Marcano, S. Páez"
EXPECTED_OBS = "OAN-SPM-OPTICAM-2m1"


class ReportGenerator:
    """
    Generate consolidated ExoFOP reports (notes) for each observation night.

    Extracts metrics from AIJ measurement tables and fit panels for all observed bands,
    incorporates predicted transit data from TTF, and formats them into a single
    text report for ExoFOP submission.
    """

    def __init__(self) -> None:
        self.base_dir = Path.cwd()
        self.data_dir = self.base_dir / "organized_data"
        self.logs_dir = self.base_dir / "logs"
        os.makedirs(self.logs_dir, exist_ok=True)

    def _extract_measurement_metrics(self, tbl_file: Path, band: str) -> dict[str, Any]:
        """Extract FWHM, Aperture, and Shift metrics from a measurement table."""
        metrics: dict[str, Any] = {"scale": PIXEL_SCALES.get(band, 0.15)}
        try:
            df = pd.read_csv(tbl_file, sep=r"\t+", engine="python", encoding="latin1")

            if "Radius" in df.columns:
                metrics["ap_px"] = float(df["Radius"].iloc[0])
            elif "Source_Radius" in df.columns:
                metrics["ap_px"] = float(df["Source_Radius"].iloc[0])

            if "Width_T1" in df.columns:
                w = df["Width_T1"].dropna()
                if not w.empty:
                    metrics["fwhm_px"] = np.median(w)

            if "X(FITS)_T1" in df.columns and "Y(FITS)_T1" in df.columns:
                xc = df["X(FITS)_T1"].dropna()
                yc = df["Y(FITS)_T1"].dropna()
                if not xc.empty and not yc.empty:
                    metrics["shift_x_std"] = np.std(xc) * metrics["scale"]
                    metrics["shift_y_std"] = np.std(yc) * metrics["scale"]
                    metrics["shift_x_max"] = np.max(np.abs(xc - np.median(xc)))
                    metrics["shift_y_max"] = np.max(np.abs(yc - np.median(yc)))

        except Exception as e:
            logger.error(f"Error extracting metrics from {tbl_file}: {e}")

        return metrics

    def _extract_fit_metrics(self, aij_dir: Path) -> dict[str, Any]:
        """Extract Depth, Tc, Rp/Rs, RMS, and Duration from fitpanel files."""
        metrics: dict[str, Any] = {}
        if not aij_dir.exists():
            return metrics

        fit_files = [
            f
            for f in aij_dir.iterdir()
            if f.is_file() and "_rel_flux_T1.txt" in f.name and "fitpanel" in f.name
        ]

        if not fit_files:
            return metrics

        fit_file = fit_files[0]
        try:
            with open(fit_file, "r") as ff:
                flines = ff.readlines()

            for line in flines:
                if "(Rp/R*)^2" in line:
                    match = re.search(r"(Fitted|Fixed|Calc|Stat)\s+([\d\.\-]+)", line)
                    if match:
                        metrics["rprs2"] = float(match.group(2))
                        metrics["depth_ppt"] = metrics["rprs2"] * 1000.0
                elif line.strip().startswith("Tc"):
                    match = re.search(r"(Fitted|Fixed|Calc|Stat)\s+([\d\.\-]+)", line)
                    if match:
                        metrics["tc"] = float(match.group(2))
                elif "t14 (h:m:s)" in line:
                    match = re.search(r"Calc\s+([\d:]+)", line)
                    if match:
                        hms = match.group(1).split(":")
                        if len(hms) >= 2:
                            metrics["duration"] = f"{int(hms[0])}:{int(hms[1]):02d}"
                elif "RMS (normalized)" in line:
                    match = re.search(r"Stat\s+([\d\.\-]+)", line)
                    if match:
                        metrics["rms_ppt"] = float(match.group(1)) * 1000.0
        except Exception as e:
            logger.error(f"Error extracting fit metrics from {fit_file}: {e}")

        return metrics

    def _get_predicted_metrics(
        self, obj_folder: Path, date: str, exofop_id: str
    ) -> dict[str, str]:
        """Read predicted transit metrics from the .dat file created by transit_info.py."""
        date_compact = date.replace("-", "")
        dat_file = (
            obj_folder
            / "exofop"
            / date
            / f"{exofop_id}-01_{date_compact}_OAN-SPM-2m1-OPTICAM_transit_times.dat"
        )

        pred = {
            "tc": "[Predicted_Tc]",
            "depth": "[Predicted_Depth]",
            "duration": "[Predicted_Duration]",
            "comment": "[Previous TTF comments]",
        }

        if dat_file.exists():
            try:
                df = pd.read_csv(dat_file, sep="\t")
                if not df.empty:
                    # Select the first transit as requested
                    row = df.iloc[0]
                    pred["tc"] = str(row["Mid(BJD)"])
                    pred["depth"] = str(row["Depth(ppt)"])
                    pred["duration"] = str(row["Duration(hrs)"])
                    pred["comment"] = str(row["Comment"])
            except Exception as e:
                logger.error(f"Error reading predicted metrics from {dat_file}: {e}")

        return pred

    def _generate_report(
        self,
        target_name: str,
        exofop_id: str,
        date: str,
        bands_data: dict[str, dict[str, Any]],
        predicted: dict[str, str],
    ) -> str:
        """Generate the consolidated report text."""
        date_dots = date.replace("-", ".")
        bands_present = sorted(bands_data.keys())
        band_str = ", ".join(bands_present)

        # Build multi-band metric strings
        aps = []
        fwhms = []
        for b in bands_present:
            m = bands_data[b]
            ap = f'{(m.get("ap_px", 0) * m["scale"]):.1f}"' if "ap_px" in m else "[Ap]"
            fwhm = (
                f'{(m.get("fwhm_px", 0) * m["scale"]):.1f}"'
                if "fwhm_px" in m
                else "[FWHM]"
            )
            aps.append(f"{b}: {ap}")
            fwhms.append(f"{b}: {fwhm}")

        ap_line = ", ".join(aps)
        fwhm_line = ", ".join(fwhms)

        obs_tc_lines = []
        depth_lines = []
        rprs2_lines = []
        rms_lines = []
        dur_lines = []

        for b in bands_present:
            m = bands_data[b]
            tc_val = f"{m['tc']:.6f}" if "tc" in m else "[Measured_Tc]"
            depth_val = f"{m['depth_ppt']:.2f}" if "depth_ppt" in m else "[Depth]"
            rprs2_val = f"{m['rprs2']:.4f}" if "rprs2" in m else "[RpRs2]"
            rms_val = f"{m['rms_ppt']:.2f}" if "rms_ppt" in m else "[RMS]"
            dur_val = m.get("duration", "[Duration]")

            obs_tc_lines.append(f"{b}: {tc_val}")
            depth_lines.append(f"{b}: {depth_val} ppt")
            rprs2_lines.append(f"{b}: {rprs2_val}")
            rms_lines.append(f"{b}: {rms_val} ppt")
            dur_lines.append(f"{b}: {dur_val}")

        obs_tc_block = "    OBSERVED Tc (BJD_TDB):  " + ", ".join(obs_tc_lines)
        depth_block = "    MEASURED DEPTH (from AIJ):  " + ", ".join(depth_lines)
        rprs2_block = "    (Rp/R*)^2 (from AIJ analysis):  " + ", ".join(rprs2_lines)
        rms_block = "    RMS of AIJ FIT: " + ", ".join(rms_lines)
        dur_block = "    MEASURED DURATION (HH:MM; from AIJ): " + ", ".join(dur_lines)

        # Shifts (approx from first available band as in original script)
        ref_b = bands_present[0]
        m_ref = bands_data[ref_b]
        shift_x_std = (
            f"{m_ref['shift_x_std']:.4f}" if "shift_x_std" in m_ref else "[X_std]"
        )
        shift_y_std = (
            f"{m_ref['shift_y_std']:.4f}" if "shift_y_std" in m_ref else "[Y_std]"
        )
        shift_x_max = (
            f"{m_ref['shift_x_max']:.4f}" if "shift_x_max" in m_ref else "[X_max]"
        )
        shift_y_max = (
            f"{m_ref['shift_y_max']:.4f}" if "shift_y_max" in m_ref else "[Y_max]"
        )

        # Timing analysis for Interpretation
        interpretation_text = ""
        try:
            # Check if predicted tc is a number
            pred_tc_val = float(predicted["tc"])
            # Get measured Tc from first band if possible
            first_m = bands_data[bands_present[0]]
            if "tc" in first_m:
                meas_tc_val = float(first_m["tc"])
                diff_min = (meas_tc_val - pred_tc_val) * 24 * 60
                direction = "after" if diff_min > 0 else "before"
                interpretation_text = f"A transit was observed {direction} the predicted one with a difference of {abs(diff_min):.2f} minutes."
        except (ValueError, TypeError):
            interpretation_text = "[Timing analysis pending]"

        template = f"""{exofop_id} (TOI {target_name}) on UT{date_dots} from {EXPECTED_OBS} in {band_str}

{OBSERVERS}/{EXPECTED_OBS} observed a full transit on {date} in {band_str} and detected a {depth_lines[0].split(": ")[1] if depth_lines else "[Depth]"} event using uncontaminated {ap_line} target apertures. [(Rp/R*)^2 (from AIJ analysis): {rprs2_lines[0].split(": ")[1] if rprs2_lines else "[RpRs2]"}]

1.  GOAL(S):
    [] Analyze if the transit is on the target star
    [] Analyze the transit chromaticity
    [] Analyze the transit timing

2.  INTERPRETATION OF RESULTS:
    {interpretation_text}


3.  APERTURE RADIUS: {ap_line}

4.  PREDICTED Tc (BJD_TDB):     {predicted["tc"]}
{obs_tc_block}

    OBSERVATION COVERAGE:
    PREDICTED DEPTH:                {predicted["depth"]} ppt
{depth_block}

{rprs2_block}

{rms_block}
    PREDICTED DURATION (HH:MM):              {predicted["duration"]}
{dur_block}

5.  FWHM: {fwhm_line}

6.  ExoFOP-TESS STATUS: All files have been uploaded to ExoFOP-TESS.

7.  OBSERVING NOTES:

    Image shift statistics (approx across bands):
             stdev.(arc-seconds): {shift_x_std} in X and {shift_y_std} in Y
             max. deviation (pixels): {shift_x_max} in X and {shift_y_max} in Y

    Detrend parameters: [e.g., AIRMASS, X(FITS)_T1, etc.]

8. PREVIOUS TTF COMMENTS: {predicted["comment"]}
"""
        return template

    def run(self) -> None:
        """Process all objects and dates to generate consolidated notes files."""
        if not self.data_dir.exists():
            logger.error(f"Data directory {self.data_dir} does not exist.")
            return

        for obj_folder in sorted(self.data_dir.iterdir()):
            if not obj_folder.is_dir():
                continue

            target_name = obj_folder.name
            meas_root = obj_folder / "measurements"
            if not meas_root.exists():
                continue

            for date_folder in sorted(meas_root.iterdir()):
                if not date_folder.is_dir():
                    continue

                date = date_folder.name
                exofop_id = get_exofop_id(target_name)
                date_compact = date.replace("-", "")

                # Collect data for all bands
                meas_files = [
                    f
                    for f in date_folder.iterdir()
                    if f.is_file()
                    and not f.name.startswith(".")
                    and f.suffix in (".tbl", ".csv")
                ]

                if not meas_files:
                    continue

                bands_data = {}
                for mf in meas_files:
                    stem = mf.stem
                    band = normalize_band(stem.split("_")[-1])

                    metrics = self._extract_measurement_metrics(mf, band)
                    aij_dir = obj_folder / "exofop" / date / "AIJ" / band
                    fit_metrics = self._extract_fit_metrics(aij_dir)
                    if fit_metrics:
                        metrics.update(fit_metrics)
                        bands_data[band] = metrics

                if not bands_data:
                    continue

                # Predicted metrics from transit_info.py
                predicted = self._get_predicted_metrics(obj_folder, date, exofop_id)

                # Naming follows TICID-01_<date>_OAN-SPM-2m1-OPTICAM_<gp-rp-ip>_notes.txt
                bands_suffix = "-".join(sorted(bands_data.keys()))
                filename = f"{exofop_id}-01_{date_compact}_OAN-SPM-2m1-OPTICAM_{bands_suffix}_notes.txt"
                out_dir = obj_folder / "exofop" / date
                out_file = out_dir / filename

                if out_file.exists():
                    logger.info(f"Skipping {out_file.name}: already exists.")
                    continue

                logger.info(
                    f"Generating consolidated report for {target_name} on {date} [{bands_suffix}]"
                )

                report_content = self._generate_report(
                    target_name, exofop_id, date, bands_data, predicted
                )

                out_dir.mkdir(parents=True, exist_ok=True)
                try:
                    with open(out_file, "w") as f:
                        f.write(report_content)
                    logger.info(f"Saved consolidated report to {out_file}")
                except Exception as e:
                    logger.error(f"Failed to write report {out_file}: {e}")
