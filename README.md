# PROFE: Pipeline de Reducción de OPTICAM para Fotometría de Exoplanetas
[![A rectangular badge, half black half purple containing the text made at Code Astro](https://img.shields.io/badge/Made%20at-Code/Astro-blueviolet.svg)](https://semaphorep.github.io/codeastro/)

#### Reduction pipeline for OPTICAM photometry of exoplanets.

A Python-based pipeline to automate preprocessing and postprocessing of data acquired with the OPTICAM instrument on the 2.1 m telescope at OAN‑SPM, aimed for implementing the data reduction method proposed by [Paez et. al (2026)](https://doi.org/10.1093/rasti/rzag021).

---

## Features

* **Preprocessing** (`profe -p`):

  * Organize and standardize FITS files by target and date.
  * Update headers: compute and insert `JD` and `UTMIDDLE` keywords.
  * Apply a 3×3 median filter for hot-pixel correction.
  * Generate an image-count summary (`images_summary.dat`).

* **Postprocessing** (`profe -o`):
  * Multi-band binned light curves with RMS measurements (PDF & CSV).
  * Single-band ExoFOP-standardized light curve plots (PNG).
  * Time-averaging noise plots (Red vs White noise characterization via MC3).
  * Altitude–Azimuth trajectory plots.
  * Centroid movement plots.
  * Aperture visualization (Field of View) plots.
  * Radial (seeing) profile plots.
  * Comparison star light curves (up to 6 stars per band).
  * Automated WCS solving via Astrometry.net (source list method).
  * Transit timing data retrieval from TESS Transit Finder (TTF).
  * Consolidated ExoFOP notes report generation.
  * Archival copy of measurement tables per band.

---

## Requirements
Dependencies defined in `pyproject.toml`:

* Python (≥3.11, <3.12)
* astropy (5.3.2)
* scipy (≥1.10, <2.0)
* matplotlib (≥3.10.3, <4.0.0)
* tqdm (≥4.67.1, <5.0.0)
* pandas (≥2.3.1, <3.0.0)
* mc3 (≥3.2.1, <4.0.0)
* photutils (≥2.2.0, <3.0.0)
* numpy (≥1.24, <2.0)
* astroquery (≥0.4.7)
* beautifulsoup4 (≥4.12.0)
* requests (≥2.31.0)

### External Setup

**WCS Solving** — Provide an [Astrometry.net](https://nova.astrometry.net/) API key:
1. Create a file named `.astrometry_key` at the root of your working directory.
2. Paste your 16-character API key inside the file.

**Transit Timing (TTF)** — Provide TESS Transit Finder credentials:
1. Create a file named `.ttf_credentials` at the root of your working directory.
2. Write your credentials in the format `username:password`.

---

## Installation

Install from PyPI:

```bash
pip install profe
```

For a development environment (includes testing, linting, and profiling tools):

```bash
git clone https://github.com/s-paez/profe.git
cd profe
pip install -e ".[dev]"
```

---

## Usage

### CLI Reference

PROFE provides a single entry point with mutually exclusive commands:

| Command | Description |
|---|---|
| `profe -p` | Run the **full preprocessing** pipeline (organize + median filter). |
| `profe --organice` | Run **only** the file reorganization and header update stage. |
| `profe --filter` | Run **only** the median filter stage. Skips if `corrected_3x3/` already exists. |
| `profe -o [TARGET]` | Run the **full postprocessing** and output generation pipeline. Optionally specify a target name to process only that target. |
| `profe man` | Display the detailed built-in manual. |
| `profe -h` | Show the quick-reference help message. |

**Optional flags:**

| Flag | Description |
|---|---|
| `-c CORES` / `--cores CORES` | Number of CPU cores for preprocessing (default: all available). Only valid with `-p`, `--organice`, or `--filter`. |
| `-o TARGET` | When a target name is provided (e.g., `profe -o "TOI-1234"`), only that target is processed. Without a target, all targets are processed. |

### Preprocessing

**PROFE** recursively searches for original FITS files inside a `data/` directory relative to where the command is executed. The pipeline organizes them, updates FITS headers (Julian Date and UTMIDDLE), and applies median filter corrections.

```bash
# Full preprocessing with all CPU cores
profe -p

# Full preprocessing with limited cores
profe -p -c 4

# Only reorganize files and update headers (no filter)
profe --organice

# Only apply the median filter (skips if corrected_3x3/ already exists)
profe --filter
```

#### Preprocessing Steps

1. **Header Update** — Computes `JD` (Julian Date at start) and `UTMIDDLE` (ISO mid-exposure) from existing `UT` and `EXPOSURE` keywords. Skips files already tagged with a PROFE HISTORY entry.
2. **File Organization** — Sorts FITS files into `organized_data/{TARGET}/raw/{DATE}/` subdirectories based on `OBJECT` and `DATE-OBS` header keywords. Creates `measurements/`, `lcs/`, and `exofop/` directories for science targets.
3. **Summary Generation** — Creates `logs/images_summary.dat` with image counts per target and date.
4. **Median Filter** — Applies a 3×3-pixel median filter (hot-pixel correction as proposed by Paez et al. 2026) and saves corrected images under `corrected_3x3/`. Adds HISTORY entries to both raw and filtered FITS headers to prevent reprocessing.


#### Output Directory Structure
After a successful preprocessing run, the `data/` contents are translated into a structured `organized_data/` hierarchy:

```text
organized_data/
└── {TARGET}/
    ├── raw/
    │   └── {DATE}/              ← Original FITS (updated headers)
    ├── corrected_3x3/
    │   └── {DATE}/              ← 3×3 median filter products
    ├── measurements/
    │   └── {DATE}/              ← AIJ .tbl/.csv files (INPUT for -o)
    │       └── times/
    │           └── times.csv    ← Optional time intervals
    ├── lcs/
    │   └── {DATE}/              ← Normalized multi-band CSVs
    ├── plots/
    │   └── {DATE}/              ← Diagnostic PDFs and PNGs
    └── exofop/
        └── {DATE}/
            ├── {band}/          ← Standardized ExoFOP products
            └── AIJ/
                └── {band}/      ← AIJ fitpanel files for report generation
```

### AstroImageJ Integration
After preprocessing, perform photometry in AstroImageJ using the files in `corrected_3x3/`. Save the measurement tables in the corresponding `measurements/{DATE}/` directory.

**Requirements for Measurement Files:**
* **Format**: `.tbl` (tab-separated) or `.csv` (comma-separated).
* **Naming**: The filename must end with the band suffix (e.g., `myfile_gp.tbl`, `obs1_rp.csv`, or `data_ip.tbl`). Supported bands: `g`, `gp`, `r`, `rp`, `i`, `ip`.

### Optional: Time Averaging & RMS Intervals
To define specific non-transit intervals for RMS calculation and noise analysis, create a `times/times.csv` file inside the date folder in `measurements/`:
* **Columns**: `init_time`, `final_time` (in JD or BJD units).
* **Usage**: If present, PROFE will use these intervals for legend RMS calculations and the time-averaging diagnostic plots. If absent, PROFE creates an empty template automatically.

### Postprocessing

Run the following to generate all scientific products:

```bash
# Generate outputs for all targets
profe -o

# Generate outputs for a specific target only
profe -o "TOI-1234"
```

The postprocessing pipeline runs the following modules **sequentially**:

| # | Module | Description |
|---|---|---|
| 1 | **AltAzGuidingPlotter** | Altitude–Azimuth polar plot and centroid displacement vs. time. |
| 2 | **LightCurvePlotter** | Multi-band 6-panel diagnostic light curves (PDF) + per-band ExoFOP PNGs + normalized CSV. |
| 3 | **TimeAveragingPlotter** | Red vs. white noise characterization using the MC3 time-averaging method (Cubillos et al. 2017). |
| 4 | **FieldViewPlotter** | Aperture visualization: source and sky annuli overlaid on a calibrated FITS image. |
| 5 | **SeeingProfilePlotter** | Radial brightness profile of the target star using `photutils.RadialProfile`. |
| 6 | **ComparisonStarsPlotter** | 6-panel light curves for up to 6 comparison stars with sigma-clipped outlier removal. |
| 7 | **AstrometrySolver** | WCS solving via Astrometry.net using local source detection (Background2D + segmentation). |
| 8 | **TransitDataManager** | Retrieves predicted transit times from the TESS Transit Finder (TTF). |
| 9 | **ReportGenerator** | Generates a consolidated ExoFOP notes text file with multi-band metrics. |

Each module checks for existing outputs before running and **skips already-processed** (target, date, band) triples.

#### Standardized ExoFOP Products
All files in `exofop/{DATE}/{BAND}/` follow the naming convention:
`{TICID}-01_{YYYYMMDD}_OAN-SPM-2m1-OPTICAM_{BAND}_{TYPE}.{EXT}`

| TYPE | Description |
|---|---|
| `_lightcurve` | High-resolution 6-panel PNG plot of the light curve. |
| `_field` | Science image showing source and sky apertures. |
| `_seeing-profile` | Radial profile of the target star. |
| `_WCS` | WCS-solved FITS with astrometric header. |
| `_compstar-lightcurves` | Diagnostic plot of up to 6 comparison star light curves. |
| `_measurements` | A copy of the original photometry table (.tbl). |

Additional products at the date level (`exofop/{DATE}/`):

| File | Description |
|---|---|
| `*_transit_times.dat` | Predicted transit ingress, mid, and egress times from TTF. |
| `*_notes.txt` | Consolidated ExoFOP report with multi-band metrics and timing analysis. |

### Logging

All pipeline runs generate timestamped log files in the `logs/` directory:

```text
logs/
├── profe_preprocess_20260420_143022.log
├── profe_organize_20260420_150112.log
├── profe_filter_20260420_151530.log
└── profe_output_20260420_160045.log
```

Each execution creates a new log file to prevent overwriting previous records.

### Manual

To see the complete detailed manual and arguments at any time, run:

```bash
profe man
```

---

## Architecture

```text
profe/
├── cli.py                  ← Central CLI entry point (argparse)
├── logger.py               ← Timestamped logging configuration
├── preprocess/
│   ├── cli.py              ← Preprocessing orchestrator
│   ├── fits_processor.py   ← Header update, file organization, summary
│   └── median_filter.py    ← 3×3 median filter (multiprocessing)
└── output/
    ├── cli.py              ← Postprocessing orchestrator
    ├── naming.py           ← ExoFOP naming conventions & TOI→TIC resolution
    ├── alt_az_centroid.py   ← Altitude–Azimuth & centroid plots
    ├── light_curves.py      ← Multi-band light curves (PDF/PNG/CSV)
    ├── correlated_noise.py  ← Time-averaging red noise analysis
    ├── field_view.py        ← Aperture visualization plots
    ├── seeing_profile.py    ← Radial profile plots
    ├── comparison_stars.py  ← Comparison star light curves
    ├── astrometry_out.py    ← WCS solving via Astrometry.net
    ├── transit_info.py      ← TTF transit data retrieval
    └── report_generator.py  ← Consolidated ExoFOP notes
```

---

## Profiling
To analyze the performance of the pipeline, you must have the development dependencies installed. Additionally, if you are testing this over a specific data directory, it is recommended to activate the virtual environment so the commands work globally:

```bash
# 1. Activate the poetry environment shell
poetry shell

# 2. Navigate to your working directory (e.g. where your data/ is) and run the profiler
# We use the `--html` flag to successfully export the interactive report
pyinstrument --html -o profe_profile.html -m profe.cli -p -c 4
```

This will generate a detailed interactive `profe_profile.html` report showing execution time for each function call.
---

## Development & Contribution

We welcome contributions to improve **PROFE**! Please follow these steps to ensure a smooth process:

1. **Fork the repository** on GitHub and clone your fork locally:
   ```bash
   git clone https://github.com/<username>/profe.git
   cd profe
2. **Create a new branch** for your feature of bugfix:
   ```bash
   git checkout -b feat/new-feature
   git checkout -b fix/issue-123
3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
4. **Enable and run pre-commit hooks** (for code style and quality checks):
   ```bash
   pre-commit install
   pre-commit run --all-files
5. **Commit and push** your changes to your fork
6. **Open a Pull Request** from your fork to the main repository. In your PR description:
    - Explain the what and why of the change
    - Reference related issues

---
## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.