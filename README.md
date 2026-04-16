# PROFE: Pipeline de Reducción de OPTICAM para Fotometría de Exoplanetas
[![A rectangular badge, half black half purple containing the text made at Code Astro](https://img.shields.io/badge/Made%20at-Code/Astro-blueviolet.svg)](https://semaphorep.github.io/codeastro/)

#### Reduction pipeline for OPTICAM photometry of exoplanets.

A Python-based pipeline to automate preprocessing and postprocessing of data acquired with the OPTICAM instrument on the 2.1 m telescope at OAN‑SPM, aimed for implementing the data reduction method proposed by [Paez et. al (2026)](https://doi.org/10.1093/rasti/rzag021).

---

## Features

* **Preprocessing** (`profe -p`):

  * Organize and standardize FITS files.
  * Update headers and compute Julian Date.
  * Apply median filter
* **Postprocessing** (`profe -o`):
  * Generated multi-band binned light curves with RMS measurements (PDF & CSV).
  * Time-averaging noise plots (Red vs White noise characterization).
  * Altitude-Azimuth trajectory and Centroid movement plots.
  * ExoFOP-standardized products:
    * Standardized single-band Light Curves (PNG).
    * Aperture visualization (Field of View) and Radial Profiles.
    * Light curves for comparison stars.
    * Automated WCS solving via Astrometry.net.
    * Integrated archival of measurement tables.

---

## Requirements
Dependencies defined in `pyproject.toml`:

* Python ($\geq 3.8.20$, $\leq 3.12$)
* astropy ($5.3.2$)
* scipy ($\geq 1.10$, $<2.0$)
* matplotlib ($\geq3.10.3$, $<4.0.0$)
* tqdm ($\geq 4.67.1$, $<5.0.0$)
* pandas ($\geq 2.3.1$, $<3.0.0$)
* mc3 ($\geq 3.2.1$, $<4.0.0$)
* photutils ($\geq 2.2.0$, $<3.0.0$)
* numpy ($\geq 1.24$, $<2.0$)
* astroquery ($\geq 0.4.7$)

### External Setup
For automated **WCS Solving**, you must provide an [Astrometry.net](https://nova.astrometry.net/) API key:
1. Create a file named `.astrometry_key` at the root of your project.
2. Paste your 16-character API key inside the file.

---

## Installation

Install from the project root:

```bash
pip install profe
```

For a development environment (include testing and linting tools):

```bash
git clone https://github.com/s-paez/profe.git
cd profe
pip install .[dev]
```

---

## Usage
### Preprocessing

**PROFE** recursively searches for original FITS files inside a `data/` directory relative to where the command is executed. The pipeline organizes them, updates FITS headers (Julian Date), and applies median filter corrections.

To execute the preprocessing:

```bash
profe -p
```

By default, the preprocessing uses all available CPU cores. You can limit the number of cores using the `-c` (or `--cores`) flag:

```bash
# Use only 4 cores
profe -p -c 4
```

#### Output Directory Structure
After a successful preprocessing run, your pipeline translates the `data/` contents into a highly structured `organized_data/` hierarchy, mapping each FITS to its extracted `{TARGET}` and observation `{DATE}`:

```text
organized_data/
└── {TARGET}/
    ├── raw/
    │   └── {DATE}/         <-- Original FITS (updated headers)
    ├── corrected_3x3/
    │   └── {DATE}/         <-- 3x3 median filter products
    ├── measurements/
    │   └── {DATE}/         <-- AIJ .tbl/.csv files (INPUT for -o)
    │       └── times/
    │           └── times.csv <-- Optional time intervals
    ├── lcs/
    │   └── {DATE}/         <-- Normalized multi-band CSVs
    ├── plots/
    │   └── {DATE}/         <-- Diagnostic PDFs and PNGs
    └── exofop/
        └── {DATE}/
            └── {band}/     <-- Standardized ExoFOP products
```

### AstroImageJ Integration
After preprocessing, perform photometry in AstroImageJ using the files in `corrected_3x3/`. Save the measurement tables in the corresponding `measurements/{DATE}/` directory.

**Requirements for Measurement Files:**
* **Format**: `.tbl` (tab-separated) or `.csv` (comma-separated).
* **Naming**: The filename must end with the band suffix (e.g., `myfile_gp.tbl`, `obs1_rp.csv`, or `data_i.tbl`). Supported bands: `g`, `gp`, `r`, `rp`, `i`, `ip`.

### Optional: Time Averaging & RMS Intervals
To define specific non-transit intervals for RMS calculation and noise analysis, create a `times/times.csv` file inside the date folder in `measurements/`:
* **Columns**: `init_time`, `final_time` (in JD or BJD units).
* **Usage**: If present, PROFE will use these intervals for legeng RMS calculations and the time-averaging diagnostic plots.

### Outputs
Run the following to generate all scientific products:

```bash
profe -o
```

#### Standardized ExoFOP Products
All files in `exofop/{DATE}/{BAND}/` follow the standardized naming convention:
`{TARGET}-01_{YYYYMMDD}_OAN-SPM-2m1-OPTICAM_{BAND}_{TYPE}.{EXT}`

| TYPE | Description |
|---|---|
| `_lightcurve` | High-resolution PNG plot of the light curve. |
| `_field` | Science image showing source and sky apertures. |
| `_seeing-profile` | Radial profile of the target star. |
| `_WCS` | WCS-solved FITS header integrated into the first science image. |
| `_compstar-lightcurves` | Diagnostic plot of up to 6 comparison star light curves. |
| `_measurements` | A copy of the original photometry table used. |

### Manual

To see the complete detailed manual and arguments at any time, run:

```bash
profe man
```

### Profiling
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