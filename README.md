# PROFE: Pipeline de Reducción de OPTICAM para Fotometría de Exoplanetas
[![A rectangular badge, half black half purple containing the text made at Code Astro](https://img.shields.io/badge/Made%20at-Code/Astro-blueviolet.svg)](https://semaphorep.github.io/codeastro/)

#### Reduction pipeline for OPTICAM photometry of exoplanets.

A Python-based pipeline to automate preprocessing and postprocessing of data acquired with the OPTICAM instrument on the 2.1 m telescope at OAN‑SPM, aimed at producing calibrated light curves and centroid analyses for transiting exoplanets implementing the data reduction methods proposed by Paez et. al (in prep.).

---

## Features

* **Preprocessing** (`profe -p`):

  * Organize and standardize FITS files.
  * Update headers and compute Julian Date.
  * Apply median filter
* **Postprocessing** (`profe -o`):

  * Plot of altitude-azimuth trajectory and centroids and movement in pixels .
  * Generate binned light curves with RMS measurements.
  * Time-averaging curves with the red and white noise in the time-series.
  * Radial profile for target star
  * Field of View with apertures for target and comparison stars

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

### Preprocessing

**PROFE** requires your original FITS files to be placed inside a `data/` directory relative to where you run the command. The pipeline will read the files from there, organize them, update their FITS headers, and apply the median filter correction.

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
    │   └── {DATE}/         <-- Original FITS (with updated headers)
    ├── corrected_3x3/
    │   └── {DATE}/         <-- Copies processed with 3x3 median filter
    ├── measurements/       <-- Data tables for postprocessing
    ├── lcs/                <-- Generated Lightcurves
    └── exofop/             <-- ExoFOP formatted outputs
```
### AstroImageJ
Once the data have been preprocessed with `profe`, it is time to perform data reduction and photometry with AstroImageJ and save the measurements tables in `.tbl` format.

### Outputs

Generate light curves (plots and files), centroid movement plots, Alt-Az trajectory, Field of View apertures, radial profile and time-averaging curves for `measurements.tbl` files:

```bash
profe -o
```

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