# PROFE: Pipeline de Reducción de OPTICAM para Fotometría de Exoplanetas
#### Reduction pipeline for OPTICAM photometry of exoplanets.

A Python-based pipeline to automate preprocessing and postprocessing of data acquired with the OPTICAM instrument on the 2.1 m telescope at OAN‑SPM, aimed at producing calibrated light curves and centroid analyses for transiting exoplanets implementing the data reduction methods proposed by Paez et. al (in prep.).

---

## Features

* **Preprocessing** (`profe_pre`):

  * Organize and standardize FITS files.
  * Update headers and compute Julian Date.
  * Apply median filter
* **Postprocessing** (`profe_out`):

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

### Preprocessing

Organize raw data from `data/` directory into the `organized_data/` directory, update time headers, perform median filter correction:

```bash
profe_pre
```
### AstroImageJ
Once the data have been preprocessed with `profe`, it is time to perform data reduction and photometry with AstroImageJ and save the measurements tables in `.tbl` format.

### Outputs

Generate light curves (plots and files), centroid movement plots, Alt-Az trajectory, Field of View apertures, radial profile and time-averaging curves for `measurements.tbl` files:

```bash
profe_out
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
