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

### Outputs

Generate light curves (plots and files), centroid movement plots, Alt-Az trajectory, Field of View apertures, radial profile and time-averaging curves for `measurements.tbl` files:

```bash
profe_out
```


<!-- ## Development & Contribution

1. Fork the repository.
2. Create a branch for your feature or fix:

   ```bash
   ```

git checkout -b feature/awesome-feature

````
3. Write tests under `tests/` and ensure they pass:
   ```bash
pytest
````

4. Format code with `black` and sort imports with `isort`.
5. Open a pull request describing your changes.

--- -->

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
