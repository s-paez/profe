# PROFE: Pipeline de Reducción de OPTICAM para Fotometría de Exoplanetas
#### Reduction pipeline for OPTICAM photometry of exoplanets.

A Python-based pipeline to automate preprocessing and postprocessing of data acquired
with the OPTICAM instrument on the 2.1 m telescope at OAN‑SPM, aimed at producing
calibrated light curves and centroid analyses for transiting exoplanets implementing
the data reduction methods proposed by Paez et. al (in prep.).

---

## Features

* **Preprocessing** (`profe_pre`):

  * Organize and standardize FITS files.
  * Update headers and compute Julian Dates.
  * Apply median filter
* **Postprocessing** (`profe_out`):

  * Altitude-Azimuth trajectory and centroids and movement in pixels .
  * Generate binned light curves with RMS measurements.
  * Time-averaging curves to see the red noise in the time-series.

---

## Requirements

* Python ≥ 3.8.20
* Dependencies defined in `pyproject.toml` / `setup.py`:

  * `numpy`$\geq$ 1.24.4
  * `pandas`$\geq$ 2.0.3
  * `astropy`$\geq$ 5.2.2
  * `matplotlib`$\geq$ 3.7.5
  * `tqdm`$\geq$ 4.67.1
  * `mc3`$\geq$ 3.1.5

---

## Installation

Install from the project root:

```bash
pip install .
```

For a development environment (include testing and linting tools):

```bash
git clone <repository-url>
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

Generate light curves (plots and files), centroid movement plots, Alt-Az trajectory and time-averaging curves for `measurements.tbl` files:

```bash
profe_out
```


---

## Project Structure

```
profe/
├── pyproject.toml    # Build configuration and dependencies
├── setup.py          # No setup for poetry
├── preprocess/       # Preprocessing subpackage
│   ├── __init__.py
│   ├── cli.py
│   ├── fits_processor.py
│   └── median_filter.py
└── postprocess/      # Postprocessing subpackage
    ├── __init__.py
    ├── cli.py
    ├── alt_az_centroid.py
    ├── correlated_noise.py
    └── light_curves.py

```

---

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
