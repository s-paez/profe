.. _usage:

Usage
=====

PROFE provides a single command-line entry point (``profe``) with mutually
exclusive modes for preprocessing and postprocessing.


CLI Reference
-------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Command
     - Description
   * - ``profe -p``
     - Run the **full preprocessing** pipeline (organize + median filter).
   * - ``profe --organice``
     - Run **only** the file reorganization and header update stage.
   * - ``profe --filter``
     - Run **only** the median filter stage. Skips if ``corrected_3x3/`` exists.
   * - ``profe -o [TARGET]``
     - Run the **full postprocessing** and output generation pipeline.
       Optionally specify a target name to process only that target.
   * - ``profe -pu [TARGET]``
     - **Prepare Upload**: Pack local output products into an intermediate JSON
       manifest and prompt for an ExoFOP Data Tag.
   * - ``profe -u [TARGET]``
     - **Upload**: Iteratively post the prepared files to ExoFOP using individual
       file endpoints to preserve exact scientific file names.
   * - ``profe man``
     - Display the detailed built-in manual.
   * - ``profe -h``
     - Show the quick-reference help message.

**Optional flags:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Description
   * - ``-c CORES`` / ``--cores CORES``
     - Number of CPU cores for preprocessing (default: all available).
       Only valid with ``-p``, ``--organice``, or ``--filter``.
   * - ``-o TARGET``
     - When a target name is provided (e.g., ``profe -o "TOI-1234"``), only
       that target is processed. Without a target, all targets are processed.


Preprocessing
-------------

PROFE recursively searches for original FITS files inside a ``data/`` directory
relative to where the command is executed.

.. code-block:: bash

   # Full preprocessing with all CPU cores
   profe -p

   # Full preprocessing limited to 4 cores
   profe -p -c 4

   # Only reorganize files and update headers (no filter)
   profe --organice

   # Only apply the median filter (skips if corrected_3x3/ exists)
   profe --filter


Preprocessing Steps
^^^^^^^^^^^^^^^^^^^

1. **Header Update** — Computes ``JD`` (Julian Date at start) and ``UTMIDDLE``
   (ISO mid-exposure) from existing ``UT`` and ``EXPOSURE`` keywords. Files
   already tagged with a PROFE ``HISTORY`` entry are skipped.

2. **File Organization** — Sorts FITS files into
   ``organized_data/{TARGET}/raw/{DATE}/`` based on ``OBJECT`` and ``DATE-OBS``
   header keywords. Creates ``measurements/``, ``lcs/``, and ``exofop/``
   directories for science targets.

3. **Summary Generation** — Creates ``logs/images_summary.dat`` with image
   counts per target and date.

4. **Median Filter** — Applies a 3×3-pixel median filter for hot-pixel
   correction (Paez et al. 2026) and saves corrected images under
   ``corrected_3x3/``. Adds ``HISTORY`` entries to both raw and filtered FITS
   headers to prevent reprocessing.


Directory Structure
^^^^^^^^^^^^^^^^^^^

After a successful preprocessing run, the ``data/`` contents are organized into:

.. code-block:: text

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
                   └── {band}/      ← AIJ fitpanel files for reports


AstroImageJ Integration
------------------------

After preprocessing, perform photometry in AstroImageJ using the files in
``corrected_3x3/``. Save the measurement tables in the corresponding
``measurements/{DATE}/`` directory.

**Requirements for measurement files:**

* **Format**: ``.tbl`` (tab-separated) or ``.csv`` (comma-separated).
* **Naming**: The filename must end with the band suffix (e.g.,
  ``myfile_gp.tbl``, ``obs1_rp.csv``, ``data_ip.tbl``).
  Supported bands: ``g``, ``gp``, ``r``, ``rp``, ``i``, ``ip``.


Optional: Time Intervals
^^^^^^^^^^^^^^^^^^^^^^^^^

To define specific non-transit intervals for RMS calculation and noise analysis,
create a ``times/times.csv`` file inside the date folder in ``measurements/``:

* **Columns**: ``init_time``, ``final_time`` (in JD or BJD units).
* If present, PROFE uses these intervals for legend RMS calculations and the
  time-averaging diagnostic plots. If absent, PROFE creates an empty template
  automatically.


Postprocessing
--------------

Run the following to generate all scientific products:

.. code-block:: bash

   # Generate outputs for all targets
   profe -o

   # Generate outputs for a specific target only
   profe -o "TOI-1234"

The postprocessing pipeline runs **9 modules sequentially**:

.. list-table::
   :header-rows: 1
   :widths: 5 30 65

   * - #
     - Module
     - Description
   * - 1
     - **AltAzGuidingPlotter**
     - Altitude–Azimuth polar plot and centroid displacement vs. time.
   * - 2
     - **LightCurvePlotter**
     - Multi-band 6-panel diagnostic light curves (PDF) + per-band ExoFOP
       PNGs + normalized CSV.
   * - 3
     - **TimeAveragingPlotter**
     - Red vs. white noise characterization using the MC3 time-averaging
       method (Cubillos et al. 2017).
   * - 4
     - **FieldViewPlotter**
     - Aperture visualization: source and sky annuli overlaid on a calibrated
       FITS image.
   * - 5
     - **SeeingProfilePlotter**
     - Radial brightness profile of the target star using
       ``photutils.RadialProfile``.
   * - 6
     - **ComparisonStarsPlotter**
     - 6-panel light curves for up to 6 comparison stars with sigma-clipped
       outlier removal.
   * - 7
     - **AstrometrySolver**
     - WCS solving via Astrometry.net using local source detection
       (Background2D + segmentation).
   * - 8
     - **TransitDataManager**
     - Retrieves predicted transit times from the TESS Transit Finder (TTF).
   * - 9
     - **ReportGenerator**
     - Generates a consolidated ExoFOP notes text file with multi-band
       metrics and timing analysis.

Each module checks for existing outputs and **skips already-processed**
(target, date, band) triples.


ExoFOP Products
^^^^^^^^^^^^^^^

All files in ``exofop/{DATE}/{BAND}/`` follow the naming convention:

``{TICID}-01_{YYYYMMDD}_OAN-SPM-2m1-OPTICAM_{BAND}_{TYPE}.{EXT}``

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - TYPE
     - Description
   * - ``_lightcurve``
     - High-resolution 6-panel PNG plot of the light curve.
   * - ``_field``
     - Science image showing source and sky apertures.
   * - ``_seeing-profile``
     - Radial profile of the target star.
   * - ``_WCS``
     - WCS-solved FITS with astrometric header.
   * - ``_compstar-lightcurves``
     - Diagnostic plot of up to 6 comparison star light curves.
   * - ``_measurements``
     - A copy of the original photometry table (``.tbl``).

**Additional products at the date level** (``exofop/{DATE}/``):

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Description
   * - ``*_transit_times.dat``
     - Predicted transit ingress, mid, and egress times from TTF.
   * - ``*_notes.txt``
     - Consolidated ExoFOP report with multi-band metrics and timing analysis.


ExoFOP Uploading
^^^^^^^^^^^^^^^^

PROFE can directly upload all standardized products to the ExoFOP single-file upload
endpoint (preserving the exact file names) for targets that have already completed
the postprocessing stage.

1. **Prepare Upload** (``profe -pu [TARGET]``):
   Scans the local ``exofop/`` directories and prompts you for a **Data Tag** for each
   pending date. It collects the valid files and generates an intermediate JSON
   metadata manifest.

2. **Upload** (``profe -u [TARGET]``):
   Reads the prepared manifests and iteratively uploads each file individually
   to ExoFOP.

   * Authenticates using your local ``.exofop_credentials`` file.
   * Automatically assigns the correct ExoFOP target and planet parameters.
   * Derives the correct ExoFOP description from each file's name (e.g., Light Curve, Field of View, WCS FITS Image).
   * Sets the 12-month proprietary period by default.

.. note::
   The upload system ensures files are only processed once. If a file already exists
   on ExoFOP or an upload fails, PROFE will log the error and continue with the remaining items.


Logging
-------

All pipeline runs generate timestamped log files in the ``logs/`` directory:

.. code-block:: text

   logs/
   ├── profe_preprocess_20260420_143022.log
   ├── profe_organize_20260420_150112.log
   ├── profe_filter_20260420_151530.log
   └── profe_output_20260420_160045.log

Each execution creates a **new log file** to preserve the full history of
pipeline runs.


Profiling
---------

To analyze the performance of the pipeline, install the development dependencies
and activate the virtual environment:

.. code-block:: bash

   # Activate the poetry environment shell
   poetry shell

   # Run the profiler (from your data/ directory)
   pyinstrument --html -o profe_profile.html -m profe.cli -p -c 4

This generates a detailed interactive ``profe_profile.html`` report showing
execution time for each function call.
