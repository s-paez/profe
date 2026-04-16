"""
Astrometry solver module via Astrometry.net API.

Finds the first calibrated FITS image per band and observation date, and
submits it to Astrometry.net to obtain a WCS-resolved copy.

Uses local source detection with photutils (Background2D subtraction +
segmentation + FWHM filtering) tuned for faint sources, then submits the
filtered source list via
``astroquery.astrometry_net.AstrometryNet.solve_from_source_list``.

Looks for the API key in `.astrometry_key` at the project root.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
from astropy.convolution import convolve
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.utils.exceptions import AstropyDeprecationWarning
from astroquery.astrometry_net import AstrometryNet, AstrometryNetClass
from astroquery.exceptions import TimeoutError as AstroqueryTimeoutError
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import (
    SourceCatalog,
    detect_sources,
    make_2dgaussian_kernel,
)


class AstrometrySolver:
    """
    Solves astronomical images for coordinates using Astrometry.net API.

    Detects sources locally using 2D background subtraction and
    segmentation, filters by FWHM, and submits the cleaned source list
    to Nova via ``solve_from_source_list``.
    """

    # Source-detection variables
    BKG_BOX_SIZE: int = 50  # Background2D box size (pixels)
    BKG_FILTER_SIZE: int = 3  # Background2D median-filter size
    FWHM_KERNEL: float = 4.0  # Gaussian kernel FWHM for convolution
    DETECT_NPIXELS: int = 5  # Minimum connected pixels for a source
    DETECT_SIGMA: float = 2.5  # Detection threshold in background RMS
    FWHM_LOW: float = 2.0  # Keep sources with FWHM > this (pixels)
    FWHM_HIGH: float = 20.0  # Keep sources with FWHM < this (pixels)
    MAX_SOURCES: int = 40  # Send at most this many sources to Nova

    def __init__(self) -> None:
        self.base_dir = Path.cwd()
        self.data_dir = self.base_dir / "organized_data"
        self.logger = logging.getLogger(__name__)
        self.api_key_path = self.base_dir / ".astrometry_key"
        self._astrometry_api: AstrometryNetClass | None = None

    def _setup_api(self) -> AstrometryNetClass | None:
        """Read the API key from disk and return a configured client."""
        if self._astrometry_api is not None:
            return self._astrometry_api

        if not self.api_key_path.exists():
            self.logger.warning(
                "No .astrometry_key file found at root. Skipping astrometry solving."
            )
            return None

        key = self.api_key_path.read_text().strip()
        if not key:
            self.logger.warning(".astrometry_key is empty. Skipping.")
            return None

        ast = AstrometryNet()
        ast.api_key = key
        ast.TIMEOUT = 120
        self._astrometry_api = ast
        return ast

    def _is_processed(
        self, obj_dir: Path, date_name: str, target_name: str, band: str
    ) -> bool:
        expected = (
            obj_dir
            / "exofop"
            / date_name
            / f"{target_name}_{date_name}_{band}_solved.fits"
        )
        return expected.exists()

    def _detect_sources(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect sources in *data* using 2D background subtraction and
        segmentation, tuned for faint stellar sources.

        Steps:
            1. Estimate and subtract the 2D background with
               ``Background2D`` (box_size=50, MedianBackground).
            2. Build a threshold map at ``DETECT_SIGMA × background_rms``.
            3. Convolve with a Gaussian kernel (FWHM=4 px) and run
               ``detect_sources`` with ``npixels=5``.
            4. Compute FWHM from ``semimajor_sigma`` and keep sources
               within the ``FWHM_LOW``–``FWHM_HIGH`` range.
            5. Sort by flux (descending) and return the top
               ``MAX_SOURCES`` (x, y) positions.
        """
        # 1. Background estimation
        sigma_clip = SigmaClip(sigma=3.0)
        bkg_estimator = MedianBackground()
        bkg = Background2D(
            data,
            box_size=(self.BKG_BOX_SIZE, self.BKG_BOX_SIZE),
            filter_size=(self.BKG_FILTER_SIZE, self.BKG_FILTER_SIZE),
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator,
        )

        data_subtracted = data - bkg.background
        threshold = self.DETECT_SIGMA * bkg.background_rms

        # Convolve with a larger kernel
        kernel = make_2dgaussian_kernel(fwhm=self.FWHM_KERNEL, size=5)
        convolved = convolve(data_subtracted, kernel)

        # Segmentation
        segm = detect_sources(convolved, threshold, npixels=self.DETECT_NPIXELS)

        if segm is None:
            self.logger.warning("Segmentation found zero sources.")
            return np.array([]), np.array([])

        # Source catalog
        cat = SourceCatalog(data_subtracted, segm)
        sources = cat.to_table()

        sources["fwhm"] = cat.semimajor_sigma.value * 2.355

        # Log FWHM distribution
        fwhm_vals = sources["fwhm"]
        self.logger.info(
            f"Source FWHM stats: n={len(fwhm_vals)}, "
            f"min={np.nanmin(fwhm_vals):.2f}, "
            f"median={np.nanmedian(fwhm_vals):.2f}, "
            f"max={np.nanmax(fwhm_vals):.2f}"
        )

        mask = (sources["fwhm"] > self.FWHM_LOW) & (sources["fwhm"] < self.FWHM_HIGH)
        filtered = sources[mask]

        self.logger.info(
            f"After FWHM filter ({self.FWHM_LOW}–{self.FWHM_HIGH} px): "
            f"{len(filtered)} of {len(sources)} sources kept"
        )

        if len(filtered) == 0:
            return np.array([]), np.array([])

        # Sort by flux (descending) and keep the brightest
        filtered.sort("segment_flux", reverse=True)
        x = filtered["xcentroid"][: self.MAX_SOURCES]
        y = filtered["ycentroid"][: self.MAX_SOURCES]

        return np.array(x), np.array(y)

    def _solve_image(
        self, api: AstrometryNetClass, fits_path: Path
    ) -> fits.Header | dict:
        """
        Detect sources locally, then submit the source list to
        Astrometry.net via ``solve_from_source_list``.

        Uses the timeout / retry pattern from the astroquery docs.

        Returns
        -------
        `~astropy.io.fits.Header` on success, empty ``dict`` on failure.
        """
        with fits.open(fits_path) as hdul:
            data = hdul[0].data.astype(float)
            hdr = hdul[0].header
            image_width = hdr["NAXIS1"]
            image_height = hdr["NAXIS2"]

        # Detect and filter sources
        x_coords, y_coords = self._detect_sources(data)
        self.logger.info(
            f"Detected {len(x_coords)} filtered sources in {fits_path.name}"
        )

        if len(x_coords) == 0:
            self.logger.warning(f"No sources detected for {fits_path.name} — skipping.")
            return {}

        # Solve settings for OPTICAM data
        solve_settings: dict = {
            "parity": 0,
            "scale_units": "degwidth",
            "scale_type": "ul",
            "scale_lower": 0.0333,
            "scale_upper": 0.1667,
            "downsample_factor": 2,
        }

        if "RA" in hdr and "DEC" in hdr:
            solve_settings["radius"] = 0.25

        # Retry loop from astroquery
        try_again = True
        submission_id = None
        wcs_header: fits.Header | dict = {}

        while try_again:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", AstropyDeprecationWarning)

                    if not submission_id:
                        wcs_header = api.solve_from_source_list(
                            x_coords,
                            y_coords,
                            image_width,
                            image_height,
                            submission_id=submission_id,
                            solve_timeout=120,
                            verbose=True,
                            **solve_settings,
                        )
                    else:
                        wcs_header = api.monitor_submission(
                            submission_id,
                            solve_timeout=120,
                            verbose=True,
                        )
            except (TimeoutError, AstroqueryTimeoutError) as exc:
                submission_id = exc.args[1]
                self.logger.warning(
                    f"Solve timed out for submission {submission_id}. Re-monitoring…"
                )
            else:
                try_again = False

        return wcs_header

    def process_object(self, obj_dir: Path) -> None:
        target = obj_dir.name

        measurements_root: Path = obj_dir / "measurements"
        if not measurements_root.exists():
            return

        api = self._setup_api()
        if api is None:
            return

        for date_folder in sorted(measurements_root.iterdir()):
            if not date_folder.is_dir():
                continue

            meas_files = [
                f
                for f in date_folder.iterdir()
                if f.is_file()
                and not f.name.startswith(".")
                and f.suffix in (".tbl", ".csv")
            ]

            exofop_dir = obj_dir / "exofop" / date_folder.name
            exofop_dir.mkdir(parents=True, exist_ok=True)

            for file_to_read in meas_files:
                band: str = file_to_read.stem.split("_")[-1]

                if self._is_processed(obj_dir, date_folder.name, target, band):
                    self.logger.info(
                        "Astrometry already solved for "
                        f"({target}, {date_folder.name}, {band})"
                    )
                    continue

                fits_dir: Path = (
                    obj_dir / "corrected_3x3" / date_folder.name / f"calibrated_{band}"
                )
                fits_cands: list[Path] = list(fits_dir.rglob("*_out.fit*"))
                if not fits_cands:
                    self.logger.warning(
                        f"No calibrated FITS in {fits_dir} for astrometry."
                    )
                    continue

                fits_to_solve = fits_cands[0]
                save_path = (
                    exofop_dir / f"{target}_{date_folder.name}_{band}_solved.fits"
                )

                self.logger.info(
                    f"Submitting {fits_to_solve.name} ({band}) to "
                    "Astrometry.net — this may take several minutes…"
                )

                try:
                    wcs_header = self._solve_image(api, fits_to_solve)

                    if wcs_header:
                        with fits.open(fits_to_solve) as hdul:
                            hdul[0].header.extend(wcs_header, update=True, strip=False)
                            hdul.writeto(save_path, overwrite=True)
                        self.logger.info(f"Astrometry complete → {save_path.name}")
                    else:
                        self.logger.warning(
                            f"Astrometry solving failed for {fits_to_solve.name}"
                        )
                except Exception as exc:
                    self.logger.error(f"Error communicating with Astrometry.net: {exc}")

    def run(self) -> None:
        """Process all objects found in ``organized_data``."""
        self.logger.info("Running Astrometry WCS Solver (astroquery)")
        for obj_dir in self.data_dir.iterdir():
            if not obj_dir.is_dir():
                continue
            if not (obj_dir / "measurements").exists():
                continue
            self.process_object(obj_dir)
