"""
Astrometry solver module via Astrometry.net API.

Finds the first calibrated FITS image per band and observation date, and
submits it to Astrometry.net to obtain a WCS-resolved copy. Uses `astroquery`
and looks for the API key in `.astrometry_key` at the project root.
"""

import logging
from pathlib import Path

from astropy.io import fits
from astroquery.astrometry_net import AstrometryNet


class AstrometrySolver:
    """
    Solves astronomical images for coordinates using Astrometry.net API.
    """

    def __init__(self) -> None:
        self.base_dir = Path.cwd()
        self.data_dir = self.base_dir / "organized_data"
        self.logger = logging.getLogger(__name__)
        self.api_key_path = self.base_dir / ".astrometry_key"
        self._astrometry_api = None

    def _setup_api(self) -> AstrometryNet:
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

        a = AstrometryNet()
        a.api_key = key
        self._astrometry_api = a
        return a

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

    def process_object(self, obj_dir: Path) -> None:
        target = obj_dir.name

        measurements_root: Path = obj_dir / "measurements"
        if not measurements_root.exists():
            return

        api = self._setup_api()
        if not api:
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
                        f"Astrometry already solved for ({obj_dir.name}, {date_folder.name}, {band})"
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
                    f"Uploading {fits_to_solve.name} ({band}) to Astrometry.net for WCS solving. This can take a few minutes..."
                )

                try:
                    # Extract RA and DEC smartly to speed up the solver
                    from astropy.coordinates import SkyCoord
                    import astropy.units as u
                    from typing import Any

                    solve_kwargs: dict[str, Any] = {"solve_timeout": 600}
                    with fits.open(fits_to_solve) as temp_hdul:
                        ra_str = temp_hdul[0].header.get("RA")
                        dec_str = temp_hdul[0].header.get("DEC")

                    if ra_str and dec_str:
                        try:
                            c = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
                            solve_kwargs["center_ra"] = c.ra.deg
                            solve_kwargs["center_dec"] = c.dec.deg
                            solve_kwargs["radius"] = 0.25
                            solve_kwargs["parity"] = (
                                2  # Images confirmed with polar inversion (Y axis)
                            )
                        except Exception as e:
                            self.logger.warning(
                                f"Could not parse RA/DEC '{ra_str}'/'{dec_str}': {e}. Falling back to Blind Solve."
                            )

                    import gzip
                    import shutil
                    import warnings
                    from astropy.utils.exceptions import AstropyDeprecationWarning

                    # Local photutils and Astrometry.net struggle to agree on scales/stars.
                    tmp_gz = fits_to_solve.with_name(fits_to_solve.name + ".gz")

                    wcs_header = None
                    try:
                        self.logger.info(
                            f"Temporarily compressing {fits_to_solve.name} to optimize HTTP upload..."
                        )
                        with open(fits_to_solve, "rb") as f_in:
                            with gzip.open(tmp_gz, "wb") as f_out:
                                shutil.copyfileobj(f_in, f_out)

                        self.logger.info(
                            f"Uploading packed image to Nova supercomputer. Parameters: {solve_kwargs}"
                        )

                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", AstropyDeprecationWarning)
                            wcs_header = api.solve_from_image(
                                str(tmp_gz), force_image_upload=True, **solve_kwargs
                            )

                            # Fallback if it fails
                            if not wcs_header and "center_ra" in solve_kwargs:
                                self.logger.warning(
                                    "Targeted solve failed. Retrying across the whole sky (Blind Solve)..."
                                )
                                wcs_header = api.solve_from_image(
                                    str(tmp_gz),
                                    force_image_upload=True,
                                    solve_timeout=600,
                                )
                    finally:
                        if tmp_gz.exists():
                            tmp_gz.unlink()  # Clean up temporary file

                    if wcs_header:
                        with fits.open(fits_to_solve) as hdul:
                            hdul[0].header.extend(wcs_header, update=True, strip=False)
                            hdul.writeto(save_path, overwrite=True)
                        self.logger.info(
                            f"Astrometry complete. Saved to {save_path.name}"
                        )
                    else:
                        self.logger.warning(
                            f"Astrometry solving timed out or failed for {fits_to_solve.name}"
                        )
                except Exception as e:
                    self.logger.error(f"Error communicating with Astrometry.net: {e}")

    def run(self) -> None:
        """
        Process all objects found in `organized_data`.
        """
        self.logger.info("Running Astrometry WCS Solver")
        for obj_dir in self.data_dir.iterdir():
            if not obj_dir.is_dir():
                continue
            if not (obj_dir / "measurements").exists():
                continue
            self.process_object(obj_dir)
