import os

import numpy as np
import pytest
from astropy.io import fits

from profe.preprocess.median_filter import MedianFilter


class TestMedianFilter:
    @pytest.fixture
    def noisy_image_setup(self, tmp_path):
        """Create a noisy FITS file and output directory structure."""
        image_path = tmp_path / "noisy.fits"
        output_path = tmp_path / "filtered.fits"

        # Create a 5x5 image with a hot pixel in the center
        data = np.zeros((5, 5))
        data[2, 2] = 100.0  # Hot pixel

        # Add some background noise
        data[0, 0] = 5.0

        hdu = fits.PrimaryHDU(data=data)
        hdu.writeto(image_path)

        return str(image_path), str(output_path)

    def test_process_image(self, noisy_image_setup):
        """Test median filter removes hot pixels."""
        image_path, output_path = noisy_image_setup

        # Filter with 3x3 window
        mf = MedianFilter(ws=3, n_processes=1)

        # Arguments: (input, output, window_size)
        args = (image_path, output_path, 3)
        result_path = mf._process_image(args)

        assert result_path == image_path
        assert os.path.exists(output_path)

        with fits.open(output_path) as hdul:
            filtered_data = hdul[0].data

            # The hot pixel (100.0) at [2,2] should be replaced by the median of its neighbors (0.0)
            # 3x3 window around [2,2] contains one 100.0 and eight 0.0s. Median is 0.0.
            assert filtered_data[2, 2] == 0.0

            # Check edge behavior (reflect mode).
            # Pixel [0,0] is 5.0. Neighbors include 0s and itself logic depending on reflection.
            # With 'reflect', [0,0] is surrounded by sufficient 0s or 5s.
            # 3x3 at corner [0,0]:
            # Data:
            # 5 0 ...
            # 0 0 ...
            # Reflected:
            # 0 0 0
            # 0 5 0 -> This row is the real row 0
            # 0 0 0
            # Median might still smooth it out depending on specific neighbors.
            # Let's focus on the hot pixel correctness first which is guaranteed.
