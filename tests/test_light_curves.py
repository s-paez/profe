import numpy as np
import pandas as pd

from profe.output.light_curves import LightCurvePlotter


class TestLightCurvePlotter:
    def test_calculate_rms_no_times(self):
        """Test RMS calculation without time intervals (global RMS)."""
        plotter = LightCurvePlotter()

        # Create synthetic data: constant flux with small jitter
        # Median should be ~1.0. RMS should be std dev.
        flux = pd.Series([1.0, 1.02, 0.98, 1.0, 1.0])
        t = pd.Series([0, 1, 2, 3, 4])
        times = None

        rms, rms_txt = plotter._calculate_rms(flux, times, t)

        # Expected calculation:
        # Median = 1.0
        # Diff = [0, 0.02, -0.02, 0, 0]
        # Diff^2 = [0, 0.0004, 0.0004, 0, 0]
        # Mean = 0.0008 / 5 = 0.00016
        # Sqrt = 0.012649

        expected_rms = np.sqrt(np.mean((flux - flux.median()) ** 2))

        assert np.isclose(rms, expected_rms)
        assert f"RMS:{rms:.4f}" in rms_txt

    def test_calculate_rms_with_times(self):
        """Test RMS calculation restricted to specific time intervals."""
        plotter = LightCurvePlotter()

        # Flux has a big dip that should be EXCLUDED by the time mask
        flux = pd.Series([1.0, 1.02, 0.98, 1.0, 0.5])  # 0.5 is at index 4 (t=4)
        t = pd.Series([0, 1, 2, 3, 4])

        # Select times 0 to 3.5 (excludes t=4)
        times = pd.DataFrame({"init_time": [0.0], "final_time": [3.5]})

        rms, rms_txt = plotter._calculate_rms(flux, times, t)

        # Code should only consider the first 4 points (approx 1.0)
        subset = flux.iloc[:4]
        expected_rms = np.sqrt(np.mean((subset - subset.median()) ** 2))

        assert np.isclose(rms, expected_rms)
        # Should be much smaller than if we included the 0.5 outlier
        assert rms < 0.1

    def test_calculate_rms_empty_mask_fallback(self):
        """Test fallback when time mask selects nothing."""
        plotter = LightCurvePlotter()

        flux = pd.Series([1.0, 1.02, 0.98])
        t = pd.Series([0, 1, 2])

        # Disjoint time interval
        times = pd.DataFrame({"init_time": [10.0], "final_time": [20.0]})

        rms, rms_txt = plotter._calculate_rms(flux, times, t)

        # Should fall back to full data
        expected_rms = np.sqrt(np.mean((flux - flux.median()) ** 2))
        assert np.isclose(rms, expected_rms)
