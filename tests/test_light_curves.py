import numpy as np
import pandas as pd

from profe.output.light_curves import LightCurvePlotter


class TestLightCurvePlotter:
    def test_calculate_rms_no_times(self):
        """Test RMS calculation without time intervals (global RMS in ppt)."""
        plotter = LightCurvePlotter()

        flux = pd.Series([1.0, 1.02, 0.98, 1.0, 1.0])
        t = pd.Series([0, 1, 2, 3, 4])
        times = None

        rms = plotter._calc_rms_in_intervals(t.values, flux.values, times)

        # Expected calculation in ppt
        expected_rms = (np.std(flux) / np.abs(np.median(flux))) * 1000

        assert np.isclose(rms, expected_rms)

    def test_calculate_rms_with_times(self):
        """Test RMS calculation restricted to specific time intervals in ppt."""
        plotter = LightCurvePlotter()

        flux = pd.Series([1.0, 1.02, 0.98, 1.0, 0.5])  # 0.5 is at index 4 (t=4)
        t = pd.Series([0, 1, 2, 3, 4])

        # Select times 0 to 3.5 (excludes t=4)
        times = pd.DataFrame({"init_time": [0.0], "final_time": [3.5]})

        rms = plotter._calc_rms_in_intervals(t.values, flux.values, times)

        # Code should only consider the first 4 points (approx 1.0)
        subset = flux.iloc[:4]
        expected_rms = (np.std(subset) / np.abs(np.median(subset))) * 1000

        assert np.isclose(rms, expected_rms)
        # Without outlier, the ppt should be small relative to outlier
        assert rms < 20.0

    def test_calculate_rms_empty_mask_fallback(self):
        """Test fallback when time mask selects nothing."""
        plotter = LightCurvePlotter()

        flux = pd.Series([1.0, 1.02, 0.98])
        t = pd.Series([0, 1, 2])

        # Disjoint time interval
        times = pd.DataFrame({"init_time": [10.0], "final_time": [20.0]})

        rms = plotter._calc_rms_in_intervals(t.values, flux.values, times)

        # Should fall back to full data
        expected_rms = (np.std(flux) / np.abs(np.median(flux))) * 1000
        assert np.isclose(rms, expected_rms)
