import numpy as np
import pandas as pd

from profe.output.correlated_noise import TimeAveragingPlotter


class TestTimeAveragingPlotter:
    def test_mask_no_times(self):
        """Test mask generation when no time intervals are provided."""
        plotter = TimeAveragingPlotter()

        data = pd.DataFrame({"BJD_TDB": [2450000.1, 2450000.2, 2450000.3]})
        times = None

        # When times is None, everything should be True (arrays of True)
        # Note: the method returns np.ones(len, dtype=bool)
        mask = plotter.mask(data, times)

        assert len(mask) == 3
        assert np.all(mask)

    def test_mask_with_intervals(self):
        """Test mask generation with specific time intervals."""
        plotter = TimeAveragingPlotter()

        # Data points at 0.1, 0.5, 0.9
        data = pd.DataFrame({"BJD_TDB": [0.1, 0.5, 0.9]})

        # Interval allows 0.0 to 0.2 AND 0.8 to 1.0
        # Should include 0.1 (True), exclude 0.5 (False), include 0.9 (True)
        times = pd.DataFrame({"init_time": [0.0, 0.8], "final_time": [0.2, 1.0]})

        mask = plotter.mask(data, times)

        assert list(mask) == [True, False, True]
