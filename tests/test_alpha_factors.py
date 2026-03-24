"""
Unit tests for the VolumeWeightedMomentum factor logic.

These tests validate the core VWAM computation directly (without requiring
a live Zipline pipeline engine) by exercising the same NumPy math used
inside the CustomFactor.compute method.
"""

import numpy as np
import pytest


def _vwam_core(closes, volumes):
    """
    Pure-NumPy replica of VolumeWeightedMomentum.compute so we can
    test it without a running Zipline pipeline.
    """
    daily_returns = np.diff(closes, axis=0) / closes[:-1]
    vol_slice = volumes[1:]

    vol_sum = np.nansum(vol_slice, axis=0)
    vol_sum[vol_sum == 0] = 1.0
    weights = vol_slice / vol_sum

    return np.nansum(weights * daily_returns, axis=0)


class TestVWAMCore:
    def test_uniform_volume_matches_simple_return(self):
        """When volume is constant every day, VWAM equals simple cumulative return."""
        n_days, n_assets = 6, 2
        closes = np.array([
            [100.0, 200.0],
            [102.0, 198.0],
            [104.0, 196.0],
            [106.0, 194.0],
            [108.0, 192.0],
            [110.0, 190.0],
        ])
        volumes = np.ones((n_days, n_assets)) * 1000.0

        result = _vwam_core(closes, volumes)

        daily_ret = np.diff(closes, axis=0) / closes[:-1]
        expected = np.mean(daily_ret, axis=0)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_high_volume_day_dominates(self):
        """A single high-volume day should dominate the signal."""
        closes = np.array([
            [100.0],
            [110.0],  # +10% on huge volume
            [111.0],  # +0.9% on tiny volume
        ])
        volumes = np.array([
            [100.0],
            [10_000.0],
            [1.0],
        ])

        result = _vwam_core(closes, volumes)

        assert result[0] > 0.09, "Signal should be dominated by the +10% high-volume day"

    def test_zero_volume_safe(self):
        """Assets with zero total volume should not produce NaN."""
        closes = np.array([[100.0], [105.0], [110.0]])
        volumes = np.zeros((3, 1))

        result = _vwam_core(closes, volumes)
        assert np.isfinite(result[0])

    def test_output_shape(self):
        """Output should be a 1-D array with one entry per asset."""
        n_days, n_assets = 10, 5
        rng = np.random.RandomState(42)
        closes = 100 + rng.randn(n_days, n_assets).cumsum(axis=0)
        closes = np.abs(closes) + 1.0
        volumes = rng.uniform(100, 10_000, size=(n_days, n_assets))

        result = _vwam_core(closes, volumes)
        assert result.shape == (n_assets,)

    def test_negative_momentum(self):
        """Consistently declining prices should yield a negative signal."""
        closes = np.array([
            [100.0],
            [95.0],
            [90.0],
            [85.0],
        ])
        volumes = np.ones((4, 1)) * 500.0

        result = _vwam_core(closes, volumes)
        assert result[0] < 0
