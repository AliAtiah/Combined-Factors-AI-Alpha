"""
Alpha factor definitions: Momentum, Mean Reversion, Overnight Sentiment,
and Volume-Weighted Momentum.
"""

import numpy as np
from zipline.pipeline import Pipeline
from zipline.pipeline.classifiers import Classifier
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import (
    AverageDollarVolume,
    CustomFactor,
    Returns,
    SimpleMovingAverage,
)
from zipline.utils.numpy_utils import int64_dtype

from config import UNIVERSE_SIZE


class Sector(Classifier):
    """Sector classifier loaded from a pre-computed numpy array."""
    dtype = int64_dtype
    window_length = 0
    inputs = ()
    missing_value = -1

    def __init__(self):
        self.data = np.load("../../data/project_7_sector/data.npy")

    def _compute(self, arrays, dates, assets, mask):
        return np.where(mask, self.data[assets], self.missing_value)


class CTO(Returns):
    """
    Close-to-Open (overnight) return.
    Hypothesis: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2554010
    """
    inputs = [USEquityPricing.open, USEquityPricing.close]

    def compute(self, today, assets, out, opens, closes):
        out[:] = (opens[-1] - closes[0]) / closes[0]


class TrailingOvernightReturns(Returns):
    """Sum of trailing overnight returns."""
    window_safe = True

    def compute(self, today, asset_ids, out, cto):
        out[:] = np.nansum(cto, axis=0)


def momentum_1yr(window_length, universe, sector):
    """1-year momentum factor, sector-neutralized and z-scored."""
    return (
        Returns(window_length=window_length, mask=universe)
        .demean(groupby=sector)
        .rank()
        .zscore()
    )


def mean_reversion_5day_sector_neutral_smoothed(window_length, universe, sector):
    """5-day mean reversion, sector-neutralized, smoothed, and z-scored."""
    unsmoothed = (
        -Returns(window_length=window_length, mask=universe)
        .demean(groupby=sector)
        .rank()
        .zscore()
    )
    return (
        SimpleMovingAverage(inputs=[unsmoothed], window_length=window_length)
        .rank()
        .zscore()
    )


def overnight_sentiment_smoothed(cto_window_length, trail_window_length, universe):
    """Smoothed overnight sentiment factor."""
    cto_out = CTO(mask=universe, window_length=cto_window_length)
    unsmoothed = (
        TrailingOvernightReturns(inputs=[cto_out], window_length=trail_window_length)
        .rank()
        .zscore()
    )
    return (
        SimpleMovingAverage(inputs=[unsmoothed], window_length=trail_window_length)
        .rank()
        .zscore()
    )


class VolumeWeightedMomentum(CustomFactor):
    """
    Volume-weighted momentum: weights each day's return by its share of
    total volume over the lookback window.  High-volume price moves carry
    more information, so this separates "confirmed" momentum from
    low-conviction drift.

    Inputs: daily close prices, daily trading volume.
    Output: single scalar per asset — the volume-weighted cumulative return.
    """
    inputs = [USEquityPricing.close, USEquityPricing.volume]
    window_safe = True

    def compute(self, today, assets, out, closes, volumes):
        daily_returns = np.diff(closes, axis=0) / closes[:-1]
        vol_slice = volumes[1:]

        vol_sum = np.nansum(vol_slice, axis=0)
        vol_sum[vol_sum == 0] = 1.0
        weights = vol_slice / vol_sum

        out[:] = np.nansum(weights * daily_returns, axis=0)


def volume_weighted_momentum_smoothed(window_length, universe, sector):
    """
    Volume-weighted momentum factor: sector-neutralized, SMA-smoothed,
    ranked, and z-scored.

    Unlike plain momentum (which weights all days equally), this factor
    amplifies price moves that occurred on high-volume days — a proxy for
    institutional conviction.
    """
    raw = (
        VolumeWeightedMomentum(window_length=window_length, mask=universe)
        .demean(groupby=sector)
        .rank()
        .zscore()
    )
    return (
        SimpleMovingAverage(inputs=[raw], window_length=20)
        .rank()
        .zscore()
    )


def build_alpha_pipeline():
    """Build a pipeline with all four alpha factors and the sector classifier."""
    universe = AverageDollarVolume(window_length=120).top(UNIVERSE_SIZE)
    sector = Sector()

    pipeline = Pipeline(screen=universe)
    pipeline.add(momentum_1yr(252, universe, sector), "Momentum_1YR")
    pipeline.add(
        mean_reversion_5day_sector_neutral_smoothed(20, universe, sector),
        "Mean_Reversion_Sector_Neutral_Smoothed",
    )
    pipeline.add(
        overnight_sentiment_smoothed(2, 10, universe),
        "Overnight_Sentiment_Smoothed",
    )
    pipeline.add(
        volume_weighted_momentum_smoothed(120, universe, sector),
        "Volume_Weighted_Momentum_Smoothed",
    )
    pipeline.add(sector, "sector_code")

    return pipeline, universe, sector
