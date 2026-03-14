"""
Feature engineering: universal quant features, regime features,
calendar features, sector one-hot encoding, and target construction.
"""

import os
import numpy as np
import pandas as pd
from zipline.pipeline.factors import (
    AnnualizedVolatility,
    AverageDollarVolume,
    CustomFactor,
    DailyReturns,
    Returns,
    SimpleMovingAverage,
)

from config import FORWARD_RETURN_WINDOW


# ---------------------------------------------------------------------------
# Regime-level custom factors
# ---------------------------------------------------------------------------

class MarketDispersion(CustomFactor):
    """Cross-sectional standard deviation of daily returns."""
    inputs = [DailyReturns()]
    window_length = 1
    window_safe = True

    def compute(self, today, assets, out, returns):
        out[:] = np.sqrt(np.nanmean((returns - np.nanmean(returns)) ** 2))


class MarketVolatility(CustomFactor):
    """Annualized volatility of the equal-weight market return."""
    inputs = [DailyReturns()]
    window_length = 1
    window_safe = True

    def compute(self, today, assets, out, returns):
        mkt_returns = np.nanmean(returns, axis=1)
        out[:] = np.sqrt(260.0 * np.nanmean((mkt_returns - np.nanmean(mkt_returns)) ** 2))


# ---------------------------------------------------------------------------
# Pipeline feature adders
# ---------------------------------------------------------------------------

def add_universal_features(pipeline, universe):
    """Add volatility and dollar-volume features to the pipeline."""
    pipeline.add(AnnualizedVolatility(window_length=20, mask=universe).rank().zscore(), "volatility_20d")
    pipeline.add(AnnualizedVolatility(window_length=120, mask=universe).rank().zscore(), "volatility_120d")
    pipeline.add(AverageDollarVolume(window_length=20, mask=universe).rank().zscore(), "adv_20d")
    pipeline.add(AverageDollarVolume(window_length=120, mask=universe).rank().zscore(), "adv_120d")


def add_regime_features(pipeline, universe):
    """Add market dispersion and volatility regime features."""
    pipeline.add(
        SimpleMovingAverage(inputs=[MarketDispersion(mask=universe)], window_length=20),
        "dispersion_20d",
    )
    pipeline.add(
        SimpleMovingAverage(inputs=[MarketDispersion(mask=universe)], window_length=120),
        "dispersion_120d",
    )
    pipeline.add(MarketVolatility(window_length=20), "market_vol_20d")
    pipeline.add(MarketVolatility(window_length=120), "market_vol_120d")


def add_date_features(df, start_date, end_date):
    """Add calendar-based features to the factor DataFrame (in-place)."""
    dates = df.index.get_level_values(0)
    df["is_Janaury"] = dates.month == 1
    df["is_December"] = dates.month == 12
    df["weekday"] = dates.weekday
    df["quarter"] = dates.quarter
    df["qtr_yr"] = df["quarter"].astype(str) + "_" + dates.year.astype(str)
    df["month_end"] = dates.isin(pd.date_range(start=start_date, end=end_date, freq="BM"))
    df["month_start"] = dates.isin(pd.date_range(start=start_date, end=end_date, freq="BMS"))
    df["qtr_end"] = dates.isin(pd.date_range(start=start_date, end=end_date, freq="BQ"))
    df["qtr_start"] = dates.isin(pd.date_range(start=start_date, end=end_date, freq="BQS"))
    return df


def add_sector_dummies(df, sector_labels_path=None):
    """One-hot encode the sector_code column."""
    if sector_labels_path is None:
        sector_labels_path = os.path.join(
            os.getcwd(), "..", "..", "data", "project_7_sector", "labels.csv"
        )
    sector_lookup = pd.read_csv(sector_labels_path, index_col="Sector_i")["Sector"].to_dict()

    sector_columns = []
    for sector_i, sector_name in sector_lookup.items():
        col = f"sector_{sector_name}"
        sector_columns.append(col)
        df[col] = df["sector_code"] == sector_i

    return df, sector_columns


def add_target(df, window=FORWARD_RETURN_WINDOW):
    """Create a shifted forward-return target column."""
    df["target"] = df.groupby(level=1)["return_5d"].shift(-window)
    return df
