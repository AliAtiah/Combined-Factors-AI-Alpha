"""
Global configuration for the Combined Factors AI Alpha project.
"""

EOD_BUNDLE_NAME = "eod-quotemedia"

UNIVERSE_SIZE = 500
UNIVERSE_END_DATE = "2016-01-05"

FACTOR_LOOKBACK_YEARS = 3
FACTOR_LOOKBACK_DAYS = 2

FORWARD_RETURN_WINDOW = 5

TRAIN_SIZE = 0.6
VALID_SIZE = 0.2
TEST_SIZE = 0.2

CLF_RANDOM_STATE = 0
N_DAYS_PER_LEAF = 10
N_STOCKS_PER_LEAF = 500

TREE_SIZES = [50, 100, 250, 500, 1000]
FINAL_N_TREES = 500

FEATURES = [
    "Mean_Reversion_Sector_Neutral_Smoothed",
    "Momentum_1YR",
    "Overnight_Sentiment_Smoothed",
    "adv_120d",
    "adv_20d",
    "dispersion_120d",
    "dispersion_20d",
    "market_vol_120d",
    "market_vol_20d",
    "volatility_20d",
    "is_Janaury",
    "is_December",
    "weekday",
    "month_end",
    "month_start",
    "qtr_end",
    "qtr_start",
]

EVAL_FACTOR_NAMES = [
    "Mean_Reversion_Sector_Neutral_Smoothed",
    "Momentum_1YR",
    "Overnight_Sentiment_Smoothed",
    "adv_120d",
    "volatility_20d",
]
