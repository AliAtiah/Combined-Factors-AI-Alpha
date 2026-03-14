# Combined Factors for Enhanced AI Alpha

Combining multiple alpha factors with machine learning ensemble methods to generate a superior AI-driven alpha signal for equity trading.

The pipeline constructs three alpha factors (Momentum, Mean Reversion, Overnight Sentiment), engineers regime and calendar features, and trains a **NoOverlapVoter** ensemble — a custom non-overlapping Random Forest that avoids look-ahead bias from overlapping forward-return targets.

## Key Results

The AI Alpha signal **outperforms each individual factor** on Sharpe ratio across train, validation, and test periods — even when the underlying factors show flat-to-negative returns.

## Project Structure

```
Combined-Factors-AI-Alpha/
├── config.py                        # Hyperparameters & feature lists
├── src/
│   ├── data/
│   │   ├── pipeline.py              # Zipline bundle registration & pricing
│   │   └── features.py              # Feature engineering (regime, calendar, sector)
│   ├── factors/
│   │   └── alpha_factors.py         # Alpha factor definitions (Momentum, MR, Overnight)
│   ├── models/
│   │   ├── classifiers.py           # Train/test split, BaggingClassifier, NoOverlapVoter
│   │   └── evaluation.py            # Sharpe ratio, Alphalens factor analysis
│   └── utils/
│       └── helpers.py               # Plotting & feature importance ranking
├── notebooks/
│   ├── analysis.ipynb               # Streamlined walkthrough (imports from src/)
│   └── Alpha-Factors-Original.ipynb # Original monolithic notebook (reference)
├── tests/
│   └── test_models.py               # Unit tests for classifiers module
├── requirements.txt
└── README.md
```

## Alpha Factors

| Factor | Description |
|--------|-------------|
| **Momentum 1YR** | 252-day return, sector-neutralized, ranked, z-scored |
| **Mean Reversion (Smoothed)** | 20-day negative return, sector-neutralized, SMA-smoothed |
| **Overnight Sentiment** | Trailing close-to-open returns, smoothed |

## Features

- **Universal Quant**: Annualized volatility (20d, 120d), average dollar volume (20d, 120d)
- **Regime**: Cross-sectional return dispersion (20d, 120d), market volatility (20d, 120d)
- **Calendar**: January/December dummies, weekday, month/quarter start/end flags
- **Sector**: One-hot encoded GICS sectors

## Models

1. **Random Forest Baseline** — Standard RF classifier with different tree counts (50–1000)
2. **BaggingClassifier** — Uses `max_samples` to reduce overlap bias
3. **NoOverlapVoter** — Ensemble of non-overlapping sub-sampled RF estimators (the final model)

### NoOverlapVoter

The overlapping forward-return targets (5-day windows) create correlated training samples. The `NoOverlapVoter` trains `n_skip_samples + 1` separate classifiers, each on a different offset of the data, and combines them via soft voting. This eliminates overlap bias while retaining full data coverage.

## Quick Start

### Prerequisites

- Python 3.6+
- Zipline data bundle (EOD QuoteMedia)

### Installation

```bash
pip install -r requirements.txt
```

### Run the Analysis

```bash
cd notebooks
jupyter notebook analysis.ipynb
```

### Run Tests

```bash
pytest tests/
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Data** | Zipline, Pandas, NumPy |
| **Factors** | Zipline Pipeline API |
| **Models** | Scikit-Learn (Random Forest, Bagging, Custom Voter) |
| **Evaluation** | Alphalens, Matplotlib |
| **Testing** | Pytest |

## License

MIT
