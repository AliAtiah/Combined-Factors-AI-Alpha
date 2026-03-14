"""
Model evaluation: predict alpha scores, compute Sharpe ratios,
and display factor comparison results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import alphalens as al


def sharpe_ratio(factor_returns, annualization_factor=np.sqrt(252)):
    """Annualized Sharpe ratio of factor returns."""
    return annualization_factor * factor_returns.mean() / factor_returns.std()


def get_factor_returns(factor_data):
    """Compute long-short factor returns for each factor."""
    ls_returns = pd.DataFrame()
    for factor, fdata in factor_data.items():
        ls_returns[factor] = al.performance.factor_returns(fdata).iloc[:, 0]
    return ls_returns


def build_factor_data(factor_df, pricing):
    """Build Alphalens factor data dict from a DataFrame of factor values."""
    return {
        name: al.utils.get_clean_factor_and_forward_returns(
            factor=data, prices=pricing, periods=[1]
        )
        for name, data in factor_df.iteritems()
    }


def show_sample_results(data, samples, classifier, factor_names, pricing):
    """
    Predict AI Alpha scores, merge with existing factors, and display
    Sharpe ratios plus cumulative return and rank autocorrelation plots.
    """
    prob_array = np.array([-1, 1])
    alpha_score = classifier.predict_proba(samples).dot(prob_array)

    alpha_label = "AI_ALPHA"
    factors_with_alpha = data.loc[samples.index].copy()
    factors_with_alpha[alpha_label] = alpha_score

    print("Cleaning Data...\n")
    factor_data = build_factor_data(
        factors_with_alpha[factor_names + [alpha_label]], pricing
    )
    print("\n-----------------------\n")

    factor_returns = get_factor_returns(factor_data)
    sr = sharpe_ratio(factor_returns)

    print("             Sharpe Ratios")
    print(sr.round(2))

    (1 + factor_returns).cumprod().plot(ylim=(0.8, 1.2))
    plt.title("Cumulative Factor Returns")
    plt.show()

    # Rank autocorrelation
    ls_fra = pd.DataFrame()
    unixt = {
        f: fd.set_index(
            pd.MultiIndex.from_tuples(
                [(x.timestamp(), y) for x, y in fd.index.values],
                names=["date", "asset"],
            )
        )
        for f, fd in factor_data.items()
    }
    for factor, fd in unixt.items():
        ls_fra[factor] = al.performance.factor_rank_autocorrelation(fd)
    ls_fra.plot(title="Factor Rank Autocorrelation", ylim=(0.8, 1.0))
    plt.show()
