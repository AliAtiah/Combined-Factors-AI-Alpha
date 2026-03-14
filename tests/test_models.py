"""
Unit tests for the classifiers module.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

from src.models.classifiers import (
    train_valid_test_split,
    non_overlapping_samples,
    bagging_classifier,
)
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


def _make_multi_index_data(n_dates=10, n_assets=3, n_features=3):
    dates = [date(2015, 1, 1) + timedelta(days=i) for i in range(n_dates)]
    assets = [f"ASSET_{i}" for i in range(n_assets)]
    index = pd.MultiIndex.from_product([dates, assets])
    n_rows = len(index)

    X = pd.DataFrame(
        np.arange(n_rows * n_features).reshape(n_rows, n_features),
        index=index,
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.arange(n_rows), index=index, name="target")
    return X, y


class TestTrainValidTestSplit:
    def test_correct_proportions(self):
        X, y = _make_multi_index_data(n_dates=10)
        X_tr, X_v, X_te, y_tr, y_v, y_te = train_valid_test_split(X, y, 0.6, 0.2, 0.2)

        total = len(X_tr) + len(X_v) + len(X_te)
        assert total == len(X)
        assert len(y_tr) == len(X_tr)
        assert len(y_v) == len(X_v)
        assert len(y_te) == len(X_te)

    def test_no_overlap(self):
        X, y = _make_multi_index_data(n_dates=10)
        X_tr, X_v, X_te, _, _, _ = train_valid_test_split(X, y, 0.6, 0.2, 0.2)

        tr_dates = set(X_tr.index.get_level_values(0))
        v_dates = set(X_v.index.get_level_values(0))
        te_dates = set(X_te.index.get_level_values(0))

        assert tr_dates.isdisjoint(v_dates)
        assert tr_dates.isdisjoint(te_dates)
        assert v_dates.isdisjoint(te_dates)


class TestNonOverlappingSamples:
    def test_reduces_data(self):
        X, y = _make_multi_index_data(n_dates=10)
        X_nol, y_nol = non_overlapping_samples(X, y, n_skip_samples=2, start_i=0)

        assert len(X_nol) < len(X)
        assert len(X_nol) == len(y_nol)


class TestBaggingClassifier:
    def test_returns_correct_type(self):
        params = {
            "criterion": "entropy",
            "min_samples_leaf": 50,
            "oob_score": True,
            "n_jobs": -1,
            "random_state": 0,
        }
        clf = bagging_classifier(100, 0.2, 1.0, params)

        assert isinstance(clf, BaggingClassifier)
        assert isinstance(clf.base_estimator, DecisionTreeClassifier)
        assert clf.max_samples == 0.2
        assert clf.n_estimators == 100
