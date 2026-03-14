"""
Classifier wrappers: train/valid/test split, non-overlapping sampling,
BaggingClassifier builder, and NoOverlapVoter ensemble.
"""

import abc
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import Bunch


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------

def train_valid_test_split(all_x, all_y, train_size, valid_size, test_size):
    """
    Split multi-indexed (date, asset) data into train / validation / test sets
    by date, preserving the temporal ordering.

    Returns (X_train, X_valid, X_test, y_train, y_valid, y_test).
    """
    dates = all_x.index.levels[0]
    n = len(dates)
    train_end = int(n * train_size)
    valid_end = int(n * (train_size + valid_size))

    train_dates = dates[:train_end]
    valid_dates = dates[train_end:valid_end]
    test_dates = dates[valid_end:]

    X_train = all_x.loc[train_dates[0]:train_dates[-1]]
    X_valid = all_x.loc[valid_dates[0]:valid_dates[-1]]
    X_test = all_x.loc[test_dates[0]:test_dates[-1]]
    y_train = all_y.loc[train_dates[0]:train_dates[-1]]
    y_valid = all_y.loc[valid_dates[0]:valid_dates[-1]]
    y_test = all_y.loc[test_dates[0]:test_dates[-1]]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


# ---------------------------------------------------------------------------
# Non-overlapping sampling
# ---------------------------------------------------------------------------

def non_overlapping_samples(x, y, n_skip_samples, start_i=0):
    """
    Sub-sample the data by taking every (n_skip_samples + 1)-th date slice,
    starting at ``start_i``.  This removes the overlap between the
    forward-return windows used as targets.
    """
    assert len(x.shape) == 2
    assert len(y.shape) == 1

    selected_dates = list(x.index.levels[0][start_i :: n_skip_samples + 1])
    return x.loc[selected_dates], y.loc[selected_dates]


# ---------------------------------------------------------------------------
# Bagging classifier builder
# ---------------------------------------------------------------------------

def bagging_classifier(n_estimators, max_samples, max_features, parameters):
    """
    Build a BaggingClassifier with a DecisionTreeClassifier base estimator.

    ``parameters`` must contain: criterion, min_samples_leaf, oob_score,
    n_jobs, random_state.
    """
    required = {"criterion", "min_samples_leaf", "oob_score", "n_jobs", "random_state"}
    assert not required - set(parameters.keys())

    base = DecisionTreeClassifier(
        criterion=parameters["criterion"],
        max_features=max_features,
        min_samples_leaf=parameters["min_samples_leaf"],
    )
    return BaggingClassifier(
        base_estimator=base,
        n_estimators=n_estimators,
        max_samples=max_samples,
        bootstrap=True,
        oob_score=parameters["oob_score"],
        n_jobs=parameters["n_jobs"],
        verbose=0,
        random_state=parameters["random_state"],
    )


# ---------------------------------------------------------------------------
# NoOverlapVoter ensemble
# ---------------------------------------------------------------------------

def calculate_oob_score(classifiers):
    """Mean out-of-bag score across fitted classifiers."""
    return np.mean([clf.oob_score_ for clf in classifiers])


def non_overlapping_estimators(x, y, classifiers, n_skip_samples):
    """Fit each classifier on a different non-overlapping offset."""
    return [
        classifiers[k].fit(*non_overlapping_samples(x, y, n_skip_samples, start_i=k))
        for k in range(len(classifiers))
    ]


class NoOverlapVoter(VotingClassifier):
    """
    Soft-voting ensemble where each base estimator is trained on a different
    non-overlapping sub-sample of the data to avoid look-ahead bias from
    overlapping forward returns.
    """

    def __init__(self, estimator, voting="soft", n_skip_samples=4):
        estimators = [(f"clf{i}", estimator) for i in range(n_skip_samples + 1)]
        self.n_skip_samples = n_skip_samples
        super().__init__(estimators, voting=voting)

    def fit(self, X, y, sample_weight=None):
        _, clfs = zip(*self.estimators)
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_

        clone_clfs = [clone(clf) for clf in clfs]
        self.estimators_ = non_overlapping_estimators(
            X, y, clone_clfs, self.n_skip_samples
        )
        self.named_estimators_ = Bunch(
            **dict(zip([name for name, _ in self.estimators], self.estimators_))
        )
        self.oob_score_ = calculate_oob_score(self.estimators_)
        return self
