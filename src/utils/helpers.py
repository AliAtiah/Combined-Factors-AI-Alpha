"""
Visualization and utility helpers.
"""

import numpy as np
import matplotlib.pyplot as plt
import graphviz
from IPython.display import Image
from sklearn.tree import export_graphviz


def plot_tree_classifier(clf, feature_names=None):
    """Render a decision tree as a PNG image."""
    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        special_characters=True,
        rotate=True,
    )
    return Image(graphviz.Source(dot_data).pipe(format="png"))


def plot(xs, ys, labels, title="", x_label="", y_label=""):
    """Plot multiple series on one axes."""
    for x, y, label in zip(xs, ys, labels):
        plt.ylim((0.5, 0.55))
        plt.plot(x, y, label=label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.show()


def rank_features_by_importance(importances, feature_names):
    """Print features sorted by importance."""
    indices = np.argsort(importances)[::-1]
    max_len = max(len(f) for f in feature_names)

    print(f"      Feature{' ' * (max_len - 8)}      Importance")
    for rank, idx in enumerate(indices):
        print(f"{rank + 1:>2}. {feature_names[idx]:<{max_len}} ({importances[idx]})")
