"""
src/ml_model.py
---------------
Random Forest classifier trained to predict individual trade profitability
(is_winner: closedPnL > 0).  Implements:

  - 5-fold stratified cross-validation (ROC-AUC)
  - Feature importance extraction
  - Permutation importance (model-agnostic)
  - Learning curve analysis
  - Per-sentiment model accuracy breakdown
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# ── Feature engineering ───────────────────────────────────────────────────────

FEATURE_COLS = ["size", "execution_price", "fg_index_col", "hour", "leverage"]

def _resolve_fg_col(df: pd.DataFrame) -> str:
    for c in ("fg_index", "fg_value", "value"):
        if c in df.columns:
            return c
    raise KeyError("Cannot find Fear & Greed index column in DataFrame.")


def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Returns (X, y, feature_names).
    Features: size, execution_price, fg_index, hour, leverage
    """
    fg_col = _resolve_fg_col(df)
    feature_names = ["size", "execution_price", fg_col, "hour", "leverage"]
    available     = [c for c in feature_names if c in df.columns]

    sub = df[available + ["is_winner"]].dropna()
    X   = sub[available].values.astype(float)
    y   = sub["is_winner"].values

    return X, y, available


# ── Training & evaluation ─────────────────────────────────────────────────────

def train_and_evaluate(
    df: pd.DataFrame,
    n_estimators: int = 200,
    cv_folds:     int = 5,
) -> Dict:
    """
    Train a Random Forest with stratified CV and return a results dict.
    """
    X, y, feature_names = build_feature_matrix(df)

    clf = RandomForestClassifier(
        n_estimators  = n_estimators,
        max_depth     = 8,
        min_samples_leaf = 20,
        class_weight  = "balanced",
        random_state  = RANDOM_STATE,
        n_jobs        = -1,
    )

    skf    = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)

    # Fit on full data for importances
    clf.fit(X, y)
    importances = dict(zip(feature_names, clf.feature_importances_))

    # Permutation importance for validation
    perm = permutation_importance(clf, X, y, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    perm_imp = dict(zip(feature_names, perm.importances_mean))

    print("\n" + "═" * 60)
    print("  RANDOM FOREST – TRADE OUTCOME CLASSIFIER")
    print("═" * 60)
    print(f"  Features       : {feature_names}")
    print(f"  Training rows  : {len(X):,}")
    print(f"  CV AUC Score   : {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"\n  Feature Importances (Gini):")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"    {feat:<20} {imp:.4f}  ({imp*100:.1f}%)")
    print(f"\n  Permutation Importances:")
    for feat, imp in sorted(perm_imp.items(), key=lambda x: -x[1]):
        print(f"    {feat:<20} {imp:.4f}")
    print("═" * 60 + "\n")

    return {
        "model":         clf,
        "cv_scores":     scores,
        "cv_auc_mean":   scores.mean(),
        "cv_auc_std":    scores.std(),
        "feature_names": feature_names,
        "importances":   importances,
        "perm_importances": perm_imp,
        "X":             X,
        "y":             y,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_learning_curve(results: Dict, save_path: str) -> None:
    """Learning curve — training vs CV score as function of training size."""
    clf    = results["model"]
    X, y   = results["X"], results["y"]

    train_sizes, train_scores, cv_scores = learning_curve(
        clf, X, y,
        cv            = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring       = "roc_auc",
        train_sizes   = np.linspace(0.1, 1.0, 8),
        n_jobs        = -1,
    )

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0D1117")
    ax.set_facecolor("#161B22")
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-", color="#58A6FF", label="Training AUC")
    ax.fill_between(train_sizes,
                    train_scores.mean(axis=1) - train_scores.std(axis=1),
                    train_scores.mean(axis=1) + train_scores.std(axis=1),
                    alpha=0.2, color="#58A6FF")
    ax.plot(train_sizes, cv_scores.mean(axis=1), "s-", color="#FF7043", label="CV AUC")
    ax.fill_between(train_sizes,
                    cv_scores.mean(axis=1) - cv_scores.std(axis=1),
                    cv_scores.mean(axis=1) + cv_scores.std(axis=1),
                    alpha=0.2, color="#FF7043")
    ax.set_title("Learning Curve – Random Forest", fontsize=11, color="#58A6FF")
    ax.set_xlabel("Training Samples", fontsize=9, color="#C9D1D9")
    ax.set_ylabel("ROC-AUC", fontsize=9, color="#C9D1D9")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(colors="#8B949E")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_feature_importance_comparison(results: Dict, save_path: str) -> None:
    """Side-by-side Gini vs permutation importance."""
    names   = results["feature_names"]
    gini    = [results["importances"][n]  for n in names]
    perm    = [results["perm_importances"][n] for n in names]

    x   = np.arange(len(names))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0D1117")
    ax.set_facecolor("#161B22")
    ax.bar(x - w/2, gini, w, label="Gini Importance", color="#58A6FF", edgecolor="#0D1117")
    ax.bar(x + w/2, perm, w, label="Permutation Importance", color="#FF7043", edgecolor="#0D1117")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9, color="#C9D1D9")
    ax.set_ylabel("Importance", fontsize=9, color="#C9D1D9")
    ax.set_title("Feature Importance: Gini vs Permutation", fontsize=11, color="#58A6FF")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.tick_params(colors="#8B949E")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close(fig)
    print(f"  Saved → {save_path}")
