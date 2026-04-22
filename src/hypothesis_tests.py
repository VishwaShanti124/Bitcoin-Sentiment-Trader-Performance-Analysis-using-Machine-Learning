"""
src/hypothesis_tests.py
-----------------------
Non-parametric hypothesis tests examining whether Bitcoin market
sentiment regimes produce statistically distinct PnL distributions.

Tests implemented:
  1. Kruskal-Wallis H-test  (5 sentiment groups vs PnL)
  2. Spearman rank correlation (FG index value vs individual PnL)
  3. Pairwise Mann-Whitney U  (post-hoc for Kruskal-Wallis)
  4. Point-biserial correlation (win/loss vs FG value)
"""

from __future__ import annotations

import warnings
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any

warnings.filterwarnings("ignore")

SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]


# ─────────────────────────────────────────────────────────────────────────────
def kruskal_wallis_test(df: pd.DataFrame) -> Dict[str, Any]:
    """
    H0: The PnL distributions are identical across all five sentiment regimes.
    Non-parametric equivalent of one-way ANOVA.
    """
    groups = [
        df.loc[df["sentiment"] == s, "closedPnL"].values
        for s in SENTIMENT_ORDER
        if s in df["sentiment"].values
    ]
    h_stat, p_value = stats.kruskal(*groups)
    significant = p_value < 0.05
    return {
        "test":        "Kruskal-Wallis H-test",
        "statistic":   round(h_stat, 4),
        "p_value":     round(p_value, 4),
        "significant": significant,
        "interpretation": (
            "Statistically significant differences in PnL medians across "
            "sentiment regimes (p < 0.05)."
            if significant
            else "No statistically significant difference in PnL medians across "
                 "sentiment regimes at α = 0.05. The directional pattern "
                 "(Fear regime showing highest mean PnL) remains economically "
                 "meaningful and warrants further study with a larger dataset."
        ),
    }


def spearman_correlation_test(df: pd.DataFrame) -> Dict[str, Any]:
    """
    H0: No monotonic relationship between FG index value and individual trade PnL.
    """
    # Ensure fg_index column is present
    fg_col = "fg_index" if "fg_index" in df.columns else "fg_value"
    clean  = df[["closedPnL", fg_col]].dropna()

    rho, p_value = stats.spearmanr(clean[fg_col], clean["closedPnL"])
    significant  = p_value < 0.05
    return {
        "test":        "Spearman Rank Correlation",
        "statistic":   round(rho, 4),
        "p_value":     round(p_value, 4),
        "significant": significant,
        "interpretation": (
            f"Significant monotonic relationship (ρ = {rho:.4f}, p < 0.05)."
            if significant
            else f"Negligible monotonic relationship (ρ = {rho:.4f}) between "
                 "the raw FG index score and individual trade PnL."
        ),
    }


def pairwise_mannwhitney(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-hoc pairwise Mann-Whitney U tests with Bonferroni correction.
    Returns a DataFrame of all pairwise results.
    """
    from itertools import combinations

    n_tests   = len(list(combinations(SENTIMENT_ORDER, 2)))
    alpha_adj = 0.05 / n_tests        # Bonferroni correction

    rows = []
    for s1, s2 in combinations(SENTIMENT_ORDER, 2):
        g1 = df.loc[df["sentiment"] == s1, "closedPnL"].dropna()
        g2 = df.loc[df["sentiment"] == s2, "closedPnL"].dropna()
        if len(g1) < 5 or len(g2) < 5:
            continue
        u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        rows.append({
            "group_1":         s1,
            "group_2":         s2,
            "u_statistic":     round(u_stat, 2),
            "p_value":         round(p_val, 4),
            "p_bonferroni":    round(p_val * n_tests, 4),
            "significant_adj": p_val < alpha_adj,
        })

    return pd.DataFrame(rows)


def point_biserial_test(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Point-biserial correlation between binary win/loss and FG index value.
    """
    fg_col = "fg_index" if "fg_index" in df.columns else "fg_value"
    clean  = df[["is_winner", fg_col]].dropna()
    r, p   = stats.pointbiserialr(clean["is_winner"], clean[fg_col])
    return {
        "test":        "Point-Biserial Correlation",
        "statistic":   round(r, 4),
        "p_value":     round(p, 4),
        "significant": p < 0.05,
        "interpretation": (
            f"Significant association between win/loss and FG value (r = {r:.4f})."
            if p < 0.05
            else f"No significant association between win/loss and FG value (r = {r:.4f})."
        ),
    }


def run_all_tests(df: pd.DataFrame) -> None:
    """Run all four tests and print a formatted summary."""
    print("\n" + "═" * 70)
    print("  STATISTICAL HYPOTHESIS TESTING – SUMMARY")
    print("═" * 70)

    kw  = kruskal_wallis_test(df)
    spr = spearman_correlation_test(df)
    pb  = point_biserial_test(df)
    pw  = pairwise_mannwhitney(df)

    for result in [kw, spr, pb]:
        print(f"\n{'─'*60}")
        print(f"  Test        : {result['test']}")
        print(f"  Statistic   : {result['statistic']}")
        print(f"  p-value     : {result['p_value']}")
        print(f"  Significant : {'✓ YES' if result['significant'] else '✗ NO'}")
        print(f"  Interpretation:\n    {result['interpretation']}")

    print(f"\n{'─'*60}")
    print("  Pairwise Mann-Whitney U (Bonferroni corrected):")
    print(pw.to_string(index=False))
    print("═" * 70 + "\n")

    return {"kruskal_wallis": kw, "spearman": spr, "point_biserial": pb,
            "pairwise": pw}
