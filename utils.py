"""
Loan Default Prediction — Credit Risk Utility Functions
Author : Kishore U. | github.com/ukishore33 | linkedin.com/in/kishore-techie
Purpose: Standalone, reusable implementations of:
         • Gini Coefficient   (Basel III PD model evaluation)
         • KS Statistic       (scorecard separation power)
         • Information Value  (feature selection — IV/WoE framework)
         • Weight of Evidence (WoE binning table)
         • Credit Grade       (A–E bucket assignment)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


# ── GINI COEFFICIENT ─────────────────────────────────────────────────────
def gini_coefficient(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Gini Coefficient = 2 * AUC - 1.

    Industry thresholds (Basel III / RBI guidance):
        > 0.60  Excellent
        > 0.40  Good       ← minimum for production deployment
        > 0.20  Acceptable
        < 0.20  Poor — do not deploy

    Args:
        y_true  : Binary ground truth array (0 = good, 1 = default)
        y_score : Model predicted probabilities for class 1

    Returns:
        Gini coefficient (float, range 0–1)
    """
    auc  = roc_auc_score(y_true, y_score)
    gini = 2 * auc - 1
    return round(float(gini), 4)


# ── KS STATISTIC ─────────────────────────────────────────────────────────
def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Kolmogorov–Smirnov (KS) Statistic.

    KS = max(cumulative TPR − cumulative FPR) across all thresholds.
    Measures the maximum separation between the Good and Bad cumulative
    distributions — the higher the KS, the better the scorecard separates
    defaulters from non-defaulters.

    Industry thresholds:
        > 0.40  Excellent
        > 0.30  Good        ← typical for deployable bank scorecards
        > 0.20  Acceptable
        < 0.20  Weak

    Args:
        y_true  : Binary ground truth array
        y_score : Model predicted probabilities for class 1

    Returns:
        KS statistic (float, range 0–1)
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ks           = float(np.max(tpr - fpr))
    return round(ks, 4)


# ── INFORMATION VALUE ────────────────────────────────────────────────────
def information_value(
    df: pd.DataFrame,
    feature: str,
    target: str = "default",
    bins: int = 10,
) -> float:
    """
    Information Value (IV) for a numeric feature.

    IV measures a feature's overall predictive power for binary classification.
    Calculated by binning the feature into quantiles, computing WoE per bin,
    and summing (Good% - Bad%) × WoE across all bins.

    IV thresholds (Siddiqi, 2006 — standard in credit scoring):
        > 0.50  Very Strong predictor
        0.30–0.50  Strong
        0.10–0.30  Medium
        0.02–0.10  Weak
        < 0.02  Useless — exclude from scorecard

    Args:
        df      : DataFrame containing feature and target columns
        feature : Name of the numeric feature column
        target  : Name of the binary target column (default = 'default')
        bins    : Number of quantile bins (default = 10)

    Returns:
        IV score (float, non-negative)
    """
    try:
        df2          = df[[feature, target]].copy().dropna()
        df2["bin"]   = pd.qcut(df2[feature], q=bins, duplicates="drop")
        grp          = df2.groupby("bin")[target].agg(bad="sum", total="count")
        grp["good"]  = grp["total"] - grp["bad"]

        total_bad    = max(grp["bad"].sum(),  1)
        total_good   = max(grp["good"].sum(), 1)

        grp["bad_pct"]  = grp["bad"]  / total_bad
        grp["good_pct"] = grp["good"] / total_good
        grp["woe"]      = np.log(
            (grp["good_pct"] + 1e-9) / (grp["bad_pct"] + 1e-9)
        )
        grp["iv_bin"]   = (grp["good_pct"] - grp["bad_pct"]) * grp["woe"]
        return round(float(grp["iv_bin"].sum()), 4)

    except Exception as exc:
        print(f"  [IV warning] {feature}: {exc}")
        return 0.0


# ── WoE BINNING TABLE ─────────────────────────────────────────────────────
def woe_table(
    df: pd.DataFrame,
    feature: str,
    target: str = "default",
    bins: int = 10,
) -> pd.DataFrame:
    """
    Generate a full Weight of Evidence (WoE) binning table for a feature.

    Used by compliance teams and risk committees to validate scorecard
    feature monotonicity — WoE should ideally trend monotonically from
    low-risk to high-risk bins.

    Returns:
        DataFrame with columns: bin, bad, good, total, bad_pct, good_pct,
                                 woe, iv_bin, cumulative_bad_pct, cumulative_good_pct
    """
    df2        = df[[feature, target]].copy().dropna()
    df2["bin"] = pd.qcut(df2[feature], q=bins, duplicates="drop")
    grp        = df2.groupby("bin", observed=True)[target].agg(
        bad="sum", total="count"
    ).reset_index()

    grp["good"]     = grp["total"] - grp["bad"]
    total_bad        = max(grp["bad"].sum(),  1)
    total_good       = max(grp["good"].sum(), 1)

    grp["bad_pct"]  = (grp["bad"]  / total_bad).round(4)
    grp["good_pct"] = (grp["good"] / total_good).round(4)
    grp["woe"]      = np.log(
        (grp["good_pct"] + 1e-9) / (grp["bad_pct"] + 1e-9)
    ).round(4)
    grp["iv_bin"]   = ((grp["good_pct"] - grp["bad_pct"]) * grp["woe"]).round(4)
    grp["cum_bad"]  = grp["bad_pct"].cumsum().round(4)
    grp["cum_good"] = grp["good_pct"].cumsum().round(4)
    grp["ks"]       = (grp["cum_good"] - grp["cum_bad"]).abs().round(4)

    return grp


# ── CREDIT GRADE ─────────────────────────────────────────────────────────
def assign_credit_grade(pd_score: float) -> str:
    """
    Map a PD score (0–100) to a credit grade A–E.

    Grade logic mirrors standard bank credit rating frameworks:
        A  (0–20)   Very Low Risk  — approve, standard terms
        B  (20–35)  Low Risk       — approve, standard terms
        C  (35–50)  Medium Risk    — approve with conditions or higher rate
        D  (50–65)  High Risk      — decline or very high rate
        E  (65–100) Very High Risk — decline

    Args:
        pd_score : Probability of Default × 100 (float, 0–100)

    Returns:
        Grade string: 'A', 'B', 'C', 'D', or 'E'
    """
    if pd_score < 20:  return "A — Very Low Risk"
    if pd_score < 35:  return "B — Low Risk"
    if pd_score < 50:  return "C — Medium Risk"
    if pd_score < 65:  return "D — High Risk"
    return "E — Very High Risk"


# ── BATCH GRADE ASSIGNMENT ────────────────────────────────────────────────
def grade_portfolio(pd_scores: pd.Series) -> pd.Series:
    """
    Apply assign_credit_grade to a full Series of PD scores.

    Args:
        pd_scores : pandas Series of PD scores (0–100)

    Returns:
        pandas Series of grade strings
    """
    return pd_scores.apply(assign_credit_grade)


# ── IV STRENGTH LABEL ────────────────────────────────────────────────────
def iv_strength_label(iv: float) -> str:
    """Return the industry-standard IV strength category."""
    if iv > 0.50: return "Very Strong (>0.5)"
    if iv > 0.30: return "Strong (0.3–0.5)"
    if iv > 0.10: return "Medium (0.1–0.3)"
    if iv > 0.02: return "Weak (0.02–0.1)"
    return "Useless (<0.02)"


if __name__ == "__main__":
    # Quick self-test with dummy data
    import numpy as np
    rng = np.random.default_rng(42)
    y   = rng.binomial(1, 0.3, 1000)
    s   = rng.uniform(0, 1, 1000)

    print(f"Gini : {gini_coefficient(y, s)}")
    print(f"KS   : {ks_statistic(y, s)}")
    print(f"Grade: {assign_credit_grade(42.5)}")
