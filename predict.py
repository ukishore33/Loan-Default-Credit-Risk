"""
Loan Default Prediction — Inference / Prediction Script
Author : Kishore U. | github.com/ukishore33 | linkedin.com/in/kishore-techie
Purpose: Score new loan applications — outputs PD score + credit grade +
         approval recommendation. Mirrors production credit decisioning flow.

Run    : python src/predict.py
         (uses example applications defined in __main__ block)
"""

from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
from sklearn.ensemble     import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from utils import assign_credit_grade, gini_coefficient, ks_statistic


FEATURES = [
    "log_income", "log_loan", "loan_tenure_m", "interest_rate",
    "existing_loans", "credit_score", "delinquency_30d", "utilisation_pct",
    "months_employed", "employment_enc", "education_enc", "residence_enc",
    "purpose_enc", "foir_pct", "ltv_proxy", "credit_band",
    "high_foir", "high_util", "delinq_flag", "combined_risk", "tenure_risk",
]

APPROVAL_LOGIC = {
    "A — Very Low Risk":  ("✅ APPROVE",       "Standard terms — low risk borrower"),
    "B — Low Risk":       ("✅ APPROVE",       "Standard terms — acceptable risk"),
    "C — Medium Risk":    ("⚠️  CONDITIONAL",  "Approve with higher rate or collateral"),
    "D — High Risk":      ("❌ DECLINE",       "Risk too high at standard terms"),
    "E — Very High Risk": ("❌ DECLINE",       "High probability of default — decline"),
}


def _engineer_single(row: dict) -> dict:
    """Feature engineering for a single raw loan application dict."""
    EMP_MAP  = {"Salaried": 3, "Self-Employed": 2, "Business": 0, "Freelancer": 1}
    EDU_MAP  = {"Post-Graduate": 2, "Professional": 3, "Graduate": 0, "Under-Graduate": 1}
    RES_MAP  = {"Own": 1, "Family": 0, "Rented": 2}
    PUR_MAP  = {
        "Business": 0, "Debt Consolidation": 1, "Education": 2,
        "Home Improvement": 3, "Medical": 4, "Personal": 5,
        "Vehicle": 6, "Wedding": 7,
    }

    r   = row.copy()
    emi = (r["loan_amount"] * (r["interest_rate"] / 1200) *
           (1 + r["interest_rate"] / 1200) ** r["loan_tenure_m"] /
           ((1 + r["interest_rate"] / 1200) ** r["loan_tenure_m"] - 1))

    r["foir_pct"]       = emi / max(r["income_monthly"], 1) * 100
    r["ltv_proxy"]      = r["loan_amount"] / max(r["income_monthly"] * 12, 1)
    r["log_income"]     = np.log1p(r["income_monthly"])
    r["log_loan"]       = np.log1p(r["loan_amount"])
    r["employment_enc"] = EMP_MAP.get(r["employment_type"], 0)
    r["education_enc"]  = EDU_MAP.get(r["education"], 0)
    r["residence_enc"]  = RES_MAP.get(r["residence_type"], 2)
    r["purpose_enc"]    = PUR_MAP.get(r["loan_purpose"], 5)
    cs                  = r["credit_score"]
    r["credit_band"]    = 4 if cs < 550 else 3 if cs < 650 else 2 if cs < 720 else 1 if cs < 780 else 0
    r["high_foir"]      = int(r["foir_pct"] > 55)
    r["high_util"]      = int(r["utilisation_pct"] > 80)
    r["delinq_flag"]    = int(r["delinquency_30d"] > 0)
    r["tenure_risk"]    = int(r["loan_tenure_m"] > 60)
    r["combined_risk"]  = r["high_foir"] + r["high_util"] + r["delinq_flag"] + min(r["existing_loans"], 3)
    return r


def _train_model() -> GradientBoostingClassifier:
    """Quick re-train on the scored dataset for inference."""
    df = pd.read_csv("data/loan_data.csv")

    le = LabelEncoder()
    df["employment_enc"] = le.fit_transform(df["employment_type"])
    df["education_enc"]  = le.fit_transform(df["education"])
    df["residence_enc"]  = le.fit_transform(df["residence_type"])
    df["purpose_enc"]    = le.fit_transform(df["loan_purpose"])
    df["log_income"]     = np.log1p(df["income_monthly"])
    df["log_loan"]       = np.log1p(df["loan_amount"])
    df["credit_band"]    = pd.cut(df["credit_score"], bins=[0,550,650,720,780,900],
                                   labels=[4,3,2,1,0]).astype(int)
    df["high_foir"]      = (df["foir_pct"] > 55).astype(int)
    df["high_util"]      = (df["utilisation_pct"] > 80).astype(int)
    df["delinq_flag"]    = (df["delinquency_30d"] > 0).astype(int)
    df["tenure_risk"]    = (df["loan_tenure_m"] > 60).astype(int)
    df["combined_risk"]  = df["high_foir"] + df["high_util"] + df["delinq_flag"] + df["existing_loans"].clip(0,3)

    X, y = df[FEATURES], df["default"]
    gb   = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, subsample=0.8, random_state=42
    )
    gb.fit(X, y)
    return gb


def predict(applications: list[dict], verbose: bool = True) -> list[dict]:
    """
    Score a list of loan application dicts.

    Each dict must contain:
        loan_id, income_monthly, loan_amount, loan_tenure_m, interest_rate,
        existing_loans, credit_score, employment_type, education, residence_type,
        delinquency_30d, utilisation_pct, months_employed, loan_purpose

    Returns:
        List of result dicts with pd_score, credit_grade, decision, reason
    """
    if not os.path.exists("data/loan_data.csv"):
        raise FileNotFoundError("Run `python src/generate_data.py` and `python src/train_model.py` first.")

    model    = _train_model()
    results  = []

    for app in applications:
        engineered = _engineer_single(app)
        X_row      = pd.DataFrame([{f: engineered[f] for f in FEATURES}])
        pd_score   = round(float(model.predict_proba(X_row)[0, 1]) * 100, 1)
        grade      = assign_credit_grade(pd_score)
        decision, reason = APPROVAL_LOGIC.get(grade, ("—", "—"))

        result = {
            "loan_id":      app.get("loan_id", "—"),
            "applicant":    app.get("applicant_name", "—"),
            "loan_amount":  f"₹{app['loan_amount']:,}",
            "credit_score": app["credit_score"],
            "foir_pct":     round(engineered["foir_pct"], 1),
            "pd_score":     pd_score,
            "credit_grade": grade,
            "decision":     decision,
            "reason":       reason,
        }
        results.append(result)

        if verbose:
            print(f"\n{'─'*60}")
            print(f"  Loan ID      : {result['loan_id']}")
            print(f"  Applicant    : {result['applicant']}")
            print(f"  Loan Amount  : {result['loan_amount']}")
            print(f"  Credit Score : {result['credit_score']}")
            print(f"  FOIR         : {result['foir_pct']}%")
            print(f"  PD Score     : {result['pd_score']} / 100")
            print(f"  Credit Grade : {result['credit_grade']}")
            print(f"  Decision     : {result['decision']}")
            print(f"  Reason       : {result['reason']}")

    return results


if __name__ == "__main__":
    # ── SAMPLE LOAN APPLICATIONS ─────────────────────────────────────────
    sample_applications = [
        {
            "loan_id":         "APP001",
            "applicant_name":  "Ramesh Kumar (Low Risk)",
            "income_monthly":  80000,
            "loan_amount":     500000,
            "loan_tenure_m":   36,
            "interest_rate":   12.5,
            "existing_loans":  0,
            "credit_score":    780,
            "employment_type": "Salaried",
            "education":       "Post-Graduate",
            "residence_type":  "Own",
            "delinquency_30d": 0,
            "utilisation_pct": 20.0,
            "months_employed": 120,
            "loan_purpose":    "Home Improvement",
        },
        {
            "loan_id":         "APP002",
            "applicant_name":  "Priya Sharma (Medium Risk)",
            "income_monthly":  45000,
            "loan_amount":     800000,
            "loan_tenure_m":   60,
            "interest_rate":   18.0,
            "existing_loans":  2,
            "credit_score":    640,
            "employment_type": "Self-Employed",
            "education":       "Graduate",
            "residence_type":  "Rented",
            "delinquency_30d": 1,
            "utilisation_pct": 65.0,
            "months_employed": 36,
            "loan_purpose":    "Business",
        },
        {
            "loan_id":         "APP003",
            "applicant_name":  "Mohammed Al-Hussain (High Risk)",
            "income_monthly":  30000,
            "loan_amount":     700000,
            "loan_tenure_m":   84,
            "interest_rate":   24.0,
            "existing_loans":  4,
            "credit_score":    480,
            "employment_type": "Freelancer",
            "education":       "Under-Graduate",
            "residence_type":  "Rented",
            "delinquency_30d": 3,
            "utilisation_pct": 90.0,
            "months_employed": 12,
            "loan_purpose":    "Debt Consolidation",
        },
    ]

    print("="*60)
    print("  LOAN DEFAULT PREDICTION — CREDIT DECISIONING ENGINE")
    print("  Author: Kishore U. | github.com/ukishore33")
    print("="*60)

    predict(sample_applications)
    print(f"\n{'─'*60}")
    print("  Run `python src/train_model.py` for full portfolio metrics.")
