"""
Loan Default Prediction — Synthetic Data Generator
Author : Kishore U. | github.com/ukishore33 | linkedin.com/in/kishore-techie
Purpose: Generates a German Credit / Lending Club-inspired synthetic loan
         portfolio with realistic default patterns, FOIR, credit scores, and
         employment-type risk differentials.
         
Run    : python src/generate_data.py
Output : data/loan_data.csv
"""

import pandas as pd
import numpy as np
import os

np.random.seed(2025)


LOAN_PURPOSES = [
    "Home Improvement", "Medical", "Wedding", "Education",
    "Business", "Debt Consolidation", "Vehicle", "Personal"
]
EMPLOYMENT_TYPES = ["Salaried", "Self-Employed", "Business", "Freelancer"]
EDUCATION_LEVELS = ["Graduate", "Post-Graduate", "Under-Graduate", "Professional"]
RESIDENCE_TYPES  = ["Own", "Rented", "Family"]
TENURES_MONTHS   = [12, 24, 36, 48, 60, 72, 84]


def compute_emi(principal: float, annual_rate: float, tenure_months: int) -> float:
    """Standard reducing-balance EMI formula."""
    r = annual_rate / 1200
    if r == 0:
        return principal / tenure_months
    return principal * r * (1 + r) ** tenure_months / ((1 + r) ** tenure_months - 1)


def assign_default(log_odds_vec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Convert log-odds to Bernoulli draws — the actual default label."""
    prob = 1 / (1 + np.exp(-log_odds_vec))
    return (rng.uniform(0, 1, len(prob)) < prob).astype(int)


def generate_loan_data(n: int = 5000, output_path: str = "data/loan_data.csv") -> pd.DataFrame:
    """
    Generate n synthetic loan records and save to CSV.

    Default-rate logic mirrors real MSME / unsecured-personal-loan portfolios:
    - Salaried + high credit score → lower default
    - Freelancer + high FOIR + delinquency history → higher default
    - Debt Consolidation loans → elevated risk (borrower already stressed)
    """
    rng = np.random.default_rng(2025)

    # ── RAW FIELDS ────────────────────────────────────────────────────────
    age              = rng.integers(21, 65, n)
    income_monthly   = rng.lognormal(10.2, 0.5, n).round(0)
    loan_amount      = rng.lognormal(11.5, 0.8, n).round(0)
    loan_tenure_m    = rng.choice(TENURES_MONTHS, n)
    interest_rate    = rng.uniform(8.5, 28.0, n).round(2)
    existing_loans   = rng.integers(0, 5, n)
    credit_score     = rng.integers(300, 850, n)
    employment_type  = rng.choice(EMPLOYMENT_TYPES, n, p=[0.50, 0.25, 0.15, 0.10])
    education        = rng.choice(EDUCATION_LEVELS, n, p=[0.45, 0.25, 0.20, 0.10])
    residence_type   = rng.choice(RESIDENCE_TYPES,  n, p=[0.35, 0.45, 0.20])
    delinquency_30d  = rng.integers(0, 6, n)
    utilisation_pct  = rng.uniform(0, 100, n).round(1)
    months_employed  = rng.integers(0, 300, n)
    loan_purpose     = rng.choice(LOAN_PURPOSES, n)

    # ── DERIVED FIELDS ────────────────────────────────────────────────────
    emi      = np.array([
        compute_emi(loan_amount[i], interest_rate[i], loan_tenure_m[i])
        for i in range(n)
    ]).round(0)
    foir_pct = (emi / np.maximum(income_monthly, 1) * 100).round(2)
    ltv_proxy = (loan_amount / (income_monthly * 12)).round(4)

    # ── DEFAULT LABEL (LOGISTIC FUNCTION OVER RISK FACTORS) ──────────────
    log_odds = (
        -3.5
        + 0.04  * (650 - credit_score) / 100        # lower score → higher risk
        + 0.80  * (foir_pct > 55).astype(float)      # over-leveraged
        + 0.60  * delinquency_30d                    # past 30-day delays
        + 0.50  * existing_loans                     # debt burden
        + 0.40  * (employment_type == "Freelancer")
        + 0.30  * (employment_type == "Self-Employed")
        + 0.25  * (utilisation_pct > 80)             # stressed credit card
        + 0.50  * (loan_purpose == "Debt Consolidation")
        - 0.30  * (education == "Post-Graduate")
        - 0.20  * (months_employed > 60)             # stable employment
        + rng.normal(0, 0.5, n)                      # idiosyncratic noise
    )
    default = assign_default(log_odds, rng)

    # ── ASSEMBLE DATAFRAME ────────────────────────────────────────────────
    df = pd.DataFrame({
        "loan_id":          [f"LN{i+1:06d}" for i in range(n)],
        "age":              age,
        "income_monthly":   income_monthly.astype(int),
        "loan_amount":      loan_amount.astype(int),
        "loan_tenure_m":    loan_tenure_m,
        "interest_rate":    interest_rate,
        "existing_loans":   existing_loans,
        "credit_score":     credit_score,
        "employment_type":  employment_type,
        "education":        education,
        "residence_type":   residence_type,
        "delinquency_30d":  delinquency_30d,
        "utilisation_pct":  utilisation_pct,
        "months_employed":  months_employed,
        "loan_purpose":     loan_purpose,
        "emi":              emi.astype(int),
        "foir_pct":         foir_pct,
        "ltv_proxy":        ltv_proxy,
        "default":          default,
    })

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    total     = len(df)
    n_default = df["default"].sum()
    print(f"✅ Generated {total:,} loans")
    print(f"   Defaults     : {n_default:,} ({n_default/total:.1%})")
    print(f"   Non-defaults : {total - n_default:,} ({1 - n_default/total:.1%})")
    print(f"   Saved to     : {output_path}")
    return df


if __name__ == "__main__":
    generate_loan_data()
