"""
Loan Default Prediction — Full Training Pipeline
Author : Kishore U. | github.com/ukishore33 | linkedin.com/in/kishore-techie
Purpose: Feature engineering → model training (LR + DT + GB) →
         credit risk evaluation (Gini, KS, IV) → portfolio scoring →
         JSON output for dashboard.

Run    : python src/train_model.py
Outputs: data/loans_scored.csv  |  model_results.json
"""

from __future__ import annotations
import json, os, sys, warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd

from sklearn.ensemble         import GradientBoostingClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.metrics          import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection  import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.tree             import DecisionTreeClassifier

# Add src/ to path so utils imports cleanly whether run from root or src/
sys.path.insert(0, os.path.dirname(__file__))
from utils import gini_coefficient, ks_statistic, information_value, iv_strength_label, grade_portfolio


# ── CONSTANTS ─────────────────────────────────────────────────────────────
DATA_IN   = "data/loan_data.csv"
DATA_OUT  = "data/loans_scored.csv"
JSON_OUT  = "model_results.json"
SEED      = 42

FEATURES = [
    "log_income", "log_loan", "loan_tenure_m", "interest_rate",
    "existing_loans", "credit_score", "delinquency_30d", "utilisation_pct",
    "months_employed", "employment_enc", "education_enc", "residence_enc",
    "purpose_enc", "foir_pct", "ltv_proxy", "credit_band",
    "high_foir", "high_util", "delinq_flag", "combined_risk", "tenure_risk",
]

IV_FEATURES = [
    "credit_score", "foir_pct", "utilisation_pct", "income_monthly",
    "delinquency_30d", "loan_amount", "existing_loans", "ltv_proxy",
    "months_employed", "interest_rate",
]


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Derive all model features from raw loan data."""
    le = LabelEncoder()
    df["employment_enc"] = le.fit_transform(df["employment_type"])
    df["education_enc"]  = le.fit_transform(df["education"])
    df["residence_enc"]  = le.fit_transform(df["residence_type"])
    df["purpose_enc"]    = le.fit_transform(df["loan_purpose"])

    df["log_income"]   = np.log1p(df["income_monthly"])
    df["log_loan"]     = np.log1p(df["loan_amount"])

    df["credit_band"]  = pd.cut(
        df["credit_score"],
        bins=[0, 550, 650, 720, 780, 900],
        labels=[4, 3, 2, 1, 0],
    ).astype(int)  # 0 = best credit, 4 = worst

    df["high_foir"]     = (df["foir_pct"]       > 55).astype(int)
    df["high_util"]     = (df["utilisation_pct"] > 80).astype(int)
    df["delinq_flag"]   = (df["delinquency_30d"] > 0).astype(int)
    df["tenure_risk"]   = (df["loan_tenure_m"]   > 60).astype(int)
    df["combined_risk"] = (
        df["high_foir"] + df["high_util"] +
        df["delinq_flag"] + df["existing_loans"].clip(0, 3)
    )
    return df


# ── MODEL EVALUATION ─────────────────────────────────────────────────────
def evaluate(name: str, df_ref: pd.DataFrame,
             y_true: np.ndarray, y_pred: np.ndarray,
             y_proba: np.ndarray) -> dict:
    """Return all classification + credit risk metrics as a dict."""
    return {
        "model":       name,
        "accuracy":    round(float(accuracy_score(y_true, y_pred)),                       4),
        "precision":   round(float(precision_score(y_true, y_pred, zero_division=0)),     4),
        "recall":      round(float(recall_score(y_true, y_pred,    zero_division=0)),     4),
        "f1":          round(float(f1_score(y_true, y_pred,        zero_division=0)),     4),
        "roc_auc":     round(float(roc_auc_score(y_true, y_proba)),                       4),
        "pr_auc":      round(float(average_precision_score(y_true, y_proba)),             4),
        "gini":        gini_coefficient(y_true, y_proba),
        "ks_stat":     ks_statistic(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────
def train(data_path: str = DATA_IN) -> dict:
    # 1. Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"'{data_path}' not found. Run `python src/generate_data.py` first."
        )
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df):,} loans | Default rate: {df['default'].mean():.1%}")

    # 2. Feature engineering
    df = engineer(df)

    X, y = df[FEATURES], df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )

    # 3. Scaling (for LR only)
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    results  = {}

    # 4a. Logistic Regression — the regulatory-compliant scorecard model
    print("\n🔄 Training Logistic Regression (scorecard)…")
    lr      = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    lr.fit(X_tr_sc, y_train)
    pr_lr   = lr.predict_proba(X_te_sc)[:, 1]
    pd_lr   = lr.predict(X_te_sc)
    results["logistic_regression"] = evaluate("Logistic Regression", df, y_test, pd_lr, pr_lr)
    print(f"   AUC {results['logistic_regression']['roc_auc']}  "
          f"Gini {results['logistic_regression']['gini']}  "
          f"KS {results['logistic_regression']['ks_stat']}")

    # 4b. Decision Tree — interpretable for risk committees
    print("🔄 Training Decision Tree…")
    dt      = DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=50,
        class_weight="balanced", random_state=SEED
    )
    dt.fit(X_train, y_train)
    pr_dt   = dt.predict_proba(X_test)[:, 1]
    pd_dt   = dt.predict(X_test)
    results["decision_tree"] = evaluate("Decision Tree", df, y_test, pd_dt, pr_dt)
    print(f"   AUC {results['decision_tree']['roc_auc']}  "
          f"Gini {results['decision_tree']['gini']}  "
          f"KS {results['decision_tree']['ks_stat']}")

    # 4c. Gradient Boosting — highest predictive power
    print("🔄 Training Gradient Boosting…")
    gb      = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, subsample=0.8, random_state=SEED
    )
    gb.fit(X_train, y_train)
    pr_gb   = gb.predict_proba(X_test)[:, 1]
    pd_gb   = gb.predict(X_test)

    # 5-fold cross-validation
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_auc  = cross_val_score(gb, X, y, cv=cv, scoring="roc_auc")

    results["gradient_boosting"] = evaluate("Gradient Boosting", df, y_test, pd_gb, pr_gb)
    results["gradient_boosting"]["cv_auc_mean"] = round(float(cv_auc.mean()), 4)
    results["gradient_boosting"]["cv_auc_std"]  = round(float(cv_auc.std()),  4)
    print(f"   AUC {results['gradient_boosting']['roc_auc']}  "
          f"Gini {results['gradient_boosting']['gini']}  "
          f"KS {results['gradient_boosting']['ks_stat']}  "
          f"CV-AUC {results['gradient_boosting']['cv_auc_mean']} ±{results['gradient_boosting']['cv_auc_std']}")

    # 5. Feature importances (GB)
    feat_imp = sorted(
        zip(FEATURES, gb.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    feature_importance = [
        {"feature": f, "importance": round(float(v), 4)} for f, v in feat_imp
    ]

    # 6. Information Value table
    print("\n🔄 Computing Information Value (IV) for features…")
    iv_table = []
    for feat in IV_FEATURES:
        iv  = information_value(df, feat, target="default")
        iv_table.append({
            "feature":             feat,
            "iv":                  iv,
            "predictive_strength": iv_strength_label(iv),
        })
    iv_table.sort(key=lambda x: -x["iv"])
    for row in iv_table[:5]:
        print(f"   {row['feature']:25s}  IV={row['iv']:.4f}  →  {row['predictive_strength']}")

    # 7. Score full portfolio
    df["pd_score"]          = (gb.predict_proba(X[FEATURES])[:, 1] * 100).round(1)
    df["credit_grade"]      = grade_portfolio(df["pd_score"])
    df["predicted_default"] = (df["pd_score"] >= 50).astype(int)
    os.makedirs(os.path.dirname(DATA_OUT) if os.path.dirname(DATA_OUT) else ".", exist_ok=True)
    df.to_csv(DATA_OUT, index=False)
    print(f"\n✅ Scored portfolio saved → {DATA_OUT}")

    # 8. Build summary
    summary = {
        "total_loans":        int(len(df)),
        "total_defaults":     int(df["default"].sum()),
        "default_rate_pct":   round(float(df["default"].mean() * 100), 2),
        "predicted_defaults": int(df["predicted_default"].sum()),
        "grade_a":            int((df["credit_grade"] == "A — Very Low Risk").sum()),
        "grade_b":            int((df["credit_grade"] == "B — Low Risk").sum()),
        "grade_c":            int((df["credit_grade"] == "C — Medium Risk").sum()),
        "grade_d":            int((df["credit_grade"] == "D — High Risk").sum()),
        "grade_e":            int((df["credit_grade"] == "E — Very High Risk").sum()),
        "avg_loan_amount":    int(df["loan_amount"].mean()),
        "avg_credit_score":   int(df["credit_score"].mean()),
        "avg_foir_pct":       round(float(df["foir_pct"].mean()), 2),
        "high_risk_loans":    int((df["pd_score"] >= 65).sum()),
    }

    output = {
        "metrics":            results,
        "feature_importance": feature_importance,
        "iv_table":           iv_table,
        "summary":            summary,
    }

    with open(JSON_OUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✅ Metrics saved → {JSON_OUT}")

    # 9. Final summary print
    gb_m = results["gradient_boosting"]
    lr_m = results["logistic_regression"]
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           CREDIT RISK MODEL — FINAL RESULTS                  ║
╠══════════════════════════════════════════════════════════════╣
║  Gradient Boosting │ AUC {gb_m['roc_auc']} │ Gini {gb_m['gini']} │ KS {gb_m['ks_stat']}        ║
║  Logistic Reg.     │ AUC {lr_m['roc_auc']} │ Gini {lr_m['gini']} │ KS {lr_m['ks_stat']}        ║
║  5-Fold CV AUC     │ {gb_m['cv_auc_mean']} ± {gb_m['cv_auc_std']}                            ║
║  High-Risk Loans   │ {summary['high_risk_loans']:,} (PD ≥ 65)                         ║
╚══════════════════════════════════════════════════════════════╝
    """)
    return output


if __name__ == "__main__":
    train()
