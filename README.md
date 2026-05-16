# 💳 Loan Default Prediction — Credit Risk Scorecard

> **End-to-end credit risk ML pipeline** simulating a production-grade Probability of Default (PD) model used by banks and NBFCs — Logistic Regression scorecard + Gradient Boosting classifier with industry-standard credit risk metrics: **Gini Coefficient 0.521 · KS Statistic 0.395 · ROC-AUC 0.761 · 5-Fold CV AUC 0.759 ± 0.009**

---

## 📌 Project Overview

This project builds a **credit risk scorecard** — the core model used by HDFC Bank, ICICI Bank, Bajaj Finance, and NBFCs to decide whether to approve or reject a loan application. It mirrors the full credit analytics workflow:

- Synthetic loan portfolio generation (German Credit / Lending Club inspired)
- Feature engineering with **FOIR, LTV proxy, credit bands, WoE encoding**
- Model training: Logistic Regression (interpretable scorecard) + Gradient Boosting (predictive power)
- Credit risk evaluation: **Gini, KS Statistic, Information Value (IV), Weight of Evidence (WoE)**
- Portfolio scoring with **5 credit grades (A to E)**
- Interactive dashboard with risk distribution, feature importance, IV table

**Relevance:** Directly applicable to Credit Risk Analyst, Risk Analytics, and Data Analyst (Finance) roles at HDFC Bank, ICICI Bank, Bajaj Finance, Moody's, CRISIL, Acuity Knowledge Partners, EXL Service.

---

## 🏗️ Repository Structure

```
loan-default-prediction/
│
├── src/
│   ├── generate_data.py        # Synthetic loan portfolio generator (5,000 loans)
│   ├── train_model.py          # Full ML pipeline — LR + Decision Tree + Gradient Boosting
│   ├── predict.py              # Inference script — score new loan applications
│   └── utils.py                # Gini, KS, IV/WoE helper functions
│
├── data/
│   ├── loan_data.csv           # Raw synthetic loan portfolio (generated)
│   └── loans_scored.csv        # Enriched with PD scores + credit grades (generated)
│
├── notebooks/
│   └── EDA_CreditRisk.ipynb    # Exploratory data analysis notebook
│
├── tests/
│   └── test_pipeline.py        # Unit tests for data generation + model pipeline
│
├── docs/
│   └── credit_risk_concepts.md # IV, WoE, Gini, KS explained in plain English
│
├── credit_risk_dashboard.html  # Interactive browser dashboard (no server needed)
├── model_results.json          # Metrics, feature importances, IV table (generated)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Standard Python gitignore
└── README.md
```

---

## 📊 Dataset — Synthetic Loan Portfolio

| Property | Value |
|---|---|
| Total Loans | 5,000 |
| Default Rate | ~33.7% (realistic for unsecured/MSME lending) |
| Loan Types | Personal, Home Improvement, Medical, Wedding, Education, Business, Vehicle |
| Employment Types | Salaried, Self-Employed, Business, Freelancer |
| Loan Range | ₹20,000 – ₹20,00,000 |
| Income Range | ₹10,000 – ₹1,50,000/month |
| Credit Score Range | 300 – 850 |

### Features Engineered

| Feature | Description | Credit Risk Rationale |
|---|---|---|
| `foir_pct` | Fixed Obligation-to-Income Ratio | EMI / Income — capacity to repay |
| `ltv_proxy` | Loan-to-Income ratio | Leverage indicator |
| `credit_band` | Bucketed credit score (0=best, 4=worst) | Bureau score risk tier |
| `high_foir` | Binary: FOIR > 55% | Industry threshold for over-leverage |
| `high_util` | Binary: Credit utilisation > 80% | Stress indicator |
| `delinq_flag` | Binary: Any 30-day past due | Behavioural default predictor |
| `combined_risk` | Sum of all binary risk flags | Composite alert score |
| `log_income` | Log-transformed monthly income | Corrects right-skewed distribution |

---

## 🤖 Models & Credit Risk Metrics

Three models were trained and compared. **Logistic Regression** is the industry standard for regulatory-compliant scorecards; **Gradient Boosting** provides maximum predictive power.

### Performance

| Metric | Gradient Boosting | Logistic Regression | Decision Tree |
|---|---|---|---|
| ROC-AUC | 0.7447 | 0.7607 | 0.6812 |
| **Gini Coefficient** | **0.4894** | **0.5213** | 0.3624 |
| **KS Statistic** | **0.3710** | **0.3948** | 0.2981 |
| PR-AUC | 0.631 | 0.644 | 0.541 |
| 5-Fold CV AUC | 0.759 ± 0.009 | 0.758 ± 0.011 | — |

### Why These Metrics?

- **Gini Coefficient** = 2 × AUC − 1. Industry benchmark: >0.3 = acceptable, >0.4 = good, >0.6 = excellent. Regulators (RBI, Basel III) explicitly require Gini reporting for PD models.
- **KS Statistic** = Maximum separation between cumulative Good and Bad distributions. >0.3 = deployable scorecard.
- **Information Value (IV)** = Measures each feature's predictive power. Used for feature selection in scorecards.

### Information Value Table

| Feature | IV Score | Predictive Strength |
|---|---|---|
| Delinquency (30-day past due) | 0.7288 | Very Strong (>0.5) |
| Existing Loans | 0.2386 | Medium (0.1–0.3) |
| Credit Score | 0.1840 | Medium (0.1–0.3) |
| FOIR % | 0.0298 | Weak (<0.1) |
| Loan Amount | 0.0140 | Weak (<0.1) |

### Credit Grade Distribution

| Grade | PD Score Range | Count | Interpretation |
|---|---|---|---|
| A — Very Low Risk | 0–20 | 1,180 | Approve immediately |
| B — Low Risk | 20–35 | 1,420 | Approve with standard terms |
| C — Medium Risk | 35–50 | 1,025 | Approve with conditions |
| D — High Risk | 50–65 | 622 | Decline or high rate |
| E — Very High Risk | 65–100 | 753 | Decline |

---

## 📊 Dashboard Features

- **8 KPI cards** — Total loans, default rate, Gini, KS, AUC, CV AUC, high-risk count
- **Credit grade distribution** — Doughnut chart (A to E)
- **Default rate by employment type** — Salaried vs Freelancer vs Business
- **Default rate by credit score band** — Visual risk gradient
- **Feature importance** — Gradient Boosting top drivers
- **Information Value table** — IV scores with predictive strength labels
- **Model comparison** — GB vs LR with confusion matrix

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data Generation | Python · Pandas · NumPy |
| ML Models | scikit-learn — LogisticRegression, DecisionTreeClassifier, GradientBoostingClassifier |
| Credit Risk Metrics | Custom Gini, KS, IV/WoE functions (Basel III aligned) |
| Feature Engineering | FOIR, LTV proxy, WoE bins, log transforms, credit bands |
| Cross-Validation | StratifiedKFold (5-fold) — preserves default rate in each fold |
| Evaluation | ROC-AUC · Gini · KS Statistic · PR-AUC · IV |
| Dashboard | Chart.js · HTML/CSS · JetBrains Mono |

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/ukishore33/loan-default-prediction.git
cd loan-default-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic data
python src/generate_data.py

# 4. Train models and generate metrics
python src/train_model.py

# 5. Score a new loan application
python src/predict.py

# 6. Open interactive dashboard
open credit_risk_dashboard.html

# 7. Run tests
python -m pytest tests/
```
##### [Live Demo](https://ukishore33.github.io/Loan-Default-Credit-Risk/credit_risk_dashboard.html)
---

## 🗂️ Key Files Explained

| File | Purpose |
|---|---|
| `src/generate_data.py` | Generates 5,000 synthetic loans with realistic default patterns |
| `src/train_model.py` | Full training pipeline — 3 models, all metrics, portfolio scoring |
| `src/utils.py` | Standalone Gini, KS, IV, WoE functions — reusable in any project |
| `src/predict.py` | Takes new loan data → outputs PD score + credit grade |
| `model_results.json` | All metrics, feature importances, IV table in structured JSON |
| `credit_risk_dashboard.html` | Interactive dashboard — open in any browser, no server |
| `docs/credit_risk_concepts.md` | Plain-English explanation of Gini, KS, IV for non-technical readers |

---

## 💡 Interview Talking Points

> *"I built a credit risk scorecard that mirrors Basel III PD model requirements. I used Logistic Regression as the primary scorecard model — because it's interpretable and regulatory-compliant — and compared it against Gradient Boosting for predictive power. The Gini of 0.52 and KS of 0.39 both exceed industry thresholds for a deployable scorecard. I also computed Information Value for all features — delinquency history was the strongest predictor at IV = 0.73, which is Very Strong by the industry scale. The model assigns every loan a credit grade from A to E, directly mirroring what HDFC Bank and Bajaj Finance use in their credit decisioning systems."*

---

## 👤 Author

**Kishore U.**
AML/KYC Compliance Analyst | Credit Risk Analytics | Data Analytics
📱 6303308133 | Bengaluru, Karnataka | Immediate Joiner
🔗 [LinkedIn](https://www.linkedin.com/in/kishore-techie/) · [GitHub](https://github.com/ukishore33)

**Skills demonstrated:** Credit Risk · Probability of Default · Scorecard Development · Gini · KS Statistic · Information Value · Python · scikit-learn · Gradient Boosting · Logistic Regression · Basel III PD Model Concepts

---

## 📜 Disclaimer

All data is **100% synthetic** — generated programmatically. No real loan, customer, or financial data was used. Built purely for portfolio demonstration.
