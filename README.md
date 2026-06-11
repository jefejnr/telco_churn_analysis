# Telco Customer Churn Analysis

End-to-end churn analysis on the IBM Telco Customer Churn dataset — covering exploratory data analysis, machine learning modelling, and actionable business recommendations.

---

## Dataset

**IBM Telco Customer Churn** (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)  
- **7,043 customers**, 21 features  
- Target: `Churn` (Yes/No) — overall rate ≈ 26.5%

Features include customer demographics, account info (tenure, contract type, payment method), and subscribed services (phone, internet, streaming, etc.).

---

## Project Structure

```
telco_churn_analysis/
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw data
├── churn_analysis.py                       # Standalone Python pipeline
├── churn_analysis.ipynb                    # Jupyter notebook (narrative version)
├── requirements.txt
├── .gitignore
└── outputs/                                # Generated charts & model results (git-ignored)
```

---

## Setup

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Analysis

**Script (headless, saves all plots to `outputs/`):**
```bash
python churn_analysis.py
```

**Notebook (interactive):**
```bash
jupyter notebook churn_analysis.ipynb
```

---

## Analysis Steps

| Step | Description |
|------|-------------|
| 1 | Data loading, type coercion, missing-value handling |
| 2 | EDA — churn distribution, numeric histograms, categorical churn rates |
| 3 | Preprocessing — label encoding, one-hot encoding, SMOTE oversampling |
| 4 | Model training — Logistic Regression, Random Forest, Gradient Boosting, XGBoost |
| 5 | Evaluation — ROC-AUC, precision/recall/F1, confusion matrix |
| 6 | Feature importance (Random Forest) |
| 7 | Business insights & recommendations |

---

## Key Findings

| Driver | Churn Rate | Action |
|--------|-----------|--------|
| Month-to-month contract | ~43% | Incentivise annual upgrades |
| Fiber optic internet | ~42% | Investigate service quality / pricing |
| Tenure < 12 months | ~50%+ | Early-life retention programme |
| Senior citizens | ~41% | Dedicated support & loyalty offers |
| Electronic check payment | ~45% | Encourage autopay migration |

**Top predictive features:** `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract_Two year`, `InternetService_Fiber optic`

---

## Model Performance (held-out test set, 20%)

| Model | ROC-AUC | F1 (Churn) |
|-------|---------|------------|
| XGBoost | ~0.84 | ~0.63 |
| Gradient Boosting | ~0.83 | ~0.62 |
| Random Forest | ~0.83 | ~0.61 |
| Logistic Regression | ~0.84 | ~0.62 |

> Exact numbers will vary slightly due to SMOTE randomness; run the script to see your results.

---

## Dependencies

- `pandas`, `numpy` — data wrangling  
- `matplotlib`, `seaborn` — visualisation  
- `scikit-learn` — preprocessing & ML models  
- `xgboost` — gradient boosted trees  
- `imbalanced-learn` — SMOTE oversampling  

---

## License

Dataset provided by IBM and distributed via Kaggle under public use terms.  
Analysis code released under the MIT License.
