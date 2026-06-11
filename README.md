# Telco Customer Churn Analysis

## Overview

This project analyses churn behaviour across 7,043 telco customers to identify the key drivers of customer loss and build a predictive model capable of flagging at-risk accounts before they leave.

The analysis follows an end-to-end data science workflow: exploratory data analysis, feature engineering, model training and comparison across four algorithms, and actionable business recommendations grounded in the findings.

---

## Dataset

IBM Telco Customer Churn — publicly available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

- 7,043 customers | 21 features
- 5,163 retained (73.4%) | 1,869 churned (26.6%)
- Class imbalance addressed using SMOTE on training data only

---

## Tools & Libraries

- Python 3.10
- pandas, numpy — data manipulation
- matplotlib, seaborn — visualisation
- scikit-learn — preprocessing, modelling, evaluation
- XGBoost — gradient boosted trees
- imbalanced-learn — SMOTE oversampling

---

## Key Findings

### Overall Churn Rate: 26.6%

Roughly 1 in 4 customers left the service — high enough to represent a serious and quantifiable business problem.

---

### Finding 1 — The first 12 months are the highest-risk window

Churned customers cluster heavily in the 0–10 month tenure range. After month 30, churn becomes rare. Tenure is also the single strongest predictor in the correlation analysis (r = -0.35) — the longer a customer stays, the less likely they ever leave. This creates a compounding retention effect: surviving the first year dramatically increases lifetime value.

---

### Finding 2 — Contract type is the most actionable lever

| Contract | Churn Rate |
|---|---|
| Month-to-month | 42.7% |
| One year | 11.3% |
| Two year | 2.8% |

Month-to-month customers churn at 15x the rate of two-year contract holders. Nudging customers toward longer commitments — particularly in months 3 to 6 — is the single highest-impact retention intervention available.

---

### Finding 3 — Fiber optic has a serious retention problem

| Internet Service | Churn Rate |
|---|---|
| Fiber optic | 41.9% |
| DSL | 19.0% |
| No internet | 7.4% |

Fiber optic customers churn at nearly six times the rate of customers with no internet service. This is the most striking finding in the dataset. Fiber optic is the premium tier, yet it drives the most churn — suggesting a significant gap between customer expectation and delivered experience, or a pricing model that does not reflect perceived value.

---

### Finding 4 — Payment method is a strong churn signal

| Payment Method | Churn Rate |
|---|---|
| Electronic check | 45.3% |
| Mailed check | 19.2% |
| Bank transfer (automatic) | 16.7% |
| Credit card (automatic) | 15.3% |

Electronic check payers churn at three times the rate of automatic payment customers. Automatic payment methods (bank transfer, credit card) cluster at the lowest churn rates across the entire dataset. Migrating customers onto autopay is both an engagement signal and a retention lever — customers who automate payment demonstrate stronger commitment to the service.

---

### Finding 5 — Life stability predicts loyalty

Customers without a partner churn at 33.0% vs 19.7% for those with one. Customers without dependents churn at 31.3% vs 15.5% with dependents. Customers with more life stability — family, long-term commitments — are significantly more likely to stay. This segment is also lower-risk to acquire and retain.

---

### Finding 6 — Senior citizens are an underserved high-churn segment

Senior citizens churn at 41.7% versus 23.7% for non-seniors — nearly double the base rate. Combined with typically lower tech confidence and higher service expectations, this segment likely warrants a dedicated support approach and targeted loyalty offers rather than standard retention outreach.

---

### Finding 7 — Protective add-ons correlate with loyalty

Customers with online security (r = -0.17) and tech support (r = -0.16) subscriptions churn at lower rates. These customers are more engaged with the product ecosystem, which increases switching cost and perceived value. Upselling protective add-ons early in the customer lifecycle may serve a dual purpose: revenue and retention.

---

### The highest-risk customer profile

A customer matching all of the following criteria is at extreme churn risk:

- Tenure under 12 months
- Month-to-month contract
- Fiber optic internet service
- Paying by electronic check
- No partner or dependents
- Senior citizen

Intervening with this profile before month 6 — through a contract upgrade offer, autopay incentive, or proactive support outreach — is where a retention programme would generate the highest return.

---

## Model Results

Four models were trained and evaluated on a held-out test set. Class imbalance was addressed using SMOTE on the training set only; the test set reflects real-world class distribution to ensure honest evaluation.

| Model | ROC-AUC |
|---|---|
| Gradient Boosting | **0.817** |
| Random Forest | 0.809 |
| Logistic Regression | 0.801 |
| XGBoost | 0.796 |

**Gradient Boosting achieved the highest ROC-AUC** and is the recommended model for production use. ROC-AUC is used as the primary metric rather than accuracy, because a model predicting no churn for every customer would still achieve 73% accuracy while being entirely useless. AUC measures the model's ability to rank churners above non-churners across all decision thresholds — which is what matters for prioritising retention outreach.

### XGBoost Confusion Matrix (held-out test set, n=1,407)

|  | Predicted: No Churn | Predicted: Churn |
|---|---|---|
| **Actual: No Churn** | 861 | 172 |
| **Actual: Churn** | 161 | 213 |

- **Recall (churners identified):** 57% — the model catches just over half of actual churners
- **Precision (churn predictions correct):** 55%
- **161 false negatives** — customers who churned without being flagged; reducing this is the primary improvement target in a production scenario

The recall/precision tradeoff can be adjusted by changing the classification threshold depending on the cost structure of the retention programme. In most cases, the cost of missing a churner who then leaves exceeds the cost of reaching out to a customer who would have stayed.

---

## Top Feature Importances (Random Forest)

| Feature | Importance |
|---|---|
| TotalCharges | 0.160 |
| tenure | 0.155 |
| MonthlyCharges | 0.130 |
| PaymentMethod_Electronic check | 0.100 |
| InternetService_Fiber optic | 0.055 |
| Partner | 0.035 |
| Contract_Two year | 0.033 |
| Dependents | 0.032 |

Tenure and charges dominate. Among categorical features, electronic check payment and fiber optic internet are by far the strongest signals — consistent with the EDA findings. Gender has negligible importance and could be excluded from a production model.

---

## Business Recommendations

| Segment | Churn Rate | Recommended Action |
|---|---|---|
| Month-to-month contracts | 42.7% | Incentivise annual upgrades at months 3 and 6 with discount or added value |
| Tenure under 12 months | 50%+ | Deploy structured early-life onboarding and proactive check-in programme |
| Fiber optic customers | 41.9% | Audit service quality and pricing perception; consider satisfaction survey at 60 days |
| Electronic check payers | 45.3% | Migrate to autopay with a billing discount or loyalty incentive |
| Senior citizens | 41.7% | Dedicated support tier, simplified communications, and loyalty offers |
| Customers without online security or tech support | Higher than average | Proactively offer protective add-ons in first 90 days |

---

## Project Structure

```
churn-analysis/
├── README.md
├── churn_analysis.ipynb
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── outputs/
│   ├── 01_churn_distribution.png
│   ├── 02_numeric_distributions.png
│   ├── 03_categorical_churn_rates.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_roc_curves.png
│   ├── 06_feature_importance.png
│   ├── 07_confusion_matrix.png
│   └── model_results.csv
└── requirements.txt
```

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook churn_analysis.ipynb
```

---

## Author

Chinenye — [LinkedIn](https://www.linkedin.com/in/) | [GitHub](https://github.com/)
