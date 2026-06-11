"""
Telco Customer Churn Analysis
Dataset: IBM Telco Customer Churn (7,043 customers)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

RANDOM_STATE = 42
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Load & inspect ──────────────────────────────────────────────────────────

def load_data(path: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)
    df.drop(columns=["customerID"], inplace=True)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


# ── 2. EDA helpers ─────────────────────────────────────────────────────────────

def plot_churn_rate(df: pd.DataFrame) -> None:
    counts = df["Churn"].value_counts()
    labels = ["No Churn", "Churn"]
    colors = ["#4C72B0", "#DD8452"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].pie(counts, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
    axes[0].set_title("Overall Churn Distribution")

    axes[1].bar(labels, counts.values, color=colors, edgecolor="white")
    axes[1].set_ylabel("Customers")
    axes[1].set_title("Churn Count")
    for i, v in enumerate(counts.values):
        axes[1].text(i, v + 30, str(v), ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_churn_distribution.png", dpi=150)
    plt.close()
    print("Churn rate:", round(df["Churn"].mean() * 100, 2), "%")


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, numeric_cols):
        for label, color in [(0, "#4C72B0"), (1, "#DD8452")]:
            subset = df[df["Churn"] == label][col]
            ax.hist(subset, bins=30, alpha=0.6, color=color,
                    label="No Churn" if label == 0 else "Churn", edgecolor="white")
        ax.set_title(col)
        ax.legend()
    plt.suptitle("Numeric Feature Distributions by Churn", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_numeric_distributions.png", dpi=150)
    plt.close()


def plot_categorical_churn_rates(df: pd.DataFrame) -> None:
    cat_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "PhoneService", "InternetService", "Contract",
        "PaperlessBilling", "PaymentMethod"
    ]
    n = len(cat_cols)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    for ax, col in zip(axes, cat_cols):
        rate = df.groupby(col)["Churn"].mean().sort_values(ascending=False)
        sns.barplot(x=rate.index, y=rate.values, ax=ax, palette="muted")
        ax.set_title(f"Churn Rate by {col}")
        ax.set_ylabel("Churn Rate")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom", fontsize=8)
    for ax in axes[n:]:
        ax.set_visible(False)
    plt.suptitle("Churn Rate by Categorical Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_categorical_churn_rates.png", dpi=150)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, encoded: pd.DataFrame) -> None:
    corr = encoded.corr()[["Churn"]].drop("Churn").sort_values("Churn", ascending=False)
    fig, ax = plt.subplots(figsize=(6, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation with Churn", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_correlation_heatmap.png", dpi=150)
    plt.close()


# ── 3. Preprocessing ───────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    df_enc = df.copy()
    le = LabelEncoder()
    binary_cols = [c for c in df_enc.select_dtypes("object").columns
                   if df_enc[c].nunique() == 2]
    for col in binary_cols:
        df_enc[col] = le.fit_transform(df_enc[col])

    df_enc = pd.get_dummies(df_enc, drop_first=True)

    X = df_enc.drop(columns=["Churn"])
    y = df_enc["Churn"]
    return X, y, df_enc


# ── 4. Modelling ───────────────────────────────────────────────────────────────

def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "XGBoost":             XGBClassifier(n_estimators=200, use_label_encoder=False,
                                             eval_metric="logloss", random_state=RANDOM_STATE),
    }


def evaluate_models(X_train, X_test, y_train, y_test, scaler) -> pd.DataFrame:
    models = build_models()
    results = []

    X_train_sc = scaler.transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (name, model) in zip(axes, models.items()):
        X_tr = X_train_sc if name == "Logistic Regression" else X_train
        X_te = X_test_sc  if name == "Logistic Regression" else X_test

        model.fit(X_tr, y_train)
        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        report = classification_report(y_test, y_pred, output_dict=True)

        results.append({
            "Model":     name,
            "ROC-AUC":   round(auc, 4),
            "Precision": round(report["1"]["precision"], 4),
            "Recall":    round(report["1"]["recall"], 4),
            "F1-Score":  round(report["1"]["f1-score"], 4),
            "Accuracy":  round(report["accuracy"], 4),
        })

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", lw=2)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(name)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")

    plt.suptitle("ROC Curves — All Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_roc_curves.png", dpi=150)
    plt.close()

    return pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)


def plot_feature_importance(X_train, y_train) -> None:
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    importance = pd.Series(rf.feature_importances_, index=X_train.columns)
    top20 = importance.nlargest(20).sort_values()

    fig, ax = plt.subplots(figsize=(9, 7))
    top20.plot(kind="barh", color="#4C72B0", ax=ax)
    ax.set_title("Top 20 Feature Importances (Random Forest)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_feature_importance.png", dpi=150)
    plt.close()


def plot_confusion_matrix(X_train, X_test, y_train, y_test) -> None:
    xgb = XGBClassifier(n_estimators=200, use_label_encoder=False,
                         eval_metric="logloss", random_state=RANDOM_STATE)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — XGBoost", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/07_confusion_matrix.png", dpi=150)
    plt.close()


# ── 5. Business insights ───────────────────────────────────────────────────────

def print_business_insights(df: pd.DataFrame) -> None:
    print("\n" + "="*60)
    print("BUSINESS INSIGHTS")
    print("="*60)

    m2m = df[df["Contract"] == "Month-to-month"]["Churn"].mean()
    one = df[df["Contract"] == "One year"]["Churn"].mean()
    two = df[df["Contract"] == "Two year"]["Churn"].mean()
    print(f"\nChurn by contract type:")
    print(f"  Month-to-month : {m2m:.1%}")
    print(f"  One year       : {one:.1%}")
    print(f"  Two year       : {two:.1%}")

    fiber = df[df["InternetService"] == "Fiber optic"]["Churn"].mean()
    dsl   = df[df["InternetService"] == "DSL"]["Churn"].mean()
    none_ = df[df["InternetService"] == "No"]["Churn"].mean()
    print(f"\nChurn by internet service:")
    print(f"  Fiber optic : {fiber:.1%}")
    print(f"  DSL         : {dsl:.1%}")
    print(f"  None        : {none_:.1%}")

    senior = df[df["SeniorCitizen"] == 1]["Churn"].mean()
    non_s  = df[df["SeniorCitizen"] == 0]["Churn"].mean()
    print(f"\nChurn by senior citizen status:")
    print(f"  Senior     : {senior:.1%}")
    print(f"  Non-senior : {non_s:.1%}")

    q1, q3 = df["tenure"].quantile([0.25, 0.75])
    low    = df[df["tenure"] <= q1]["Churn"].mean()
    high   = df[df["tenure"] >= q3]["Churn"].mean()
    print(f"\nChurn by tenure:")
    print(f"  Low tenure  (<={q1:.0f} mo) : {low:.1%}")
    print(f"  High tenure (>={q3:.0f} mo) : {high:.1%}")


# ── 6. Main pipeline ───────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df = load_data()
    print(f"Shape: {df.shape}  |  Churn rate: {df['Churn'].mean():.1%}")

    print("\nRunning EDA...")
    plot_churn_rate(df)
    plot_numeric_distributions(df)
    plot_categorical_churn_rates(df)

    print("Preprocessing...")
    X, y, df_enc = preprocess(df)
    plot_correlation_heatmap(df, df_enc)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    scaler = StandardScaler().fit(X_train_res)

    print("\nTraining & evaluating models...")
    results = evaluate_models(X_train_res, X_test, y_train_res, y_test, scaler)
    print("\nModel Comparison:")
    print(results.to_string(index=False))

    print("\nGenerating feature importance & confusion matrix...")
    plot_feature_importance(X_train_res, y_train_res)
    plot_confusion_matrix(X_train_res, X_test, y_train_res, y_test)

    print_business_insights(df)

    results.to_csv(f"{OUTPUT_DIR}/model_results.csv", index=False)
    print(f"\nAll outputs saved to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
