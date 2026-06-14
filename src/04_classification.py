"""
04_classification.py
Classification Analysis for Telco Customer Churn

Models:
- Logistic Regression (baseline, balanced, SMOTE)
- Gaussian Naïve Bayes (baseline, SMOTE)

Outputs:
- Model artefacts (joblib)
- Metrics DataFrame
- Feature importance from best Logistic Regression
"""

import pickle
import warnings

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_data():
    """Load train/test splits and feature names."""
    X_train = pd.read_pickle(DATA_PROCESSED / "X_train.pkl")
    X_test = pd.read_pickle(DATA_PROCESSED / "X_test.pkl")
    y_train = pd.read_pickle(DATA_PROCESSED / "y_train.pkl")
    y_test = pd.read_pickle(DATA_PROCESSED / "y_test.pkl")

    with open(DATA_PROCESSED / "feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    print(f"[INFO] X_train: {X_train.shape}  |  X_test: {X_test.shape}")
    print(f"[INFO] y_train distribution:\n{y_train.value_counts().to_string()}")
    print(f"[INFO] Feature count: {len(feature_names)}")
    return X_train, X_test, y_train, y_test, feature_names


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a fitted model on the test set.

    Returns a dict of metrics and prints the classification report.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    roc = roc_auc_score(y_test, y_proba)

    print(f"\n{'─' * 60}")
    print(f"  {model_name}")
    print(f"{'─' * 60}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(f"  ROC AUC : {roc:.4f}")
    print(f"{'─' * 60}")

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC AUC": roc,
    }


def train_all_models(X_train, X_test, y_train, y_test):
    """Train all classification models and return metrics + model dict."""

    results = []
    models_dict = {}

    # ── Prepare SMOTE-resampled training data (done once) ────────────────
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(
        f"\n[INFO] SMOTE resampled training set: "
        f"{X_train_smote.shape[0]:,} samples "
        f"(original {X_train.shape[0]:,})"
    )

    # ── 1. Logistic Regression – Baseline ────────────────────────────────
    lr_base = LogisticRegression(max_iter=1000, random_state=42)
    lr_base.fit(X_train, y_train)
    models_dict["LogReg_Baseline"] = lr_base
    results.append(evaluate_model(lr_base, X_test, y_test, "LogReg_Baseline"))

    # ── 2. Logistic Regression – Balanced ────────────────────────────────
    lr_bal = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr_bal.fit(X_train, y_train)
    models_dict["LogReg_Balanced"] = lr_bal
    results.append(evaluate_model(lr_bal, X_test, y_test, "LogReg_Balanced"))

    # ── 3. Logistic Regression – SMOTE ───────────────────────────────────
    lr_smote = LogisticRegression(max_iter=1000, random_state=42)
    lr_smote.fit(X_train_smote, y_train_smote)
    models_dict["LogReg_SMOTE"] = lr_smote
    results.append(evaluate_model(lr_smote, X_test, y_test, "LogReg_SMOTE"))

    # ── 4. Gaussian NB – Baseline ────────────────────────────────────────
    gnb_base = GaussianNB()
    gnb_base.fit(X_train, y_train)
    models_dict["NaiveBayes_Baseline"] = gnb_base
    results.append(evaluate_model(gnb_base, X_test, y_test, "NaiveBayes_Baseline"))

    # ── 5. Gaussian NB – SMOTE ───────────────────────────────────────────
    gnb_smote = GaussianNB()
    gnb_smote.fit(X_train_smote, y_train_smote)
    models_dict["NaiveBayes_SMOTE"] = gnb_smote
    results.append(evaluate_model(gnb_smote, X_test, y_test, "NaiveBayes_SMOTE"))

    return results, models_dict


def select_best_models(results, models_dict):
    """
    Select the best Logistic Regression and best Naive Bayes
    based on F1-Score for the churn class.
    """
    lr_results = [r for r in results if r["Model"].startswith("LogReg")]
    nb_results = [r for r in results if r["Model"].startswith("NaiveBayes")]

    best_lr_name = max(lr_results, key=lambda r: r["F1-Score"])["Model"]
    best_nb_name = max(nb_results, key=lambda r: r["F1-Score"])["Model"]

    print(f"\n[BEST] Logistic Regression : {best_lr_name}")
    print(f"[BEST] Naïve Bayes        : {best_nb_name}")

    return models_dict[best_lr_name], best_lr_name, models_dict[best_nb_name], best_nb_name


def extract_feature_importance(model, feature_names):
    """
    Extract feature importance (absolute coefficients) from a fitted
    Logistic Regression model.

    Returns a DataFrame sorted by importance (descending).
    """
    coefs = model.coef_[0]
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": np.abs(coefs)}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    return importance_df


def save_artifacts(
    best_lr,
    best_nb,
    models_dict,
    metrics_df,
    importance_df,
):
    """Persist all models and processed data."""
    # Best Logistic Regression
    joblib.dump(best_lr, MODELS_DIR / "logreg_best.pkl")
    print(f"[SAVED] {MODELS_DIR / 'logreg_best.pkl'}")

    # Best Naive Bayes
    joblib.dump(best_nb, MODELS_DIR / "naive_bayes_best.pkl")
    print(f"[SAVED] {MODELS_DIR / 'naive_bayes_best.pkl'}")

    # All models dict
    joblib.dump(models_dict, MODELS_DIR / "all_models.pkl")
    print(f"[SAVED] {MODELS_DIR / 'all_models.pkl'}")

    # Metrics DataFrame
    metrics_df.to_pickle(DATA_PROCESSED / "model_metrics.pkl")
    print(f"[SAVED] {DATA_PROCESSED / 'model_metrics.pkl'}")

    # Feature importance
    importance_df.to_pickle(DATA_PROCESSED / "feature_importance.pkl")
    print(f"[SAVED] {DATA_PROCESSED / 'feature_importance.pkl'}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Classification Analysis – Telco Customer Churn")
    print("=" * 65)

    # 1. Load data
    X_train, X_test, y_train, y_test, feature_names = load_data()

    # 2–3. Train & evaluate all models
    results, models_dict = train_all_models(X_train, X_test, y_train, y_test)

    # 4. Comparison table
    metrics_df = pd.DataFrame(results)
    print("\n" + "=" * 65)
    print("  Model Comparison")
    print("=" * 65)
    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # 5–6. Select best models
    best_lr, best_lr_name, best_nb, best_nb_name = select_best_models(results, models_dict)

    # 9–10. Feature importance from best LogReg
    importance_df = extract_feature_importance(best_lr, feature_names)
    print(f"\n── Top 10 Feature Importances ({best_lr_name}) ──────────────")
    print(importance_df.head(10).to_string(index=False))

    # 7–8, 10. Save all artefacts
    print("\n── Saving Artifacts ──────────────────────────────────────────")
    save_artifacts(best_lr, best_nb, models_dict, metrics_df, importance_df)

    print("\n[DONE] Classification analysis complete.\n")


if __name__ == "__main__":
    main()
