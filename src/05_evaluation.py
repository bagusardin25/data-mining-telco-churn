"""
05_evaluation.py — Evaluasi & Interpretasi Model
=================================================
Memuat model terlatih dan data uji, menghasilkan visualisasi evaluasi
(confusion matrix, ROC curve, perbandingan metrik, feature importance),
serta mencetak ringkasan rekomendasi bisnis.
"""

import warnings
warnings.filterwarnings("ignore")

import pickle
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
FIG_DIR = BASE_DIR / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Styling ──────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.family": "sans-serif",
    "font.size": 10,
})

# ═════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════

def _load_pickle(path: Path):
    """Load a pickle (.pkl) file using pandas or plain pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_joblib(path: Path):
    """Load a joblib-serialized object."""
    return joblib.load(path)


# ═════════════════════════════════════════════════════════════════════════
#  1. Confusion Matrix Heatmaps (all models, side-by-side)
# ═════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(
    all_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Create a subplot grid with confusion matrices for every model."""
    model_names = list(all_models.keys())
    n = len(model_names)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, name in enumerate(model_names):
        model = all_models[name]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            ax=axes[idx],
            cbar=False,
        )
        axes[idx].set_title(name, fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = FIG_DIR / "eval_confusion_matrices.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Confusion matrices saved -> {save_path}")


# ═════════════════════════════════════════════════════════════════════════
#  2. ROC Curve Comparison
# ═════════════════════════════════════════════════════════════════════════

def plot_roc_curves(
    all_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Plot ROC curves for all models on a single figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette("husl", len(all_models))

    for (name, model), color in zip(all_models.items(), colors):
        # Try predict_proba first, fall back to decision_function
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            print(f"  ⚠ {name}: no probability output — skipped ROC.")
            continue

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    save_path = FIG_DIR / "eval_roc_curves.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] ROC curves saved -> {save_path}")


# ═════════════════════════════════════════════════════════════════════════
#  3. Model Comparison Bar Chart
# ═════════════════════════════════════════════════════════════════════════

def plot_model_comparison(metrics_df: pd.DataFrame) -> None:
    """Grouped bar chart comparing metrics across models."""
    metric_cols = [c for c in metrics_df.columns if c != "Model"]
    n_metrics = len(metric_cols)
    n_models = len(metrics_df)

    x = np.arange(n_metrics)
    width = 0.8 / n_models
    colors = sns.color_palette("Set2", n_models)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (_, row) in enumerate(metrics_df.iterrows()):
        values = [row[c] for c in metric_cols]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=row["Model"], color=colors[i])
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_cols, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_path = FIG_DIR / "eval_model_comparison.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Model comparison chart saved -> {save_path}")


# ═════════════════════════════════════════════════════════════════════════
#  4. Feature Importance (top 15)
# ═════════════════════════════════════════════════════════════════════════

def plot_feature_importance(fi_df: pd.DataFrame) -> None:
    """Horizontal bar chart of the top 15 most important features."""
    top = fi_df.sort_values("importance", ascending=False).head(15)
    top = top.sort_values("importance", ascending=True)  # for horizontal barh

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette("viridis", len(top))
    ax.barh(top["feature"], top["importance"], color=colors)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title("Top 15 Feature Importance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_path = FIG_DIR / "eval_feature_importance.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Feature importance chart saved -> {save_path}")


# ═════════════════════════════════════════════════════════════════════════
#  5. Comprehensive Summary
# ═════════════════════════════════════════════════════════════════════════

def print_summary(
    metrics_df: pd.DataFrame,
    fi_df: pd.DataFrame,
    all_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cluster_labels=None,
    df_with_clusters: pd.DataFrame = None,
) -> None:
    """Print a detailed evaluation summary with business recommendations."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("  COMPREHENSIVE EVALUATION SUMMARY")
    print(f"{sep}\n")

    # ── Best model recommendation ────────────────────────────────────
    # Column name from 04_classification.py is "F1-Score"
    f1_col = "F1-Score" if "F1-Score" in metrics_df.columns else "F1"
    best_row = metrics_df.loc[metrics_df[f1_col].idxmax()]
    best_name = best_row["Model"]
    print("▸ BEST MODEL RECOMMENDATION")
    print(f"  Model  : {best_name}")
    for col in [c for c in metrics_df.columns if c != "Model"]:
        print(f"  {col:12s}: {best_row[col]:.4f}")
    print()

    # ── Classification report for best model ─────────────────────────
    if best_name in all_models:
        y_pred = all_models[best_name].predict(X_test)
        print("▸ CLASSIFICATION REPORT (Best Model)")
        print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # ── Top churn factors ────────────────────────────────────────────
    top5 = fi_df.sort_values("importance", ascending=False).head(5)
    print("▸ TOP 5 CHURN FACTORS")
    for i, (_, r) in enumerate(top5.iterrows(), 1):
        print(f"  {i}. {r['feature']:30s} (importance: {r['importance']:.4f})")
    print()

    # ── Clustering insights ──────────────────────────────────────────
    if df_with_clusters is not None and "Cluster" in df_with_clusters.columns:
        print("▸ CUSTOMER SEGMENTATION INSIGHTS (K-Means)")
        for cl in sorted(df_with_clusters["Cluster"].unique()):
            segment = df_with_clusters[df_with_clusters["Cluster"] == cl]
            churn_col = "Churn" if "Churn" in segment.columns else None
            churn_rate = segment[churn_col].mean() * 100 if churn_col else float("nan")
            print(f"  Cluster {cl}: {len(segment):>5d} customers | Churn rate: {churn_rate:.1f}%")
        print()

    # ── Business recommendations ─────────────────────────────────────
    print("▸ BUSINESS RECOMMENDATIONS")
    recommendations = [
        "1. Prioritas retensi pada pelanggan kontrak Month-to-month — kelompok ini "
        "memiliki risiko churn tertinggi. Tawarkan insentif perpanjangan kontrak.",
        "2. Tingkatkan bundling layanan (OnlineSecurity, TechSupport, OnlineBackup) "
        "untuk meningkatkan stickiness pelanggan.",
        "3. Evaluasi strategi harga untuk pelanggan dengan MonthlyCharges tinggi — "
        "pertimbangkan diskon loyalitas atau paket hemat.",
        "4. Fokuskan program onboarding di 12 bulan pertama (tenure rendah = risiko "
        "churn tinggi).",
        "5. Gunakan model prediksi churn untuk scoring proaktif dan trigger "
        "kampanye retensi otomatis.",
    ]
    for rec in recommendations:
        print(f"  {rec}")

    print(f"\n{sep}")
    print("  Evaluation complete. All figures saved to reports/figures/")
    print(f"{sep}\n")


# ═════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Loading data & models …")

    # ── Load data ────────────────────────────────────────────────────
    X_test = _load_pickle(DATA_DIR / "X_test.pkl")
    y_test = _load_pickle(DATA_DIR / "y_test.pkl")
    metrics_df = _load_pickle(DATA_DIR / "model_metrics.pkl")
    fi_df = _load_pickle(DATA_DIR / "feature_importance.pkl")

    # Optional files (may not yet exist)
    try:
        cluster_labels = _load_pickle(DATA_DIR / "cluster_labels.pkl")
    except FileNotFoundError:
        cluster_labels = None

    try:
        df_with_clusters = _load_pickle(DATA_DIR / "df_with_clusters.pkl")
    except FileNotFoundError:
        df_with_clusters = None

    # ── Load models ──────────────────────────────────────────────────
    all_models = _load_joblib(MODEL_DIR / "all_models.pkl")

    print(f"  Models loaded: {list(all_models.keys())}")
    print(f"  Test set size: {len(X_test)}")
    print()

    # ── Generate plots ───────────────────────────────────────────────
    plot_confusion_matrices(all_models, X_test, y_test)
    plot_roc_curves(all_models, X_test, y_test)
    plot_model_comparison(metrics_df)
    plot_feature_importance(fi_df)

    # ── Print summary ────────────────────────────────────────────────
    print_summary(
        metrics_df=metrics_df,
        fi_df=fi_df,
        all_models=all_models,
        X_test=X_test,
        y_test=y_test,
        cluster_labels=cluster_labels,
        df_with_clusters=df_with_clusters,
    )


if __name__ == "__main__":
    main()
