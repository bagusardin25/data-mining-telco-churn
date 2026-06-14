"""
03_clustering_kmeans.py
K-Means Clustering Analysis for Telco Customer Churn

Performs:
- Elbow method to determine optimal k
- Silhouette score analysis
- K-Means clustering with optimal k
- Cluster profiling and visualization
- PCA 2D visualization
"""

import os
import pickle
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
MODELS_DIR = BASE_DIR / "models"

# Ensure output directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Plot settings ────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150


def load_data():
    """Load pre-processed clustering data and the cleaned DataFrame."""
    X_clustering = pd.read_pickle(DATA_PROCESSED / "X_clustering.pkl")
    df_cleaned = pd.read_pickle(DATA_PROCESSED / "df_cleaned.pkl")
    print(f"[INFO] X_clustering shape: {X_clustering.shape}")
    print(f"[INFO] df_cleaned shape : {df_cleaned.shape}")
    return X_clustering, df_cleaned


def elbow_method(X, k_range=range(2, 11)):
    """Compute and plot inertia for each k (Elbow Method)."""
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        print(f"  k={k:>2d}  |  Inertia = {km.inertia_:,.2f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), inertias, "bo-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax.set_ylabel("Inertia", fontsize=12)
    ax.set_title("Elbow Method – Inertia vs k", fontsize=14, fontweight="bold")
    ax.set_xticks(list(k_range))
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "clustering_elbow.png")
    plt.close(fig)
    print(f"[SAVED] {FIGURES_DIR / 'clustering_elbow.png'}")
    return inertias


def silhouette_analysis(X, k_range=range(2, 11)):
    """Compute and plot silhouette scores for each k."""
    scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)
        print(f"  k={k:>2d}  |  Silhouette = {score:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), scores, "rs-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax.set_ylabel("Silhouette Score", fontsize=12)
    ax.set_title("Silhouette Score vs k", fontsize=14, fontweight="bold")
    ax.set_xticks(list(k_range))
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "clustering_silhouette.png")
    plt.close(fig)
    print(f"[SAVED] {FIGURES_DIR / 'clustering_silhouette.png'}")
    return scores


def fit_kmeans(X, optimal_k=3):
    """Fit final K-Means model with the chosen k."""
    print(f"\n[INFO] Fitting final K-Means with k={optimal_k} ...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    print(f"[INFO] Cluster sizes:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cl, cnt in zip(unique, counts):
        print(f"  Cluster {cl}: {cnt:,} samples ({cnt / len(cluster_labels) * 100:.1f}%)")
    return kmeans, cluster_labels


def profile_clusters(df, key_features=None):
    """
    Print and visualise mean feature values per cluster.

    Parameters
    ----------
    df : DataFrame with a 'Cluster' column already attached.
    key_features : list of feature names to profile (defaults provided).
    """
    if key_features is None:
        key_features = ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]

    profile = df.groupby("Cluster")[key_features].mean()
    print("\n── Cluster Profiles (mean values) ────────────────────────────")
    print(profile.to_string())
    print("──────────────────────────────────────────────────────────────\n")

    # ── Bar chart of profiles for key numeric features ────────────────────
    plot_features = [f for f in key_features if f != "Churn"]
    profile_plot = df.groupby("Cluster")[plot_features].mean()

    fig, axes = plt.subplots(1, len(plot_features), figsize=(5 * len(plot_features), 5))
    if len(plot_features) == 1:
        axes = [axes]
    palette = sns.color_palette("Set2", n_colors=profile_plot.index.nunique())
    for ax, feat in zip(axes, plot_features):
        profile_plot[feat].plot(kind="bar", ax=ax, color=palette, edgecolor="black")
        ax.set_title(feat, fontsize=13, fontweight="bold")
        ax.set_xlabel("Cluster", fontsize=11)
        ax.set_ylabel("Mean Value", fontsize=11)
        ax.set_xticklabels(profile_plot.index, rotation=0)
    plt.suptitle("Cluster Profiles – Key Features", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "clustering_profiles.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {FIGURES_DIR / 'clustering_profiles.png'}")

    # ── Churn rate per cluster ────────────────────────────────────────────
    churn_rate = df.groupby("Cluster")["Churn"].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        churn_rate.index.astype(str),
        churn_rate.values,
        color=sns.color_palette("Set2", n_colors=len(churn_rate)),
        edgecolor="black",
    )
    for bar, val in zip(bars, churn_rate.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Churn Rate (%)", fontsize=12)
    ax.set_title("Churn Rate per Cluster", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "clustering_churn_rate.png")
    plt.close(fig)
    print(f"[SAVED] {FIGURES_DIR / 'clustering_churn_rate.png'}")

    return profile


def pca_visualization(X, cluster_labels):
    """Reduce to 2D with PCA and scatter‑plot the clusters."""
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_
    print(
        f"[INFO] PCA explained variance: PC1={explained[0]:.2%}, PC2={explained[1]:.2%}, "
        f"Total={sum(explained):.2%}"
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=cluster_labels,
        cmap="Set2",
        alpha=0.6,
        s=15,
        edgecolors="none",
    )
    legend = ax.legend(
        *scatter.legend_elements(),
        title="Cluster",
        loc="best",
        fontsize=10,
    )
    ax.add_artist(legend)
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} variance)", fontsize=12)
    ax.set_title("K-Means Clusters – PCA 2D Projection", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "clustering_pca.png")
    plt.close(fig)
    print(f"[SAVED] {FIGURES_DIR / 'clustering_pca.png'}")


def save_artifacts(kmeans_model, cluster_labels, df_with_clusters):
    """Persist model and processed data."""
    # K-Means model
    joblib.dump(kmeans_model, MODELS_DIR / "kmeans.pkl")
    print(f"[SAVED] {MODELS_DIR / 'kmeans.pkl'}")

    # Cluster labels (numpy array)
    with open(DATA_PROCESSED / "cluster_labels.pkl", "wb") as f:
        pickle.dump(cluster_labels, f)
    print(f"[SAVED] {DATA_PROCESSED / 'cluster_labels.pkl'}")

    # DataFrame with Cluster column
    df_with_clusters.to_pickle(DATA_PROCESSED / "df_with_clusters.pkl")
    print(f"[SAVED] {DATA_PROCESSED / 'df_with_clusters.pkl'}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main(optimal_k: int = 3):
    print("=" * 65)
    print("  K-Means Clustering Analysis – Telco Customer Churn")
    print("=" * 65)

    # 1. Load data
    X_clustering, df_cleaned = load_data()

    # 2. Elbow method
    print("\n── Elbow Method ──────────────────────────────────────────────")
    elbow_method(X_clustering)

    # 3. Silhouette analysis
    print("\n── Silhouette Analysis ───────────────────────────────────────")
    silhouette_analysis(X_clustering)

    # 4 & 5. Fit final model
    kmeans, cluster_labels = fit_kmeans(X_clustering, optimal_k=optimal_k)

    # 6. Attach cluster labels to df_cleaned
    df_with_clusters = df_cleaned.copy()
    df_with_clusters["Cluster"] = cluster_labels

    # 7. Profile clusters
    print("\n── Cluster Profiling ─────────────────────────────────────────")
    profile_clusters(df_with_clusters)

    # 8. PCA visualization
    print("\n── PCA Visualization ─────────────────────────────────────────")
    pca_visualization(X_clustering, cluster_labels)

    # 9. Save artifacts
    print("\n── Saving Artifacts ──────────────────────────────────────────")
    save_artifacts(kmeans, cluster_labels, df_with_clusters)

    print("\n[DONE] Clustering analysis complete.\n")


if __name__ == "__main__":
    main(optimal_k=3)
