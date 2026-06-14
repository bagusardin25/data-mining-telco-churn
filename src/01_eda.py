"""
01_eda.py - Exploratory Data Analysis for Telco Customer Churn Dataset
======================================================================
Generates summary statistics and saves visualizations to reports/figures/.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw',
                             'WA_Fn-UseC_-Telco-Customer-Churn.csv')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures')
DPI = 150
sns.set_style('whitegrid')
PALETTE = sns.color_palette('Set2')


def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV and perform basic type fixes."""
    df = pd.read_csv(path)
    # TotalCharges contains blank strings → convert to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def plot_churn_distribution(df: pd.DataFrame, out_dir: str) -> None:
    """Bar chart of Churn counts with percentage annotations."""
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df['Churn'].value_counts()
    total = len(df)
    bars = ax.bar(counts.index, counts.values, color=[PALETTE[0], PALETTE[1]],
                  edgecolor='black', linewidth=0.5)
    for bar, count in zip(bars, counts.values):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11,
                fontweight='bold')
    ax.set_title('Churn Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Churn', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'eda_churn_distribution.png'), dpi=DPI)
    plt.close(fig)
    print('[OK] Saved eda_churn_distribution.png')


def plot_numeric_distributions(df: pd.DataFrame, out_dir: str) -> None:
    """Histograms for tenure, MonthlyCharges, TotalCharges in a 2×2 grid."""
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = axes.flatten()
    for idx, col in enumerate(numeric_cols):
        ax = axes_flat[idx]
        ax.hist(df[col].dropna(), bins=30, color=PALETTE[idx], edgecolor='black',
                linewidth=0.5, alpha=0.85)
        ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
    # Hide the unused 4th subplot
    axes_flat[3].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'eda_numeric_distributions.png'), dpi=DPI)
    plt.close(fig)
    print('[OK] Saved eda_numeric_distributions.png')


def plot_boxplots(df: pd.DataFrame, out_dir: str) -> None:
    """Boxplots of numeric features grouped by Churn."""
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, col in enumerate(numeric_cols):
        sns.boxplot(x='Churn', y=col, data=df, ax=axes[idx],
                    palette=[PALETTE[0], PALETTE[1]], width=0.5)
        axes[idx].set_title(f'{col} by Churn', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'eda_boxplots.png'), dpi=DPI)
    plt.close(fig)
    print('[OK] Saved eda_boxplots.png')


def plot_churn_by_category(df: pd.DataFrame, col: str, filename: str,
                           out_dir: str, figsize=(8, 5)) -> None:
    """Generic churn-rate bar chart for a categorical column."""
    churn_rate = df.groupby(col)['Churn'].apply(
        lambda x: (x == 'Yes').mean() * 100
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(churn_rate.index, churn_rate.values, color=PALETTE[:len(churn_rate)],
                  edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, churn_rate.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10,
                fontweight='bold')
    ax.set_title(f'Churn Rate by {col}', fontsize=14, fontweight='bold')
    ax.set_xlabel(col, fontsize=12)
    ax.set_ylabel('Churn Rate (%)', fontsize=12)
    ax.set_ylim(0, churn_rate.max() + 10)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=DPI)
    plt.close(fig)
    print(f'[OK] Saved {filename}')


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    """Correlation heatmap of numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                linewidths=0.5, ax=ax, square=True)
    ax.set_title('Correlation Heatmap (Numeric Features)', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'eda_correlation_heatmap.png'), dpi=DPI)
    plt.close(fig)
    print('[OK] Saved eda_correlation_heatmap.png')


def plot_churn_by_tenure_group(df: pd.DataFrame, out_dir: str) -> None:
    """Churn rate grouped by tenure bins."""
    bins = [0, 12, 24, 48, 60, np.inf]
    labels = ['0-12', '12-24', '24-48', '48-60', '60+']
    df_copy = df.copy()
    df_copy['tenure_group'] = pd.cut(df_copy['tenure'], bins=bins, labels=labels,
                                     right=True)
    churn_rate = df_copy.groupby('tenure_group', observed=False)['Churn'].apply(
        lambda x: (x == 'Yes').mean() * 100
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(churn_rate.index.astype(str), churn_rate.values,
                  color=PALETTE[:len(churn_rate)], edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, churn_rate.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10,
                fontweight='bold')
    ax.set_title('Churn Rate by Tenure Group', fontsize=14, fontweight='bold')
    ax.set_xlabel('Tenure (months)', fontsize=12)
    ax.set_ylabel('Churn Rate (%)', fontsize=12)
    ax.set_ylim(0, churn_rate.max() + 10)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'eda_churn_by_tenure_group.png'), dpi=DPI)
    plt.close(fig)
    print('[OK] Saved eda_churn_by_tenure_group.png')


def plot_churn_by_services(df: pd.DataFrame, out_dir: str) -> None:
    """Churn rate by various service features."""
    service_cols = [
        'OnlineSecurity', 'TechSupport', 'OnlineBackup',
        'DeviceProtection', 'StreamingTV', 'StreamingMovies',
    ]
    # For each service, compute churn rate when the service is "Yes" vs "No"
    records = []
    for col in service_cols:
        for val in ['Yes', 'No']:
            subset = df[df[col] == val]
            if len(subset) == 0:
                continue
            rate = (subset['Churn'] == 'Yes').mean() * 100
            records.append({'Service': col, 'Subscribed': val, 'Churn Rate (%)': rate})
    svc_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Service', y='Churn Rate (%)', hue='Subscribed', data=svc_df,
                palette=[PALETTE[1], PALETTE[0]], edgecolor='black', linewidth=0.5,
                ax=ax)
    # Annotate bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', fontsize=8, fontweight='bold',
                     padding=2)
    ax.set_title('Churn Rate by Service Subscription', fontsize=14,
                 fontweight='bold')
    ax.set_xlabel('Service Feature', fontsize=12)
    ax.set_ylabel('Churn Rate (%)', fontsize=12)
    ax.legend(title='Subscribed')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'eda_churn_by_services.png'), dpi=DPI)
    plt.close(fig)
    print('[OK] Saved eda_churn_by_services.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Resolve paths
    raw_path = os.path.abspath(RAW_DATA_PATH)
    fig_dir = os.path.abspath(FIGURES_DIR)
    os.makedirs(fig_dir, exist_ok=True)

    print('=' * 60)
    print('  Telco Customer Churn — Exploratory Data Analysis')
    print('=' * 60)

    # 1. Load data
    df = load_data(raw_path)
    print(f'\nDataset shape: {df.shape}')

    # 2. Info
    print('\n--- Dataset Info ---')
    df.info()

    # 3. Describe
    print('\n--- Descriptive Statistics (Numeric) ---')
    print(df.describe().to_string())

    # 4. Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'Missing': missing, 'Pct (%)': missing_pct})
    missing_df = missing_df[missing_df['Missing'] > 0]
    print('\n--- Missing Values ---')
    if missing_df.empty:
        print('No missing values (after type conversion, check TotalCharges).')
    else:
        print(missing_df.to_string())

    # 5. Duplicates
    dup_count = df.duplicated().sum()
    print(f'\nDuplicate rows: {dup_count}')

    # 6. Churn distribution summary
    print('\n--- Churn Distribution ---')
    churn_counts = df['Churn'].value_counts()
    churn_pcts = df['Churn'].value_counts(normalize=True) * 100
    for label in churn_counts.index:
        print(f'  {label}: {churn_counts[label]} ({churn_pcts[label]:.2f}%)')

    # 7. Generate all plots
    print('\n--- Generating Visualizations ---')
    plot_churn_distribution(df, fig_dir)
    plot_numeric_distributions(df, fig_dir)
    plot_boxplots(df, fig_dir)
    plot_churn_by_category(df, 'Contract', 'eda_churn_by_contract.png', fig_dir)
    plot_churn_by_category(df, 'InternetService', 'eda_churn_by_internet.png', fig_dir)
    plot_churn_by_category(df, 'PaymentMethod', 'eda_churn_by_payment.png', fig_dir,
                           figsize=(10, 5))
    plot_correlation_heatmap(df, fig_dir)
    plot_churn_by_tenure_group(df, fig_dir)
    plot_churn_by_services(df, fig_dir)

    print('\n' + '=' * 60)
    print('  EDA complete — all figures saved to reports/figures/')
    print('=' * 60)


if __name__ == '__main__':
    main()
