"""
02_preprocessing.py - Data Preprocessing for Telco Customer Churn Dataset
=========================================================================
Cleans, encodes, scales, splits, and saves processed data artifacts.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw',
                             'WA_Fn-UseC_-Telco-Customer-Churn.csv')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Columns to label-encode as binary 0/1
BINARY_MAP_YES_NO = [
    'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn',
]

# Columns to one-hot encode
OHE_COLUMNS = ['InternetService', 'Contract', 'PaymentMethod']

# Numeric columns to scale
NUMERIC_SCALE_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV, fix types, handle missing values, drop identifiers."""
    df = pd.read_csv(path)

    # 1. Convert TotalCharges to numeric (blank strings → NaN)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # 2. Fill missing TotalCharges with median
    median_total = df['TotalCharges'].median()
    df['TotalCharges'].fillna(median_total, inplace=True)
    print(f'[INFO] TotalCharges NaN filled with median = {median_total:.2f}')

    # 3. Drop customerID
    df.drop(columns=['customerID'], inplace=True)
    print('[INFO] Dropped customerID column')

    return df


def replace_no_service(df: pd.DataFrame) -> pd.DataFrame:
    """Replace 'No internet service' and 'No phone service' with 'No'."""
    replace_cols = [
        'MultipleLines',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
    ]
    for col in replace_cols:
        df[col] = df[col].replace({
            'No internet service': 'No',
            'No phone service': 'No',
        })
    print('[INFO] Replaced "No internet service" / "No phone service" with "No"')
    return df


def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode binary columns to 0/1."""
    # gender: Female=0, Male=1
    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1}).astype(int)

    # Yes/No columns → No=0, Yes=1
    for col in BINARY_MAP_YES_NO:
        df[col] = df[col].map({'No': 0, 'Yes': 1}).astype(int)

    # SeniorCitizen is already 0/1 — keep as-is
    print('[INFO] Binary columns label-encoded (0/1)')
    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode multi-category columns with drop_first=True."""
    df = pd.get_dummies(df, columns=OHE_COLUMNS, drop_first=True, dtype=int)
    print(f'[INFO] One-hot encoded: {OHE_COLUMNS}')
    return df


def split_and_scale(df: pd.DataFrame):
    """
    Separate features/target, train-test split, scale numeric columns.
    Returns X_train, X_test, y_train, y_test, X_clustering, scaler, feature_names.
    """
    # Separate features and target
    y = df['Churn']
    X = df.drop(columns=['Churn'])
    feature_names = list(X.columns)

    # Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f'[INFO] Train/test split: {X_train.shape[0]} / {X_test.shape[0]}  '
          f'(test_size={TEST_SIZE}, random_state={RANDOM_STATE})')

    # StandardScaler on numeric columns only
    scaler = StandardScaler()
    X_train[NUMERIC_SCALE_COLS] = scaler.fit_transform(X_train[NUMERIC_SCALE_COLS])
    X_test[NUMERIC_SCALE_COLS] = scaler.transform(X_test[NUMERIC_SCALE_COLS])

    # Full scaled X for clustering (all rows)
    X_all = X.copy()
    X_all[NUMERIC_SCALE_COLS] = scaler.transform(X_all[NUMERIC_SCALE_COLS])

    print('[INFO] StandardScaler fitted on train, applied to train/test/clustering')

    return X_train, X_test, y_train, y_test, X_all, scaler, feature_names


def save_artifacts(X_train, X_test, y_train, y_test, X_clustering,
                   df_cleaned, scaler, feature_names) -> None:
    """Persist all processed artifacts to disk."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    processed = os.path.abspath(PROCESSED_DIR)
    models = os.path.abspath(MODELS_DIR)

    # DataFrames / Series → pd.to_pickle
    X_train.to_pickle(os.path.join(processed, 'X_train.pkl'))
    X_test.to_pickle(os.path.join(processed, 'X_test.pkl'))
    y_train.to_pickle(os.path.join(processed, 'y_train.pkl'))
    y_test.to_pickle(os.path.join(processed, 'y_test.pkl'))
    X_clustering.to_pickle(os.path.join(processed, 'X_clustering.pkl'))
    df_cleaned.to_pickle(os.path.join(processed, 'df_cleaned.pkl'))

    # Feature names → pickle
    with open(os.path.join(processed, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)

    # Scaler → joblib (sklearn object)
    joblib.dump(scaler, os.path.join(models, 'scaler.pkl'))

    print(f'[INFO] Saved processed data to {processed}')
    print(f'[INFO] Saved scaler to {models}/scaler.pkl')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    raw_path = os.path.abspath(RAW_DATA_PATH)

    print('=' * 60)
    print('  Telco Customer Churn — Data Preprocessing')
    print('=' * 60)

    # Step 1-3: Load, type-fix, impute, drop ID
    df = load_and_clean(raw_path)

    # Step 4: Replace "No internet/phone service" → "No"
    df = replace_no_service(df)

    # Step 5-6: Binary encoding
    df = encode_binary(df)

    # Step 7: One-hot encoding
    df = one_hot_encode(df)

    # Keep a copy of the fully-encoded dataframe (before split)
    df_cleaned = df.copy()

    # Step 8-10: Split, scale, get clustering set
    X_train, X_test, y_train, y_test, X_clustering, scaler, feature_names = \
        split_and_scale(df)

    # Step 11: Save everything
    save_artifacts(X_train, X_test, y_train, y_test, X_clustering,
                   df_cleaned, scaler, feature_names)

    # Print summary
    print('\n--- Summary ---')
    print(f'df_cleaned shape  : {df_cleaned.shape}')
    print(f'X_train shape     : {X_train.shape}')
    print(f'X_test shape      : {X_test.shape}')
    print(f'y_train shape     : {y_train.shape}')
    print(f'y_test shape      : {y_test.shape}')
    print(f'X_clustering shape: {X_clustering.shape}')

    print('\n--- Class Distribution (y_train) ---')
    dist = y_train.value_counts()
    for label in dist.index:
        pct = dist[label] / len(y_train) * 100
        print(f'  {label}: {dist[label]} ({pct:.2f}%)')

    print('\n--- Feature Names ---')
    for i, name in enumerate(feature_names, 1):
        print(f'  {i:2d}. {name}')

    print('\n' + '=' * 60)
    print('  Preprocessing complete')
    print('=' * 60)


if __name__ == '__main__':
    main()
