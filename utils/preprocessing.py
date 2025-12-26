"""
Preprocessing utilities for fraud detection project.
Final stable version (NaN + datetime + SMOTE safe).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')


# =====================================================
# Utility Functions
# =====================================================

def remove_datetime_columns(X: pd.DataFrame) -> pd.DataFrame:
    datetime_cols = X.select_dtypes(include=['datetime64[ns]']).columns
    if len(datetime_cols) > 0:
        print(f"Removing datetime columns: {list(datetime_cols)}")
        X = X.drop(columns=datetime_cols)
    return X


def clean_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    """
    1. Drop columns that are entirely NaN
    2. Fill remaining NaNs with median
    """
    # Drop all-NaN columns
    all_nan_cols = X.columns[X.isna().all()]
    if len(all_nan_cols) > 0:
        print(f"Dropping all-NaN columns: {list(all_nan_cols)}")
        X = X.drop(columns=all_nan_cols)

    # Fill remaining NaNs
    X = X.fillna(X.median(numeric_only=True))

    return X


# =====================================================
# Imbalance Handling
# =====================================================

class ImbalanceHandler:

    @staticmethod
    def apply_smote(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
        print("Applying SMOTE...")
        smote = SMOTE(random_state=random_state)
        X_res, y_res, *_ = smote.fit_resample(X, y)
        X_res = np.asarray(X_res)
        y_res = np.asarray(y_res).ravel()
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=getattr(y, "name", None))

    @staticmethod
    def apply_random_undersampling(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
        print("Applying Random Undersampling...")
        rus = RandomUnderSampler(random_state=random_state)
        X_res, y_res, *_ = rus.fit_resample(X, y)
        X_res = np.asarray(X_res)
        y_res = np.asarray(y_res).ravel()
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=getattr(y, "name", None))

    @staticmethod
    def apply_smote_tomek(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
        print("Applying SMOTE + Tomek...")
        smt = SMOTETomek(random_state=random_state)
        X_res, y_res, *_ = smt.fit_resample(X, y)
        X_res = np.asarray(X_res)
        y_res = np.asarray(y_res).ravel()
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=getattr(y, "name", None))


# =====================================================
# Scaling
# =====================================================

class DataScaler:

    @staticmethod
    def apply_standard_scaling(X_train, X_test=None):
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )

        return X_train_scaled, X_test_scaled, scaler

    @staticmethod
    def apply_minmax_scaling(X_train, X_test=None):
        scaler = MinMaxScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )

        return X_train_scaled, X_test_scaled, scaler


# =====================================================
# Feature Preparation
# =====================================================

def prepare_features_for_modeling(df: pd.DataFrame, target_col='class'):
    print("Preparing features...")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Remove identifier columns
    drop_cols = ['user_id', 'device_id', 'ip_address']
    drop_cols = [c for c in drop_cols if c in X.columns]
    X = X.drop(columns=drop_cols)

    # Encode categorical columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Replace infinities
    X = X.replace([np.inf, -np.inf], np.nan)

    return X, y


# =====================================================
# FULL PIPELINE
# =====================================================

def full_preprocessing_pipeline(
    df: pd.DataFrame,
    target_col='class',
    sampling_strategy='smote',
    scaling_method='standard'
) -> Dict[str, Any]:

    print("Starting preprocessing pipeline...")

    # Prepare features
    X, y = prepare_features_for_modeling(df, target_col)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Remove datetime columns
    X_train = remove_datetime_columns(X_train)
    X_test = remove_datetime_columns(X_test)

    # 🔥 CLEAN NaNs PROPERLY
    X_train = clean_missing_values(X_train)
    X_test = clean_missing_values(X_test)

    # Final safety check
    if X_train.isna().sum().sum() != 0:
        raise ValueError("NaN values still present after cleaning")

    # Handle imbalance
    if sampling_strategy == 'smote':
        X_train, y_train = ImbalanceHandler.apply_smote(X_train, y_train)
    elif sampling_strategy == 'undersample':
        X_train, y_train = ImbalanceHandler.apply_random_undersampling(X_train, y_train)
    elif sampling_strategy == 'smote_tomek':
        X_train, y_train = ImbalanceHandler.apply_smote_tomek(X_train, y_train)

    # Scaling
    if scaling_method == 'standard':
        X_train, X_test, scaler = DataScaler.apply_standard_scaling(X_train, X_test)
    elif scaling_method == 'minmax':
        X_train, X_test, scaler = DataScaler.apply_minmax_scaling(X_train, X_test)
    else:
        scaler = None

    print("✅ Preprocessing completed successfully!")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': list(X_train.columns)
    }
