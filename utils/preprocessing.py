"""
Preprocessing utilities for fraud detection project.
Contains functions for handling class imbalance, scaling, and data transformation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class ImbalanceHandler:
    """Class for handling class imbalance in fraud detection datasets."""
    
    @staticmethod
    def analyze_class_distribution(y: pd.Series) -> Dict[str, Any]:
        """
        Analyze the class distribution in the target variable.
        
        Args:
            y: Target variable series
        
        Returns:
            Dictionary with class distribution statistics
        """
        class_counts = y.value_counts()
        class_props = y.value_counts(normalize=True)
        
        stats = {
            'class_counts': class_counts.to_dict(),
            'class_proportions': class_props.to_dict(),
            'imbalance_ratio': class_counts.max() / class_counts.min(),
            'minority_class_percentage': class_props.min() * 100
        }
        
        print("Class Distribution Analysis:")
        print(f"Class counts: {stats['class_counts']}")
        print(f"Class proportions: {stats['class_proportions']}")
        print(f"Imbalance ratio: {stats['imbalance_ratio']:.2f}")
        print(f"Minority class percentage: {stats['minority_class_percentage']:.2f}%")
        
        return stats
    
    @staticmethod
    def apply_smote(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE (Synthetic Minority Oversampling Technique) to balance classes.
        
        Args:
            X: Feature matrix
            y: Target variable
            random_state: Random state for reproducibility
        
        Returns:
            Balanced feature matrix and target variable
        """
        print("Applying SMOTE oversampling...")
        
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print(f"Original shape: {X.shape}")
        print(f"Resampled shape: {X_resampled.shape}")
        
        # Analyze new distribution
        ImbalanceHandler.analyze_class_distribution(pd.Series(y_resampled))
        
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
    
    @staticmethod
    def apply_random_undersampling(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply random undersampling to balance classes.
        
        Args:
            X: Feature matrix
            y: Target variable
            random_state: Random state for reproducibility
        
        Returns:
            Balanced feature matrix and target variable
        """
        print("Applying Random Undersampling...")
        
        rus = RandomUnderSampler(random_state=random_state)
        X_resampled, y_resampled = rus.fit_resample(X, y)
        
        print(f"Original shape: {X.shape}")
        print(f"Resampled shape: {X_resampled.shape}")
        
        # Analyze new distribution
        ImbalanceHandler.analyze_class_distribution(pd.Series(y_resampled))
        
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
    
    @staticmethod
    def apply_smote_tomek(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE + Tomek links for balanced sampling.
        
        Args:
            X: Feature matrix
            y: Target variable
            random_state: Random state for reproducibility
        
        Returns:
            Balanced feature matrix and target variable
        """
        print("Applying SMOTE + Tomek links...")
        
        smote_tomek = SMOTETomek(random_state=random_state)
        X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
        
        print(f"Original shape: {X.shape}")
        print(f"Resampled shape: {X_resampled.shape}")
        
        # Analyze new distribution
        ImbalanceHandler.analyze_class_distribution(pd.Series(y_resampled))
        
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)

class DataScaler:
    """Class for data scaling and normalization."""
    
    @staticmethod
    def apply_standard_scaling(X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
        """
        Apply standard scaling (z-score normalization) to features.
        
        Args:
            X_train: Training feature matrix
            X_test: Test feature matrix (optional)
        
        Returns:
            Scaled training and test sets, and the fitted scaler
        """
        print("Applying Standard Scaling...")
        
        scaler = StandardScaler()
        
        # Fit on training data only
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
        
        print(f"Features scaled: {len(X_train.columns)}")
        
        return X_train_scaled, X_test_scaled, scaler
    
    @staticmethod
    def apply_minmax_scaling(X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
        """
        Apply MinMax scaling to features.
        
        Args:
            X_train: Training feature matrix
            X_test: Test feature matrix (optional)
        
        Returns:
            Scaled training and test sets, and the fitted scaler
        """
        print("Applying MinMax Scaling...")
        
        scaler = MinMaxScaler()
        
        # Fit on training data only
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
        
        print(f"Features scaled: {len(X_train.columns)}")
        
        return X_train_scaled, X_test_scaled, scaler

class DataSplitter:
    """Class for splitting data into training and testing sets."""
    
    @staticmethod
    def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
                   random_state: int = 42, stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility
            stratify: Whether to stratify the split based on target variable
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        print(f"Splitting data: {100*(1-test_size):.0f}% train, {100*test_size:.0f}% test")
        
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=stratify_param
        )
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        # Analyze class distribution in splits
        print("\nTraining set class distribution:")
        ImbalanceHandler.analyze_class_distribution(y_train)
        
        print("\nTest set class distribution:")
        ImbalanceHandler.analyze_class_distribution(y_test)
        
        return X_train, X_test, y_train, y_test

def prepare_features_for_modeling(df: pd.DataFrame, target_col: str = 'class') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features for machine learning modeling.
    
    Args:
        df: Input dataframe with all features
        target_col: Name of the target column
    
    Returns:
        Feature matrix (X) and target variable (y)
    """
    print("Preparing features for modeling...")
    
    # Separate features and target
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Remove non-numeric columns that shouldn't be used for modeling
    exclude_cols = ['user_id', 'device_id', 'ip_address', 'signup_time', 'purchase_time']
    exclude_cols = [col for col in exclude_cols if col in X.columns]
    
    if exclude_cols:
        print(f"Excluding columns: {exclude_cols}")
        X = X.drop(columns=exclude_cols)
    
    # Handle any remaining categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print(f"Converting categorical columns to numeric: {list(categorical_cols)}")
        le = LabelEncoder()
        for col in categorical_cols:
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Fill any remaining NaN values
    X = X.fillna(X.median())
    
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    
    return X, y

def full_preprocessing_pipeline(df: pd.DataFrame, target_col: str = 'class', 
                              sampling_strategy: str = 'smote', 
                              scaling_method: str = 'standard') -> Dict[str, Any]:
    """
    Complete preprocessing pipeline for fraud detection data.
    
    Args:
        df: Input dataframe
        target_col: Name of the target column
        sampling_strategy: 'smote', 'undersample', or 'smote_tomek'
        scaling_method: 'standard' or 'minmax'
    
    Returns:
        Dictionary containing processed datasets and fitted transformers
    """
    print("Starting full preprocessing pipeline...")
    
    # Prepare features
    X, y = prepare_features_for_modeling(df, target_col)
    
    # Split data
    splitter = DataSplitter()
    X_train, X_test, y_train, y_test = splitter.split_data(X, y)
    
    # Handle class imbalance (only on training data)
    imbalance_handler = ImbalanceHandler()
    
    if sampling_strategy == 'smote':
        X_train_balanced, y_train_balanced = imbalance_handler.apply_smote(X_train, y_train)
    elif sampling_strategy == 'undersample':
        X_train_balanced, y_train_balanced = imbalance_handler.apply_random_undersampling(X_train, y_train)
    elif sampling_strategy == 'smote_tomek':
        X_train_balanced, y_train_balanced = imbalance_handler.apply_smote_tomek(X_train, y_train)
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # Scale features
    scaler_class = DataScaler()
    
    if scaling_method == 'standard':
        X_train_scaled, X_test_scaled, scaler = scaler_class.apply_standard_scaling(X_train_balanced, X_test)
    elif scaling_method == 'minmax':
        X_train_scaled, X_test_scaled, scaler = scaler_class.apply_minmax_scaling(X_train_balanced, X_test)
    else:
        X_train_scaled, X_test_scaled, scaler = X_train_balanced, X_test, None
    
    print("Preprocessing pipeline completed!")
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train_balanced,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': list(X.columns),
        'original_shapes': {
            'X_train': X_train.shape,
            'X_test': X_test.shape,
            'X_train_balanced': X_train_balanced.shape
        }
    }
