"""
Feature engineering utilities for fraud detection project.
Contains functions for creating time-based features, transaction patterns, and other derived features.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Class for feature engineering operations."""
    
    @staticmethod
    def create_time_features(df: pd.DataFrame, timestamp_col: str = 'purchase_time') -> pd.DataFrame:
        """
        Create time-based features from timestamp column.
        
        Args:
            df: Input dataframe
            timestamp_col: Name of the timestamp column
        
        Returns:
            Dataframe with additional time features
        """
        df_features = df.copy()
        
        if timestamp_col in df_features.columns:
            # Ensure timestamp is datetime
            df_features[timestamp_col] = pd.to_datetime(df_features[timestamp_col])
            
            # Extract time components
            df_features['hour_of_day'] = df_features[timestamp_col].dt.hour
            df_features['day_of_week'] = df_features[timestamp_col].dt.dayofweek
            df_features['day_of_month'] = df_features[timestamp_col].dt.day
            df_features['month'] = df_features[timestamp_col].dt.month
            df_features['year'] = df_features[timestamp_col].dt.year
            
            # Create time period categories
            df_features['time_period'] = pd.cut(
                df_features['hour_of_day'],
                bins=[0, 6, 12, 18, 24],
                labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                include_lowest=True
            )
            
            # Weekend indicator
            df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
            
            print(f"Time features created: hour_of_day, day_of_week, time_period, is_weekend")
        
        return df_features
    
    @staticmethod
    def calculate_time_since_signup(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate time difference between signup and purchase.
        
        Args:
            df: Input dataframe with signup_time and purchase_time columns
        
        Returns:
            Dataframe with time_since_signup feature
        """
        df_features = df.copy()
        
        if 'signup_time' in df_features.columns and 'purchase_time' in df_features.columns:
            # Ensure both are datetime
            df_features['signup_time'] = pd.to_datetime(df_features['signup_time'])
            df_features['purchase_time'] = pd.to_datetime(df_features['purchase_time'])
            
            # Calculate time difference in hours
            df_features['time_since_signup'] = (
                df_features['purchase_time'] - df_features['signup_time']
            ).dt.total_seconds() / 3600
            
            # Create categories for time since signup
            df_features['signup_category'] = pd.cut(
                df_features['time_since_signup'],
                bins=[0, 1, 24, 168, 720, float('inf')],
                labels=['<1hr', '1-24hr', '1-7days', '1-30days', '>30days'],
                include_lowest=True
            )
            
            print(f"Time since signup calculated (in hours)")
        
        return df_features
    
    @staticmethod
    def create_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create transaction-based features like frequency and velocity.
        
        Args:
            df: Input dataframe
        
        Returns:
            Dataframe with transaction features
        """
        df_features = df.copy()
        
        # User-based features
        if 'user_id' in df_features.columns:
            user_stats = df_features.groupby('user_id').agg({
                'purchase_value': ['count', 'sum', 'mean', 'std'],
                'purchase_time': ['min', 'max']
            }).reset_index()
            
            # Flatten column names
            user_stats.columns = ['user_id', 'user_transaction_count', 'user_total_spent',
                                'user_avg_transaction', 'user_transaction_std',
                                'user_first_transaction', 'user_last_transaction']
            
            # Calculate user activity span
            user_stats['user_activity_span'] = (
                user_stats['user_last_transaction'] - user_stats['user_first_transaction']
            ).dt.total_seconds() / 3600
            
            # Transaction velocity (transactions per hour)
            user_stats['user_transaction_velocity'] = (
                user_stats['user_transaction_count'] / (user_stats['user_activity_span'] + 1)
            )
            
            # Merge back to main dataframe
            df_features = df_features.merge(user_stats, on='user_id', how='left')
        
        # Device-based features
        if 'device_id' in df_features.columns:
            device_stats = df_features.groupby('device_id').agg({
                'user_id': 'nunique',
                'purchase_value': ['count', 'mean']
            }).reset_index()
            
            device_stats.columns = ['device_id', 'device_user_count', 
                                  'device_transaction_count', 'device_avg_transaction']
            
            df_features = df_features.merge(device_stats, on='device_id', how='left')
        
        # IP-based features
        if 'ip_address' in df_features.columns:
            ip_stats = df_features.groupby('ip_address').agg({
                'user_id': 'nunique',
                'purchase_value': ['count', 'mean']
            }).reset_index()
            
            ip_stats.columns = ['ip_address', 'ip_user_count',
                              'ip_transaction_count', 'ip_avg_transaction']
            
            df_features = df_features.merge(ip_stats, on='ip_address', how='left')
        
        print(f"Transaction features created for users, devices, and IP addresses")
        
        return df_features
    
    @staticmethod
    def create_purchase_value_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on purchase value patterns.
        
        Args:
            df: Input dataframe
        
        Returns:
            Dataframe with purchase value features
        """
        df_features = df.copy()
        
        if 'purchase_value' in df_features.columns:
            # Log transformation for skewed data
            df_features['purchase_value_log'] = np.log1p(df_features['purchase_value'])
            
            # Purchase value categories
            df_features['purchase_category'] = pd.cut(
                df_features['purchase_value'],
                bins=[0, 50, 200, 500, 1000, float('inf')],
                labels=['Low', 'Medium', 'High', 'Very_High', 'Extreme'],
                include_lowest=True
            )
            
            # Z-score for outlier detection
            df_features['purchase_value_zscore'] = (
                df_features['purchase_value'] - df_features['purchase_value'].mean()
            ) / df_features['purchase_value'].std()
            
            # Round number indicator (often suspicious)
            df_features['is_round_amount'] = (df_features['purchase_value'] % 10 == 0).astype(int)
            
            print(f"Purchase value features created")
        
        return df_features
    
    @staticmethod
    def encode_categorical_features(df: pd.DataFrame, categorical_cols: List[str] = None) -> pd.DataFrame:
        """
        Encode categorical features using appropriate methods.
        
        Args:
            df: Input dataframe
            categorical_cols: List of categorical columns to encode
        
        Returns:
            Dataframe with encoded categorical features
        """
        df_encoded = df.copy()
        
        if categorical_cols is None:
            categorical_cols = ['source', 'browser', 'sex', 'country', 'time_period', 
                             'signup_category', 'purchase_category']
        
        # Filter to existing columns
        categorical_cols = [col for col in categorical_cols if col in df_encoded.columns]
        
        for col in categorical_cols:
            # For high cardinality columns, use frequency encoding
            if df_encoded[col].nunique() > 10:
                freq_map = df_encoded[col].value_counts().to_dict()
                df_encoded[f'{col}_frequency'] = df_encoded[col].map(freq_map)
            else:
                # One-hot encoding for low cardinality
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
        
        print(f"Categorical features encoded: {categorical_cols}")
        
        return df_encoded

def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps to the dataset.
    
    Args:
        df: Input dataframe
    
    Returns:
        Dataframe with all engineered features
    """
    print("Starting comprehensive feature engineering...")
    
    feature_eng = FeatureEngineer()
    
    # Apply all feature engineering steps
    df_features = feature_eng.create_time_features(df)
    df_features = feature_eng.calculate_time_since_signup(df_features)
    df_features = feature_eng.create_transaction_features(df_features)
    df_features = feature_eng.create_purchase_value_features(df_features)
    df_features = feature_eng.encode_categorical_features(df_features)
    
    print(f"Feature engineering complete. Final shape: {df_features.shape}")
    
    return df_features
