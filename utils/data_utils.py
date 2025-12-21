"""
Data utility functions for fraud detection project.
Contains functions for data loading, cleaning, and preprocessing.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Class for loading and initial processing of fraud detection datasets."""
    
    def __init__(self, data_path: str = "data/"):
        self.data_path = data_path
    
    def load_fraud_data(self) -> pd.DataFrame:
        """Load the main fraud detection dataset."""
        try:
            df = pd.read_csv(f"{self.data_path}Fraud_Data.csv")
            print(f"Fraud data loaded: {df.shape}")
            return df
        except FileNotFoundError:
            print("Fraud_Data.csv not found in data directory")
            return pd.DataFrame()
    
    def load_ip_country_data(self) -> pd.DataFrame:
        """Load IP to country mapping dataset."""
        try:
            df = pd.read_csv(f"{self.data_path}IpAddress_to_Country.csv")
            print(f"IP-Country data loaded: {df.shape}")
            return df
        except FileNotFoundError:
            print("IpAddress_to_Country.csv not found in data directory")
            return pd.DataFrame()
    
    def load_creditcard_data(self) -> pd.DataFrame:
        """Load credit card fraud dataset."""
        try:
            df = pd.read_csv(f"{self.data_path}creditcard.csv")
            print(f"Credit card data loaded: {df.shape}")
            return df
        except FileNotFoundError:
            print("creditcard.csv not found in data directory")
            return pd.DataFrame()

class DataCleaner:
    """Class for data cleaning operations."""
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input dataframe
            strategy: 'drop', 'mean', 'median', 'mode', or 'forward_fill'
        """
        print(f"Missing values before cleaning:\n{df.isnull().sum()}")
        
        if strategy == 'drop':
            df_clean = df.dropna()
        elif strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_clean = df.copy()
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        elif strategy == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_clean = df.copy()
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        elif strategy == 'mode':
            df_clean = df.fillna(df.mode().iloc[0])
        elif strategy == 'forward_fill':
            df_clean = df.ffill()
        else:
            df_clean = df.copy()
        
        print(f"Missing values after cleaning:\n{df_clean.isnull().sum()}")
        return df_clean
    
    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows from the dataset."""
        initial_shape = df.shape
        df_clean = df.drop_duplicates()
        final_shape = df_clean.shape
        
        print(f"Duplicates removed: {initial_shape[0] - final_shape[0]} rows")
        print(f"Shape before: {initial_shape}, Shape after: {final_shape}")
        
        return df_clean
    
    @staticmethod
    def correct_data_types(df: pd.DataFrame) -> pd.DataFrame:
        """Correct data types for common columns."""
        df_clean = df.copy()
        
        # Convert timestamp columns if they exist
        timestamp_cols = ['signup_time', 'purchase_time']
        for col in timestamp_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        
        # Convert categorical columns
        categorical_cols = ['source', 'browser', 'sex']
        for col in categorical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype('category')
        
        # Ensure target variable is integer
        if 'class' in df_clean.columns:
            df_clean['class'] = df_clean['class'].astype(int)
        if 'Class' in df_clean.columns:
            df_clean['Class'] = df_clean['Class'].astype(int)
        
        return df_clean

def ip_to_int(ip_address: str) -> int:
    """Convert IP address string to integer."""
    try:
        parts = ip_address.split('.')
        return (int(parts[0]) << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])
    except:
        return 0

def merge_with_geolocation(fraud_df: pd.DataFrame, ip_country_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge fraud data with IP-to-country mapping for geolocation analysis.
    
    Args:
        fraud_df: Main fraud dataset
        ip_country_df: IP to country mapping dataset
    
    Returns:
        Merged dataframe with country information
    """
    # Convert IP addresses to integers
    fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_int)
    
    # Sort IP country data for efficient merging
    ip_country_df = ip_country_df.sort_values('lower_bound_ip_address')
    
    # Merge using pandas merge_asof for range-based joining
    merged_df = pd.merge_asof(
        fraud_df.sort_values('ip_int'),
        ip_country_df[['lower_bound_ip_address', 'upper_bound_ip_address', 'country']],
        left_on='ip_int',
        right_on='lower_bound_ip_address',
        direction='backward'
    )
    
    # Filter out rows where IP is not within the range
    merged_df = merged_df[
        (merged_df['ip_int'] >= merged_df['lower_bound_ip_address']) &
        (merged_df['ip_int'] <= merged_df['upper_bound_ip_address'])
    ]
    
    # Clean up temporary columns
    merged_df = merged_df.drop(['ip_int', 'lower_bound_ip_address', 'upper_bound_ip_address'], axis=1)
    
    print(f"Merged dataset shape: {merged_df.shape}")
    print(f"Countries found: {merged_df['country'].nunique()}")
    
    return merged_df