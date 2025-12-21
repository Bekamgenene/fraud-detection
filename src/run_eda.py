"""
Standalone EDA script for fraud detection analysis.
Run this to generate comprehensive exploratory data analysis.
"""

import sys
import os
sys.path.append('../')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.visualization import EDAVisualizer

def run_comprehensive_eda(df, target_col='class'):
    """
    Run comprehensive EDA analysis for fraud detection dataset.
    """
    print("=" * 60)
    print("COMPREHENSIVE EXPLORATORY DATA ANALYSIS REPORT")
    print("=" * 60)
    
    visualizer = EDAVisualizer()
    
    # Basic dataset info
    print(f"\nDataset Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\nMissing Values:")
        print(missing_values[missing_values > 0])
    else:
        print("\nNo missing values found!")
    
    # Class distribution
    print(f"\nTarget Variable ({target_col}) Distribution:")
    print(df[target_col].value_counts())
    print(f"Fraud Rate: {df[target_col].mean():.4f}")
    
    # Plot class distribution
    visualizer.plot_class_distribution(df[target_col])
    
    # Numerical features analysis
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    if numerical_cols:
        print(f"\nNumerical Features: {len(numerical_cols)}")
        print(numerical_cols)
        
        # Plot distributions for key features
        key_numerical = ['purchase_value', 'age']
        available_numerical = [col for col in key_numerical if col in numerical_cols]
        
        if available_numerical:
            visualizer.plot_numerical_distributions(df, available_numerical, target_col)
        
        # Correlation matrix for numerical features
        if len(numerical_cols) > 1:
            visualizer.plot_correlation_matrix(df[numerical_cols + [target_col]])
    
    # Categorical features analysis
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if categorical_cols:
        print(f"\nCategorical Features: {len(categorical_cols)}")
        print(categorical_cols)
        
        # Plot distributions for key categorical features
        key_categorical = ['source', 'browser', 'sex']
        available_categorical = [col for col in key_categorical if col in categorical_cols]
        
        if available_categorical:
            visualizer.plot_categorical_distributions(df, available_categorical, target_col)
    
    # Time-based analysis if available
    if 'hour_of_day' in df.columns:
        visualizer.plot_time_patterns(df, 'hour_of_day', target_col)
    
    if 'day_of_week' in df.columns:
        visualizer.plot_time_patterns(df, 'day_of_week', target_col)
    
    # Purchase value analysis
    if 'purchase_value' in df.columns:
        plt.figure(figsize=(15, 5))
        
        # Purchase value distribution by class
        plt.subplot(1, 3, 1)
        for class_val in df[target_col].unique():
            subset = df[df[target_col] == class_val]['purchase_value']
            plt.hist(subset, alpha=0.7, label=f'Class {class_val}', bins=30)
        plt.title('Purchase Value Distribution by Class')
        plt.xlabel('Purchase Value')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Box plot
        plt.subplot(1, 3, 2)
        df.boxplot(column='purchase_value', by=target_col, ax=plt.gca())
        plt.title('Purchase Value by Class')
        plt.suptitle('')
        
        # Fraud rate by purchase value ranges
        plt.subplot(1, 3, 3)
        df['purchase_range'] = pd.cut(df['purchase_value'], bins=10)
        fraud_by_range = df.groupby('purchase_range')[target_col].mean()
        fraud_by_range.plot(kind='bar', rot=45)
        plt.title('Fraud Rate by Purchase Value Range')
        plt.ylabel('Fraud Rate')
        
        plt.tight_layout()
        plt.show()
    
    print("\n" + "=" * 60)
    print("EDA REPORT COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    # This script can be run independently
    print("EDA script ready to use!")
    print("Import this function in your notebook: from src.run_eda import run_comprehensive_eda")
