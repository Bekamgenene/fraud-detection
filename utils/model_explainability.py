"""
Model explainability utilities for fraud detection project.
Contains SHAP-based explainability functions for model interpretation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SHAPExplainer:
    """Class for SHAP-based model explainability and interpretation."""
    
    def __init__(self):
        self.explainers = {}
        self.shap_values = {}
        
    def create_explainer(self, model, model_name: str, X_train: pd.DataFrame, 
                        model_type: str = 'auto') -> shap.Explainer:
        """
        Create SHAP explainer for a given model.
        
        Args:
            model: Trained model
            model_name: Name of the model
            X_train: Training data for background
            model_type: Type of explainer ('tree', 'linear', 'kernel', 'auto')
            
        Returns:
            SHAP explainer object
        """
        print(f"Creating SHAP explainer for {model_name}...")
        
        if model_type == 'auto':
            # Auto-detect model type
            if hasattr(model, 'tree_'):
                model_type = 'tree'
            elif hasattr(model, 'coef_'):
                model_type = 'linear'
            else:
                model_type = 'kernel'
        
        try:
            if model_type == 'tree':
                # For tree-based models (Random Forest, XGBoost, LightGBM)
                explainer = shap.TreeExplainer(model)
            elif model_type == 'linear':
                # For linear models (Logistic Regression)
                explainer = shap.LinearExplainer(model, X_train)
            else:
                # For other models (use kernel explainer with sample)
                background = shap.sample(X_train, min(100, len(X_train)))
                explainer = shap.KernelExplainer(model.predict_proba, background)
            
            self.explainers[model_name] = explainer
            print(f"✓ SHAP explainer created for {model_name}")
            
            return explainer
            
        except Exception as e:
            print(f"Error creating explainer for {model_name}: {str(e)}")
            return None
    
    def calculate_shap_values(self, explainer: shap.Explainer, X_test: pd.DataFrame, 
                             model_name: str, max_samples: int = 1000) -> np.ndarray:
        """
        Calculate SHAP values for test data.
        
        Args:
            explainer: SHAP explainer object
            X_test: Test data
            model_name: Name of the model
            max_samples: Maximum number of samples to explain
            
        Returns:
            SHAP values array
        """
        print(f"Calculating SHAP values for {model_name}...")
        
        # Limit samples for computational efficiency
        if len(X_test) > max_samples:
            X_sample = X_test.sample(n=max_samples, random_state=42)
        else:
            X_sample = X_test
        
        try:
            if isinstance(explainer, shap.TreeExplainer):
                shap_values = explainer.shap_values(X_sample)
                # For binary classification, take positive class
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:
                shap_values = explainer.shap_values(X_sample)
                if len(shap_values.shape) == 3:
                    shap_values = shap_values[:, :, 1]
            
            self.shap_values[model_name] = {
                'shap_values': shap_values,
                'data': X_sample,
                'feature_names': X_sample.columns.tolist()
            }
            
            print(f"✓ SHAP values calculated for {model_name}")
            return shap_values
            
        except Exception as e:
            print(f"Error calculating SHAP values for {model_name}: {str(e)}")
            return None
    
    def plot_summary_plot(self, model_name: str, plot_type: str = 'dot', 
                         max_display: int = 20, figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Create SHAP summary plot.
        
        Args:
            model_name: Name of the model
            plot_type: Type of plot ('dot', 'bar', 'violin')
            max_display: Maximum number of features to display
            figsize: Figure size
        """
        if model_name not in self.shap_values:
            print(f"SHAP values not available for {model_name}")
            return
        
        shap_data = self.shap_values[model_name]
        
        plt.figure(figsize=figsize)
        
        if plot_type == 'bar':
            shap.summary_plot(
                shap_data['shap_values'], 
                shap_data['data'],
                plot_type='bar',
                max_display=max_display,
                show=False
            )
        else:
            shap.summary_plot(
                shap_data['shap_values'], 
                shap_data['data'],
                max_display=max_display,
                show=False
            )
        
        plt.title(f'SHAP Summary Plot - {model_name.replace("_", " ").title()}')
        plt.tight_layout()
        plt.show()
    
    def plot_waterfall_plot(self, model_name: str, sample_idx: int = 0, 
                           figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Create SHAP waterfall plot for a single prediction.
        
        Args:
            model_name: Name of the model
            sample_idx: Index of the sample to explain
            figsize: Figure size
        """
        if model_name not in self.shap_values:
            print(f"SHAP values not available for {model_name}")
            return
        
        shap_data = self.shap_values[model_name]
        explainer = self.explainers[model_name]
        
        plt.figure(figsize=figsize)
        
        try:
            # Create explanation object
            if hasattr(explainer, 'expected_value'):
                expected_value = explainer.expected_value
                if isinstance(expected_value, list):
                    expected_value = expected_value[1]
            else:
                expected_value = 0
            
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_data['shap_values'][sample_idx],
                    base_values=expected_value,
                    data=shap_data['data'].iloc[sample_idx],
                    feature_names=shap_data['feature_names']
                ),
                show=False
            )
            
            plt.title(f'SHAP Waterfall Plot - {model_name.replace("_", " ").title()} (Sample {sample_idx})')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error creating waterfall plot: {str(e)}")
    
    def plot_force_plot(self, model_name: str, sample_idx: int = 0) -> None:
        """
        Create SHAP force plot for a single prediction.
        
        Args:
            model_name: Name of the model
            sample_idx: Index of the sample to explain
        """
        if model_name not in self.shap_values:
            print(f"SHAP values not available for {model_name}")
            return
        
        shap_data = self.shap_values[model_name]
        explainer = self.explainers[model_name]
        
        try:
            if hasattr(explainer, 'expected_value'):
                expected_value = explainer.expected_value
                if isinstance(expected_value, list):
                    expected_value = expected_value[1]
            else:
                expected_value = 0
            
            force_plot = shap.force_plot(
                expected_value,
                shap_data['shap_values'][sample_idx],
                shap_data['data'].iloc[sample_idx],
                feature_names=shap_data['feature_names']
            )
            
            return force_plot
            
        except Exception as e:
            print(f"Error creating force plot: {str(e)}")
            return None
    
    def plot_dependence_plot(self, model_name: str, feature_name: str, 
                            interaction_feature: str = None,
                            figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Create SHAP dependence plot for a specific feature.
        
        Args:
            model_name: Name of the model
            feature_name: Name of the feature to plot
            interaction_feature: Feature to color by for interactions
            figsize: Figure size
        """
        if model_name not in self.shap_values:
            print(f"SHAP values not available for {model_name}")
            return
        
        shap_data = self.shap_values[model_name]
        
        plt.figure(figsize=figsize)
        
        try:
            shap.dependence_plot(
                feature_name,
                shap_data['shap_values'],
                shap_data['data'],
                interaction_index=interaction_feature,
                show=False
            )
            
            plt.title(f'SHAP Dependence Plot - {feature_name} ({model_name.replace("_", " ").title()})')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error creating dependence plot: {str(e)}")
    
    def analyze_feature_importance(self, model_name: str, top_n: int = 10) -> pd.DataFrame:
        """
        Analyze global feature importance using SHAP values.
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance analysis
        """
        if model_name not in self.shap_values:
            print(f"SHAP values not available for {model_name}")
            return pd.DataFrame()
        
        shap_data = self.shap_values[model_name]
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_data['shap_values']).mean(axis=0)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': shap_data['feature_names'],
            'mean_abs_shap': mean_abs_shap,
            'mean_shap': shap_data['shap_values'].mean(axis=0),
            'std_shap': shap_data['shap_values'].std(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        
        print(f"Top {top_n} Most Important Features ({model_name}):")
        print("-" * 50)
        
        top_features = importance_df.head(top_n)
        for idx, row in top_features.iterrows():
            print(f"{row['feature']:25}: {row['mean_abs_shap']:.4f}")
        
        return importance_df
    
    def interpret_fraud_drivers(self, model_name: str, top_n: int = 10) -> Dict[str, Any]:
        """
        Interpret key drivers of fraud based on SHAP analysis.
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to analyze
            
        Returns:
            Dictionary with fraud driver insights
        """
        if model_name not in self.shap_values:
            print(f"SHAP values not available for {model_name}")
            return {}
        
        shap_data = self.shap_values[model_name]
        importance_df = self.analyze_feature_importance(model_name, top_n)
        
        insights = {
            'model_name': model_name,
            'top_fraud_drivers': [],
            'protective_factors': [],
            'feature_interactions': []
        }
        
        print(f"\nFRAUD DRIVER ANALYSIS - {model_name.upper()}")
        print("=" * 60)
        
        # Analyze top features
        for idx, row in importance_df.head(top_n).iterrows():
            feature = row['feature']
            mean_shap = row['mean_shap']
            
            if mean_shap > 0:
                insights['top_fraud_drivers'].append({
                    'feature': feature,
                    'impact': mean_shap,
                    'interpretation': f"Higher {feature} increases fraud probability"
                })
            else:
                insights['protective_factors'].append({
                    'feature': feature,
                    'impact': abs(mean_shap),
                    'interpretation': f"Higher {feature} decreases fraud probability"
                })
        
        # Print insights
        print("\nTop Fraud Risk Factors:")
        print("-" * 30)
        for driver in insights['top_fraud_drivers'][:5]:
            print(f"• {driver['feature']}: {driver['interpretation']}")
        
        print("\nTop Protective Factors:")
        print("-" * 30)
        for factor in insights['protective_factors'][:5]:
            print(f"• {factor['feature']}: {factor['interpretation']}")
        
        return insights
    
    def create_comprehensive_explanation(self, model_name: str, 
                                       sample_indices: List[int] = None) -> None:
        """
        Create comprehensive SHAP explanation with multiple visualizations.
        
        Args:
            model_name: Name of the model
            sample_indices: List of sample indices to explain in detail
        """
        if model_name not in self.shap_values:
            print(f"SHAP values not available for {model_name}")
            return
        
        print(f"COMPREHENSIVE SHAP ANALYSIS - {model_name.upper()}")
        print("=" * 60)
        
        # 1. Summary plot
        print("\n1. Global Feature Importance (Summary Plot)")
        self.plot_summary_plot(model_name, plot_type='dot')
        
        # 2. Bar plot
        print("\n2. Feature Importance Bar Plot")
        self.plot_summary_plot(model_name, plot_type='bar')
        
        # 3. Feature importance analysis
        print("\n3. Feature Importance Analysis")
        importance_df = self.analyze_feature_importance(model_name)
        
        # 4. Fraud driver interpretation
        print("\n4. Fraud Driver Interpretation")
        insights = self.interpret_fraud_drivers(model_name)
        
        # 5. Individual predictions (if sample indices provided)
        if sample_indices:
            print("\n5. Individual Prediction Explanations")
            for idx in sample_indices[:3]:  # Limit to 3 samples
                print(f"\nSample {idx}:")
                self.plot_waterfall_plot(model_name, idx)
        
        # 6. Dependence plots for top features
        print("\n6. Feature Dependence Analysis")
        top_features = importance_df.head(3)['feature'].tolist()
        for feature in top_features:
            self.plot_dependence_plot(model_name, feature)
        
        return insights

def explain_best_model(model, model_name: str, X_train: pd.DataFrame, 
                      X_test: pd.DataFrame, y_test: pd.Series,
                      sample_indices: List[int] = None) -> Dict[str, Any]:
    """
    Complete SHAP explanation pipeline for the best model.
    
    Args:
        model: Best performing model
        model_name: Name of the model
        X_train: Training features
        X_test: Test features
        y_test: Test target
        sample_indices: Specific samples to explain
        
    Returns:
        Dictionary with explanation insights
    """
    print("STARTING COMPREHENSIVE MODEL EXPLAINABILITY ANALYSIS")
    print("=" * 70)
    
    explainer_obj = SHAPExplainer()
    
    # Create explainer
    explainer = explainer_obj.create_explainer(model, model_name, X_train)
    
    if explainer is None:
        print("Failed to create SHAP explainer")
        return {}
    
    # Calculate SHAP values
    shap_values = explainer_obj.calculate_shap_values(explainer, X_test, model_name)
    
    if shap_values is None:
        print("Failed to calculate SHAP values")
        return {}
    
    # Generate comprehensive explanation
    insights = explainer_obj.create_comprehensive_explanation(model_name, sample_indices)
    
    print("\n" + "=" * 70)
    print("MODEL EXPLAINABILITY ANALYSIS COMPLETED")
    
    return insights
