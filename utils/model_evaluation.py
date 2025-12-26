"""
Model evaluation utilities for fraud detection project.
Contains functions for evaluating models with appropriate metrics for imbalanced data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score
)
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Class for comprehensive model evaluation with imbalanced data metrics."""
    
    def __init__(self):
        self.evaluation_results = {}
    
    def evaluate_single_model(self, model, model_name: str, X_test: pd.DataFrame, 
                             y_test: pd.Series, X_train: pd.DataFrame = None, 
                             y_train: pd.Series = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single model.
        
        Args:
            model: Trained model
            model_name: Name of the model
            X_test: Test features
            y_test: Test target
            X_train: Training features (optional, for training metrics)
            y_train: Training target (optional, for training metrics)
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Evaluating {model_name}...")
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba)
        }
        
        # Training metrics if provided
        if X_train is not None and y_train is not None:
            y_train_pred = model.predict(X_train)
            y_train_pred_proba = model.predict_proba(X_train)[:, 1]
            
            metrics.update({
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'train_f1_score': f1_score(y_train, y_train_pred),
                'train_roc_auc': roc_auc_score(y_train, y_train_pred_proba),
                'train_pr_auc': average_precision_score(y_train, y_train_pred_proba)
            })
        
        # Store results
        self.evaluation_results[model_name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'true_labels': y_test
        }
        
        print(f"✓ {model_name} evaluation completed")
        
        return metrics
    
    def evaluate_multiple_models(self, models: Dict[str, Any], X_test: pd.DataFrame, 
                                y_test: pd.Series, X_train: pd.DataFrame = None, 
                                y_train: pd.Series = None) -> pd.DataFrame:
        """
        Evaluate multiple models and return comparison DataFrame.
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test target
            X_train: Training features (optional)
            y_train: Training target (optional)
            
        Returns:
            DataFrame with model comparison metrics
        """
        print("Evaluating multiple models...")
        print("=" * 50)
        
        all_metrics = []
        
        for model_name, model in models.items():
            metrics = self.evaluate_single_model(
                model, model_name, X_test, y_test, X_train, y_train
            )
            all_metrics.append(metrics)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(all_metrics)
        comparison_df = comparison_df.set_index('model_name')
        
        print("\n" + "=" * 50)
        print("Model evaluation completed!")
        
        return comparison_df
    
    def plot_confusion_matrices(self, models: Dict[str, Any], X_test: pd.DataFrame, 
                               y_test: pd.Series, figsize: Tuple[int, int] = (15, 5)) -> None:
        """
        Plot confusion matrices for multiple models.
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test target
            figsize: Figure size
        """
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, model) in enumerate(models.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Legitimate', 'Fraud'],
                       yticklabels=['Legitimate', 'Fraud'])
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}\nConfusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, models: Dict[str, Any], X_test: pd.DataFrame, 
                       y_test: pd.Series, figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot ROC curves for multiple models.
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test target
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        for model_name, model in models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, label=f'{model_name.replace("_", " ").title()} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_precision_recall_curves(self, models: Dict[str, Any], X_test: pd.DataFrame, 
                                    y_test: pd.Series, figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot Precision-Recall curves for multiple models.
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test target
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Baseline (random classifier for imbalanced data)
        baseline = y_test.mean()
        plt.axhline(y=baseline, color='k', linestyle='--', 
                   label=f'Baseline (AP = {baseline:.3f})')
        
        for model_name, model in models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            ap_score = average_precision_score(y_test, y_pred_proba)
            
            plt.plot(recall, precision, 
                    label=f'{model_name.replace("_", " ").title()} (AP = {ap_score:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_feature_importance(self, model, feature_names: List[str], 
                               model_name: str, top_n: int = 20, 
                               figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot feature importance for tree-based models.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            model_name: Name of the model
            top_n: Number of top features to show
            figsize: Figure size
        """
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance_scores = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importance_scores = np.abs(model.coef_[0])
        else:
            print(f"Feature importance not available for {model_name}")
            return
        
        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=True).tail(top_n)
        
        plt.figure(figsize=figsize)
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title(f'Top {top_n} Feature Importances - {model_name.replace("_", " ").title()}')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
    
    def generate_classification_report(self, models: Dict[str, Any], X_test: pd.DataFrame, 
                                     y_test: pd.Series) -> None:
        """
        Generate detailed classification reports for all models.
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test target
        """
        print("DETAILED CLASSIFICATION REPORTS")
        print("=" * 60)
        
        for model_name, model in models.items():
            y_pred = model.predict(X_test)
            
            print(f"\n{model_name.upper().replace('_', ' ')}")
            print("-" * 40)
            print(classification_report(y_test, y_pred, 
                                      target_names=['Legitimate', 'Fraud']))
    
    def identify_best_model(self, comparison_df: pd.DataFrame, 
                           primary_metric: str = 'f1_score',
                           secondary_metrics: List[str] = ['pr_auc', 'recall']) -> str:
        """
        Identify the best model based on specified metrics.
        
        Args:
            comparison_df: DataFrame with model comparison metrics
            primary_metric: Primary metric for model selection
            secondary_metrics: Secondary metrics for tie-breaking
            
        Returns:
            Name of the best model
        """
        print(f"Identifying best model based on {primary_metric}...")
        
        # Sort by primary metric
        sorted_df = comparison_df.sort_values(primary_metric, ascending=False)
        
        # Get top models
        best_score = sorted_df[primary_metric].iloc[0]
        top_models = sorted_df[sorted_df[primary_metric] >= best_score * 0.99]  # Within 1% of best
        
        if len(top_models) > 1:
            print(f"Multiple models within 1% of best {primary_metric}. Using secondary metrics...")
            
            # Use secondary metrics for tie-breaking
            for metric in secondary_metrics:
                if metric in top_models.columns:
                    top_models = top_models.sort_values(metric, ascending=False)
                    break
        
        best_model = top_models.index[0]
        
        print(f"Best model: {best_model}")
        print(f"Best {primary_metric}: {sorted_df.loc[best_model, primary_metric]:.4f}")
        
        return best_model
    
    def create_model_comparison_summary(self, comparison_df: pd.DataFrame) -> None:
        """
        Create a comprehensive model comparison summary.
        
        Args:
            comparison_df: DataFrame with model comparison metrics
        """
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)
        
        # Key metrics for fraud detection
        key_metrics = ['f1_score', 'precision', 'recall', 'pr_auc', 'roc_auc']
        
        print("\nKey Metrics Comparison:")
        print("-" * 30)
        
        summary_df = comparison_df[key_metrics].round(4)
        
        # Add ranking for each metric
        for metric in key_metrics:
            summary_df[f'{metric}_rank'] = summary_df[metric].rank(ascending=False)
        
        print(summary_df[key_metrics])
        
        # Best model for each metric
        print("\nBest Model for Each Metric:")
        print("-" * 30)
        for metric in key_metrics:
            best_model = summary_df[metric].idxmax()
            best_score = summary_df.loc[best_model, metric]
            print(f"{metric:12}: {best_model:15} ({best_score:.4f})")
        
        # Overall ranking (average rank across key metrics)
        rank_cols = [f'{metric}_rank' for metric in key_metrics]
        summary_df['avg_rank'] = summary_df[rank_cols].mean(axis=1)
        overall_ranking = summary_df.sort_values('avg_rank')
        
        print(f"\nOverall Ranking (by average rank):")
        print("-" * 30)
        for idx, (model, row) in enumerate(overall_ranking.iterrows(), 1):
            print(f"{idx}. {model:15} (avg rank: {row['avg_rank']:.2f})")

def evaluate_models_comprehensive(models: Dict[str, Any], X_train: pd.DataFrame, 
                                 y_train: pd.Series, X_test: pd.DataFrame, 
                                 y_test: pd.Series, feature_names: List[str]) -> Tuple[pd.DataFrame, str]:
    """
    Comprehensive evaluation pipeline for multiple models.
    
    Args:
        models: Dictionary of trained models
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        feature_names: List of feature names
        
    Returns:
        Tuple of (comparison DataFrame, best model name)
    """
    evaluator = ModelEvaluator()
    
    # Evaluate all models
    comparison_df = evaluator.evaluate_multiple_models(
        models, X_test, y_test, X_train, y_train
    )
    
    # Generate visualizations
    print("\nGenerating evaluation visualizations...")
    evaluator.plot_confusion_matrices(models, X_test, y_test)
    evaluator.plot_roc_curves(models, X_test, y_test)
    evaluator.plot_precision_recall_curves(models, X_test, y_test)
    
    # Feature importance for each model
    for model_name, model in models.items():
        evaluator.plot_feature_importance(model, feature_names, model_name)
    
    # Classification reports
    evaluator.generate_classification_report(models, X_test, y_test)
    
    # Model comparison summary
    evaluator.create_model_comparison_summary(comparison_df)
    
    # Identify best model
    best_model = evaluator.identify_best_model(comparison_df)
    
    return comparison_df, best_model
