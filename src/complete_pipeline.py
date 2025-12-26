"""
Complete fraud detection pipeline script.
Runs the entire workflow from data loading to model training and explainability.
"""

import sys
import os
sys.path.append('../')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from utils.data_utils import DataLoader, DataCleaner, merge_with_geolocation
from utils.feature_engineering import create_all_features
from utils.preprocessing import full_preprocessing_pipeline
from utils.model_training import ModelTrainer
from utils.model_evaluation import evaluate_models_comprehensive
from utils.model_explainability import explain_best_model

def run_complete_pipeline(data_path='../data/', models_to_train=None, 
                         hyperparameter_tuning=True, save_models=True):
    """
    Run the complete fraud detection pipeline.
    
    Args:
        data_path: Path to data directory
        models_to_train: List of models to train
        hyperparameter_tuning: Whether to perform hyperparameter tuning
        save_models: Whether to save trained models
    
    Returns:
        Dictionary with results for each dataset
    """
    if models_to_train is None:
        models_to_train = ['logistic_regression', 'random_forest', 'xgboost']
    
    print("=" * 80)
    print("FRAUD DETECTION COMPLETE PIPELINE")
    print("=" * 80)
    
    results = {}
    
    # Initialize data loader
    data_loader = DataLoader(data_path=data_path)
    cleaner = DataCleaner()
    
    # Process Fraud Data
    print("\n1. PROCESSING FRAUD DATASET")
    print("-" * 50)
    
    fraud_df = data_loader.load_fraud_data()
    
    if not fraud_df.empty:
        # Data cleaning and feature engineering
        fraud_df_clean = cleaner.handle_missing_values(fraud_df, strategy='drop')
        fraud_df_clean = cleaner.remove_duplicates(fraud_df_clean)
        fraud_df_clean = cleaner.correct_data_types(fraud_df_clean)
        fraud_df_features = create_all_features(fraud_df_clean)
        
        # Preprocessing
        fraud_processed = full_preprocessing_pipeline(
            fraud_df_features,
            target_col='class',
            sampling_strategy='smote',
            scaling_method='standard'
        )
        
        # Model training
        print("\nTraining models on fraud dataset...")
        trainer = ModelTrainer(random_state=42)
        fraud_models = trainer.train_all_models(
            fraud_processed['X_train'],
            fraud_processed['y_train'],
            models_to_train=models_to_train,
            hyperparameter_tuning=hyperparameter_tuning
        )
        
        # Model evaluation
        print("\nEvaluating fraud detection models...")
        fraud_comparison, fraud_best_model = evaluate_models_comprehensive(
            fraud_models,
            fraud_processed['X_train'],
            fraud_processed['y_train'],
            fraud_processed['X_test'],
            fraud_processed['y_test'],
            fraud_processed['feature_names']
        )
        
        # SHAP explainability
        print("\nGenerating SHAP explanations...")
        best_model = fraud_models[fraud_best_model]
        fraud_insights = explain_best_model(
            best_model,
            fraud_best_model,
            fraud_processed['X_train'],
            fraud_processed['X_test'],
            fraud_processed['y_test']
        )
        
        results['fraud'] = {
            'processed_data': fraud_processed,
            'models': fraud_models,
            'comparison': fraud_comparison,
            'best_model': fraud_best_model,
            'insights': fraud_insights
        }
        
        print(f"\n✓ Fraud dataset processing completed. Best model: {fraud_best_model}")
    
    # Process Credit Card Data
    print("\n2. PROCESSING CREDIT CARD DATASET")
    print("-" * 50)
    
    creditcard_df = data_loader.load_creditcard_data()
    
    if not creditcard_df.empty:
        # Preprocessing (credit card data is already clean)
        cc_processed = full_preprocessing_pipeline(
            creditcard_df,
            target_col='Class',
            sampling_strategy='smote',
            scaling_method='standard'
        )
        
        # Model training
        print("\nTraining models on credit card dataset...")
        cc_trainer = ModelTrainer(random_state=42)
        cc_models = cc_trainer.train_all_models(
            cc_processed['X_train'],
            cc_processed['y_train'],
            models_to_train=models_to_train,
            hyperparameter_tuning=hyperparameter_tuning
        )
        
        # Model evaluation
        print("\nEvaluating credit card fraud models...")
        cc_comparison, cc_best_model = evaluate_models_comprehensive(
            cc_models,
            cc_processed['X_train'],
            cc_processed['y_train'],
            cc_processed['X_test'],
            cc_processed['y_test'],
            cc_processed['feature_names']
        )
        
        # SHAP explainability
        print("\nGenerating SHAP explanations...")
        cc_best_model_obj = cc_models[cc_best_model]
        cc_insights = explain_best_model(
            cc_best_model_obj,
            cc_best_model,
            cc_processed['X_train'],
            cc_processed['X_test'],
            cc_processed['y_test']
        )
        
        results['creditcard'] = {
            'processed_data': cc_processed,
            'models': cc_models,
            'comparison': cc_comparison,
            'best_model': cc_best_model,
            'insights': cc_insights
        }
        
        print(f"\n✓ Credit card dataset processing completed. Best model: {cc_best_model}")
    
    # Save models if requested
    if save_models:
        save_pipeline_results(results)
    
    # Generate final report
    generate_final_report(results)
    
    print("\n" + "=" * 80)
    print("COMPLETE PIPELINE EXECUTION FINISHED")
    print("=" * 80)
    
    return results

def save_pipeline_results(results):
    """Save trained models and preprocessing objects."""
    import joblib
    
    # Create models directory
    os.makedirs('../models', exist_ok=True)
    
    print("\nSaving models and preprocessing objects...")
    
    for dataset_name, dataset_results in results.items():
        if 'models' in dataset_results:
            best_model_name = dataset_results['best_model']
            best_model = dataset_results['models'][best_model_name]
            processed_data = dataset_results['processed_data']
            
            # Save best model
            model_path = f'../models/{dataset_name}_best_model_{best_model_name}.pkl'
            joblib.dump(best_model, model_path)
            
            # Save scaler
            scaler_path = f'../models/{dataset_name}_scaler.pkl'
            joblib.dump(processed_data['scaler'], scaler_path)
            
            # Save feature names
            features_path = f'../models/{dataset_name}_feature_names.txt'
            with open(features_path, 'w') as f:
                for feature in processed_data['feature_names']:
                    f.write(f"{feature}\n")
            
            print(f"✓ {dataset_name} model saved: {best_model_name}")
    
    print("All models and preprocessing objects saved!")

def generate_final_report(results):
    """Generate comprehensive final report."""
    print("\n" + "=" * 80)
    print("FINAL FRAUD DETECTION ANALYSIS REPORT")
    print("=" * 80)
    
    for dataset_name, dataset_results in results.items():
        if 'comparison' in dataset_results:
            print(f"\n{dataset_name.upper()} DATASET RESULTS")
            print("-" * 50)
            
            comparison = dataset_results['comparison']
            best_model = dataset_results['best_model']
            
            print(f"Best Model: {best_model}")
            print(f"Dataset Shape: {dataset_results['processed_data']['X_test'].shape}")
            
            # Key metrics
            best_metrics = comparison.loc[best_model]
            print(f"\nPerformance Metrics:")
            print(f"  F1-Score:   {best_metrics['f1_score']:.4f}")
            print(f"  Precision:  {best_metrics['precision']:.4f}")
            print(f"  Recall:     {best_metrics['recall']:.4f}")
            print(f"  PR-AUC:     {best_metrics['pr_auc']:.4f}")
            print(f"  ROC-AUC:    {best_metrics['roc_auc']:.4f}")
            
            # Model comparison
            print(f"\nAll Models Comparison:")
            key_metrics = ['f1_score', 'precision', 'recall', 'pr_auc']
            print(comparison[key_metrics].round(4))
            
            # SHAP insights
            if 'insights' in dataset_results and dataset_results['insights']:
                insights = dataset_results['insights']
                
                if 'top_fraud_drivers' in insights:
                    print(f"\nTop Fraud Risk Factors:")
                    for i, driver in enumerate(insights['top_fraud_drivers'][:5], 1):
                        print(f"  {i}. {driver['feature']}")
                
                if 'protective_factors' in insights:
                    print(f"\nTop Protective Factors:")
                    for i, factor in enumerate(insights['protective_factors'][:5], 1):
                        print(f"  {i}. {factor['feature']}")
    
    # Business recommendations
    print(f"\nBUSINESS RECOMMENDATIONS")
    print("-" * 30)
    print("1. Deploy best performing models for real-time fraud scoring")
    print("2. Implement monitoring based on SHAP-identified risk factors")
    print("3. Set fraud thresholds balancing detection rate vs false positives")
    print("4. Regular model retraining to adapt to evolving fraud patterns")
    print("5. Use explainability insights for fraud investigation workflows")
    
    print(f"\nTECHNICAL RECOMMENDATIONS")
    print("-" * 30)
    print("1. Implement A/B testing framework for model comparison")
    print("2. Set up model performance monitoring and drift detection")
    print("3. Create automated retraining pipeline")
    print("4. Develop real-time feature engineering pipeline")
    print("5. Implement model versioning and rollback capabilities")

if __name__ == "__main__":
    # Run complete pipeline
    results = run_complete_pipeline(
        data_path='../data/',
        models_to_train=['logistic_regression', 'random_forest', 'xgboost'],
        hyperparameter_tuning=True,
        save_models=True
    )
    
    print("\nPipeline execution completed successfully!")
    print(f"Results available for: {list(results.keys())}")
