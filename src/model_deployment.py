"""
Model deployment utilities for fraud detection system.
Contains functions for loading trained models and making predictions.
"""

import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionPredictor:
    """Class for loading and using trained fraud detection models."""
    
    def __init__(self, model_path: str, scaler_path: str, features_path: str):
        """
        Initialize the predictor with trained model and preprocessing objects.
        
        Args:
            model_path: Path to trained model file
            scaler_path: Path to fitted scaler object
            features_path: Path to feature names file
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Load feature names
        with open(features_path, 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
        
        print(f"Model loaded successfully!")
        print(f"Features: {len(self.feature_names)}")
        print(f"Model type: {type(self.model).__name__}")
    
    def predict_fraud_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud probability for input transactions.
        
        Args:
            X: Input features DataFrame
            
        Returns:
            Array of fraud probabilities
        """
        # Ensure features are in correct order
        X_ordered = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X_ordered)
        
        # Get probabilities
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return probabilities
    
    def predict_fraud_binary(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Make binary fraud predictions.
        
        Args:
            X: Input features DataFrame
            threshold: Decision threshold
            
        Returns:
            Array of binary predictions (0=legitimate, 1=fraud)
        """
        probabilities = self.predict_fraud_probability(X)
        return (probabilities >= threshold).astype(int)
    
    def predict_with_explanation(self, X: pd.DataFrame, 
                               threshold: float = 0.5) -> Dict[str, Any]:
        """
        Make predictions with detailed explanation.
        
        Args:
            X: Input features DataFrame
            threshold: Decision threshold
            
        Returns:
            Dictionary with predictions and explanations
        """
        probabilities = self.predict_fraud_probability(X)
        predictions = (probabilities >= threshold).astype(int)
        
        results = {
            'predictions': predictions,
            'probabilities': probabilities,
            'threshold': threshold,
            'fraud_count': predictions.sum(),
            'fraud_rate': predictions.mean(),
            'high_risk_transactions': np.where(probabilities >= 0.8)[0].tolist(),
            'medium_risk_transactions': np.where((probabilities >= 0.5) & (probabilities < 0.8))[0].tolist(),
            'low_risk_transactions': np.where(probabilities < 0.5)[0].tolist()
        }
        
        return results

def load_fraud_model(dataset_type: str = 'fraud') -> FraudDetectionPredictor:
    """
    Load trained fraud detection model.
    
    Args:
        dataset_type: Type of dataset ('fraud' or 'creditcard')
        
    Returns:
        FraudDetectionPredictor instance
    """
    import os
    
    models_dir = '../models'
    
    # Find model files
    model_files = [f for f in os.listdir(models_dir) if f.startswith(f'{dataset_type}_best_model')]
    
    if not model_files:
        raise FileNotFoundError(f"No trained model found for {dataset_type} dataset")
    
    model_path = os.path.join(models_dir, model_files[0])
    scaler_path = os.path.join(models_dir, f'{dataset_type}_scaler.pkl')
    features_path = os.path.join(models_dir, f'{dataset_type}_feature_names.txt')
    
    return FraudDetectionPredictor(model_path, scaler_path, features_path)

def batch_fraud_detection(transactions_df: pd.DataFrame, 
                         dataset_type: str = 'fraud',
                         threshold: float = 0.5) -> pd.DataFrame:
    """
    Perform batch fraud detection on multiple transactions.
    
    Args:
        transactions_df: DataFrame with transaction features
        dataset_type: Type of model to use ('fraud' or 'creditcard')
        threshold: Decision threshold
        
    Returns:
        DataFrame with original data plus fraud predictions
    """
    # Load model
    predictor = load_fraud_model(dataset_type)
    
    # Make predictions
    results = predictor.predict_with_explanation(transactions_df, threshold)
    
    # Add predictions to original DataFrame
    output_df = transactions_df.copy()
    output_df['fraud_probability'] = results['probabilities']
    output_df['fraud_prediction'] = results['predictions']
    output_df['risk_level'] = pd.cut(
        results['probabilities'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    return output_df

def real_time_fraud_check(transaction_data: Dict[str, Any],
                         dataset_type: str = 'fraud',
                         threshold: float = 0.5) -> Dict[str, Any]:
    """
    Real-time fraud check for a single transaction.
    
    Args:
        transaction_data: Dictionary with transaction features
        dataset_type: Type of model to use
        threshold: Decision threshold
        
    Returns:
        Dictionary with fraud assessment
    """
    # Convert to DataFrame
    transaction_df = pd.DataFrame([transaction_data])
    
    # Load model and predict
    predictor = load_fraud_model(dataset_type)
    probability = predictor.predict_fraud_probability(transaction_df)[0]
    prediction = int(probability >= threshold)
    
    # Determine risk level
    if probability >= 0.8:
        risk_level = 'High'
        recommendation = 'Block transaction and investigate'
    elif probability >= 0.5:
        risk_level = 'Medium'
        recommendation = 'Flag for manual review'
    else:
        risk_level = 'Low'
        recommendation = 'Approve transaction'
    
    return {
        'transaction_id': transaction_data.get('transaction_id', 'unknown'),
        'fraud_probability': float(probability),
        'fraud_prediction': prediction,
        'risk_level': risk_level,
        'recommendation': recommendation,
        'threshold_used': threshold,
        'model_type': dataset_type
    }

def model_performance_monitoring(test_data: pd.DataFrame, 
                               true_labels: pd.Series,
                               dataset_type: str = 'fraud') -> Dict[str, float]:
    """
    Monitor model performance on new data.
    
    Args:
        test_data: Test features
        true_labels: True fraud labels
        dataset_type: Type of model to use
        
    Returns:
        Dictionary with performance metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score
    )
    
    # Load model and predict
    predictor = load_fraud_model(dataset_type)
    probabilities = predictor.predict_fraud_probability(test_data)
    predictions = predictor.predict_fraud_binary(test_data)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions),
        'recall': recall_score(true_labels, predictions),
        'f1_score': f1_score(true_labels, predictions),
        'roc_auc': roc_auc_score(true_labels, probabilities),
        'pr_auc': average_precision_score(true_labels, probabilities),
        'fraud_detection_rate': recall_score(true_labels, predictions),
        'false_positive_rate': 1 - precision_score(true_labels, predictions)
    }
    
    return metrics

if __name__ == "__main__":
    # Example usage
    print("Fraud Detection Model Deployment Example")
    print("=" * 50)
    
    try:
        # Load model
        predictor = load_fraud_model('fraud')
        
        # Example transaction (you would replace this with real data)
        example_transaction = {
            'purchase_value': 50.0,
            'age': 35,
            'hour_of_day': 14,
            'day_of_week': 2,
            'time_since_signup': 720.0,
            # Add other required features...
        }
        
        # Real-time prediction
        result = real_time_fraud_check(example_transaction)
        
        print(f"Transaction Assessment:")
        print(f"Fraud Probability: {result['fraud_probability']:.4f}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Recommendation: {result['recommendation']}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the model training pipeline first to generate trained models.")
    except Exception as e:
        print(f"Error: {e}")
