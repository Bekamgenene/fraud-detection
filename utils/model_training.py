"""
Model training utilities for fraud detection project.
Contains classes for training Logistic Regression and Ensemble models.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from xgboost import XGBClassifier
from typing import Dict, Any, Tuple, Optional
import warnings

# Optional LightGBM import
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Skipping LightGBM models.")
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Class for training different machine learning models."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 hyperparameter_tuning: bool = True) -> LogisticRegression:
        """
        Train Logistic Regression model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Trained Logistic Regression model
        """
        print("Training Logistic Regression model...")
        
        if hyperparameter_tuning:
            # Define parameter grid for hyperparameter tuning
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000, 2000]
            }
            
            # Create base model
            base_model = LogisticRegression(random_state=self.random_state, class_weight='balanced')
            
            # Perform grid search with cross-validation
            grid_search = GridSearchCV(
                base_model, 
                param_grid, 
                cv=5, 
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Store best parameters
            self.best_params['logistic_regression'] = grid_search.best_params_
            model = grid_search.best_estimator_
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
            
        else:
            # Train with default parameters
            model = LogisticRegression(
                random_state=self.random_state,
                class_weight='balanced',
                max_iter=1000
            )
            model.fit(X_train, y_train)
        
        # Store the model
        self.models['logistic_regression'] = model
        print("Logistic Regression training completed!")
        
        return model
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                           hyperparameter_tuning: bool = True) -> RandomForestClassifier:
        """
        Train Random Forest model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Trained Random Forest model
        """
        print("Training Random Forest model...")
        
        if hyperparameter_tuning:
            # Define parameter grid for hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
            
            # Create base model
            base_model = RandomForestClassifier(
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
            
            # Perform grid search with cross-validation
            grid_search = GridSearchCV(
                base_model, 
                param_grid, 
                cv=3,  # Reduced CV folds for faster training
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Store best parameters
            self.best_params['random_forest'] = grid_search.best_params_
            model = grid_search.best_estimator_
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
            
        else:
            # Train with default parameters
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        # Store the model
        self.models['random_forest'] = model
        print("Random Forest training completed!")
        
        return model
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     hyperparameter_tuning: bool = True) -> XGBClassifier:
        """
        Train XGBoost model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Trained XGBoost model
        """
        print("Training XGBoost model...")
        
        # Calculate scale_pos_weight for class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        if hyperparameter_tuning:
            # Define parameter grid for hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            # Create base model
            base_model = XGBClassifier(
                random_state=self.random_state,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss'
            )
            
            # Perform grid search with cross-validation
            grid_search = GridSearchCV(
                base_model, 
                param_grid, 
                cv=3,  # Reduced CV folds for faster training
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Store best parameters
            self.best_params['xgboost'] = grid_search.best_params_
            model = grid_search.best_estimator_
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
            
        else:
            # Train with default parameters
            model = XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss'
            )
            model.fit(X_train, y_train)
        
        # Store the model
        self.models['xgboost'] = model
        print("XGBoost training completed!")
        
        return model
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray, 
                      tune_hyperparameters: bool = False) -> Any:
        """
        Train LightGBM classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Trained LightGBM model or None if LightGBM not available
        """
        if not LIGHTGBM_AVAILABLE:
            print("LightGBM not available. Skipping LightGBM training.")
            return None
            
        if tune_hyperparameters:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.01, 0.1],
                'num_leaves': [31, 50]
            }
            
            lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
            grid_search = GridSearchCV(
                lgb_model, param_grid, cv=3, scoring='f1',
                n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_
        else:
            # Train with default parameters
            model = lgb.LGBMClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced',
                verbose=-1
            )
            model.fit(X_train, y_train)
        
        # Store the model
        self.models['lightgbm'] = model
        print("LightGBM training completed!")
        
        return model
    
    def get_model(self, model_name: str):
        """Get a trained model by name."""
        return self.models.get(model_name)
    
    def get_best_params(self, model_name: str) -> Dict[str, Any]:
        """Get best parameters for a model."""
        return self.best_params.get(model_name, {})
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        models_to_train: list = None, 
                        hyperparameter_tuning: bool = True) -> Dict[str, Any]:
        """
        Train all specified models.
        
        Args:
            X_train: Training features
            y_train: Training target
            models_to_train: List of models to train
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary of trained models
        """
        if models_to_train is None:
            models_to_train = ['logistic_regression', 'random_forest', 'xgboost']
        
        print(f"Training {len(models_to_train)} models...")
        print("=" * 60)
        
        trained_models = {}
        
        for model_name in models_to_train:
            print(f"\n{model_name.upper().replace('_', ' ')}")
            print("-" * 40)
            
            if model_name == 'logistic_regression':
                model = self.train_logistic_regression(X_train, y_train, hyperparameter_tuning)
            elif model_name == 'random_forest':
                model = self.train_random_forest(X_train, y_train, hyperparameter_tuning)
            elif model_name == 'xgboost':
                model = self.train_xgboost(X_train, y_train, hyperparameter_tuning)
            elif model_name == 'lightgbm':
                model = self.train_lightgbm(X_train, y_train, hyperparameter_tuning)
            else:
                print(f"Unknown model: {model_name}")
                continue
            
            trained_models[model_name] = model
        
        print("\n" + "=" * 60)
        print("All models training completed!")
        
        return trained_models

def cross_validate_models(models: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series,
                         cv_folds: int = 5, scoring: str = 'f1') -> Dict[str, Dict[str, float]]:
    """
    Perform cross-validation on multiple models.
    
    Args:
        models: Dictionary of trained models
        X_train: Training features
        y_train: Training target
        cv_folds: Number of cross-validation folds
        scoring: Scoring metric
        
    Returns:
        Dictionary with cross-validation results
    """
    print(f"Performing {cv_folds}-fold cross-validation...")
    
    cv_results = {}
    
    for model_name, model in models.items():
        print(f"Cross-validating {model_name}...")
        
        scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring)
        
        cv_results[model_name] = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist()
        }
        
        print(f"{model_name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return cv_results
