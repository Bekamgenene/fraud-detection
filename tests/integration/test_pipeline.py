"""
Integration tests for the complete fraud detection pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.complete_pipeline import run_complete_pipeline
from utils.data_utils import DataLoader, DataCleaner
from utils.preprocessing import full_preprocessing_pipeline
from utils.model_training import ModelTrainer


@pytest.mark.integration
class TestCompletePipeline:
    """Test the complete fraud detection pipeline."""
    
    @pytest.fixture
    def temp_pipeline_data(self):
        """Create temporary data for pipeline testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Create realistic fraud data
        np.random.seed(42)
        n_samples = 500
        
        fraud_data = pd.DataFrame({
            'user_id': range(n_samples),
            'signup_time': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
            'purchase_time': pd.date_range('2023-01-02', periods=n_samples, freq='1H'),
            'purchase_value': np.random.lognormal(3, 1, n_samples),
            'device_id': [f'device_{i%50}' for i in range(n_samples)],
            'source': np.random.choice(['SEO', 'Ads', 'Direct'], n_samples),
            'browser': np.random.choice(['Chrome', 'Firefox', 'Safari'], n_samples),
            'sex': np.random.choice(['M', 'F'], n_samples),
            'age': np.random.randint(18, 80, n_samples),
            'ip_address': [f'192.168.{i%256}.{(i*7)%256}' for i in range(n_samples)],
            'class': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        })
        fraud_data.to_csv(Path(temp_dir) / 'Fraud_Data.csv', index=False)
        
        # Create credit card data
        cc_features = {}
        for i in range(1, 29):
            cc_features[f'V{i}'] = np.random.normal(0, 1, n_samples)
        cc_features['Time'] = np.random.uniform(0, 172800, n_samples)
        cc_features['Amount'] = np.random.lognormal(3, 1, n_samples)
        cc_features['Class'] = np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
        
        cc_data = pd.DataFrame(cc_features)
        cc_data.to_csv(Path(temp_dir) / 'creditcard.csv', index=False)
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_data_loading_and_cleaning_pipeline(self, temp_pipeline_data):
        """Test data loading and cleaning integration."""
        loader = DataLoader(data_path=temp_pipeline_data + '/')
        cleaner = DataCleaner()
        
        # Load data
        fraud_df = loader.load_fraud_data()
        assert not fraud_df.empty
        
        # Clean data
        fraud_clean = cleaner.handle_missing_values(fraud_df, strategy='drop')
        fraud_clean = cleaner.remove_duplicates(fraud_clean)
        fraud_clean = cleaner.correct_data_types(fraud_clean)
        
        assert not fraud_clean.empty
        assert fraud_clean['class'].dtype == int
    
    def test_preprocessing_pipeline_integration(self, sample_fraud_data):
        """Test the complete preprocessing pipeline."""
        result = full_preprocessing_pipeline(
            sample_fraud_data,
            target_col='class',
            sampling_strategy='smote',
            scaling_method='standard'
        )
        
        # Check all required keys are present
        required_keys = ['X_train', 'X_test', 'y_train', 'y_test', 'scaler', 'feature_names']
        for key in required_keys:
            assert key in result
        
        # Check data shapes
        assert result['X_train'].shape[0] > 0
        assert result['X_test'].shape[0] > 0
        assert len(result['y_train']) == result['X_train'].shape[0]
        assert len(result['y_test']) == result['X_test'].shape[0]
    
    def test_model_training_integration(self, sample_features_targets):
        """Test model training integration."""
        X, y = sample_features_targets
        trainer = ModelTrainer(random_state=42)
        
        models = trainer.train_all_models(
            X, y,
            models_to_train=['logistic_regression', 'random_forest'],
            hyperparameter_tuning=False
        )
        
        assert len(models) == 2
        assert 'logistic_regression' in models
        assert 'random_forest' in models
        
        # Test predictions
        for model_name, model in models.items():
            predictions = model.predict(X)
            assert len(predictions) == len(y)
            assert all(pred in [0, 1] for pred in predictions)
    
    @pytest.mark.slow
    def test_complete_pipeline_small_dataset(self, temp_pipeline_data):
        """Test the complete pipeline with a small dataset."""
        # Run pipeline with minimal configuration
        results = run_complete_pipeline(
            data_path=temp_pipeline_data + '/',
            models_to_train=['logistic_regression'],
            hyperparameter_tuning=False,
            save_models=False
        )
        
        # Check results structure
        assert isinstance(results, dict)
        
        # Check fraud dataset results if available
        if 'fraud' in results:
            fraud_results = results['fraud']
            assert 'models' in fraud_results
            assert 'comparison' in fraud_results
            assert 'best_model' in fraud_results
        
        # Check credit card dataset results if available
        if 'creditcard' in results:
            cc_results = results['creditcard']
            assert 'models' in cc_results
            assert 'comparison' in cc_results
            assert 'best_model' in cc_results


@pytest.mark.integration
class TestDataFlowIntegration:
    """Test data flow between different components."""
    
    def test_data_loader_to_preprocessing(self, temp_data_dir):
        """Test data flow from loader to preprocessing."""
        loader = DataLoader(data_path=temp_data_dir + '/')
        cleaner = DataCleaner()
        
        # Load and clean data
        df = loader.load_fraud_data()
        df_clean = cleaner.handle_missing_values(df)
        df_clean = cleaner.remove_duplicates(df_clean)
        df_clean = cleaner.correct_data_types(df_clean)
        
        # Ensure data can be used for preprocessing
        assert not df_clean.empty
        assert 'class' in df_clean.columns
    
    def test_preprocessing_to_model_training(self, sample_fraud_data):
        """Test data flow from preprocessing to model training."""
        # Preprocess data
        processed = full_preprocessing_pipeline(
            sample_fraud_data,
            target_col='class',
            sampling_strategy='none',  # Skip resampling for speed
            scaling_method='standard'
        )
        
        # Train model with processed data
        trainer = ModelTrainer(random_state=42)
        model = trainer.train_logistic_regression(
            processed['X_train'],
            processed['y_train'],
            hyperparameter_tuning=False
        )
        
        # Test model can make predictions
        predictions = model.predict(processed['X_test'])
        assert len(predictions) == len(processed['y_test'])
        assert all(pred in [0, 1] for pred in predictions)
