"""
Pytest configuration and fixtures for fraud detection tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from utils.data_utils import DataLoader, DataCleaner
from utils.model_training import ModelTrainer


@pytest.fixture
def sample_fraud_data():
    """Create sample fraud detection data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'user_id': range(n_samples),
        'signup_time': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
        'purchase_time': pd.date_range('2023-01-02', periods=n_samples, freq='1H'),
        'purchase_value': np.random.lognormal(3, 1, n_samples),
        'device_id': [f'device_{i%100}' for i in range(n_samples)],
        'source': np.random.choice(['SEO', 'Ads', 'Direct'], n_samples),
        'browser': np.random.choice(['Chrome', 'Firefox', 'Safari'], n_samples),
        'sex': np.random.choice(['M', 'F'], n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'ip_address': [f'192.168.{i%256}.{(i*7)%256}' for i in range(n_samples)],
        'class': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])  # 10% fraud
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_creditcard_data():
    """Create sample credit card data for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 28
    
    # Create V1-V28 features (PCA components)
    features = {}
    for i in range(1, n_features + 1):
        features[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    features['Time'] = np.random.uniform(0, 172800, n_samples)  # 48 hours in seconds
    features['Amount'] = np.random.lognormal(3, 1, n_samples)
    features['Class'] = np.random.choice([0, 1], n_samples, p=[0.998, 0.002])  # 0.2% fraud
    
    return pd.DataFrame(features)


@pytest.fixture
def temp_data_dir():
    """Create temporary directory with sample data files."""
    temp_dir = tempfile.mkdtemp()
    
    # Create sample CSV files
    fraud_data = pd.DataFrame({
        'user_id': [1, 2, 3],
        'purchase_value': [100, 200, 300],
        'class': [0, 1, 0]
    })
    fraud_data.to_csv(Path(temp_dir) / 'Fraud_Data.csv', index=False)
    
    ip_data = pd.DataFrame({
        'lower_bound_ip_address': [0, 1000000, 2000000],
        'upper_bound_ip_address': [999999, 1999999, 2999999],
        'country': ['US', 'UK', 'CA']
    })
    ip_data.to_csv(Path(temp_dir) / 'IpAddress_to_Country.csv', index=False)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def config():
    """Create test configuration."""
    return Config()


@pytest.fixture
def data_loader(temp_data_dir):
    """Create DataLoader instance with temp data."""
    return DataLoader(data_path=temp_data_dir + '/')


@pytest.fixture
def data_cleaner():
    """Create DataCleaner instance."""
    return DataCleaner()


@pytest.fixture
def model_trainer():
    """Create ModelTrainer instance."""
    return ModelTrainer(random_state=42)


@pytest.fixture
def mock_model():
    """Create mock ML model for testing."""
    model = Mock()
    model.fit.return_value = model
    model.predict.return_value = np.array([0, 1, 0, 1])
    model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]])
    return model


@pytest.fixture
def sample_features_targets():
    """Create sample X, y data for model testing."""
    np.random.seed(42)
    X = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 100),
        'feature_2': np.random.normal(0, 1, 100),
        'feature_3': np.random.uniform(0, 1, 100)
    })
    y = pd.Series(np.random.choice([0, 1], 100, p=[0.8, 0.2]))
    return X, y
