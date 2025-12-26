"""
Configuration management for fraud detection system.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data handling."""
    data_path: str = "data/raw/"
    fraud_data_file: str = "Fraud_Data.csv"
    ip_country_file: str = "IpAddress_to_Country.csv"
    creditcard_file: str = "creditcard.csv"
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True


@dataclass
class ModelConfig:
    """Configuration for model training."""
    models_to_train: List[str] = field(default_factory=lambda: [
        'logistic_regression', 'random_forest', 'xgboost'
    ])
    hyperparameter_tuning: bool = True
    cv_folds: int = 5
    scoring_metric: str = 'f1'
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    sampling_strategy: str = 'smote'  # 'smote', 'undersample', 'smote_tomek', 'none'
    scaling_method: str = 'standard'  # 'standard', 'minmax', 'none'
    handle_missing: str = 'drop'  # 'drop', 'mean', 'median', 'mode'
    remove_duplicates: bool = True


@dataclass
class OutputConfig:
    """Configuration for outputs and logging."""
    models_path: str = "models/"
    logs_path: str = "logs/"
    reports_path: str = "reports/"
    save_models: bool = True
    save_reports: bool = True
    log_level: str = "INFO"


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    def __post_init__(self):
        """Create necessary directories."""
        for path in [self.output.models_path, self.output.logs_path, self.output.reports_path]:
            Path(path).mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    """Load configuration with environment variable overrides."""
    config = Config()
    
    # Override with environment variables if available
    if os.getenv('DATA_PATH'):
        config.data.data_path = os.getenv('DATA_PATH')
    if os.getenv('MODELS_PATH'):
        config.output.models_path = os.getenv('MODELS_PATH')
    if os.getenv('LOG_LEVEL'):
        config.output.log_level = os.getenv('LOG_LEVEL')
    
    return config
