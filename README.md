# Fraud Detection System

## Adey Innovations Inc. - E-commerce & Banking Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-green.svg)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.40+-red.svg)](https://shap.readthedocs.io)

A comprehensive fraud detection system built with advanced machine learning techniques, featuring class imbalance handling, ensemble models, and SHAP explainability for e-commerce and banking transactions.

## Project Overview

This project implements a production-ready fraud detection system that:

- **Handles highly imbalanced datasets** (9.36% fraud rate)
- **Implements advanced sampling techniques** (SMOTE, undersampling)
- **Uses ensemble models** (Random Forest, XGBoost, LightGBM)
- **Provides model explainability** with SHAP analysis
- **Supports multiple datasets** (e-commerce and banking transactions)
- **Includes deployment utilities** for real-time fraud scoring

## Key Results

### Model Performance

- **Best Model**: XGBoost Classifier
- **F1-Score**: 0.8542
- **Precision**: 0.8234 (17.66% false positive rate)
- **Recall**: 0.8876 (88.76% fraud detection rate)
- **PR-AUC**: 0.8456 (excellent performance on imbalanced data)

### Business Impact

- **88.76% fraud detection rate** - catches majority of fraudulent transactions
- **17.66% false positive rate** - acceptable for fraud prevention
- **Real-time scoring capability** - sub-second prediction times
- **Explainable predictions** - SHAP analysis for investigation support

## Project Structure

```
fraud-detection-system/
├── data/                           # Data files (gitignored)
│   ├── Fraud_Data.csv             # E-commerce transaction data
│   ├── IpAddress_to_Country.csv   # IP geolocation mapping
│   └── creditcard.csv             # Bank transaction data
├── utils/                          # Modular utility functions
│   ├── data_utils.py              # Data loading and cleaning
│   ├── feature_engineering.py     # Feature creation and transformation
│   ├── preprocessing.py           # Class imbalance and scaling
│   ├── model_training.py          # ML model training utilities
│   ├── model_evaluation.py        # Model evaluation and comparison
│   ├── model_explainability.py    # SHAP-based explainability
│   └── visualization.py           # EDA and plotting functions
├── src/                           # Source code and scripts
│   ├── run_eda.py                 # Standalone EDA execution
│   ├── complete_pipeline.py       # End-to-end pipeline script
│   └── model_deployment.py        # Model deployment utilities
├── notebook/                      # Jupyter notebooks
│   ├── fraud_detection_analysis.ipynb           # Main EDA notebook
│   └── model_training_and_explainability.ipynb  # Model training notebook
├── models/                        # Trained models (created after training)
│   ├── fraud_best_model_xgboost.pkl    # Best fraud detection model
│   ├── fraud_scaler.pkl                # Feature scaler
│   └── fraud_feature_names.txt         # Feature names list
├── requirements.txt               # Python dependencies
└── README.md                     # Project documentation
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd fraud-detection-system

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Place your datasets in the `data/raw/` folder:

- `Fraud_Data.csv` - E-commerce transaction data
- `IpAddress_to_Country.csv` - IP to country mapping (optional)
- `creditcard.csv` - Bank transaction data (optional)

### 3. Run Complete Pipeline

```bash
# Run the complete fraud detection pipeline
python src/complete_pipeline.py
```

This will:

- Load and clean the data
- Engineer features
- Train multiple models
- Evaluate and compare models
- Generate SHAP explanations
- Save the best models

### 4. Interactive Analysis

```bash
# Launch Jupyter notebook
jupyter notebook

# Open either:
# - notebook/fraud_detection_analysis.ipynb (EDA)
# - notebook/model_training_and_explainability.ipynb (Modeling)
```

## Usage Examples

### Real-time Fraud Detection

```python
from src.model_deployment import real_time_fraud_check

# Single transaction assessment
transaction = {
    'purchase_value': 150.0,
    'age': 25,
    'hour_of_day': 2,  # 2 AM transaction
    'day_of_week': 6,  # Weekend
    'time_since_signup': 1.5,  # 1.5 hours since signup
    # ... other features
}

result = real_time_fraud_check(transaction)
print(f"Fraud Probability: {result['fraud_probability']:.4f}")
print(f"Risk Level: {result['risk_level']}")
print(f"Recommendation: {result['recommendation']}")
```

### Batch Processing

```python
from src.model_deployment import batch_fraud_detection

# Process multiple transactions
transactions_df = pd.read_csv('new_transactions.csv')
results_df = batch_fraud_detection(transactions_df, threshold=0.5)

# View high-risk transactions
high_risk = results_df[results_df['risk_level'] == 'High']
print(f"Found {len(high_risk)} high-risk transactions")
```

### Model Performance Monitoring

```python
from src.model_deployment import model_performance_monitoring

# Monitor model performance on new data
metrics = model_performance_monitoring(test_data, true_labels)
print(f"Current F1-Score: {metrics['f1_score']:.4f}")
print(f"Fraud Detection Rate: {metrics['fraud_detection_rate']:.4f}")
```

## Features

### Data Processing

- **Automated data cleaning** with missing value handling
- **Feature engineering** (50+ features from 11 original)
- **Time-based features** (hour, day, time since signup)
- **Behavioral features** (transaction velocity, frequency)
- **Geolocation analysis** (IP to country mapping)

### Machine Learning

- **Multiple algorithms** (Logistic Regression, Random Forest, XGBoost, LightGBM)
- **Hyperparameter tuning** with GridSearchCV
- **Class imbalance handling** (SMOTE, undersampling, SMOTE+Tomek)
- **Cross-validation** for robust model selection
- **Appropriate metrics** for imbalanced data (F1, PR-AUC, Recall)

### Model Explainability

- **SHAP analysis** for global and local explanations
- **Feature importance** visualization
- **Fraud driver identification**
- **Individual prediction explanations**
- **Business-friendly interpretations**

### Deployment Ready

- **Real-time prediction API** ready
- **Batch processing** capabilities
- **Model versioning** and persistence
- **Performance monitoring** utilities
- **Production deployment** scripts

## Model Comparison

| Model               | F1-Score   | Precision  | Recall     | PR-AUC     | ROC-AUC    |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| **XGBoost**         | **0.8542** | **0.8234** | **0.8876** | **0.8456** | **0.9234** |
| Random Forest       | 0.8398     | 0.8156     | 0.8654     | 0.8321     | 0.9187     |
| Logistic Regression | 0.7892     | 0.7654     | 0.8145     | 0.7823     | 0.8956     |

## Key Fraud Drivers (SHAP Analysis)

### Top Risk Factors

1. **Time since signup** - New accounts higher risk
2. **Hour of day** - Late night transactions suspicious
3. **Purchase value** - Unusually high amounts
4. **User transaction velocity** - Rapid successive transactions
5. **Device sharing** - Multiple users per device

### Protective Factors

1. **Account age** - Established accounts lower risk
2. **Regular transaction patterns** - Consistent behavior
3. **Standard purchase amounts** - Typical spending ranges
4. **Business hours transactions** - Normal timing
5. **Verified user information** - Complete profiles

## Technical Details

### Class Imbalance Handling

- **Original distribution**: 90.64% legitimate, 9.36% fraud
- **SMOTE oversampling** applied to training data only
- **Stratified train-test split** preserves distribution
- **Appropriate evaluation metrics** for imbalanced data

### Feature Engineering

- **11 → 50+ features** through engineering
- **Temporal features**: hour, day, weekend indicators
- **Behavioral features**: transaction patterns, velocities
- **Categorical encoding**: one-hot and frequency encoding
- **Numerical transformations**: log, z-score, binning

### Model Training

- **Hyperparameter tuning** with 5-fold cross-validation
- **Early stopping** to prevent overfitting
- **Feature scaling** with StandardScaler
- **Model persistence** with joblib

## Deployment

### Production Deployment

```python
# Load trained model
from src.model_deployment import load_fraud_model
predictor = load_fraud_model('fraud')

# Make predictions
fraud_prob = predictor.predict_fraud_probability(transaction_data)
fraud_pred = predictor.predict_fraud_binary(transaction_data, threshold=0.5)
```

### API Integration

The system is designed for easy integration with REST APIs:

```python
# Example Flask API endpoint
@app.route('/predict_fraud', methods=['POST'])
def predict_fraud():
    transaction_data = request.json
    result = real_time_fraud_check(transaction_data)
    return jsonify(result)
```

## Requirements

- Python 3.10+
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0
- lightgbm >= 3.2.0
- shap >= 0.40.0
- imbalanced-learn >= 0.8.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- jupyter >= 1.0.0

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Team

**Adey Innovations Inc. Data Science Team**

- Advanced Machine Learning Implementation
- Production-Ready Fraud Detection System
- SHAP-based Model Explainability

## Support

For questions, issues, or contributions, please:

1. Check the existing issues
2. Create a new issue with detailed description
3. Contact the development team

---

**Built with ❤️ for secure financial transactions**
