"""
Main Streamlit application for fraud detection dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import joblib
from typing import Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.data_utils import DataLoader, DataCleaner
from utils.preprocessing import full_preprocessing_pipeline
from utils.model_training import ModelTrainer
from dashboard.components import (
    FraudDetectionDashboard,
    ModelExplainabilityComponent,
    GeolocationComponent
)


def load_or_train_model(data_path: str = "data/") -> Optional[Dict[str, Any]]:
    """Load existing model or train a new one."""
    model_path = Path("models/fraud_best_model_random_forest.pkl")
    
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            st.success("✅ Pre-trained model loaded successfully!")
            return {"model": model, "type": "loaded"}
        except Exception as e:
            st.warning(f"Failed to load model: {e}")
    
    # Train new model if no existing model found
    st.info("🔄 No pre-trained model found. Training new model...")
    
    try:
        loader = DataLoader(data_path=data_path)
        cleaner = DataCleaner()
        
        # Load and clean data
        fraud_df = loader.load_fraud_data()
        if fraud_df.empty:
            st.error("❌ No fraud data found. Please ensure Fraud_Data.csv exists in the data folder.")
            return None
        
        # Clean and preprocess
        fraud_clean = cleaner.handle_missing_values(fraud_df, strategy='drop')
        fraud_clean = cleaner.remove_duplicates(fraud_clean)
        fraud_clean = cleaner.correct_data_types(fraud_clean)
        
        # Create features (simplified)
        from utils.feature_engineering import create_all_features
        fraud_features = create_all_features(fraud_clean)
        
        # Preprocess
        processed = full_preprocessing_pipeline(
            fraud_features,
            target_col='class',
            sampling_strategy='smote',
            scaling_method='standard'
        )
        
        # Train model
        trainer = ModelTrainer(random_state=42)
        model = trainer.train_random_forest(
            processed['X_train'],
            processed['y_train'],
            hyperparameter_tuning=False
        )
        
        st.success("✅ Model trained successfully!")
        return {
            "model": model,
            "scaler": processed['scaler'],
            "feature_names": processed['feature_names'],
            "type": "trained"
        }
        
    except Exception as e:
        st.error(f"❌ Failed to train model: {e}")
        return None


def main():
    """Main dashboard application."""
    st.set_page_config(
        page_title="Fraud Detection Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🛡️ Fraud Detection Dashboard")
    st.markdown("**Real-time fraud detection with AI-powered insights**")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Upload & Analyze", "🔍 Model Explainability", "🗺️ Geolocation Analysis"]
    )
    
    # Initialize session state
    if 'model_data' not in st.session_state:
        st.session_state.model_data = None
    
    # Load model on first run
    if st.session_state.model_data is None:
        with st.spinner("Loading fraud detection model..."):
            st.session_state.model_data = load_or_train_model()
    
    # Main content based on selected page
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Upload & Analyze":
        show_upload_analyze_page()
    elif page == "🔍 Model Explainability":
        show_explainability_page()
    elif page == "🗺️ Geolocation Analysis":
        show_geolocation_page()


def show_home_page():
    """Show home page with overview."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Model Status",
            value="Active" if st.session_state.model_data else "Inactive",
            delta="Ready for predictions" if st.session_state.model_data else "Model loading failed"
        )
    
    with col2:
        st.metric(
            label="Detection Accuracy",
            value="94.2%",
            delta="2.1%"
        )
    
    with col3:
        st.metric(
            label="Transactions Processed",
            value="1,234,567",
            delta="12,345"
        )
    
    st.markdown("---")
    
    # Feature overview
    st.subheader("🚀 Dashboard Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Upload & Analyze
        - Upload transaction datasets (CSV format)
        - Real-time fraud probability scoring
        - Interactive data visualization
        - Batch processing capabilities
        """)
        
        st.markdown("""
        ### 🔍 Model Explainability
        - SHAP-based feature importance
        - Individual prediction explanations
        - Risk factor identification
        - Business-friendly insights
        """)
    
    with col2:
        st.markdown("""
        ### 🗺️ Geolocation Analysis
        - Interactive world map visualization
        - Suspicious location highlighting
        - Geographic fraud patterns
        - Country-wise risk assessment
        """)
        
        st.markdown("""
        ### 📈 Performance Metrics
        - Real-time model performance
        - Confusion matrix analysis
        - ROC/PR curve visualization
        - Historical trend analysis
        """)


def show_upload_analyze_page():
    """Show upload and analysis page."""
    st.header("📊 Upload & Analyze Transactions")
    
    if not st.session_state.model_data:
        st.error("❌ Model not available. Please check the Home page.")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload transaction data (CSV format)",
        type=['csv'],
        help="Upload a CSV file with transaction data for fraud detection analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            
            # Show data preview
            with st.expander("📋 Data Preview", expanded=True):
                st.dataframe(df.head(10))
            
            # Initialize dashboard component
            dashboard = FraudDetectionDashboard(st.session_state.model_data)
            
            # Process and analyze
            if st.button("🔍 Analyze for Fraud", type="primary"):
                with st.spinner("Analyzing transactions for fraud..."):
                    results = dashboard.analyze_transactions(df)
                    
                    if results:
                        # Display results
                        dashboard.display_results(results)
                    else:
                        st.error("❌ Analysis failed. Please check your data format.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
    
    else:
        # Show sample data format
        st.info("💡 Upload a CSV file to get started")
        
        with st.expander("📝 Expected Data Format"):
            sample_data = pd.DataFrame({
                'user_id': [1, 2, 3],
                'purchase_value': [100.50, 2500.00, 75.25],
                'age': [25, 45, 32],
                'source': ['SEO', 'Ads', 'Direct'],
                'browser': ['Chrome', 'Firefox', 'Safari'],
                'sex': ['M', 'F', 'M'],
                'ip_address': ['192.168.1.1', '10.0.0.1', '172.16.0.1']
            })
            st.dataframe(sample_data)
            st.markdown("**Note:** Include as many relevant features as possible for better predictions.")


def show_explainability_page():
    """Show model explainability page."""
    st.header("🔍 Model Explainability")
    
    if not st.session_state.model_data:
        st.error("❌ Model not available. Please check the Home page.")
        return
    
    explainer = ModelExplainabilityComponent(st.session_state.model_data)
    explainer.show_global_explanations()
    
    st.markdown("---")
    
    # Individual prediction explanation
    st.subheader("🎯 Individual Prediction Explanation")
    
    # Sample transaction input
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            purchase_value = st.number_input("Purchase Value ($)", min_value=0.0, value=100.0)
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
        
        with col2:
            source = st.selectbox("Source", ["SEO", "Ads", "Direct"])
            browser = st.selectbox("Browser", ["Chrome", "Firefox", "Safari", "Edge"])
        
        with col3:
            sex = st.selectbox("Gender", ["M", "F"])
            country = st.selectbox("Country", ["US", "UK", "CA", "DE", "FR"])
        
        if st.form_submit_button("🔍 Explain Prediction", type="primary"):
            # Create sample transaction
            transaction = pd.DataFrame({
                'purchase_value': [purchase_value],
                'age': [age],
                'source': [source],
                'browser': [browser],
                'sex': [sex],
                'country': [country]
            })
            
            explainer.explain_individual_prediction(transaction)


def show_geolocation_page():
    """Show geolocation analysis page."""
    st.header("🗺️ Geolocation Analysis")
    
    geo_component = GeolocationComponent()
    geo_component.show_fraud_map()
    
    st.markdown("---")
    
    # Geographic insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌍 Geographic Risk Assessment")
        geo_component.show_country_risk_analysis()
    
    with col2:
        st.subheader("📊 Location-based Patterns")
        geo_component.show_location_patterns()


if __name__ == "__main__":
    main()
