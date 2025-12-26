"""
Dashboard components for fraud detection system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import shap
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')


class FraudDetectionDashboard:
    """Main dashboard component for fraud detection."""
    
    def __init__(self, model_data: Dict[str, Any]):
        self.model_data = model_data
        self.model = model_data.get('model')
        self.scaler = model_data.get('scaler')
        self.feature_names = model_data.get('feature_names', [])
    
    def analyze_transactions(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Analyze transactions for fraud probability."""
        try:
            # Basic preprocessing
            df_processed = self._preprocess_for_prediction(df)
            
            if df_processed is None:
                return None
            
            # Make predictions
            if hasattr(self.model, 'predict_proba'):
                fraud_probabilities = self.model.predict_proba(df_processed)[:, 1]
            else:
                fraud_probabilities = self.model.predict(df_processed)
            
            predictions = (fraud_probabilities > 0.5).astype(int)
            
            # Create results
            results_df = df.copy()
            results_df['fraud_probability'] = fraud_probabilities
            results_df['prediction'] = predictions
            results_df['risk_level'] = self._categorize_risk(fraud_probabilities)
            
            return {
                'results_df': results_df,
                'fraud_count': int(predictions.sum()),
                'total_count': len(predictions),
                'avg_fraud_prob': float(fraud_probabilities.mean()),
                'high_risk_count': int((fraud_probabilities > 0.7).sum())
            }
            
        except Exception as e:
            st.error(f"Error in analysis: {e}")
            return None
    
    def _preprocess_for_prediction(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Preprocess data for model prediction."""
        try:
            # Create basic features if they don't exist
            df_processed = df.copy()
            
            # Handle missing required columns
            required_features = ['purchase_value', 'age']
            for feature in required_features:
                if feature not in df_processed.columns:
                    if feature == 'purchase_value':
                        df_processed[feature] = 100.0  # Default value
                    elif feature == 'age':
                        df_processed[feature] = 30  # Default age
            
            # Encode categorical variables
            categorical_cols = ['source', 'browser', 'sex']
            for col in categorical_cols:
                if col in df_processed.columns:
                    df_processed[col] = pd.Categorical(df_processed[col]).codes
                else:
                    df_processed[col] = 0  # Default category
            
            # Select only numeric columns for prediction
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_numeric = df_processed[numeric_cols]
            
            # Handle any remaining missing values
            df_numeric = df_numeric.fillna(df_numeric.mean())
            
            # Scale if scaler is available
            if self.scaler:
                # Ensure we have the right number of features
                if len(df_numeric.columns) != len(self.feature_names):
                    # Add missing features with default values
                    for feature in self.feature_names:
                        if feature not in df_numeric.columns:
                            df_numeric[feature] = 0
                    
                    # Reorder columns to match training
                    df_numeric = df_numeric[self.feature_names]
                
                df_scaled = pd.DataFrame(
                    self.scaler.transform(df_numeric),
                    columns=df_numeric.columns,
                    index=df_numeric.index
                )
                return df_scaled
            
            return df_numeric
            
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            return None
    
    def _categorize_risk(self, probabilities: np.ndarray) -> List[str]:
        """Categorize fraud probabilities into risk levels."""
        risk_levels = []
        for prob in probabilities:
            if prob < 0.3:
                risk_levels.append("Low")
            elif prob < 0.7:
                risk_levels.append("Medium")
            else:
                risk_levels.append("High")
        return risk_levels
    
    def display_results(self, results: Dict[str, Any]):
        """Display analysis results."""
        results_df = results['results_df']
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Transactions",
                f"{results['total_count']:,}",
                delta=None
            )
        
        with col2:
            fraud_rate = (results['fraud_count'] / results['total_count']) * 100
            st.metric(
                "Fraud Detected",
                f"{results['fraud_count']:,}",
                delta=f"{fraud_rate:.1f}%"
            )
        
        with col3:
            st.metric(
                "Avg Fraud Probability",
                f"{results['avg_fraud_prob']:.1%}",
                delta=None
            )
        
        with col4:
            st.metric(
                "High Risk Transactions",
                f"{results['high_risk_count']:,}",
                delta=f"{(results['high_risk_count']/results['total_count']*100):.1f}%"
            )
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Fraud probability distribution
            fig_hist = px.histogram(
                results_df,
                x='fraud_probability',
                nbins=30,
                title='Fraud Probability Distribution',
                labels={'fraud_probability': 'Fraud Probability', 'count': 'Number of Transactions'}
            )
            fig_hist.add_vline(x=0.5, line_dash="dash", line_color="red", 
                              annotation_text="Decision Threshold")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Risk level pie chart
            risk_counts = results_df['risk_level'].value_counts()
            fig_pie = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title='Risk Level Distribution',
                color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Detailed results table
        st.subheader("📋 Detailed Results")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            risk_filter = st.multiselect(
                "Filter by Risk Level",
                options=['Low', 'Medium', 'High'],
                default=['Medium', 'High']
            )
        
        with col2:
            show_only_fraud = st.checkbox("Show only predicted fraud", value=False)
        
        # Apply filters
        filtered_df = results_df.copy()
        if risk_filter:
            filtered_df = filtered_df[filtered_df['risk_level'].isin(risk_filter)]
        if show_only_fraud:
            filtered_df = filtered_df[filtered_df['prediction'] == 1]
        
        # Display filtered table
        st.dataframe(
            filtered_df.sort_values('fraud_probability', ascending=False),
            use_container_width=True
        )
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="fraud_detection_results.csv",
            mime="text/csv"
        )


class ModelExplainabilityComponent:
    """Component for model explainability features."""
    
    def __init__(self, model_data: Dict[str, Any]):
        self.model_data = model_data
        self.model = model_data.get('model')
        self.feature_names = model_data.get('feature_names', [])
    
    def show_global_explanations(self):
        """Show global model explanations."""
        st.subheader("🌍 Global Feature Importance")
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                # Tree-based model feature importance
                importances = self.model.feature_importances_
                feature_names = self.feature_names[:len(importances)]
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=True)
                
                fig = px.bar(
                    importance_df.tail(15),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Top 15 Most Important Features',
                    labels={'importance': 'Feature Importance', 'feature': 'Features'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance insights
                st.markdown("### 💡 Key Insights")
                top_features = importance_df.tail(5)['feature'].tolist()
                st.write(f"**Top risk factors:** {', '.join(reversed(top_features))}")
                
            else:
                st.info("Feature importance not available for this model type.")
                
        except Exception as e:
            st.error(f"Error generating global explanations: {e}")
    
    def explain_individual_prediction(self, transaction: pd.DataFrame):
        """Explain individual prediction using SHAP or LIME."""
        try:
            # Preprocess transaction
            dashboard = FraudDetectionDashboard(self.model_data)
            processed_transaction = dashboard._preprocess_for_prediction(transaction)
            
            if processed_transaction is None:
                st.error("Failed to process transaction data.")
                return
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                fraud_prob = self.model.predict_proba(processed_transaction)[0, 1]
            else:
                fraud_prob = self.model.predict(processed_transaction)[0]
            
            prediction = "FRAUD" if fraud_prob > 0.5 else "LEGITIMATE"
            
            # Display prediction
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Fraud Probability",
                    f"{fraud_prob:.1%}",
                    delta=None
                )
            
            with col2:
                color = "red" if prediction == "FRAUD" else "green"
                st.markdown(f"**Prediction:** <span style='color: {color}'>{prediction}</span>", 
                           unsafe_allow_html=True)
            
            # Feature contributions (simplified)
            if hasattr(self.model, 'feature_importances_') and len(processed_transaction.columns) > 0:
                feature_values = processed_transaction.iloc[0]
                feature_importances = self.model.feature_importances_[:len(feature_values)]
                
                # Calculate feature contributions (simplified approach)
                contributions = feature_values * feature_importances
                
                contrib_df = pd.DataFrame({
                    'feature': processed_transaction.columns,
                    'value': feature_values.values,
                    'contribution': contributions
                }).sort_values('contribution', key=abs, ascending=False)
                
                # Show top contributing features
                st.subheader("🔍 Feature Contributions")
                
                top_contrib = contrib_df.head(10)
                fig = px.bar(
                    top_contrib,
                    x='contribution',
                    y='feature',
                    orientation='h',
                    title='Top 10 Feature Contributions to Prediction',
                    color='contribution',
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Explanation text
                st.markdown("### 📝 Explanation")
                positive_features = contrib_df[contrib_df['contribution'] > 0].head(3)
                negative_features = contrib_df[contrib_df['contribution'] < 0].head(3)
                
                if len(positive_features) > 0:
                    st.write("**Factors increasing fraud risk:**")
                    for _, row in positive_features.iterrows():
                        st.write(f"- {row['feature']}: {row['value']:.2f}")
                
                if len(negative_features) > 0:
                    st.write("**Factors decreasing fraud risk:**")
                    for _, row in negative_features.iterrows():
                        st.write(f"- {row['feature']}: {row['value']:.2f}")
            
        except Exception as e:
            st.error(f"Error explaining prediction: {e}")


class GeolocationComponent:
    """Component for geolocation visualization."""
    
    def show_fraud_map(self):
        """Show interactive fraud map."""
        st.subheader("🗺️ Global Fraud Activity Map")
        
        # Generate sample geolocation data
        sample_data = self._generate_sample_geo_data()
        
        # Create folium map
        m = folium.Map(location=[20, 0], zoom_start=2)
        
        # Add markers for fraud locations
        for _, row in sample_data.iterrows():
            color = 'red' if row['is_fraud'] else 'green'
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=max(3, row['transaction_count'] / 10),
                popup=f"Country: {row['country']}<br>"
                      f"Transactions: {row['transaction_count']}<br>"
                      f"Fraud Rate: {row['fraud_rate']:.1%}",
                color=color,
                fill=True,
                opacity=0.7
            ).add_to(m)
        
        # Display map
        map_data = st_folium(m, width=700, height=500)
        
        # Legend
        st.markdown("""
        **Legend:**
        - 🔴 Red circles: High fraud rate locations
        - 🟢 Green circles: Low fraud rate locations
        - Circle size: Number of transactions
        """)
    
    def show_country_risk_analysis(self):
        """Show country-wise risk analysis."""
        sample_data = self._generate_sample_geo_data()
        
        # Sort by fraud rate
        risk_data = sample_data.sort_values('fraud_rate', ascending=False)
        
        fig = px.bar(
            risk_data.head(10),
            x='fraud_rate',
            y='country',
            orientation='h',
            title='Top 10 Countries by Fraud Rate',
            labels={'fraud_rate': 'Fraud Rate (%)', 'country': 'Country'},
            color='fraud_rate',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def show_location_patterns(self):
        """Show location-based fraud patterns."""
        sample_data = self._generate_sample_geo_data()
        
        # Transaction volume vs fraud rate scatter plot
        fig = px.scatter(
            sample_data,
            x='transaction_count',
            y='fraud_rate',
            size='transaction_count',
            color='fraud_rate',
            hover_data=['country'],
            title='Transaction Volume vs Fraud Rate by Location',
            labels={
                'transaction_count': 'Number of Transactions',
                'fraud_rate': 'Fraud Rate (%)'
            },
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _generate_sample_geo_data(self) -> pd.DataFrame:
        """Generate sample geolocation data for demonstration."""
        np.random.seed(42)
        
        countries = [
            ('United States', 39.8283, -98.5795),
            ('United Kingdom', 55.3781, -3.4360),
            ('Germany', 51.1657, 10.4515),
            ('France', 46.2276, 2.2137),
            ('Canada', 56.1304, -106.3468),
            ('Australia', -25.2744, 133.7751),
            ('Japan', 36.2048, 138.2529),
            ('Brazil', -14.2350, -51.9253),
            ('India', 20.5937, 78.9629),
            ('China', 35.8617, 104.1954)
        ]
        
        data = []
        for country, lat, lon in countries:
            transaction_count = np.random.randint(100, 10000)
            fraud_rate = np.random.uniform(0.001, 0.05)  # 0.1% to 5% fraud rate
            is_fraud = fraud_rate > 0.02  # High fraud if > 2%
            
            data.append({
                'country': country,
                'latitude': lat,
                'longitude': lon,
                'transaction_count': transaction_count,
                'fraud_rate': fraud_rate,
                'is_fraud': is_fraud
            })
        
        return pd.DataFrame(data)
