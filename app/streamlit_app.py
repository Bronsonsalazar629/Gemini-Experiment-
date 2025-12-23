"""
Clinical Fairness Intervention System - Streamlit Demo UI

Interactive web interface for fairness analysis and intervention generation.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import networkx as nx

# Import our modules
from causal_analysis import CausalAnalyzer
from bias_detection import BiasDetector
from intervention_engine import InterventionEngine
from code_generator import CodeGenerator

# Page config
st.set_page_config(
    page_title="Clinical Fairness Intervention System",
    page_icon="scales",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Session state initialization
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'bias_metrics' not in st.session_state:
    st.session_state.bias_metrics = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None


def load_data(uploaded_file):
    """Load and cache data."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        # Load default demo data
        default_path = Path(__file__).parent.parent / "data" / "sample" / "demo_data.csv"
        if default_path.exists():
            return pd.read_csv(default_path)
    return None


def train_demo_model(data, feature_cols, target_col='referral'):
    """Train a simple model for demonstration."""
    # Encode categorical features
    data_encoded = data.copy()

    encoders = {}
    for col in ['race', 'gender', 'insurance_type']:
        if col in data.columns:
            le = LabelEncoder()
            data_encoded[f'{col}_encoded'] = le.fit_transform(data[col])
            encoders[col] = le

    # Prepare features
    feature_cols_encoded = [
        'age', 'race_encoded', 'gender_encoded', 'creatinine_level',
        'chronic_conditions', 'insurance_type_encoded', 'prior_visits',
        'distance_to_hospital'
    ]

    X = data_encoded[feature_cols_encoded + ['race']]  # Include race for analysis
    y = data_encoded[target_col]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train[feature_cols_encoded], y_train)

    return model, X_train, X_test, y_train, y_test, encoders


def plot_fairness_metrics(metrics, sensitive_attr):
    """Create visualizations for fairness metrics."""
    group_metrics = metrics['group_metrics']

    # Extract data for plotting
    groups = list(group_metrics.keys())
    positive_rates = [group_metrics[g]['positive_rate'] for g in groups]
    accuracies = [group_metrics[g]['accuracy'] for g in groups]
    tprs = [group_metrics[g]['true_positive_rate'] for g in groups]
    fprs = [group_metrics[g]['false_positive_rate'] for g in groups]

    # Create subplots
    col1, col2 = st.columns(2)

    with col1:
        # Positive rate by group
        fig1 = go.Figure(data=[
            go.Bar(x=groups, y=positive_rates, marker_color='#1f77b4')
        ])
        fig1.update_layout(
            title="Positive Prediction Rate by Group",
            xaxis_title=sensitive_attr.capitalize(),
            yaxis_title="Positive Rate",
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Accuracy by group
        fig2 = go.Figure(data=[
            go.Bar(x=groups, y=accuracies, marker_color='#2ca02c')
        ])
        fig2.update_layout(
            title="Accuracy by Group",
            xaxis_title=sensitive_attr.capitalize(),
            yaxis_title="Accuracy",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

    # TPR vs FPR comparison
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name='TPR', x=groups, y=tprs, marker_color='#ff7f0e'))
    fig3.add_trace(go.Bar(name='FPR', x=groups, y=fprs, marker_color='#d62728'))
    fig3.update_layout(
        title="True Positive Rate vs False Positive Rate by Group",
        xaxis_title=sensitive_attr.capitalize(),
        yaxis_title="Rate",
        barmode='group',
        height=400
    )
    st.plotly_chart(fig3, use_container_width=True)


def display_fairness_summary(metrics):
    """Display summary of fairness criteria."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        dp_diff = metrics['demographic_parity']['difference']
        status = "PASS" if dp_diff < 0.1 else "FAIL"
        st.metric(
            "Demographic Parity",
            f"{dp_diff:.3f}",
            delta=status,
            delta_color="normal" if dp_diff < 0.1 else "inverse"
        )

    with col2:
        eo_diff = metrics['equalized_odds']['average_difference']
        status = "PASS" if eo_diff < 0.1 else "FAIL"
        st.metric(
            "Equalized Odds",
            f"{eo_diff:.3f}",
            delta=status,
            delta_color="normal" if eo_diff < 0.1 else "inverse"
        )

    with col3:
        eop_diff = metrics['equal_opportunity']['difference']
        status = "PASS" if eop_diff < 0.1 else "FAIL"
        st.metric(
            "Equal Opportunity",
            f"{eop_diff:.3f}",
            delta=status,
            delta_color="normal" if eop_diff < 0.1 else "inverse"
        )

    with col4:
        acc = metrics['overall_accuracy']
        st.metric(
            "Overall Accuracy",
            f"{acc:.3f}",
            delta="Model Performance"
        )


def main():
    # Header
    st.markdown('<div class="main-header">Clinical Fairness Intervention System</div>', unsafe_allow_html=True)
    st.markdown("*Detect and remediate bias in clinical ML models using causal inference and automated interventions*")

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        # Data upload
        st.subheader("1. Data")
        uploaded_file = st.file_uploader("Upload Clinical Data (CSV)", type=['csv'])

        if uploaded_file is None:
            st.info("Using default demo dataset (MIMIC-style)")

        # Load data
        data = load_data(uploaded_file)
        if data is not None:
            st.session_state.data = data
            st.success(f"Loaded {len(data)} records")

        # Settings
        st.subheader("2. Settings")
        if st.session_state.data is not None:
            sensitive_attr = st.selectbox(
                "Protected Attribute",
                options=['race', 'gender'],
                index=0
            )
            outcome = st.selectbox(
                "Outcome Variable",
                options=['referral'],
                index=0
            )
        else:
            sensitive_attr = 'race'
            outcome = 'referral'

        # API Configuration
        st.subheader("3. LLM Configuration")
        use_live_api = st.checkbox("Use Live Gemini API", value=False,
                                   help="Uncheck to use cached demo (recommended for presentations)")

        if use_live_api:
            api_key = st.text_input("Gemini API Key", type="password",
                                   value="AIzaSyBjUWfo4dP2KnO9wTgVgkLK1Bqdl5Y3D4E")
            st.warning("Live API enabled - may have latency during demo")
        else:
            api_key = None
            st.success("Using cached results for reliable demo")

    # Main content
    if st.session_state.data is None:
        st.warning("Please upload data or use the default dataset")
        return

    data = st.session_state.data

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Data Overview",
        "Bias Analysis",
        "Interventions",
        "Generated Code"
    ])

    # Tab 1: Data Overview
    with tab1:
        st.header("Data Overview")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Features", len(data.columns))
        with col3:
            st.metric("Outcome Rate", f"{data[outcome].mean():.1%}")

        st.subheader("Sample Data")
        st.dataframe(data.head(20), use_container_width=True)

        st.subheader("Data Distribution")
        col1, col2 = st.columns(2)

        with col1:
            # Distribution by protected attribute
            fig = px.histogram(data, x=sensitive_attr, color=outcome,
                             title=f"Outcome Distribution by {sensitive_attr.capitalize()}")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Age distribution
            fig = px.histogram(data, x='age', color=outcome,
                             title="Outcome Distribution by Age")
            st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Bias Analysis
    with tab2:
        st.header("Bias Detection & Causal Analysis")

        if st.button("Run Analysis", type="primary", use_container_width=True):
            with st.spinner("Training model and analyzing bias..."):
                # Train model
                feature_cols = [
                    'age', 'race_encoded', 'gender_encoded', 'creatinine_level',
                    'chronic_conditions', 'insurance_type_encoded', 'prior_visits',
                    'distance_to_hospital'
                ]

                model, X_train, X_test, y_train, y_test, encoders = train_demo_model(
                    data, feature_cols, outcome
                )
                st.session_state.model = model

                # Bias detection
                detector = BiasDetector([sensitive_attr])
                metrics = detector.compute_fairness_metrics(
                    model, X_test, y_test, sensitive_attr
                )
                st.session_state.bias_metrics = metrics

                # Causal analysis
                analyzer = CausalAnalyzer(data, sensitive_attr, outcome)
                graph_result = analyzer.infer_causal_graph(
                    use_cache=not use_live_api,
                    llm_enhanced=use_live_api
                )

                # Unpack dict result
                graph = graph_result['graph']
                explanation = graph_result['llm_explanation']

                # Store in session
                st.session_state.causal_graph = graph
                st.session_state.causal_explanation = explanation

                st.success("Analysis complete!")

        # Display results
        if st.session_state.bias_metrics is not None:
            st.subheader("Fairness Metrics Summary")
            display_fairness_summary(st.session_state.bias_metrics)

            st.subheader("Detailed Metrics by Group")
            plot_fairness_metrics(st.session_state.bias_metrics, sensitive_attr)

            # Show detailed report
            with st.expander("View Detailed Bias Report"):
                detector = BiasDetector([sensitive_attr])
                report = detector.generate_bias_report(
                    st.session_state.bias_metrics,
                    sensitive_attr
                )
                st.code(report)

    # Tab 3: Interventions
    with tab3:
        st.header("Fairness Interventions")

        if st.session_state.bias_metrics is None:
            st.warning("Please run bias analysis first")
        else:
            if st.button("Generate Recommendations", type="primary", use_container_width=True):
                with st.spinner("Generating intervention recommendations..."):
                    engine = InterventionEngine()
                    recommendations = engine.suggest_interventions(
                        st.session_state.bias_metrics,
                        max_recommendations=5
                    )
                    st.session_state.recommendations = recommendations

            if st.session_state.recommendations:
                st.subheader("Recommended Interventions")

                for rec in st.session_state.recommendations:
                    with st.expander(f"#{rec.priority}: {rec.name} ({rec.category})", expanded=(rec.priority == 1)):
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.markdown(f"**Description:** {rec.description}")
                            st.markdown(f"**Expected Impact:** {rec.expected_impact}")

                        with col2:
                            st.markdown(f"**Complexity:** {rec.complexity}")
                            st.markdown(f"**Preserves Accuracy:** {'Yes' if rec.preserves_accuracy else 'No'}")

                        if rec.clinical_rationale:
                            st.markdown(f"**Clinical Rationale:** {rec.clinical_rationale}")

                        st.markdown("**Parameters:**")
                        st.json(rec.parameters)

    # Tab 4: Generated Code
    with tab4:
        st.header("Generated Intervention Code")

        if st.session_state.recommendations:
            selected_intervention = st.selectbox(
                "Select Intervention",
                options=[rec.name for rec in st.session_state.recommendations],
                index=0
            )

            if st.button("Generate Code", type="primary", use_container_width=True):
                with st.spinner(f"Generating code for {selected_intervention}..."):
                    generator = CodeGenerator(api_key=api_key if use_live_api else None)
                    code_result = generator.generate_fix_code(selected_intervention)

                    st.success("Code generated!")

                    st.subheader("Required Imports")
                    st.code("\n".join(code_result.imports), language="python")

                    st.subheader("Implementation Code")
                    st.code(code_result.code, language="python")

                    st.subheader("Usage Example")
                    st.code(code_result.usage_example, language="python")

                    st.info(f"Estimated Runtime: {code_result.estimated_runtime}")

                    # Download button
                    full_code = "\n".join(code_result.imports) + "\n\n" + code_result.code + "\n\n# Usage Example:\n" + code_result.usage_example
                    st.download_button(
                        label="Download Code",
                        data=full_code,
                        file_name=f"{selected_intervention.replace(' ', '_').lower()}_intervention.py",
                        mime="text/x-python"
                    )
        else:
            st.info("Generate intervention recommendations first")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Clinical Fairness Intervention System | Built for Healthcare AI Hackathon | "
        "Powered by Gemini 3, AIF360, and Fairlearn"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
