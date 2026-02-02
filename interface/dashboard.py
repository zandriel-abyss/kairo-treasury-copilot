#!/usr/bin/env python3
"""
Kairo Treasury Optimization Dashboard

Interactive Streamlit dashboard for the Kairo Treasury Copilot.
Provides visualization and interactive analysis of treasury optimization recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add modules directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.data_ingest import get_data_loader
from modules.recommendation_engine import create_recommendation_engine, PaymentScenario
from modules.fx_model import create_fx_predictor
from modules.behavior_forecast import create_behavior_forecaster
from modules.netting_optimizer import create_netting_optimizer
from modules.simulation_engine import create_treasury_simulator, DEFAULT_SCENARIOS

# Configure page
st.set_page_config(
    page_title="Kairo Treasury Copilot",
    page_icon="💰",
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
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-high {
        background-color: #d4edda;
        border-left-color: #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-medium {
        background-color: #fff3cd;
        border-left-color: #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class KairoDashboard:
    """Streamlit dashboard for Kairo Treasury Copilot."""

    def __init__(self):
        """Initialize the dashboard with data and models."""
        self.initialize_session_state()
        self.load_data_and_models()

    def initialize_session_state(self):
        """Initialize Streamlit session state."""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.selected_payment = None
            st.session_state.portfolio_forecast = None

    def load_data_and_models(self):
        """Load data and initialize models."""
        try:
            with st.spinner("🔄 Initializing Kairo Treasury Copilot..."):
                # Load data
                self.data_loader = get_data_loader()
                self.ap_data = self.data_loader.load_ap_data()
                self.ar_data = self.data_loader.load_ar_data()
                self.fx_data = self.data_loader.load_fx_data()

                # Initialize models
                self.recommendation_engine = create_recommendation_engine(
                    self.fx_data, self.ap_data, self.ar_data
                )
                self.fx_predictor = create_fx_predictor(self.fx_data)
                self.behavior_forecaster = create_behavior_forecaster(self.ap_data, self.ar_data)
                self.netting_optimizer = create_netting_optimizer(self.ap_data, self.ar_data)
                self.treasury_simulator = create_treasury_simulator(
                    self.ap_data, self.ar_data, self.fx_data
                )

                st.session_state.initialized = True

        except Exception as e:
            st.error(f"❌ Failed to initialize Kairo: {str(e)}")
            st.stop()

    def render_sidebar(self):
        """Render the sidebar with navigation."""
        st.sidebar.title("🎯 Kairo Treasury Copilot")
        st.sidebar.markdown("---")

        # Navigation
        page = st.sidebar.radio(
            "Navigation",
            ["🏠 Dashboard", "💱 FX Analysis", "👥 Behavior Insights",
             "🔄 Netting Opportunities", "💡 Recommendations", "🎲 Simulation"]
        )

        st.sidebar.markdown("---")

        # Quick stats
        st.sidebar.subheader("📊 Quick Stats")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("AP Transactions", f"{len(self.ap_data):,}")
        with col2:
            st.metric("AR Transactions", f"{len(self.ar_data):,}")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            total_ap = self.ap_data['amount'].sum()
            st.metric("Total Payables", f"${total_ap:,.0f}")
        with col2:
            total_ar = self.ar_data['amount'].sum()
            st.metric("Total Receivables", f"${total_ar:,.0f}")

        return page

    def render_dashboard(self):
        """Render the main dashboard page."""
        st.markdown('<h1 class="main-header">🏠 Kairo Treasury Dashboard</h1>', unsafe_allow_html=True)
        st.markdown("AI-powered treasury optimization for smarter payment timing decisions")

        # Check if system is initialized
        if not hasattr(self, 'ap_data') or self.ap_data is None:
            st.warning("⚠️ System initializing... Please wait a moment and refresh if needed.")
            return

        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.subheader("💰 Portfolio Health")
            net_position = self.ar_data['amount'].sum() - self.ap_data['amount'].sum()
            st.metric("Net Position", f"${net_position:,.0f}", delta=f"{net_position/len(self.ap_data)*100:.1f}% per transaction")

        with col2:
            st.subheader("🎯 Upcoming Payments (30d)")
            upcoming = self.data_loader.get_upcoming_payments(30)
            st.metric("Payments Due", len(upcoming), f"${upcoming['amount'].sum():,.0f} total")

        with col3:
            st.subheader("📈 FX Prediction Accuracy")
            # Show recent FX model performance
            perf = self.fx_predictor.get_historical_performance('USD/EUR', 7)
            accuracy = perf.get('hit_rate', 0.5) * 100
            st.metric("Model Accuracy", f"{accuracy:.1f}%", "directional")

        with col4:
            st.subheader("🔄 Netting Potential")
            netting = self.netting_optimizer.analyze_netting_opportunities(30)
            efficiency = netting.get('netting_efficiency', 0) * 100
            st.metric("Netting Efficiency", f"{efficiency:.1f}%")

        # Charts row
        col1, col2 = st.columns(2)

        with col1:
            self.render_payment_timeline_chart()

        with col2:
            self.render_fx_rates_chart()

        # Recent recommendations
        st.subheader("💡 Recent Recommendations")
        self.render_recent_recommendations()

    def render_payment_timeline_chart(self):
        """Render payment timeline visualization."""
        st.subheader("📅 Payment Timeline (Next 30 Days)")

        # Get upcoming payments and receipts
        upcoming_payments = self.data_loader.get_upcoming_payments(30)
        upcoming_receipts = self.data_loader.get_expected_receipts(30)

        if len(upcoming_payments) == 0 and len(upcoming_receipts) == 0:
            st.info("No upcoming transactions in the next 30 days")
            return

        # Create timeline data
        timeline_data = []

        for _, payment in upcoming_payments.iterrows():
            timeline_data.append({
                'date': payment['due_date'],
                'amount': payment['amount'],
                'type': 'Payment',
                'counterparty': payment['vendor'],
                'currency': payment['currency']
            })

        for _, receipt in upcoming_receipts.iterrows():
            timeline_data.append({
                'date': receipt['expected_payment_date'],
                'amount': receipt['amount'],
                'type': 'Receipt',
                'counterparty': receipt['customer'],
                'currency': receipt['currency']
            })

        if timeline_data:
            df = pd.DataFrame(timeline_data)

            # Group by week and type
            df['week'] = df['date'].dt.to_period('W').dt.start_time
            weekly = df.groupby(['week', 'type'])['amount'].sum().reset_index()

            fig = px.bar(weekly, x='week', y='amount', color='type',
                        title="Weekly Cash Flow Forecast",
                        labels={'amount': 'Amount ($)', 'week': 'Week'},
                        barmode='group')

            fig.update_layout(height=300)
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No transaction data available for timeline")

    def render_fx_rates_chart(self):
        """Render FX rates visualization."""
        st.subheader("💱 FX Rate Trends")

        # Get available currency pairs
        currency_pairs = self.fx_data['currency_pair'].unique()

        selected_pair = st.selectbox("Select Currency Pair", currency_pairs,
                                   key="fx_pair_select")

        if selected_pair:
            pair_data = self.fx_data[self.fx_data['currency_pair'] == selected_pair].copy()
            pair_data = pair_data.sort_values('date')

            # Create subplot with rate and volatility
            fig = make_subplots(rows=2, cols=1,
                              subplot_titles=['Exchange Rate', 'Volatility (30-day)'],
                              shared_xaxes=True)

            # Rate chart
            fig.add_trace(
                go.Scatter(x=pair_data['date'], y=pair_data['rate'],
                          mode='lines', name='Rate'),
                row=1, col=1
            )

            # Volatility chart
            fig.add_trace(
                go.Scatter(x=pair_data['date'], y=pair_data['volatility_30d'],
                          mode='lines', name='Volatility', line=dict(color='orange')),
                row=2, col=1
            )

            fig.update_layout(height=400, showlegend=False)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Rate", row=1, col=1)
            fig.update_yaxes(title_text="Volatility", row=2, col=1)

            st.plotly_chart(fig, width='stretch')

    def render_recent_recommendations(self):
        """Render recent recommendation cards."""
        # Get a few sample recommendations
        try:
            recommendations = self.recommendation_engine.get_portfolio_recommendations(7)[:3]

            if recommendations:
                for rec in recommendations:
                    confidence_class = "confidence-high" if rec.confidence_score > 0.8 else "confidence-medium" if rec.confidence_score > 0.6 else "confidence-low"

                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>{rec.scenario.invoice_id}</strong> - {rec.scenario.vendor}<br>
                        <span class="{confidence_class}">Confidence: {rec.confidence_score:.1%}</span><br>
                        <small>{rec.recommended_action}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recent recommendations available")

        except Exception as e:
            st.error(f"Could not load recommendations: {str(e)}")

    def render_fx_analysis_page(self):
        """Render FX analysis page."""
        st.markdown('<h1 class="main-header">💱 FX Rate Analysis</h1>', unsafe_allow_html=True)

        # Check if system is initialized
        if not hasattr(self, 'fx_data') or self.fx_data is None:
            st.error("❌ System not initialized. Please refresh the page.")
            return

        try:
            currency_pairs = self.fx_data['currency_pair'].unique()
            if len(currency_pairs) == 0:
                st.error("❌ No currency pairs available.")
                return
        except Exception as e:
            st.error(f"❌ Error loading currency pairs: {str(e)}")
            return

        col1, col2 = st.columns([2, 1])

        with col1:
            selected_pair = st.selectbox("Currency Pair", currency_pairs, key="fx_analysis_pair")

            days_ahead = st.slider("Prediction Horizon (days)", 1, 30, 7)
            include_shap = st.checkbox("Include SHAP Explainability Analysis", value=False)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔮 Generate FX Prediction"):
                    with st.spinner("Analyzing FX trends..." if not include_shap else "Analyzing FX trends with SHAP explainability..."):
                        try:
                            prediction = self.fx_predictor.predict_fx_rate(
                                selected_pair, datetime.now(), days_ahead=days_ahead,
                                include_shap=include_shap
                            )

                            # Display prediction results
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Current Rate", f"{prediction['current_rate']:.4f}")
                            col2.metric("Predicted Rate", f"{prediction['predicted_rate']:.4f}",
                                      f"{prediction['rate_change_pct']:+.2f}%")
                            col3.metric("Confidence", f"{prediction['prediction_confidence']:.1%}")

                            # Recommendation
                            if prediction['is_delay_favorable']:
                                st.success("💡 Delay payments favorable for this currency")
                            else:
                                st.warning("⚠️ Pay on schedule - delay not recommended")

                            # Reasoning
                            st.subheader("📋 Analysis Details")
                            st.info(prediction['reasoning'])

                            # SHAP Analysis
                            if include_shap and 'shap_analysis' in prediction:
                                shap_data = prediction['shap_analysis']
                                if 'error' not in shap_data:
                                    st.subheader("🔍 SHAP Explainability Analysis")
                                    st.info(shap_data['shap_explanation'])

                                    # Feature importance
                                    st.subheader("🎯 Key Factors")
                                    top_features = shap_data['top_features'][:5]

                                    for feature_name, shap_value in top_features:
                                        impact = "📈 Pushing upward" if shap_value > 0 else "📉 Pulling downward"
                                        strength = "Strong" if abs(shap_value) > 0.01 else "Moderate"
                                        readable_name = self._get_readable_feature_name(feature_name)

                                        col1, col2, col3 = st.columns([2, 1, 1])
                                        col1.write(f"**{readable_name}**")
                                        col2.write(f"{strength}")
                                        col3.write(f"{impact}")

                                    st.metric("SHAP Confidence", f"{shap_data['shap_confidence']:.1%}")
                                else:
                                    st.warning(f"SHAP analysis not available: {shap_data['error']}")

                        except Exception as e:
                            st.error(f"Prediction failed: {str(e)}")

            with col2:
                if st.button("📊 Show SHAP Feature Importance"):
                    with st.spinner("Calculating global feature importance..."):
                        try:
                            importance_data = self.fx_predictor.get_shap_feature_importance(
                                selected_pair, 'random_forest'
                            )

                            if 'error' not in importance_data:
                                st.subheader("🌍 Global Feature Importance")

                                # Create bar chart of feature importance
                                top_features = importance_data['top_features'][:10]
                                features = [self._get_readable_feature_name(f[0]) for f in top_features]
                                importance_values = [f[1] for f in top_features]

                                fig = px.bar(
                                    x=importance_values,
                                    y=features,
                                    orientation='h',
                                    title=f"Feature Importance for {selected_pair}",
                                    labels={'x': 'Importance', 'y': 'Feature'}
                                )
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, width='stretch')

                                # Summary stats
                                st.subheader("📈 Summary Statistics")
                                col1, col2 = st.columns(2)
                                col1.metric("Top Feature", f"{features[0]}")
                                col2.metric("Top Importance", f"{importance_values[0]:.3f}")

                            else:
                                st.error(f"Could not calculate feature importance: {importance_data['error']}")

                        except Exception as e:
                            st.error(f"Feature importance analysis failed: {str(e)}")

        with col2:
            st.subheader("📊 Model Performance")
            perf = self.fx_predictor.get_historical_performance(selected_pair, 7)
            if 'error' not in perf:
                st.metric("Directional Accuracy", f"{perf['hit_rate']:.1%}")
                st.metric("Avg Volatility", f"{perf['avg_volatility']:.1%}")
            else:
                st.info("Performance data not available")

    def render_behavior_insights_page(self):
        """Render behavior analysis page."""
        st.markdown('<h1 class="main-header">👥 Payment Behavior Insights</h1>', unsafe_allow_html=True)

        # Check if system is initialized
        if not hasattr(self, 'behavior_forecaster') or self.behavior_forecaster is None:
            st.error("❌ System not initialized. Please refresh the page.")
            return

        tab1, tab2 = st.tabs(["Payment Behavior", "Collection Behavior"])

        with tab1:
            self.render_payment_behavior_analysis()

        with tab2:
            self.render_collection_behavior_analysis()

    def render_payment_behavior_analysis(self):
        """Render payment behavior analysis."""
        st.subheader("📈 Payment Behavior Patterns")

        # Get summary statistics
        summary = self.behavior_forecaster.analyze_payment_patterns()

        if 'error' in summary:
            st.error(summary['error'])
            return

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Payment Lag", f"{summary['avg_payment_lag_days']:.1f} days")
        col2.metric("Early Payment Rate", f"{summary['early_payment_rate']:.1%}")
        col3.metric("Late Payment Rate", f"{summary['late_payment_rate']:.1%}")
        col4.metric("Predictability", f"{summary['predictability_score']:.1%}")

        # Insights
        st.subheader("🔍 Key Insights")
        for insight in summary['insights']:
            st.write(f"• {insight}")

        # Monthly patterns
        st.subheader("📅 Monthly Payment Patterns")
        monthly_data = summary['monthly_patterns']
        if monthly_data:
            df = pd.DataFrame(monthly_data).T
            df.index.name = 'Month'
            df = df.reset_index()

            fig = px.bar(df, x='Month', y='mean', title="Average Payment Lag by Month",
                        labels={'mean': 'Avg Lag (days)'})
            st.plotly_chart(fig, width='stretch')

    def render_collection_behavior_analysis(self):
        """Render collection behavior analysis."""
        st.subheader("📈 Collection Behavior Patterns")

        # Get summary statistics
        summary = self.behavior_forecaster.analyze_collection_patterns()

        if 'error' in summary:
            st.error(summary['error'])
            return

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Collection Lag", f"{summary['avg_collection_lag_days']:.1f} days")
        col2.metric("Early Collection Rate", f"{summary['early_collection_rate']:.1%}")
        col3.metric("Late Collection Rate", f"{summary['late_collection_rate']:.1%}")
        col4.metric("Predictability", f"{summary['predictability_score']:.1%}")

        # Insights
        st.subheader("🔍 Key Insights")
        for insight in summary['insights']:
            st.write(f"• {insight}")

    def render_netting_page(self):
        """Render netting opportunities page."""
        st.markdown('<h1 class="main-header">🔄 Netting Opportunities</h1>', unsafe_allow_html=True)

        # Check if system is initialized
        if not hasattr(self, 'netting_optimizer') or self.netting_optimizer is None:
            st.error("❌ System not initialized. Please refresh the page.")
            return

        days_ahead = st.slider("Analysis Period (days)", 7, 90, 30)

        if st.button("🔍 Analyze Netting Opportunities"):
            with st.spinner("Analyzing netting opportunities..."):
                try:
                    analysis = self.netting_optimizer.analyze_netting_opportunities(days_ahead)

                    # Key metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Netting Efficiency", f"{analysis['netting_efficiency']:.1%}")
                    col2.metric("Offset Potential", f"${analysis['total_offset_potential']:,.0f}")
                    col3.metric("Residual Exposure", f"${analysis['total_residual_exposure']:,.0f}")

                    # Recommendations
                    st.subheader("💡 Recommendations")
                    for rec in analysis['recommendations']:
                        st.info(rec)

                    # Currency breakdown
                    st.subheader("📊 Currency Breakdown")
                    bucket_results = analysis.get('time_bucket_results', {})
                    if bucket_results:
                        latest_bucket = list(bucket_results.keys())[-1]
                        currency_data = bucket_results[latest_bucket].get('currency_breakdown', {})

                        breakdown_data = []
                        for currency, data in currency_data.items():
                            breakdown_data.append({
                                'Currency': currency,
                                'Outflows': data['outflows'],
                                'Inflows': data['inflows'],
                                'Net Position': data['net_position'],
                                'Offset Potential': data['offset_potential']
                            })

                        if breakdown_data:
                            df = pd.DataFrame(breakdown_data)
                            st.dataframe(df, width='stretch')

                except Exception as e:
                    st.error(f"Netting analysis failed: {str(e)}")

    def render_recommendations_page(self):
        """Render recommendations page."""
        st.markdown('<h1 class="main-header">💡 Payment Recommendations</h1>', unsafe_allow_html=True)

        # Check if system is initialized
        if not hasattr(self, 'recommendation_engine') or self.recommendation_engine is None:
            st.error("❌ System not initialized. Please refresh the page.")
            return

        # Portfolio recommendations
        st.subheader("📋 Portfolio Recommendations")

        days_ahead = st.slider("Look ahead (days)", 7, 90, 30, key="rec_days")

        if st.button("🔄 Generate Recommendations", key="gen_recs"):
            with st.spinner("Generating recommendations..."):
                try:
                    recommendations = self.recommendation_engine.get_portfolio_recommendations(days_ahead)

                    if recommendations:
                        st.success(f"Generated {len(recommendations)} recommendations")

                        # Display recommendations
                        for i, rec in enumerate(recommendations[:10], 1):
                            confidence_class = "confidence-high" if rec.confidence_score > 0.8 else "confidence-medium" if rec.confidence_score > 0.6 else "confidence-low"
                            confidence_label = "High" if rec.confidence_score > 0.8 else "Medium" if rec.confidence_score > 0.6 else "Low"

                            with st.expander(f"#{i} {rec.scenario.invoice_id} - {rec.scenario.vendor} ({confidence_label} Confidence)"):
                                col1, col2 = st.columns([2, 1])

                                with col1:
                                    st.markdown(f"**Recommendation:** {rec.recommended_action}")
                                    st.markdown(f"**Amount:** ${rec.scenario.amount:,.2f} {rec.scenario.currency}")
                                    st.markdown(f"**Due Date:** {rec.scenario.due_date.strftime('%Y-%m-%d')}")
                                    st.markdown(f"**Confidence:** {rec.confidence_score:.1%}")

                                    if rec.reasoning:
                                        st.markdown("**Reasoning:**")
                                        st.info(rec.reasoning)

                                with col2:
                                    st.metric("FX Impact", f"{rec.expected_fx_impact:+.2f}%")
                                    st.metric("Netting Impact", f"{rec.expected_netting_impact:.1%}")

                                    if rec.alternative_options:
                                        st.markdown("**Alternatives:**")
                                        for alt in rec.alternative_options[:2]:
                                            st.markdown(f"• {alt['action']} ({alt['confidence_score']:.1%})")

                        # Export option
                        if st.button("📥 Export Recommendations to CSV"):
                            export_data = []
                            for rec in recommendations:
                                export_data.append({
                                    'invoice_id': rec.scenario.invoice_id,
                                    'vendor': rec.scenario.vendor,
                                    'amount': rec.scenario.amount,
                                    'currency': rec.scenario.currency,
                                    'due_date': rec.scenario.due_date,
                                    'recommendation': rec.recommended_action,
                                    'confidence': rec.confidence_score,
                                    'fx_impact': rec.expected_fx_impact,
                                    'netting_impact': rec.expected_netting_impact,
                                    'reasoning': rec.reasoning[:100] + "..." if len(rec.reasoning) > 100 else rec.reasoning
                                })

                            df = pd.DataFrame(export_data)
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name="kairo_recommendations.csv",
                                mime="text/csv",
                                key="download_recs"
                            )

                    else:
                        st.info("No recommendations available for the selected period")

                except Exception as e:
                    st.error(f"Failed to generate recommendations: {str(e)}")

    def render_simulation_page(self):
        """Render simulation page for comprehensive treasury strategy simulation."""
        st.markdown('<h1 class="main-header">🎲 Treasury Strategy Simulation</h1>', unsafe_allow_html=True)

        st.markdown("""
        Run comprehensive simulations of different payment timing strategies and hedging approaches
        to optimize your treasury operations.
        """)

        # Check if system is initialized
        if not hasattr(self, 'treasury_simulator') or self.treasury_simulator is None:
            st.error("❌ System not initialized. Please refresh the page.")
            return

        # Scenario selection
        st.subheader("🎛️ Simulation Scenarios")

        col1, col2 = st.columns(2)

        with col1:
            available_scenarios = list(DEFAULT_SCENARIOS.keys())
            selected_scenarios = st.multiselect(
                "Select Scenarios to Compare",
                options=available_scenarios,
                default=['current_behavior', 'ai_optimized'],
                format_func=lambda x: DEFAULT_SCENARIOS[x].name
            )

        with col2:
            time_horizon = st.slider("Time Horizon (days)", 30, 180, 90)

        # Run simulation
        if st.button("🚀 Run Treasury Simulation", key="run_treasury_sim"):
            if not selected_scenarios:
                st.error("Please select at least one scenario")
                return

            with st.spinner("Running comprehensive treasury simulation..."):
                try:
                    # Prepare scenarios
                    scenarios_to_run = [DEFAULT_SCENARIOS[name] for name in selected_scenarios]

                    # Use historical data for demonstration (since we don't have future payments)
                    historical_payments = self.ap_data[
                        (self.ap_data['due_date'] >= '2023-01-01') &
                        (self.ap_data['due_date'] <= '2023-03-31')
                    ].copy()

                    if len(historical_payments) == 0:
                        st.error("No historical data available for simulation")
                        return

                    # Run simulation comparison
                    results = self.treasury_simulator.run_portfolio_comparison(
                        scenarios_to_run, days_ahead=time_horizon
                    )

                    # Override with historical data
                    for scenario in scenarios_to_run:
                        results[scenario.name] = self.treasury_simulator.run_payment_timing_simulation(
                            scenario, historical_payments
                        )

                    # Generate comparison report
                    report_df = self.treasury_simulator.generate_scenario_report(results)

                    st.success("Simulation Complete!")

                    # Display results
                    st.subheader("📊 Simulation Results")

                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    best_scenario = report_df.loc[report_df['FX Savings (%)'].idxmax()]
                    col1.metric("Best FX Strategy", best_scenario['Scenario'])
                    col2.metric("Max FX Savings", f"{best_scenario['FX Savings (%)']:.1f}%")

                    lowest_risk = report_df.loc[report_df['Risk Exposure (%)'].idxmin()]
                    col3.metric("Lowest Risk", lowest_risk['Scenario'])
                    col4.metric("Risk Level", f"{lowest_risk['Risk Exposure (%)']:.1f}%")

                    # Detailed comparison table
                    st.subheader("🔍 Detailed Comparison")
                    st.dataframe(report_df.style.highlight_max(axis=0, subset=['FX Savings (%)', 'Confidence Score'])
                               .highlight_min(axis=0, subset=['Risk Exposure (%)', 'Working Capital Cost']))

                    # Scenario details
                    st.subheader("📋 Scenario Details")

                    for scenario_name, result in results.items():
                        with st.expander(f"🎯 {scenario_name} - Confidence: {result.confidence_score:.1%}"):
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Payments", result.total_payments)
                            col2.metric("Total Amount", f"${result.total_amount:,.0f}")
                            col3.metric("Avg Delay", f"{result.avg_payment_delay:.1f}d")
                            col4.metric("FX Savings", f"{result.fx_savings_pct:.1f}%")

                            # Cash flow metrics
                            st.subheader("Cash Flow Impact")
                            cf = result.cash_flow_impact
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Total Outflow", f"${cf.get('total_outflow', 0):,.0f}")
                            col2.metric("Peak Weekly", f"${cf.get('peak_weekly_outflow', 0):,.0f}")
                            col3.metric("Volatility", f"{cf.get('cash_flow_volatility', 0):.2f}")

                            # Recommendations
                            if result.recommendations:
                                st.subheader("💡 Recommendations")
                                for rec in result.recommendations:
                                    st.write(f"• {rec}")

                    # Export option
                    if st.button("📥 Export Simulation Results"):
                        csv = report_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name="kairo_simulation_results.csv",
                            mime="text/csv",
                            key="download_sim"
                        )

                except Exception as e:
                    st.error(f"Simulation failed: {str(e)}")
                    st.exception(e)

        # Individual payment simulation (legacy)
        st.markdown("---")
        st.subheader("🔍 Individual Payment Simulation")

        # Use historical data for individual simulation
        historical_payments = self.ap_data[
            (self.ap_data['due_date'] >= '2023-01-01') &
            (self.ap_data['due_date'] <= '2023-03-31')
        ]

        if len(historical_payments) > 0:
            selected_invoice = st.selectbox(
                "Select Invoice for Individual Analysis",
                options=[f"{row['invoice_id']} - {row['vendor']} (${row['amount']:,.0f} {row['currency']})"
                        for _, row in historical_payments.head(10).iterrows()],
                key="individual_sim"
            )

            if selected_invoice:
                invoice_id = selected_invoice.split(' - ')[0]
                payment_data = historical_payments[historical_payments['invoice_id'] == invoice_id].iloc[0]

                col1, col2, col3 = st.columns(3)
                col1.metric("Invoice ID", payment_data['invoice_id'])
                col2.metric("Vendor", payment_data['vendor'])
                col3.metric("Amount", f"${payment_data['amount']:,.0f} {payment_data['currency']}")

                if st.button("🔮 Analyze Individual Payment", key="analyze_individual"):
                    with st.spinner("Analyzing payment timing options..."):
                        try:
                            base_date = payment_data['due_date']
                            payment_options = [
                                base_date,
                                base_date + timedelta(days=3),
                                base_date + timedelta(days=7),
                                base_date + timedelta(days=14),
                            ]

                            scenario = PaymentScenario(
                                invoice_id=payment_data['invoice_id'],
                                vendor=payment_data['vendor'],
                                amount=payment_data['amount'],
                                currency=payment_data['currency'],
                                due_date=base_date,
                                payment_options=payment_options
                            )

                            recommendation = self.recommendation_engine.analyze_payment_scenario(scenario)

                            st.subheader("💡 AI Recommendation")
                            st.markdown(f"""
                            <div class="recommendation-high">
                                <h4>{recommendation.recommended_action}</h4>
                                <p>Confidence: {recommendation.confidence_score:.1%}</p>
                                <p>FX Impact: {recommendation.expected_fx_impact:+.2f}%</p>
                            </div>
                            """, unsafe_allow_html=True)

                        except Exception as e:
                            st.error(f"Individual analysis failed: {str(e)}")

    def _get_readable_feature_name(self, feature_name: str) -> str:
        """Convert technical feature names to human-readable descriptions."""
        name_mapping = {
            'lag_1d': 'Yesterday\'s rate',
            'lag_7d': 'Rate from 7 days ago',
            'lag_30d': 'Rate from 30 days ago',
            'rolling_mean_7d': '7-day average rate',
            'rolling_mean_30d': '30-day average rate',
            'rolling_std_7d': '7-day rate volatility',
            'rolling_std_30d': '30-day rate volatility',
            'pct_change_1d': 'Daily rate change',
            'pct_change_7d': 'Weekly rate change',
            'pct_change_30d': 'Monthly rate change',
            'momentum_7d': '7-day momentum',
            'momentum_30d': '30-day momentum',
            'volatility_7d': 'Short-term volatility',
            'volatility_30d': 'Long-term volatility'
        }

        return name_mapping.get(feature_name, feature_name.replace('_', ' ').title())

    def run(self):
        """Run the dashboard."""
        page = self.render_sidebar()

        if page == "🏠 Dashboard":
            self.render_dashboard()
        elif page == "💱 FX Analysis":
            self.render_fx_analysis_page()
        elif page == "👥 Behavior Insights":
            self.render_behavior_insights_page()
        elif page == "🔄 Netting Opportunities":
            self.render_netting_page()
        elif page == "💡 Recommendations":
            self.render_recommendations_page()
        elif page == "🎲 Simulation":
            self.render_simulation_page()


def main():
    """Main entry point."""
    dashboard = KairoDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()