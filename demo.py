#!/usr/bin/env python3
"""
Kairo Treasury Copilot - Interactive Demo

Comprehensive demonstration of all Kairo Treasury Copilot features.
Shows the complete workflow from data loading to AI-powered recommendations.
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add modules directory to path
sys.path.append(str(Path(__file__).parent / 'modules'))

from modules.data_ingest import get_data_loader
from modules.fx_model import create_fx_predictor
from modules.behavior_forecast import create_behavior_forecaster
from modules.netting_optimizer import create_netting_optimizer
from modules.recommendation_engine import create_recommendation_engine, PaymentScenario
from modules.simulation_engine import create_treasury_simulator, DEFAULT_SCENARIOS

class KairoDemo:
    """Interactive demo of Kairo Treasury Copilot features."""

    def __init__(self):
        """Initialize demo with all components."""
        print("🎯 KAIRO TREASURY COPILOT - FEATURE DEMO")
        print("=" * 60)

        self.loader = None
        self.fx_predictor = None
        self.behavior_forecaster = None
        self.netting_optimizer = None
        self.recommendation_engine = None
        self.simulator = None

    def initialize_system(self):
        """Initialize all system components."""
        print("\n🔄 Initializing Kairo Treasury Copilot...")

        try:
            # Load data
            self.loader = get_data_loader()
            ap_data = self.loader.load_ap_data()
            ar_data = self.loader.load_ar_data()
            fx_data = self.loader.load_fx_data()

            print(f"📊 Loaded {len(ap_data)} AP transactions, {len(ar_data)} AR transactions")
            print(f"💱 Loaded FX data for {len(fx_data['currency_pair'].unique())} currency pairs")

            # Initialize components
            self.fx_predictor = create_fx_predictor(fx_data)
            self.behavior_forecaster = create_behavior_forecaster(ap_data, ar_data)
            self.netting_optimizer = create_netting_optimizer(ap_data, ar_data)
            self.recommendation_engine = create_recommendation_engine(fx_data, ap_data, ar_data)
            self.simulator = create_treasury_simulator(ap_data, ar_data, fx_data)

            print("✅ System initialized successfully!")
            return True

        except Exception as e:
            print(f"❌ Initialization failed: {str(e)}")
            return False

    def demo_data_ingestion(self):
        """Demonstrate data ingestion capabilities."""
        print("\n📊 DEMO: Data Ingestion & Processing")
        print("-" * 40)

        # Show data summaries
        ap_summary = self.loader.get_ap_summary()
        ar_summary = self.loader.get_ar_summary()

        print("🏦 Accounts Payable Summary:")
        print(f"  • Total invoices: {ap_summary['total_invoices']:,}")
        print(f"  • Total amount: ${ap_summary['total_amount']:,.0f}")
        print(f"  • Average payment lag: {ap_summary['avg_payment_lag']:.1f} days")
        print(f"  • Early payment rate: {ap_summary['early_payment_rate']:.1%}")

        print("\n💰 Accounts Receivable Summary:")
        print(f"  • Total invoices: {ar_summary['total_invoices']:,}")
        print(f"  • Total amount: ${ar_summary['total_amount']:,.0f}")
        print(f"  • Average collection lag: {ar_summary['avg_collection_lag']:.1f} days")
        print(f"  • Early collection rate: {ar_summary['early_collection_rate']:.1%}")

        # Show upcoming transactions
        upcoming_payments = self.loader.get_upcoming_payments(30)
        upcoming_receipts = self.loader.get_expected_receipts(30)

        print(f"\n📅 Upcoming (30 days): {len(upcoming_payments)} payments, {len(upcoming_receipts)} receipts")

    def demo_fx_predictions(self):
        """Demonstrate FX prediction capabilities."""
        print("\n💱 DEMO: FX Rate Predictions")
        print("-" * 32)

        currencies = ['USD/EUR', 'USD/GBP', 'USD/JPY']

        for currency in currencies:
            print(f"\n📈 Predicting {currency} for next 7 days...")

            # Regular prediction
            result = self.fx_predictor.predict_fx_rate(currency, datetime.now(), days_ahead=7)
            print(f"  Current rate: {result['current_rate']:.4f}")
            print(f"  Predicted rate: {result['predicted_rate']:.4f}")
            print(f"  Expected change: {result['rate_change_pct']:+.2f}%")
            print(f"  Delay favorable: {result['is_delay_favorable']}")
            print(f"  Model confidence: {result['prediction_confidence']:.1%}")

            # SHAP analysis
            print(f"  🔍 SHAP Analysis:")
            shap_result = self.fx_predictor.predict_with_shap(currency, datetime.now(), days_ahead=7)
            if 'error' not in shap_result:
                print(f"    Explanation: {shap_result['shap_explanation']}")
                print(f"    SHAP Confidence: {shap_result['confidence_score']:.1%}")
            else:
                print(f"    SHAP unavailable: {shap_result['error']}")

            time.sleep(0.5)  # Brief pause for readability

    def demo_behavior_analysis(self):
        """Demonstrate behavioral analysis."""
        print("\n👥 DEMO: Payment Behavior Analysis")
        print("-" * 36)

        # Payment behavior
        print("🏦 Payment Behavior Insights:")
        payment_patterns = self.behavior_forecaster.analyze_payment_patterns()
        if 'error' not in payment_patterns:
            print(f"  • Average lag: {payment_patterns['avg_payment_lag_days']:.1f} days")
            print(f"  • Predictability: {payment_patterns['predictability_score']:.1%}")
            print(f"  • Early payment rate: {payment_patterns['early_payment_rate']:.1%}")
            print("  Key insights:")
            for insight in payment_patterns['insights'][:2]:
                print(f"    - {insight}")

        # Collection behavior
        print("\n💰 Collection Behavior Insights:")
        collection_patterns = self.behavior_forecaster.analyze_collection_patterns()
        if 'error' not in collection_patterns:
            print(f"  • Average lag: {collection_patterns['avg_collection_lag_days']:.1f} days")
            print(f"  • Predictability: {collection_patterns['predictability_score']:.1%}")
            print(f"  • Early collection rate: {collection_patterns['early_collection_rate']:.1%}")
            print("  Key insights:")
            for insight in collection_patterns['insights'][:2]:
                print(f"    - {insight}")

    def demo_netting_optimization(self):
        """Demonstrate netting optimization."""
        print("\n🔄 DEMO: Netting Optimization")
        print("-" * 30)

        # Use historical data for demonstration
        analysis = self.netting_optimizer.analyze_netting_opportunities(30)
        print(f"Netting Efficiency: {analysis['netting_efficiency']:.1%}")
        print(f"Total Offset Potential: ${analysis['total_offset_potential']:,.0f}")
        print(f"Total Residual Exposure: ${analysis['total_residual_exposure']:,.0f}")

        print("\n💡 Recommendations:")
        for rec in analysis['recommendations'][:2]:
            print(f"  • {rec}")

    def demo_ai_recommendations(self):
        """Demonstrate AI-powered recommendations."""
        print("\n🤖 DEMO: AI-Powered Recommendations")
        print("-" * 38)

        # Get a sample payment for analysis
        ap_data = self.loader.load_ap_data()
        sample_payment = ap_data.head(1).iloc[0]

        print("📄 Analyzing sample payment:")
        print(f"  Invoice: {sample_payment['invoice_id']}")
        print(f"  Vendor: {sample_payment['vendor']}")
        print(f"  Amount: ${sample_payment['amount']:,.0f} {sample_payment['currency']}")
        print(f"  Due Date: {sample_payment['due_date'].strftime('%Y-%m-%d')}")

        # Create payment scenario
        payment_options = [
            sample_payment['due_date'],
            sample_payment['due_date'] + timedelta(days=3),
            sample_payment['due_date'] + timedelta(days=7),
            sample_payment['due_date'] + timedelta(days=14),
        ]

        scenario = PaymentScenario(
            invoice_id=sample_payment['invoice_id'],
            vendor=sample_payment['vendor'],
            amount=sample_payment['amount'],
            currency=sample_payment['currency'],
            due_date=sample_payment['due_date'],
            payment_options=payment_options
        )

        # Get recommendation
        recommendation = self.recommendation_engine.analyze_payment_scenario(scenario)

        print("\n🎯 AI Recommendation:")
        print(f"  {recommendation.recommended_action}")
        print(f"  Confidence: {recommendation.confidence_score:.1%}")
        print(f"  FX Impact: {recommendation.expected_fx_impact:+.2f}%")
        print(f"  Netting Impact: {recommendation.expected_netting_impact:.1%}")

        if recommendation.alternative_options:
            print("\n🔄 Alternative Options:")
            for i, alt in enumerate(recommendation.alternative_options[:2], 1):
                print(f"  {i}. {alt['action']} (Confidence: {alt['confidence_score']:.1%})")

        if recommendation.historical_examples:
            print("\n📚 Similar Historical Cases:")
            for example in recommendation.historical_examples[:2]:
                print(f"  • {example['invoice_id']}: {example['outcome']}")

    def demo_treasury_simulation(self):
        """Demonstrate treasury strategy simulation."""
        print("\n🎲 DEMO: Treasury Strategy Simulation")
        print("-" * 38)

        # Run scenario comparison using historical data
        scenarios = [
            DEFAULT_SCENARIOS['current_behavior'],
            DEFAULT_SCENARIOS['ai_optimized'],
            DEFAULT_SCENARIOS['conservative_delay']
        ]

        print("Comparing payment strategies over 30 days...")

        # Use historical data for the demo
        ap_data = self.loader.load_ap_data()
        historical_payments = ap_data[
            (ap_data['due_date'] >= '2023-01-01') &
            (ap_data['due_date'] <= '2023-03-31')
        ].copy()

        if len(historical_payments) > 0:
            results = self.simulator.run_portfolio_comparison(scenarios, 30)

            # Override with historical data
            for scenario in scenarios:
                results[scenario.name] = self.simulator.run_payment_timing_simulation(
                    scenario, historical_payments
                )

            # Generate comparison report
            report = self.simulator.generate_scenario_report(results)

            print("\n📊 Scenario Comparison Results:")
            print("Strategy                  | FX Savings | Risk | Confidence")
            print("-" * 55)

            for _, row in report.iterrows():
                strategy = row['Scenario'][:20].ljust(20)
                fx_savings = f"{row['FX Savings (%)']:+.1f}%".rjust(10)
                risk = f"{row['Risk Exposure (%)']:.1f}%".rjust(4)
                confidence = f"{row['Confidence Score']:.1%}".rjust(10)
                print(f"{strategy} | {fx_savings} | {risk} | {confidence}")

            # Show best strategy
            best_fx = report.loc[report['FX Savings (%)'].idxmax(), 'Scenario']
            lowest_risk = report.loc[report['Risk Exposure (%)'].idxmin(), 'Scenario']

            print(f"\n🏆 Best FX Strategy: {best_fx}")
            print(f"🛡️  Lowest Risk Strategy: {lowest_risk}")

        else:
            print("No historical data available for simulation demo.")

    def demo_portfolio_recommendations(self):
        """Demonstrate portfolio-level recommendations."""
        print("\n📊 DEMO: Portfolio Recommendations")
        print("-" * 35)

        # Generate portfolio recommendations
        recommendations = self.recommendation_engine.get_portfolio_recommendations(14)

        if recommendations:
            print(f"Generated {len(recommendations)} payment recommendations for next 14 days")

            # Show top 3 recommendations
            print("\nTop Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"{i}. {rec.recommended_action}")
                print(".1%")
                print(".2f")
                print()

        else:
            print("No upcoming payments found for portfolio analysis.")

    def run_full_demo(self):
        """Run the complete feature demonstration."""
        print("🎬 Starting Kairo Treasury Copilot Feature Demo")
        print("This demo showcases all major capabilities of the system.\n")

        # Initialize system
        if not self.initialize_system():
            return False

        # Run all demos
        demos = [
            ("Data Ingestion", self.demo_data_ingestion),
            ("FX Predictions", self.demo_fx_predictions),
            ("Behavior Analysis", self.demo_behavior_analysis),
            ("Netting Optimization", self.demo_netting_optimization),
            ("AI Recommendations", self.demo_ai_recommendations),
            ("Treasury Simulation", self.demo_treasury_simulation),
            ("Portfolio Analysis", self.demo_portfolio_recommendations),
        ]

        for demo_name, demo_func in demos:
            try:
                demo_func()
                print(f"\n✅ {demo_name} demo completed")
            except Exception as e:
                print(f"\n❌ {demo_name} demo failed: {str(e)}")

            time.sleep(1)  # Brief pause between demos

        # Final summary
        print("\n" + "=" * 60)
        print("🎉 KAIRO TREASURY COPILOT DEMO COMPLETE!")
        print("=" * 60)
        print("\n✨ Demonstrated Features:")
        print("  • 📊 Data ingestion and processing")
        print("  • 💱 FX rate prediction with SHAP explainability")
        print("  • 👥 Payment behavior analysis")
        print("  • 🔄 Natural hedging optimization")
        print("  • 🤖 AI-powered payment recommendations")
        print("  • 🎲 Treasury strategy simulation")
        print("  • 📈 Portfolio-level optimization")
        print("\n🚀 Ready for production use!")
        print("\n💡 Next Steps:")
        print("  • Start CLI: python interface/interface_cli.py")
        print("  • Start Dashboard: streamlit run interface/dashboard.py")
        print("  • Run Tests: python tests/test_system_integration.py")

        return True

def main():
    """Main demo entry point."""
    demo = KairoDemo()
    success = demo.run_full_demo()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())