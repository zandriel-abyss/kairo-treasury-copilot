#!/usr/bin/env python3
"""
Kairo Treasury Optimization Copilot - CLI Interface

Command-line interface for the Kairo AI treasury assistant.
Provides interactive access to payment recommendations, FX analysis,
and portfolio optimization features.
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd

# Add modules directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.data_ingest import get_data_loader
from modules.recommendation_engine import create_recommendation_engine, PaymentScenario
from modules.fx_model import FXPredictor, create_fx_predictor
from modules.behavior_forecast import create_behavior_forecaster
from modules.netting_optimizer import create_netting_optimizer


class KairoCLI:
    """Command-line interface for Kairo treasury optimization."""

    def __init__(self):
        """Initialize the CLI with all necessary components."""
        print("🚀 Initializing Kairo Treasury Optimization Copilot...")
        print()

        try:
            # Load data
            self.data_loader = get_data_loader()
            self.ap_data = self.data_loader.load_ap_data()
            self.ar_data = self.data_loader.load_ar_data()
            self.fx_data = self.data_loader.load_fx_data()

            # Initialize components
            self.recommendation_engine = create_recommendation_engine(
                self.fx_data, self.ap_data, self.ar_data
            )
            self.fx_predictor = create_fx_predictor(self.fx_data)
            self.behavior_forecaster = create_behavior_forecaster(self.ap_data, self.ar_data)
            self.netting_optimizer = create_netting_optimizer(self.ap_data, self.ar_data)

            print("✅ Kairo initialized successfully!")
            print(f"📊 Loaded {len(self.ap_data)} AP transactions, {len(self.ar_data)} AR transactions")
            print(f"💱 FX data for {len(self.fx_data['currency_pair'].unique())} currency pairs")
            print()

        except Exception as e:
            print(f"❌ Failed to initialize Kairo: {str(e)}")
            print("Please ensure data files are present in the 'data' directory.")
            sys.exit(1)

    def run(self):
        """Run the main CLI loop."""
        while True:
            self._display_main_menu()
            choice = input("Enter your choice (1-7): ").strip()

            if choice == '1':
                self._analyze_specific_payment()
            elif choice == '2':
                self._portfolio_recommendations()
            elif choice == '3':
                self._fx_analysis()
            elif choice == '4':
                self._behavior_analysis()
            elif choice == '5':
                self._netting_analysis()
            elif choice == '6':
                self._export_recommendations()
            elif choice == '7':
                print("👋 Thank you for using Kairo! Goodbye.")
                break
            else:
                print("❌ Invalid choice. Please enter 1-7.")
                print()

    def _display_main_menu(self):
        """Display the main menu."""
        print("🎯 KAIRO TREASURY OPTIMIZATION COPILOT")
        print("=" * 50)
        print("1. 📄 Analyze Specific Payment")
        print("2. 📊 Portfolio Recommendations")
        print("3. 💱 FX Rate Analysis")
        print("4. 👥 Payment Behavior Analysis")
        print("5. 🔄 Netting Opportunities")
        print("6. 💾 Export Recommendations")
        print("7. 🚪 Exit")
        print()

    def _analyze_specific_payment(self):
        """Analyze a specific payment scenario."""
        print("📄 ANALYZE SPECIFIC PAYMENT")
        print("-" * 30)

        # Show upcoming payments
        upcoming = self.data_loader.get_upcoming_payments(30)
        if len(upcoming) == 0:
            print("No upcoming payments found in the next 30 days.")
            print("Using historical data for demonstration...")
            # Use a recent payment for demo
            sample_payment = self.ap_data.head(1).iloc[0]
        else:
            print("Upcoming payments:")
            for i, (_, payment) in enumerate(upcoming.head(5).iterrows(), 1):
                print(f"{i}. {payment['invoice_id']} - {payment['vendor']} - "
                      f"${payment['amount']:,.0f} {payment['currency']} - "
                      f"Due: {payment['due_date'].strftime('%Y-%m-%d')}")
            print()

            choice = input("Enter payment number to analyze (or press Enter for first): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(upcoming):
                sample_payment = upcoming.iloc[int(choice) - 1]
            else:
                sample_payment = upcoming.iloc[0]

        print(f"Analyzing: {sample_payment['invoice_id']}")
        print(f"Vendor: {sample_payment['vendor']}")
        print(f"Amount: ${sample_payment['amount']:,.2f} {sample_payment['currency']}")
        print(f"Due Date: {sample_payment['due_date'].strftime('%Y-%m-%d')}")
        print()

        # Create payment options
        due_date = sample_payment['due_date']
        payment_options = [
            due_date,  # On time
            due_date + timedelta(days=3),   # Short delay
            due_date + timedelta(days=7),   # Week delay
            due_date + timedelta(days=14),  # Two week delay
        ]

        scenario = PaymentScenario(
            invoice_id=sample_payment['invoice_id'],
            vendor=sample_payment['vendor'],
            amount=sample_payment['amount'],
            currency=sample_payment['currency'],
            due_date=due_date,
            payment_options=payment_options
        )

        try:
            recommendation = self.recommendation_engine.analyze_payment_scenario(scenario)

            self._display_recommendation(recommendation)

            # Ask for override
            print()
            override = input("Would you like to override this recommendation? (y/n): ").strip().lower()
            if override == 'y':
                self._handle_override(recommendation)

        except Exception as e:
            print(f"❌ Error analyzing payment: {str(e)}")

        input("\nPress Enter to continue...")

    def _portfolio_recommendations(self):
        """Display portfolio-level recommendations."""
        print("📊 PORTFOLIO RECOMMENDATIONS")
        print("-" * 35)

        days_ahead = input("Analyze payments for how many days ahead? (default 30): ").strip()
        try:
            days_ahead = int(days_ahead) if days_ahead else 30
        except ValueError:
            days_ahead = 30

        print(f"Analyzing portfolio for next {days_ahead} days...")
        print()

        try:
            recommendations = self.recommendation_engine.get_portfolio_recommendations(days_ahead)

            if not recommendations:
                print("No upcoming payments found in the specified period.")
                print("Try analyzing a longer period or check data availability.")
            else:
                print(f"📋 Found {len(recommendations)} payment recommendations")
                print()

                # Display top recommendations
                for i, rec in enumerate(recommendations[:10], 1):  # Show top 10
                    print(f"{i}. {rec.recommended_action}")
                    print(f"   Confidence: {rec.confidence_score:.1%} | Invoice: {rec.scenario.invoice_id}")
                    print()

                if len(recommendations) > 10:
                    print(f"... and {len(recommendations) - 10} more recommendations")
                    print()

                # Summary statistics
                high_conf = sum(1 for r in recommendations if r.confidence_score > 0.8)
                delay_recs = sum(1 for r in recommendations if "Delay payment" in r.recommended_action)

                print("📈 Summary:")
                print(f"• Total recommendations: {len(recommendations)}")
                print(f"• High confidence (>80%): {high_conf}")
                print(f"• Delay recommendations: {delay_recs}")
                print()

        except Exception as e:
            print(f"❌ Error generating portfolio recommendations: {str(e)}")

        input("\nPress Enter to continue...")

    def _fx_analysis(self):
        """Perform FX rate analysis."""
        print("💱 FX RATE ANALYSIS")
        print("-" * 20)

        # Show available currency pairs
        currency_pairs = self.data_loader.get_currency_pairs()
        print("Available currency pairs:")
        for i, pair in enumerate(currency_pairs[:10], 1):
            print(f"{i}. {pair}")
        if len(currency_pairs) > 10:
            print(f"... and {len(currency_pairs) - 10} more")
        print()

        pair_choice = input("Enter currency pair (e.g., USD/EUR) or number: ").strip()

        # Handle number input
        if pair_choice.isdigit():
            idx = int(pair_choice) - 1
            if 0 <= idx < len(currency_pairs):
                selected_pair = currency_pairs[idx]
            else:
                print("Invalid number.")
                return
        else:
            selected_pair = pair_choice.upper()

        if selected_pair not in currency_pairs:
            print(f"Currency pair {selected_pair} not found.")
            return

        days_ahead = input("Predict for how many days ahead? (default 7): ").strip()
        try:
            days_ahead = int(days_ahead) if days_ahead else 7
        except ValueError:
            days_ahead = 7

        include_shap = input("Include SHAP explainability analysis? (y/n, default n): ").strip().lower() == 'y'

        print(f"\nPredicting {selected_pair} for {days_ahead} days ahead...")
        if include_shap:
            print("Including SHAP explainability analysis...")
        print()

        try:
            prediction = self.fx_predictor.predict_fx_rate(
                selected_pair, datetime.now(), days_ahead=days_ahead,
                include_shap=include_shap
            )

            print(f"📈 FX Prediction for {selected_pair}")
            print(f"Current Rate: {prediction['current_rate']:.4f}")
            print(f"Predicted Rate: {prediction['predicted_rate']:.4f}")
            print(f"Expected Change: {prediction['rate_change_pct']:+.2f}%")
            print(f"Confidence: {prediction['prediction_confidence']:.1%}")
            print(f"Model: {prediction['model_type']}")
            print()
            print("Reasoning:")
            print(prediction['reasoning'])
            print()

            if prediction['is_delay_favorable']:
                print("💡 Delay favorable: Consider delaying payments in this currency")
            else:
                print("⚠️  Delay not favorable: Pay on schedule to avoid FX losses")

            # SHAP Analysis
            if include_shap and 'shap_analysis' in prediction:
                shap_data = prediction['shap_analysis']
                if 'error' not in shap_data:
                    print("\n🔍 SHAP EXPLAINABILITY ANALYSIS")
                    print("-" * 35)
                    print("Explanation:")
                    print(shap_data['shap_explanation'])
                    print()
                    print("Top Factors:")
                    for i, (feature_name, shap_value) in enumerate(shap_data['top_features'][:5], 1):
                        impact = "📈 Pushing upward" if shap_value > 0 else "📉 Pulling downward"
                        strength = "Strong" if abs(shap_value) > 0.01 else "Moderate"
                        readable_name = self._get_readable_feature_name(feature_name)
                        print(f"{i}. {readable_name}: {strength} {impact}")
                    print()
                    print(f"SHAP Confidence: {shap_data['shap_confidence']:.1%}")
                else:
                    print(f"\n⚠️ SHAP analysis not available: {shap_data['error']}")

        except Exception as e:
            print(f"❌ Error predicting FX: {str(e)}")

        input("\nPress Enter to continue...")

    def _behavior_analysis(self):
        """Perform payment behavior analysis."""
        print("👥 PAYMENT BEHAVIOR ANALYSIS")
        print("-" * 32)

        analysis_type = input("Analyze (1) Payment behavior or (2) Collection behavior? ").strip()

        if analysis_type == '1':
            # Payment behavior
            print("\nPayment Behavior Analysis")
            print("-" * 25)

            vendor_input = input("Enter vendor name (or press Enter for all): ").strip()
            currency_input = input("Enter currency (or press Enter for all): ").strip()

            vendor = vendor_input if vendor_input else None
            currency = currency_input if currency_input else None

            try:
                analysis = self.behavior_forecaster.analyze_payment_patterns(
                    vendor=vendor, currency=currency
                )

                if 'error' in analysis:
                    print(f"❌ {analysis['error']}")
                else:
                    print(f"📊 Analysis for: {analysis['entity']} ({analysis['currency']})")
                    print(f"Total payments: {analysis['total_payments']}")
                    print(f"Average lag: {analysis['avg_payment_lag_days']:.1f} days")
                    print(f"Predictability: {analysis['predictability_score']:.1%}")
                    print(f"Early payment rate: {analysis['early_payment_rate']:.1%}")
                    print(f"Late payment rate: {analysis['late_payment_rate']:.1%}")
                    print()
                    print("Key insights:")
                    for insight in analysis['insights']:
                        print(f"• {insight}")

            except Exception as e:
                print(f"❌ Error analyzing payment behavior: {str(e)}")

        elif analysis_type == '2':
            # Collection behavior
            print("\nCollection Behavior Analysis")
            print("-" * 28)

            customer_input = input("Enter customer name (or press Enter for all): ").strip()
            currency_input = input("Enter currency (or press Enter for all): ").strip()

            customer = customer_input if customer_input else None
            currency = currency_input if currency_input else None

            try:
                analysis = self.behavior_forecaster.analyze_collection_patterns(
                    customer=customer, currency=currency
                )

                if 'error' in analysis:
                    print(f"❌ {analysis['error']}")
                else:
                    print(f"📊 Analysis for: {analysis['entity']} ({analysis['currency']})")
                    print(f"Total collections: {analysis['total_collections']}")
                    print(f"Average lag: {analysis['avg_collection_lag_days']:.1f} days")
                    print(f"Predictability: {analysis['predictability_score']:.1%}")
                    print(f"Early collection rate: {analysis['early_collection_rate']:.1%}")
                    print(f"Late collection rate: {analysis['late_collection_rate']:.1%}")
                    print()
                    print("Key insights:")
                    for insight in analysis['insights']:
                        print(f"• {insight}")

            except Exception as e:
                print(f"❌ Error analyzing collection behavior: {str(e)}")

        else:
            print("❌ Invalid choice.")

        input("\nPress Enter to continue...")

    def _netting_analysis(self):
        """Perform netting opportunity analysis."""
        print("🔄 NETTING OPPORTUNITIES")
        print("-" * 25)

        days_ahead = input("Analyze for how many days ahead? (default 30): ").strip()
        try:
            days_ahead = int(days_ahead) if days_ahead else 30
        except ValueError:
            days_ahead = 30

        currency_focus = input("Focus on specific currency? (press Enter for all): ").strip()
        currency_focus = currency_focus if currency_focus else None

        print(f"\nAnalyzing netting opportunities for next {days_ahead} days...")
        if currency_focus:
            print(f"Currency focus: {currency_focus}")
        print()

        try:
            analysis = self.netting_optimizer.analyze_netting_opportunities(
                time_window_days=days_ahead, currency_focus=currency_focus
            )

            print(f"📊 Netting Analysis Results")
            print(f"Netting Efficiency: {analysis['netting_efficiency']:.1%}")
            print(f"Total Offset Potential: ${analysis['total_offset_potential']:,.0f}")
            print(f"Total Residual Exposure: ${analysis['total_residual_exposure']:,.0f}")
            print()

            if analysis['recommendations']:
                print("💡 Recommendations:")
                for rec in analysis['recommendations']:
                    print(f"• {rec}")
                print()

            # Show currency breakdown if available
            bucket_results = analysis.get('time_bucket_results', {})
            if bucket_results:
                print("Currency Breakdown (Latest Period):")
                latest_bucket = list(bucket_results.keys())[-1]  # Get most recent
                currency_data = bucket_results[latest_bucket].get('currency_breakdown', {})

                for currency, data in list(currency_data.items())[:5]:  # Show top 5
                    offset_pct = (data['offset_potential'] / (data['outflows'] + data['inflows'])) * 100 if (data['outflows'] + data['inflows']) > 0 else 0
                    print(f"• {currency}: ${data['offset_potential']:,.0f} offset potential ({offset_pct:.1%})")

        except Exception as e:
            print(f"❌ Error analyzing netting: {str(e)}")

        input("\nPress Enter to continue...")

    def _export_recommendations(self):
        """Export recommendations to file."""
        print("💾 EXPORT RECOMMENDATIONS")
        print("-" * 27)

        days_ahead = input("Export recommendations for how many days ahead? (default 30): ").strip()
        try:
            days_ahead = int(days_ahead) if days_ahead else 30
        except ValueError:
            days_ahead = 30

        filename = input("Enter filename (default: kairo_recommendations.csv): ").strip()
        if not filename:
            filename = "kairo_recommendations.csv"

        print(f"\nGenerating recommendations for next {days_ahead} days...")
        print(f"Exporting to: {filename}")
        print()

        try:
            recommendations = self.recommendation_engine.get_portfolio_recommendations(days_ahead)

            if not recommendations:
                print("No recommendations to export.")
                return

            # Convert to DataFrame for export
            export_data = []
            for rec in recommendations:
                export_data.append({
                    'recommendation_id': rec.recommendation_id,
                    'invoice_id': rec.scenario.invoice_id,
                    'vendor': rec.scenario.vendor,
                    'amount': rec.scenario.amount,
                    'currency': rec.scenario.currency,
                    'due_date': rec.scenario.due_date,
                    'recommended_action': rec.recommended_action,
                    'confidence_score': rec.confidence_score,
                    'expected_fx_impact': rec.expected_fx_impact,
                    'expected_netting_impact': rec.expected_netting_impact,
                    'reasoning': rec.reasoning[:200] + "..." if len(rec.reasoning) > 200 else rec.reasoning,  # Truncate long text
                    'historical_examples_count': len(rec.historical_examples),
                    'override_allowed': rec.override_allowed
                })

            df = pd.DataFrame(export_data)
            df.to_csv(filename, index=False)

            print(f"✅ Successfully exported {len(recommendations)} recommendations to {filename}")
            print()
            print("Sample of exported data:")
            print(df.head(3).to_string(index=False))

        except Exception as e:
            print(f"❌ Error exporting recommendations: {str(e)}")

        input("\nPress Enter to continue...")

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

    def _display_recommendation(self, recommendation):
        """Display a recommendation in a formatted way."""
        print("🎯 RECOMMENDATION")
        print("-" * 15)
        print(recommendation.recommended_action)
        print()

        print("📊 DETAILS")
        print("-" * 10)
        print(f"Invoice: {recommendation.scenario.invoice_id}")
        print(f"Vendor: {recommendation.scenario.vendor}")
        print(f"Amount: ${recommendation.scenario.amount:,.2f} {recommendation.scenario.currency}")
        print(f"Due Date: {recommendation.scenario.due_date.strftime('%Y-%m-%d')}")
        print(f"Confidence: {recommendation.confidence_score:.1%}")
        print(f"FX Impact: {recommendation.expected_fx_impact:+.2f}%")
        print(f"Netting Impact: {recommendation.expected_netting_impact:.1%}")
        print()

        print("💭 REASONING")
        print("-" * 12)
        print(recommendation.reasoning)
        print()

        if recommendation.alternative_options:
            print("🔄 ALTERNATIVES")
            print("-" * 14)
            for i, alt in enumerate(recommendation.alternative_options[:3], 1):
                print(f"{i}. {alt['action']}")
                print(f"   Confidence: {alt['confidence_score']:.1%}")
            print()

        if recommendation.historical_examples:
            print("📚 SIMILAR CASES")
            print("-" * 16)
            for example in recommendation.historical_examples[:3]:
                print(f"• {example['invoice_id']}: {example['outcome']}")
            print()

        if recommendation.pain_protocol_suggestion:
            pain = recommendation.pain_protocol_suggestion
            print("🔗 PAIN PROTOCOL")
            print("-" * 16)
            print(f"Preferred Rail: {pain['preferred_rail']}")
            if pain.get('delay_fallback'):
                print(f"Fallback: {pain['delay_fallback']}")
            print()

    def _handle_override(self, recommendation):
        """Handle user override of recommendation."""
        print("⚠️  RECOMMENDATION OVERRIDE")
        print("-" * 25)
        print("Current recommendation:")
        print(f"  {recommendation.recommended_action}")
        print()

        print("Available options:")
        for i, alt in enumerate(recommendation.alternative_options, 1):
            print(f"{i}. {alt['action']}")
        print(f"{len(recommendation.alternative_options) + 1}. Custom timing")
        print()

        choice = input("Enter your choice: ").strip()

        if choice.isdigit():
            choice_num = int(choice)
            if 1 <= choice_num <= len(recommendation.alternative_options):
                selected = recommendation.alternative_options[choice_num - 1]
                print(f"✅ Override accepted: {selected['action']}")
                # In production, this would log the override and reasoning
            elif choice_num == len(recommendation.alternative_options) + 1:
                custom_date = input("Enter custom payment date (YYYY-MM-DD): ").strip()
                try:
                    datetime.strptime(custom_date, '%Y-%m-%d')
                    print(f"✅ Custom timing accepted: Pay on {custom_date}")
                except ValueError:
                    print("❌ Invalid date format.")
            else:
                print("❌ Invalid choice.")
        else:
            print("❌ Invalid input.")

        print()


def main():
    """Main entry point for the CLI."""
    try:
        cli = KairoCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        print("Please check your data files and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()