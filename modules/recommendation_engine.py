"""
Recommendation Engine for Kairo Treasury Optimization

Integrates FX predictions, behavioral analysis, and netting optimization
to provide actionable treasury recommendations with explainability.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
try:
    # Try relative imports first (when run as module)
    from .fx_model import FXPredictor
    from .behavior_forecast import BehaviorForecaster
    from .netting_optimizer import NettingOptimizer
except ImportError:
    # Fall back to absolute imports (when run directly)
    from fx_model import FXPredictor
    from behavior_forecast import BehaviorForecaster
    from netting_optimizer import NettingOptimizer


@dataclass
class PaymentScenario:
    """Represents a payment decision scenario."""
    invoice_id: str
    vendor: str
    amount: float
    currency: str
    due_date: datetime
    payment_options: List[datetime]  # Different possible payment dates


@dataclass
class Recommendation:
    """Represents a treasury recommendation."""
    recommendation_id: str
    scenario: PaymentScenario
    recommended_action: str
    confidence_score: float
    expected_fx_impact: float
    expected_netting_impact: float
    reasoning: str
    alternative_options: List[Dict]
    historical_examples: List[Dict]
    override_allowed: bool
    pain_protocol_suggestion: Optional[Dict] = None


class RecommendationEngine:
    """
    Central engine that integrates all Kairo modules to provide
    comprehensive treasury optimization recommendations.
    """

    def __init__(self, fx_predictor: FXPredictor,
                 behavior_forecaster: BehaviorForecaster,
                 netting_optimizer: NettingOptimizer):
        """
        Initialize the recommendation engine.

        Args:
            fx_predictor: Trained FX prediction model
            behavior_forecaster: Behavioral analysis model
            netting_optimizer: Netting optimization model
        """
        self.fx_predictor = fx_predictor
        self.behavior_forecaster = behavior_forecaster
        self.netting_optimizer = netting_optimizer

    def analyze_payment_scenario(self, scenario: PaymentScenario) -> Recommendation:
        """
        Analyze a payment scenario and provide comprehensive recommendations.

        Args:
            scenario: Payment scenario to analyze

        Returns:
            Detailed recommendation with confidence scores and reasoning
        """
        recommendation_id = f"REC-{scenario.invoice_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Analyze each payment option
        option_analyses = []

        for payment_date in scenario.payment_options:
            analysis = self._analyze_payment_option(scenario, payment_date)
            option_analyses.append(analysis)

        # Select best recommendation
        best_option = max(option_analyses, key=lambda x: x['composite_score'])

        # Generate alternative options (top 3)
        alternative_options = sorted(option_analyses,
                                   key=lambda x: x['composite_score'],
                                   reverse=True)[1:4]

        # Find historical examples
        historical_examples = self._find_historical_examples(scenario, best_option['payment_date'])

        # Generate PAIN protocol suggestion (future integration)
        pain_suggestion = self._generate_pain_protocol_suggestion(best_option)

        # Create recommendation
        recommendation = Recommendation(
            recommendation_id=recommendation_id,
            scenario=scenario,
            recommended_action=self._format_recommended_action(best_option),
            confidence_score=best_option['composite_score'],
            expected_fx_impact=best_option['fx_impact']['expected_change_pct'],
            expected_netting_impact=best_option['netting_impact']['score'],
            reasoning=self._generate_reasoning(best_option, option_analyses),
            alternative_options=self._format_alternatives(alternative_options),
            historical_examples=historical_examples,
            override_allowed=True,
            pain_protocol_suggestion=pain_suggestion
        )

        return recommendation

    def _analyze_payment_option(self, scenario: PaymentScenario, payment_date: datetime) -> Dict:
        """Analyze a specific payment timing option."""
        days_delay = (payment_date - scenario.due_date).days

        # FX Impact Analysis
        fx_impact = self._calculate_fx_impact(scenario, payment_date)

        # Netting Impact Analysis
        netting_impact = self._calculate_netting_impact(scenario, payment_date)

        # Behavioral Analysis
        behavioral_score = self._calculate_behavioral_score(scenario, payment_date)

        # Working Capital Cost
        wc_cost = self._calculate_working_capital_cost(scenario, days_delay)

        # Composite Score (weighted combination)
        # Weights: FX (40%), Netting (30%), Behavioral (20%), WC Cost (10%)
        composite_score = (
            0.4 * fx_impact['score'] +
            0.3 * netting_impact['score'] +
            0.2 * behavioral_score +
            0.1 * (1 - wc_cost['relative_cost'])  # Lower cost is better
        )

        return {
            'payment_date': payment_date,
            'days_delay': days_delay,
            'fx_impact': fx_impact,
            'netting_impact': netting_impact,
            'behavioral_score': behavioral_score,
            'wc_cost': wc_cost,
            'composite_score': composite_score
        }

    def _calculate_fx_impact(self, scenario: PaymentScenario, payment_date: datetime) -> Dict:
        """Calculate FX impact of payment timing decision."""
        try:
            # Get relevant currency pair
            base_currency = 'USD'  # Assuming USD as base for recommendations
            currency_pair = f"{base_currency}/{scenario.currency}"

            if currency_pair not in self.fx_predictor.fx_data['currency_pair'].unique():
                # Try reverse pair
                currency_pair = f"{scenario.currency}/{base_currency}"

            if currency_pair not in self.fx_predictor.fx_data['currency_pair'].unique():
                return {
                    'score': 0.5,  # Neutral
                    'expected_change_pct': 0.0,
                    'confidence': 0.5,
                    'is_favorable': True,
                    'reasoning': f"No FX data available for {scenario.currency}"
                }

            # Predict FX rate change
            days_ahead = (payment_date - datetime.now()).days
            if days_ahead < 0:  # Past date
                days_ahead = 1

            prediction = self.fx_predictor.predict_fx_rate(
                currency_pair, datetime.now(), days_ahead=min(days_ahead, 30)
            )

            # For payment in foreign currency, favorable if rate decreases (gets more base currency per unit)
            is_favorable = prediction['rate_change_pct'] < 0 if scenario.currency != base_currency else prediction['is_delay_favorable']

            # Convert to 0-1 score (1 = very favorable, 0 = unfavorable)
            fx_score = 0.5 + (prediction['rate_change_pct'] * -10)  # Scale percentage to score
            fx_score = max(0, min(1, fx_score))

            return {
                'score': fx_score,
                'expected_change_pct': prediction['rate_change_pct'],
                'confidence': prediction['prediction_confidence'],
                'is_favorable': is_favorable,
                'reasoning': prediction['reasoning']
            }

        except Exception as e:
            return {
                'score': 0.5,
                'expected_change_pct': 0.0,
                'confidence': 0.3,
                'is_favorable': True,
                'reasoning': f"FX analysis unavailable: {str(e)}"
            }

    def _calculate_netting_impact(self, scenario: PaymentScenario, payment_date: datetime) -> Dict:
        """Calculate netting impact of payment timing."""
        try:
            # Simplified netting impact calculation
            # In production, would use full netting optimizer

            # Check if payment timing aligns with receipts in same currency
            receipts_same_period = self.behavior_forecaster.ar_data[
                (self.behavior_forecaster.ar_data['currency'] == scenario.currency) &
                (self.behavior_forecaster.ar_data['expected_payment_date'] >= payment_date - timedelta(days=3)) &
                (self.behavior_forecaster.ar_data['expected_payment_date'] <= payment_date + timedelta(days=3))
            ]

            potential_offset = receipts_same_period['amount'].sum()
            offset_ratio = min(scenario.amount, potential_offset) / scenario.amount

            # Higher offset ratio = better netting (score closer to 1)
            netting_score = offset_ratio * 0.8 + 0.2  # Base score of 0.2

            return {
                'score': netting_score,
                'potential_offset_amount': potential_offset,
                'offset_ratio': offset_ratio,
                'reasoning': f"Payment timing aligns with ${potential_offset:,.0f} in receipts"
            }

        except Exception as e:
            return {
                'score': 0.5,
                'potential_offset_amount': 0,
                'offset_ratio': 0,
                'reasoning': f"Netting analysis unavailable: {str(e)}"
            }

    def _calculate_behavioral_score(self, scenario: PaymentScenario, payment_date: datetime) -> float:
        """Calculate behavioral appropriateness score."""
        try:
            # Get vendor payment pattern
            vendor_pattern = self.behavior_forecaster.analyze_payment_patterns(vendor=scenario.vendor)

            if 'error' in vendor_pattern:
                return 0.5  # Neutral score

            avg_lag = vendor_pattern['avg_payment_lag_days']
            std_lag = vendor_pattern['std_payment_lag_days']
            predictability = vendor_pattern['predictability_score']

            # Calculate how typical this payment timing is
            days_from_due = (payment_date - scenario.due_date).days
            deviation_from_norm = abs(days_from_due - avg_lag)

            # Score based on deviation (lower deviation = higher score)
            deviation_score = max(0, 1 - (deviation_from_norm / (std_lag + 1)))

            # Weight by predictability
            behavioral_score = deviation_score * predictability + (1 - predictability) * 0.5

            return behavioral_score

        except Exception:
            return 0.5

    def _calculate_working_capital_cost(self, scenario: PaymentScenario, days_delay: int) -> Dict:
        """Calculate working capital cost of delaying payment."""
        if days_delay <= 0:
            return {'cost_amount': 0, 'relative_cost': 0, 'reasoning': 'No delay, no working capital cost'}

        # Simplified WC cost calculation
        # In production: interest_rate * amount * days_delay / 365
        daily_interest_rate = 0.0003  # ~0.03% per day (10% annual rate)
        cost_amount = scenario.amount * daily_interest_rate * days_delay

        # Relative cost (0-1 scale)
        max_reasonable_delay = 30
        relative_cost = min(days_delay / max_reasonable_delay, 1)

        return {
            'cost_amount': cost_amount,
            'relative_cost': relative_cost,
            'reasoning': f"${cost_amount:.2f} working capital cost for {days_delay} day delay"
        }

    def _format_recommended_action(self, best_option: Dict) -> str:
        """Format the recommended action as human-readable text."""
        payment_date = best_option['payment_date']
        days_delay = best_option['days_delay']

        if days_delay == 0:
            action = f"Pay on due date: {payment_date.strftime('%Y-%m-%d')}"
        elif days_delay > 0:
            action = f"Delay payment by {days_delay} days until {payment_date.strftime('%Y-%m-%d')}"
        else:
            action = f"Pay early by {abs(days_delay)} days on {payment_date.strftime('%Y-%m-%d')}"

        # Add key benefits
        benefits = []
        if best_option['fx_impact']['is_favorable']:
            benefits.append(f"FX benefit: {best_option['fx_impact']['expected_change_pct']:+.2f}%")
        if best_option['netting_impact']['offset_ratio'] > 0.5:
            benefits.append(f"Strong netting: {best_option['netting_impact']['offset_ratio']:.0%} offset")

        if benefits:
            action += f" ({'; '.join(benefits[:2])})"

        return action

    def _generate_reasoning(self, best_option: Dict, all_options: List[Dict]) -> str:
        """Generate comprehensive reasoning for the recommendation."""
        reasoning_parts = []

        # Overall score
        confidence_level = "high" if best_option['composite_score'] > 0.7 else "medium" if best_option['composite_score'] > 0.4 else "low"
        reasoning_parts.append(f"Recommendation confidence: {confidence_level} ({best_option['composite_score']:.1%})")

        # FX reasoning
        fx = best_option['fx_impact']
        if fx['confidence'] > 0.6:
            reasoning_parts.append(f"FX analysis: {fx['reasoning']}")
        else:
            reasoning_parts.append("FX analysis: Limited confidence in FX prediction")

        # Netting reasoning
        netting = best_option['netting_impact']
        reasoning_parts.append(netting['reasoning'])

        # Working capital consideration
        wc = best_option['wc_cost']
        if wc['cost_amount'] > 0:
            reasoning_parts.append(f"Working capital cost: {wc['reasoning']}")
        else:
            reasoning_parts.append("No working capital cost (on-time or early payment)")

        # Comparison to alternatives
        if len(all_options) > 1:
            best_score = best_option['composite_score']
            next_best = max(opt['composite_score'] for opt in all_options if opt != best_option)
            score_diff = best_score - next_best
            if score_diff > 0.1:
                reasoning_parts.append(f"Strongly preferred over alternatives (score difference: {score_diff:.1%})")
            elif score_diff > 0.05:
                reasoning_parts.append(f"Moderately preferred over alternatives")

        return " ".join(reasoning_parts)

    def _format_alternatives(self, alternatives: List[Dict]) -> List[Dict]:
        """Format alternative options for presentation."""
        formatted = []

        for alt in alternatives:
            formatted.append({
                'action': self._format_recommended_action(alt),
                'confidence_score': alt['composite_score'],
                'key_tradeoffs': self._identify_tradeoffs(alt)
            })

        return formatted

    def _identify_tradeoffs(self, option: Dict) -> str:
        """Identify key tradeoffs of an alternative option."""
        tradeoffs = []

        if option['wc_cost']['relative_cost'] > 0.7:
            tradeoffs.append("Higher working capital cost")
        if option['fx_impact']['score'] < 0.4:
            tradeoffs.append("Unfavorable FX impact")
        if option['netting_impact']['score'] < 0.4:
            tradeoffs.append("Poor netting alignment")

        if not tradeoffs:
            tradeoffs.append("Balanced considerations")

        return "; ".join(tradeoffs)

    def _find_historical_examples(self, scenario: PaymentScenario, recommended_date: datetime) -> List[Dict]:
        """Find historical examples of similar payment decisions."""
        examples = []

        try:
            # Find similar payments by vendor and amount range
            similar_payments = self.behavior_forecaster.ap_data[
                (self.behavior_forecaster.ap_data['vendor'] == scenario.vendor) &
                (self.behavior_forecaster.ap_data['amount'].between(scenario.amount * 0.5, scenario.amount * 1.5)) &
                (self.behavior_forecaster.ap_data['currency'] == scenario.currency)
            ]

            for _, payment in similar_payments.head(3).iterrows():
                # Calculate what happened vs recommended timing
                actual_delay = (payment['actual_payment_date'] - payment['due_date']).days
                recommended_delay = (recommended_date - scenario.due_date).days

                timing_diff = abs(actual_delay - recommended_delay)
                similarity_score = max(0, 1 - (timing_diff / 30))  # Similarity based on timing difference

                if similarity_score > 0.3:  # Only include reasonably similar examples
                    examples.append({
                        'invoice_id': payment['invoice_id'],
                        'historical_date': payment['actual_payment_date'],
                        'amount': payment['amount'],
                        'similarity_score': similarity_score,
                        'outcome': self._assess_historical_outcome(payment, recommended_delay)
                    })

        except Exception:
            pass

        return examples[:3]  # Return up to 3 examples

    def _assess_historical_outcome(self, historical_payment: pd.Series, recommended_delay: int) -> str:
        """Assess the outcome of a historical payment decision."""
        actual_delay = (historical_payment['actual_payment_date'] - historical_payment['due_date']).days

        if abs(actual_delay - recommended_delay) <= 3:
            return "Similar timing - neutral outcome"
        elif actual_delay < recommended_delay:
            return "Paid earlier than recommended - may have missed FX opportunity"
        else:
            return "Paid later than recommended - potentially better FX outcome"

    def _generate_pain_protocol_suggestion(self, best_option: Dict) -> Dict:
        """Generate PAIN protocol integration suggestion."""
        # Placeholder for future PAIN protocol integration
        delay_days = best_option['days_delay']

        if delay_days > 7:
            # Suggest stablecoin rail with delay fallback
            return {
                'preferred_rail': 'stablecoin',
                'delay_fallback': f"wait {delay_days} days + retry on bank wire",
                'reasoning': 'Long delay allows stablecoin settlement with bank wire backup'
            }
        elif delay_days > 0:
            # Suggest immediate stablecoin
            return {
                'preferred_rail': 'stablecoin',
                'delay_fallback': None,
                'reasoning': 'Short delay suitable for stablecoin settlement'
            }
        else:
            # Suggest immediate bank wire
            return {
                'preferred_rail': 'bank_wire',
                'delay_fallback': None,
                'reasoning': 'Immediate payment best suited for traditional rails'
            }

    def get_portfolio_recommendations(self, days_ahead: int = 30) -> List[Recommendation]:
        """
        Generate recommendations for upcoming payments in the portfolio.

        Args:
            days_ahead: Number of days to look ahead

        Returns:
            List of recommendations for upcoming payments
        """
        # Get upcoming payments
        upcoming_payments = self.behavior_forecaster.ap_data[
            (self.behavior_forecaster.ap_data['due_date'] >= datetime.now()) &
            (self.behavior_forecaster.ap_data['due_date'] <= datetime.now() + timedelta(days=days_ahead))
        ]

        recommendations = []

        for _, payment in upcoming_payments.iterrows():
            # Create payment options (due date, +3, +7, +14 days)
            payment_options = [
                payment['due_date'],
                payment['due_date'] + timedelta(days=3),
                payment['due_date'] + timedelta(days=7),
                payment['due_date'] + timedelta(days=14)
            ]

            scenario = PaymentScenario(
                invoice_id=payment['invoice_id'],
                vendor=payment['vendor'],
                amount=payment['amount'],
                currency=payment['currency'],
                due_date=payment['due_date'],
                payment_options=payment_options
            )

            try:
                recommendation = self.analyze_payment_scenario(scenario)
                recommendations.append(recommendation)
            except Exception as e:
                print(f"Error generating recommendation for {payment['invoice_id']}: {str(e)}")
                continue

        # Sort by confidence score (highest first)
        recommendations.sort(key=lambda x: x.confidence_score, reverse=True)

        return recommendations


def create_recommendation_engine(fx_data: pd.DataFrame, ap_data: pd.DataFrame,
                               ar_data: pd.DataFrame) -> RecommendationEngine:
    """Factory function to create recommendation engine with all dependencies."""
    fx_predictor = FXPredictor(fx_data)
    behavior_forecaster = BehaviorForecaster(ap_data, ar_data)
    netting_optimizer = NettingOptimizer(ap_data, ar_data)

    return RecommendationEngine(fx_predictor, behavior_forecaster, netting_optimizer)


if __name__ == "__main__":
    # Example usage and testing
    from data_ingest import get_data_loader

    # Load data
    loader = get_data_loader()
    ap_data = loader.load_ap_data()
    ar_data = loader.load_ar_data()
    fx_data = loader.load_fx_data()

    # Create recommendation engine
    engine = create_recommendation_engine(fx_data, ap_data, ar_data)

    print("Kairo Recommendation Engine Demo")
    print("=" * 40)

    # Test with a sample payment scenario
    # Use historical data for demonstration since we don't have future payments
    sample_payment = ap_data.head(1).iloc[0]

    payment_options = [
        sample_payment['due_date'],
        sample_payment['due_date'] + timedelta(days=3),
        sample_payment['due_date'] + timedelta(days=7),
    ]

    scenario = PaymentScenario(
        invoice_id=sample_payment['invoice_id'],
        vendor=sample_payment['vendor'],
        amount=sample_payment['amount'],
        currency=sample_payment['currency'],
        due_date=sample_payment['due_date'],
        payment_options=payment_options
    )

    print(f"Analyzing payment: {scenario.invoice_id}")
    print(f"Vendor: {scenario.vendor}")
    print(f"Amount: ${scenario.amount:,.2f} {scenario.currency}")
    print(f"Due date: {scenario.due_date.strftime('%Y-%m-%d')}")
    print()

    try:
        recommendation = engine.analyze_payment_scenario(scenario)

        print("RECOMMENDATION:")
        print(f"{recommendation.recommended_action}")
        print()
        print("DETAILED ANALYSIS:")
        print(f"Confidence Score: {recommendation.confidence_score:.1%}")
        print(f"Expected FX Impact: {recommendation.expected_fx_impact:.2f}%")
        print(f"Expected Netting Impact: {recommendation.expected_netting_impact:.2f}")
        print()
        print("REASONING:")
        print(recommendation.reasoning)
        print()

        if recommendation.alternative_options:
            print("ALTERNATIVE OPTIONS:")
            for i, alt in enumerate(recommendation.alternative_options[:2], 1):
                print(f"{i}. {alt['action']} (Confidence: {alt['confidence_score']:.1%})")
            print()

        if recommendation.historical_examples:
            print("HISTORICAL EXAMPLES:")
            for example in recommendation.historical_examples:
                print(f"• {example['invoice_id']}: {example['outcome']} (Similarity: {example['similarity_score']:.1%})")
            print()

        if recommendation.pain_protocol_suggestion:
            pain = recommendation.pain_protocol_suggestion
            print("PAIN PROTOCOL SUGGESTION:")
            print(f"Preferred Rail: {pain['preferred_rail']}")
            if pain['delay_fallback']:
                print(f"Delay Fallback: {pain['delay_fallback']}")
            print(f"Reasoning: {pain['reasoning']}")

    except Exception as e:
        print(f"Error generating recommendation: {str(e)}")
        print("This may be due to limited historical data or model constraints.")