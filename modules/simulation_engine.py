"""
Simulation Engine for Kairo Treasury Optimization

Provides what-if analysis capabilities for payment timing decisions,
portfolio optimization scenarios, and treasury strategy simulations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
try:
    # Try relative imports first (when run as module)
    from .recommendation_engine import create_recommendation_engine, PaymentScenario
except ImportError:
    # Fall back to absolute imports (when run directly)
    from recommendation_engine import create_recommendation_engine, PaymentScenario


@dataclass
class SimulationScenario:
    """Represents a simulation scenario configuration."""
    name: str
    description: str
    payment_strategy: str  # 'current', 'optimized', 'delay_all', 'pay_early', 'custom'
    fx_hedging_level: float  # 0.0 to 1.0 (0 = no hedging, 1 = full hedging)
    time_horizon_days: int
    risk_tolerance: float  # 0.0 to 1.0 (0 = risk averse, 1 = risk seeking)


@dataclass
class SimulationResult:
    """Results from a treasury simulation."""
    scenario: SimulationScenario
    total_payments: int
    total_amount: float
    avg_payment_delay: float
    fx_savings_pct: float
    working_capital_cost: float
    netting_efficiency: float
    risk_exposure: float
    confidence_score: float
    cash_flow_impact: Dict[str, float]
    recommendations: List[str]


class TreasurySimulator:
    """
    Advanced simulation engine for treasury optimization scenarios.

    Provides comprehensive what-if analysis for payment timing strategies,
    hedging decisions, and portfolio optimization.
    """

    def __init__(self, ap_data: pd.DataFrame, ar_data: pd.DataFrame,
                 fx_data: pd.DataFrame, recommendation_engine):
        """
        Initialize the treasury simulator.

        Args:
            ap_data: Accounts Payable data
            ar_data: Accounts Receivable data
            fx_data: FX rates data
            recommendation_engine: Initialized recommendation engine
        """
        self.ap_data = ap_data.copy()
        self.ar_data = ar_data.copy()
        self.fx_data = fx_data.copy()
        self.recommendation_engine = recommendation_engine

        # Simulation parameters
        self.base_fx_rate = 1.0  # USD baseline
        self.discount_rate = 0.05  # 5% annual working capital cost

    def run_payment_timing_simulation(self, scenario: SimulationScenario,
                                    payment_subset: Optional[pd.DataFrame] = None) -> SimulationResult:
        """
        Run a payment timing simulation scenario.

        Args:
            scenario: Simulation scenario configuration
            payment_subset: Optional subset of payments to simulate

        Returns:
            Comprehensive simulation results
        """
        # Get payments for simulation
        if payment_subset is not None:
            payments = payment_subset.copy()
        else:
            # Get upcoming payments within time horizon
            cutoff_date = datetime.now() + timedelta(days=scenario.time_horizon_days)
            payments = self.ap_data[
                (self.ap_data['due_date'] >= datetime.now()) &
                (self.ap_data['due_date'] <= cutoff_date)
            ].copy()

        if len(payments) == 0:
            return self._create_empty_result(scenario)

        # Apply payment strategy
        simulated_payments = self._apply_payment_strategy(payments, scenario)

        # Calculate simulation metrics
        metrics = self._calculate_simulation_metrics(simulated_payments, scenario)

        # Generate recommendations
        recommendations = self._generate_simulation_recommendations(metrics, scenario)

        return SimulationResult(
            scenario=scenario,
            total_payments=len(simulated_payments),
            total_amount=simulated_payments['amount'].sum(),
            avg_payment_delay=simulated_payments['simulated_delay_days'].mean(),
            fx_savings_pct=metrics['fx_savings'],
            working_capital_cost=metrics['working_capital_cost'],
            netting_efficiency=metrics['netting_efficiency'],
            risk_exposure=metrics['risk_exposure'],
            confidence_score=metrics['confidence_score'],
            cash_flow_impact=metrics['cash_flow_impact'],
            recommendations=recommendations
        )

    def _apply_payment_strategy(self, payments: pd.DataFrame, scenario: SimulationScenario) -> pd.DataFrame:
        """Apply the specified payment strategy to the payment dataset."""
        df = payments.copy()

        if scenario.payment_strategy == 'current':
            # Use historical payment behavior
            df['simulated_payment_date'] = df['due_date'] + pd.to_timedelta(df['payment_lag_days'], unit='D')
            df['simulated_delay_days'] = df['payment_lag_days']

        elif scenario.payment_strategy == 'optimized':
            # Use AI recommendations for each payment
            df = self._apply_optimized_strategy(df)

        elif scenario.payment_strategy == 'delay_all':
            # Delay all payments by a fixed amount (based on risk tolerance)
            delay_days = int(scenario.risk_tolerance * 30)  # 0-30 days based on risk tolerance
            df['simulated_payment_date'] = df['due_date'] + timedelta(days=delay_days)
            df['simulated_delay_days'] = delay_days

        elif scenario.payment_strategy == 'pay_early':
            # Pay early to capture discounts (opposite of delay)
            early_days = int((1 - scenario.risk_tolerance) * 10)  # 0-10 days early
            df['simulated_payment_date'] = df['due_date'] - timedelta(days=early_days)
            df['simulated_delay_days'] = -early_days

        elif scenario.payment_strategy == 'custom':
            # Custom strategy - could be extended
            df['simulated_payment_date'] = df['due_date']
            df['simulated_delay_days'] = 0

        else:
            raise ValueError(f"Unknown payment strategy: {scenario.payment_strategy}")

        return df

    def _apply_optimized_strategy(self, payments: pd.DataFrame) -> pd.DataFrame:
        """Apply AI-optimized payment timing to each payment."""
        optimized_payments = []

        for _, payment in payments.iterrows():
            try:
                # Create payment scenario with multiple options
                payment_options = [
                    payment['due_date'],  # On time
                    payment['due_date'] + timedelta(days=3),
                    payment['due_date'] + timedelta(days=7),
                    payment['due_date'] + timedelta(days=14),
                ]

                scenario = PaymentScenario(
                    invoice_id=payment['invoice_id'],
                    vendor=payment['vendor'],
                    amount=payment['amount'],
                    currency=payment['currency'],
                    due_date=payment['due_date'],
                    payment_options=payment_options
                )

                # Get recommendation
                recommendation = self.recommendation_engine.analyze_payment_scenario(scenario)

                # Extract recommended payment date from recommendation text
                recommended_date = self._parse_recommended_date(recommendation.recommended_action, payment['due_date'])

                optimized_payments.append({
                    **payment.to_dict(),
                    'simulated_payment_date': recommended_date,
                    'simulated_delay_days': (recommended_date - payment['due_date']).days,
                    'optimization_confidence': recommendation.confidence_score,
                    'expected_fx_impact': recommendation.expected_fx_impact
                })

            except Exception as e:
                # Fallback to current behavior if optimization fails
                optimized_payments.append({
                    **payment.to_dict(),
                    'simulated_payment_date': payment['due_date'] + timedelta(days=payment.get('payment_lag_days', 0)),
                    'simulated_delay_days': payment.get('payment_lag_days', 0),
                    'optimization_confidence': 0.5,
                    'expected_fx_impact': 0.0
                })

        return pd.DataFrame(optimized_payments)

    def _parse_recommended_date(self, recommendation_text: str, due_date: datetime) -> datetime:
        """Parse the recommended payment date from recommendation text."""
        text = recommendation_text.lower()

        if 'due date' in text or 'on time' in text:
            return due_date
        elif 'delay' in text:
            # Extract delay days
            import re
            delay_match = re.search(r'delay.*by (\d+) days', text)
            if delay_match:
                delay_days = int(delay_match.group(1))
                return due_date + timedelta(days=delay_days)

        # Default to due date
        return due_date

    def _calculate_simulation_metrics(self, payments: pd.DataFrame, scenario: SimulationScenario) -> Dict[str, Any]:
        """Calculate comprehensive metrics for the simulation."""
        # FX Savings Calculation
        fx_savings = self._calculate_fx_savings(payments, scenario)

        # Working Capital Cost
        wc_cost = self._calculate_working_capital_cost(payments)

        # Netting Efficiency
        netting_efficiency = self._calculate_netting_efficiency(payments, scenario)

        # Risk Exposure
        risk_exposure = self._calculate_risk_exposure(payments, scenario)

        # Confidence Score
        confidence_score = self._calculate_simulation_confidence(payments, scenario)

        # Cash Flow Impact
        cash_flow_impact = self._calculate_cash_flow_impact(payments)

        return {
            'fx_savings': fx_savings,
            'working_capital_cost': wc_cost,
            'netting_efficiency': netting_efficiency,
            'risk_exposure': risk_exposure,
            'confidence_score': confidence_score,
            'cash_flow_impact': cash_flow_impact
        }

    def _calculate_fx_savings(self, payments: pd.DataFrame, scenario: SimulationScenario) -> float:
        """Calculate FX savings from payment timing optimization."""
        total_base_fx_impact = 0
        total_amount = 0

        for _, payment in payments.iterrows():
            # Simplified FX impact calculation
            # In production, would use actual FX rate predictions
            currency_multiplier = 1.0
            if payment['currency'] != 'USD':
                # Assume some FX volatility impact
                delay_impact = payment.get('expected_fx_impact', 0)
                currency_multiplier = 1 + (delay_impact / 100)

            # Apply hedging factor (less FX impact if hedged)
            effective_fx_impact = currency_multiplier * (1 - scenario.fx_hedging_level)

            total_base_fx_impact += payment['amount'] * (effective_fx_impact - 1)
            total_amount += payment['amount']

        return (total_base_fx_impact / total_amount) * 100 if total_amount > 0 else 0

    def _calculate_working_capital_cost(self, payments: pd.DataFrame) -> float:
        """Calculate working capital cost of payment delays."""
        total_cost = 0

        for _, payment in payments.iterrows():
            delay_days = payment.get('simulated_delay_days', 0)
            if delay_days > 0:
                # Daily cost based on discount rate
                daily_cost = self.discount_rate / 365
                cost = payment['amount'] * daily_cost * delay_days
                total_cost += cost

        return total_cost

    def _calculate_netting_efficiency(self, payments: pd.DataFrame, scenario: SimulationScenario) -> float:
        """Calculate netting efficiency for the simulated payments."""
        # Group payments by currency and week
        payments['payment_week'] = payments['simulated_payment_date'].dt.to_period('W').dt.start_time

        weekly_netting = payments.groupby(['currency', 'payment_week'])['amount'].sum()

        # Calculate potential offset with receipts
        # Simplified: assume some receipts in same period
        total_outflows = payments['amount'].sum()
        estimated_offset = total_outflows * 0.3  # Assume 30% natural offset

        return (estimated_offset / total_outflows) * 100 if total_outflows > 0 else 0

    def _calculate_risk_exposure(self, payments: pd.DataFrame, scenario: SimulationScenario) -> float:
        """Calculate risk exposure based on payment timing and hedging."""
        # Base risk from payment delays
        avg_delay = payments['simulated_delay_days'].mean()
        delay_risk = min(avg_delay / 30, 1)  # Normalize to 0-1

        # FX risk based on currency exposure and hedging
        currency_risk = len(payments['currency'].unique()) / 5  # More currencies = more risk

        # Combined risk (lower hedging = higher risk)
        combined_risk = (delay_risk + currency_risk) / 2
        effective_risk = combined_risk * (1 - scenario.fx_hedging_level)

        return effective_risk * 100  # Return as percentage

    def _calculate_simulation_confidence(self, payments: pd.DataFrame, scenario: SimulationScenario) -> float:
        """Calculate overall confidence in simulation results."""
        # Average optimization confidence if available
        if 'optimization_confidence' in payments.columns:
            avg_confidence = payments['optimization_confidence'].mean()
        else:
            avg_confidence = 0.7  # Default for non-optimized strategies

        # Adjust based on scenario parameters
        strategy_confidence = {
            'optimized': 1.0,
            'current': 0.8,
            'delay_all': 0.6,
            'pay_early': 0.6,
            'custom': 0.5
        }.get(scenario.payment_strategy, 0.5)

        return (avg_confidence + strategy_confidence) / 2

    def _calculate_cash_flow_impact(self, payments: pd.DataFrame) -> Dict[str, float]:
        """Calculate cash flow impact by time period."""
        # Group payments by week
        payments['week'] = payments['simulated_payment_date'].dt.to_period('W').dt.start_time
        weekly_cash_flow = payments.groupby('week')['amount'].sum()

        # Calculate key metrics
        total_outflow = payments['amount'].sum()
        peak_weekly_outflow = weekly_cash_flow.max()
        cash_flow_volatility = weekly_cash_flow.std() / weekly_cash_flow.mean() if weekly_cash_flow.mean() > 0 else 0

        return {
            'total_outflow': total_outflow,
            'peak_weekly_outflow': peak_weekly_outflow,
            'cash_flow_volatility': cash_flow_volatility,
            'weeks_covered': len(weekly_cash_flow)
        }

    def _generate_simulation_recommendations(self, metrics: Dict[str, Any],
                                           scenario: SimulationScenario) -> List[str]:
        """Generate recommendations based on simulation results."""
        recommendations = []

        # FX Savings recommendation
        if metrics['fx_savings'] > 0.5:
            recommendations.append(f"💰 Expected FX savings: {metrics['fx_savings']:.2f}% - Consider implementing this strategy")
        elif metrics['fx_savings'] < -0.5:
            recommendations.append(f"⚠️ Potential FX losses: {abs(metrics['fx_savings']):.2f}% - Review hedging strategy")

        # Working capital recommendation
        if metrics['working_capital_cost'] > 1000:
            recommendations.append(f"💸 Working capital cost: ${metrics['working_capital_cost']:,.0f} - Balance against FX benefits")

        # Risk recommendation
        if metrics['risk_exposure'] > 70:
            recommendations.append("⚠️ High risk exposure - Consider increasing hedging or shortening payment delays")
        elif metrics['risk_exposure'] < 30:
            recommendations.append("✅ Low risk exposure - Strategy appears conservative")

        # Netting recommendation
        if metrics['netting_efficiency'] > 40:
            recommendations.append("🔄 Good netting efficiency - Natural hedges reducing FX exposure")

        # Confidence recommendation
        if metrics['confidence_score'] < 0.6:
            recommendations.append("🤔 Low confidence in results - Consider gathering more historical data")

        return recommendations if recommendations else ["✅ Simulation completed - Review metrics above for insights"]

    def _create_empty_result(self, scenario: SimulationScenario) -> SimulationResult:
        """Create an empty result for scenarios with no payments."""
        return SimulationResult(
            scenario=scenario,
            total_payments=0,
            total_amount=0,
            avg_payment_delay=0,
            fx_savings_pct=0,
            working_capital_cost=0,
            netting_efficiency=0,
            risk_exposure=0,
            confidence_score=0,
            cash_flow_impact={},
            recommendations=["No payments found in the specified time horizon"]
        )

    def run_portfolio_comparison(self, scenarios: List[SimulationScenario],
                               days_ahead: int = 90) -> Dict[str, SimulationResult]:
        """
        Run multiple scenarios and compare results.

        Args:
            scenarios: List of scenarios to compare
            days_ahead: Time horizon for analysis

        Returns:
            Dictionary mapping scenario names to results
        """
        results = {}

        # Get payment subset for all scenarios
        cutoff_date = datetime.now() + timedelta(days=days_ahead)
        payment_subset = self.ap_data[
            (self.ap_data['due_date'] >= datetime.now()) &
            (self.ap_data['due_date'] <= cutoff_date)
        ]

        for scenario in scenarios:
            try:
                result = self.run_payment_timing_simulation(scenario, payment_subset)
                results[scenario.name] = result
            except Exception as e:
                print(f"Error running scenario {scenario.name}: {str(e)}")
                results[scenario.name] = self._create_empty_result(scenario)

        return results

    def generate_scenario_report(self, results: Dict[str, SimulationResult]) -> pd.DataFrame:
        """Generate a comparison report from multiple scenario results."""
        report_data = []

        for scenario_name, result in results.items():
            report_data.append({
                'Scenario': scenario_name,
                'Payments': result.total_payments,
                'Total Amount': result.total_amount,
                'Avg Delay (days)': result.avg_payment_delay,
                'FX Savings (%)': result.fx_savings_pct,
                'Working Capital Cost': result.working_capital_cost,
                'Netting Efficiency (%)': result.netting_efficiency,
                'Risk Exposure (%)': result.risk_exposure,
                'Confidence Score': result.confidence_score,
                'Cash Flow Volatility': result.cash_flow_impact.get('cash_flow_volatility', 0)
            })

        return pd.DataFrame(report_data)


def create_treasury_simulator(ap_data: pd.DataFrame, ar_data: pd.DataFrame,
                            fx_data: pd.DataFrame) -> TreasurySimulator:
    """Factory function to create treasury simulator."""
    recommendation_engine = create_recommendation_engine(fx_data, ap_data, ar_data)
    return TreasurySimulator(ap_data, ar_data, fx_data, recommendation_engine)


# Predefined scenario templates
DEFAULT_SCENARIOS = {
    'current_behavior': SimulationScenario(
        name='Current Behavior',
        description='Maintain current payment patterns based on historical behavior',
        payment_strategy='current',
        fx_hedging_level=0.5,
        time_horizon_days=90,
        risk_tolerance=0.5
    ),

    'ai_optimized': SimulationScenario(
        name='AI Optimized',
        description='Use AI recommendations for optimal payment timing',
        payment_strategy='optimized',
        fx_hedging_level=0.3,
        time_horizon_days=90,
        risk_tolerance=0.6
    ),

    'conservative_delay': SimulationScenario(
        name='Conservative Delay',
        description='Delay all payments by moderate amounts with full hedging',
        payment_strategy='delay_all',
        fx_hedging_level=0.8,
        time_horizon_days=90,
        risk_tolerance=0.3
    ),

    'aggressive_delay': SimulationScenario(
        name='Aggressive Delay',
        description='Maximize payment delays with minimal hedging',
        payment_strategy='delay_all',
        fx_hedging_level=0.2,
        time_horizon_days=90,
        risk_tolerance=0.8
    ),

    'early_payment': SimulationScenario(
        name='Early Payment',
        description='Pay early to maintain relationships with minimal delays',
        payment_strategy='pay_early',
        fx_hedging_level=0.6,
        time_horizon_days=90,
        risk_tolerance=0.2
    )
}


if __name__ == "__main__":
    # Example usage
    from data_ingest import get_data_loader

    # Load data
    loader = get_data_loader()
    ap_data = loader.load_ap_data()
    ar_data = loader.load_ar_data()
    fx_data = loader.load_fx_data()

    # Create simulator
    simulator = create_treasury_simulator(ap_data, ar_data, fx_data)

    # Run a sample scenario using historical data for demonstration
    # Filter for payments in a past period
    historical_payments = ap_data[
        (ap_data['due_date'] >= '2023-01-01') &
        (ap_data['due_date'] <= '2023-03-31')
    ].copy()

    if len(historical_payments) > 0:
        scenario = DEFAULT_SCENARIOS['ai_optimized']
        print(f"Running simulation: {scenario.name}")
        print(f"Description: {scenario.description}")
        print(f"Using {len(historical_payments)} historical payments for demonstration")
        print("-" * 50)

        result = simulator.run_payment_timing_simulation(scenario, historical_payments)
    else:
        print("No historical payments found for demonstration")
        result = simulator.run_payment_timing_simulation(DEFAULT_SCENARIOS['ai_optimized'])

    print("Results:")
    print(f"Total payments: {result.total_payments}")
    print(f"Total amount: ${result.total_amount:,.0f}")
    print(".1f")
    print(".2f")
    print(f"Working capital cost: ${result.working_capital_cost:,.0f}")
    print(".1f")
    print(".1f")
    print(".1%")
    print()

    print("Recommendations:")
    for rec in result.recommendations:
        print(f"• {rec}")

    print()

    # Run comparison using historical data
    print("Running scenario comparison...")
    scenarios_to_compare = [
        DEFAULT_SCENARIOS['current_behavior'],
        DEFAULT_SCENARIOS['ai_optimized'],
        DEFAULT_SCENARIOS['conservative_delay']
    ]

    # Use historical data for comparison
    historical_payments = ap_data[
        (ap_data['due_date'] >= '2023-01-01') &
        (ap_data['due_date'] <= '2023-03-31')
    ].copy()

    comparison_results = simulator.run_portfolio_comparison(scenarios_to_compare, 90)

    # Override with historical data for each scenario
    for scenario in scenarios_to_compare:
        if len(historical_payments) > 0:
            comparison_results[scenario.name] = simulator.run_payment_timing_simulation(scenario, historical_payments)
    report = simulator.generate_scenario_report(comparison_results)

    print("Scenario Comparison:")
    print(report.to_string(index=False))