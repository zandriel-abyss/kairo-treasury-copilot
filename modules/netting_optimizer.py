"""
Netting Optimization Module for Kairo Treasury Optimization

Identifies natural hedging opportunities by matching cash inflows and outflows
across currencies and time periods to minimize FX exposure.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class NettingOptimizationError(Exception):
    """Custom exception for netting optimization errors"""
    pass


class NettingOptimizer:
    """
    Optimizes cash flow netting to identify natural hedging opportunities.

    Matches payments and receipts by currency and timing to reduce FX exposure
    and provide recommendations for treasury optimization.
    """

    def __init__(self, ap_data: pd.DataFrame, ar_data: pd.DataFrame):
        """
        Initialize the netting optimizer.

        Args:
            ap_data: Accounts Payable data with behavioral forecasts
            ar_data: Accounts Receivable data with behavioral forecasts
        """
        self.ap_data = ap_data.copy()
        self.ar_data = ar_data.copy()

        # Validate data structure
        self._validate_data()

    def _validate_data(self):
        """Validate that required columns exist in the data."""
        ap_required = ['invoice_id', 'vendor', 'amount', 'currency', 'due_date', 'actual_payment_date']
        ar_required = ['invoice_id', 'customer', 'amount', 'currency', 'expected_payment_date', 'actual_payment_date']

        ap_missing = [col for col in ap_required if col not in self.ap_data.columns]
        ar_missing = [col for col in ar_required if col not in self.ar_data.columns]

        if ap_missing:
            raise NettingOptimizationError(f"AP data missing required columns: {ap_missing}")
        if ar_missing:
            raise NettingOptimizationError(f"AR data missing required columns: {ar_missing}")

    def analyze_netting_opportunities(self, time_window_days: int = 30,
                                    currency_focus: Optional[str] = None) -> Dict:
        """
        Analyze netting opportunities within a time window.

        Args:
            time_window_days: Number of days to look ahead
            currency_focus: Specific currency to analyze (optional)

        Returns:
            Dictionary with netting analysis results
        """
        # Get forecasted cash flows
        outflows = self._get_forecasted_outflows(time_window_days, currency_focus)
        inflows = self._get_forecasted_inflows(time_window_days, currency_focus)

        # Combine into time buckets (weekly for analysis)
        time_buckets = self._create_time_buckets(time_window_days)

        netting_results = {}
        total_offset_potential = 0
        total_residual_exposure = 0

        for bucket_start, bucket_end in time_buckets:
            bucket_outflows = outflows[
                (outflows['expected_date'] >= bucket_start) &
                (outflows['expected_date'] <= bucket_end)
            ]

            bucket_inflows = inflows[
                (inflows['expected_date'] >= bucket_start) &
                (inflows['expected_date'] <= bucket_end)
            ]

            # Analyze netting by currency
            bucket_netting = self._analyze_bucket_netting(bucket_outflows, bucket_inflows, bucket_start)

            netting_results[f"{bucket_start.strftime('%Y-%m-%d')} to {bucket_end.strftime('%Y-%m-%d')}"] = bucket_netting

            total_offset_potential += bucket_netting['offset_potential']
            total_residual_exposure += bucket_netting['residual_exposure']

        # Overall netting efficiency
        total_gross_exposure = sum(abs(result['gross_exposure']) for result in netting_results.values())
        netting_efficiency = (total_offset_potential / total_gross_exposure) if total_gross_exposure > 0 else 0

        return {
            'analysis_period_days': time_window_days,
            'currency_focus': currency_focus or 'all_currencies',
            'netting_efficiency': netting_efficiency,
            'total_offset_potential': total_offset_potential,
            'total_residual_exposure': total_residual_exposure,
            'time_bucket_results': netting_results,
            'recommendations': self._generate_netting_recommendations(netting_efficiency, netting_results)
        }

    def _get_forecasted_outflows(self, days_ahead: int, currency: Optional[str] = None) -> pd.DataFrame:
        """Get forecasted payment outflows."""
        cutoff_date = datetime.now() + timedelta(days=days_ahead)
        outflows = self.ap_data[
            (self.ap_data['due_date'] >= datetime.now()) &
            (self.ap_data['due_date'] <= cutoff_date)
        ].copy()

        if currency:
            outflows = outflows[outflows['currency'] == currency]

        # For simplicity, use due_date as expected payment date
        # In production, this would use behavioral forecasts
        outflows['expected_date'] = outflows['due_date']
        outflows['flow_type'] = 'outflow'
        outflows['counterparty'] = outflows['vendor']

        return outflows[['invoice_id', 'counterparty', 'amount', 'currency', 'expected_date', 'flow_type']]

    def _get_forecasted_inflows(self, days_ahead: int, currency: Optional[str] = None) -> pd.DataFrame:
        """Get forecasted collection inflows."""
        cutoff_date = datetime.now() + timedelta(days=days_ahead)
        inflows = self.ar_data[
            (self.ar_data['expected_payment_date'] >= datetime.now()) &
            (self.ar_data['expected_payment_date'] <= cutoff_date)
        ].copy()

        if currency:
            inflows = inflows[inflows['currency'] == currency]

        # For simplicity, use expected_payment_date as expected collection date
        inflows['expected_date'] = inflows['expected_payment_date']
        inflows['flow_type'] = 'inflow'
        inflows['counterparty'] = inflows['customer']

        return inflows[['invoice_id', 'counterparty', 'amount', 'currency', 'expected_date', 'flow_type']]

    def _create_time_buckets(self, total_days: int, bucket_size_days: int = 7) -> List[Tuple[datetime, datetime]]:
        """Create time buckets for analysis."""
        buckets = []
        current_date = datetime.now()

        while current_date < datetime.now() + timedelta(days=total_days):
            bucket_end = min(current_date + timedelta(days=bucket_size_days - 1),
                           datetime.now() + timedelta(days=total_days))
            buckets.append((current_date, bucket_end))
            current_date = bucket_end + timedelta(days=1)

        return buckets

    def _analyze_bucket_netting(self, outflows: pd.DataFrame, inflows: pd.DataFrame,
                              bucket_start: datetime) -> Dict:
        """Analyze netting opportunities within a specific time bucket."""
        # Group by currency
        currency_netting = {}

        all_currencies = set(outflows['currency'].unique()) | set(inflows['currency'].unique())

        bucket_gross_exposure = 0
        bucket_offset_potential = 0
        bucket_residual_exposure = 0

        for currency in all_currencies:
            curr_outflows = outflows[outflows['currency'] == currency]['amount'].sum()
            curr_inflows = inflows[inflows['currency'] == currency]['amount'].sum()

            # Net position for this currency
            net_position = curr_inflows - curr_outflows

            # Offset potential is the minimum of inflows and outflows
            offset_potential = min(curr_inflows, curr_outflows)
            residual_exposure = abs(net_position)

            currency_netting[currency] = {
                'outflows': curr_outflows,
                'inflows': curr_inflows,
                'net_position': net_position,
                'offset_potential': offset_potential,
                'residual_exposure': residual_exposure
            }

            bucket_offset_potential += offset_potential
            bucket_residual_exposure += residual_exposure
            bucket_gross_exposure += curr_outflows + curr_inflows

        return {
            'bucket_start': bucket_start,
            'currency_breakdown': currency_netting,
            'gross_exposure': bucket_gross_exposure,
            'offset_potential': bucket_offset_potential,
            'residual_exposure': bucket_residual_exposure,
            'netting_efficiency': bucket_offset_potential / bucket_gross_exposure if bucket_gross_exposure > 0 else 0
        }

    def identify_cross_currency_opportunities(self, time_window_days: int = 30) -> Dict:
        """
        Identify cross-currency netting opportunities using FX rates.

        Args:
            time_window_days: Number of days to analyze

        Returns:
            Dictionary with cross-currency netting opportunities
        """
        # This is a simplified version - in production would use current FX rates
        # to find offsetting flows in different currencies

        outflows = self._get_forecasted_outflows(time_window_days)
        inflows = self._get_forecasted_inflows(time_window_days)

        # Group by time buckets and currency
        opportunities = []

        for (flow_date, currency), group in pd.concat([outflows, inflows]).groupby(['expected_date', 'currency']):
            outflow_amount = group[group['flow_type'] == 'outflow']['amount'].sum()
            inflow_amount = group[group['flow_type'] == 'inflow']['amount'].sum()

            if outflow_amount > 0 and inflow_amount > 0:
                opportunities.append({
                    'date': flow_date,
                    'currency': currency,
                    'outflow_amount': outflow_amount,
                    'inflow_amount': inflow_amount,
                    'offset_potential': min(outflow_amount, inflow_amount)
                })

        # Sort by offset potential
        opportunities.sort(key=lambda x: x['offset_potential'], reverse=True)

        total_offset_potential = sum(opp['offset_potential'] for opp in opportunities)

        return {
            'time_window_days': time_window_days,
            'total_opportunities': len(opportunities),
            'total_offset_potential': total_offset_potential,
            'top_opportunities': opportunities[:10],  # Top 10 opportunities
            'recommendations': self._generate_cross_currency_recommendations(opportunities)
        }

    def optimize_payment_timing(self, target_currency: str, time_window_days: int = 30) -> Dict:
        """
        Optimize payment timing to maximize natural hedging.

        Args:
            target_currency: Currency to optimize for
            time_window_days: Time window to consider

        Returns:
            Dictionary with payment timing optimization recommendations
        """
        outflows = self._get_forecasted_outflows(time_window_days, target_currency)
        inflows = self._get_forecasted_inflows(time_window_days, target_currency)

        if len(outflows) == 0:
            return {'error': f'No outflows found for {target_currency}'}

        # Find timing windows where delaying payments could improve netting
        timing_optimization = []

        for _, payment in outflows.iterrows():
            payment_date = payment['expected_date']
            payment_amount = payment['amount']

            # Check netting efficiency if paid on different dates
            original_netting = self._calculate_netting_for_date(target_currency, payment_date, outflows, inflows)

            # Test delaying by 3, 7, 14 days
            delay_options = []
            for delay_days in [3, 7, 14]:
                delayed_date = payment_date + timedelta(days=delay_days)

                # Remove this payment from original date and add to delayed date
                adjusted_outflows = outflows.copy()
                adjusted_outflows.loc[adjusted_outflows['invoice_id'] == payment['invoice_id'], 'expected_date'] = delayed_date

                delayed_netting = self._calculate_netting_for_date(target_currency, payment_date, adjusted_outflows, inflows)

                improvement = delayed_netting - original_netting
                delay_options.append({
                    'delay_days': delay_days,
                    'delayed_date': delayed_date,
                    'netting_improvement': improvement,
                    'fx_savings_estimate': improvement * 0.005  # Rough estimate: 0.5% FX benefit per unit offset
                })

            # Find best delay option
            best_delay = max(delay_options, key=lambda x: x['netting_improvement'])

            timing_optimization.append({
                'invoice_id': payment['invoice_id'],
                'counterparty': payment['counterparty'],
                'amount': payment_amount,
                'original_date': payment_date,
                'recommended_delay': best_delay['delay_days'] if best_delay['netting_improvement'] > 0 else 0,
                'recommended_date': best_delay['delayed_date'] if best_delay['netting_improvement'] > 0 else payment_date,
                'expected_netting_improvement': best_delay['netting_improvement'],
                'estimated_fx_benefit': best_delay['fx_savings_estimate']
            })

        # Sort by potential benefit
        timing_optimization.sort(key=lambda x: x['expected_netting_improvement'], reverse=True)

        total_potential_benefit = sum(opt['estimated_fx_benefit'] for opt in timing_optimization)

        return {
            'target_currency': target_currency,
            'time_window_days': time_window_days,
            'total_payments_analyzed': len(timing_optimization),
            'total_potential_fx_benefit': total_potential_benefit,
            'timing_recommendations': timing_optimization,
            'summary': self._generate_timing_optimization_summary(timing_optimization)
        }

    def _calculate_netting_for_date(self, currency: str, date: datetime,
                                  outflows: pd.DataFrame, inflows: pd.DataFrame) -> float:
        """Calculate netting efficiency for a specific date."""
        date_outflows = outflows[
            (outflows['currency'] == currency) &
            (outflows['expected_date'] == date)
        ]['amount'].sum()

        date_inflows = inflows[
            (inflows['currency'] == currency) &
            (inflows['expected_date'] == date)
        ]['amount'].sum()

        return min(date_outflows, date_inflows)

    def _generate_netting_recommendations(self, overall_efficiency: float, bucket_results: Dict) -> List[str]:
        """Generate netting recommendations based on analysis."""
        recommendations = []

        if overall_efficiency > 0.7:
            recommendations.append("Excellent natural hedging through netting - consider reducing FX hedges.")
        elif overall_efficiency > 0.4:
            recommendations.append("Moderate netting opportunities available - optimize payment timing.")
        else:
            recommendations.append("Limited netting opportunities - maintain comprehensive FX hedging strategy.")

        # Check for timing-specific recommendations
        high_efficiency_buckets = [bucket for bucket, result in bucket_results.items()
                                 if result['netting_efficiency'] > 0.6]

        if high_efficiency_buckets:
            recommendations.append(f"Focus hedging efforts outside of high-netting periods: {', '.join(high_efficiency_buckets[:3])}")

        return recommendations

    def _generate_cross_currency_recommendations(self, opportunities: List[Dict]) -> List[str]:
        """Generate cross-currency netting recommendations."""
        recommendations = []

        if not opportunities:
            recommendations.append("No significant cross-currency netting opportunities identified.")
            return recommendations

        total_opportunities = len(opportunities)
        high_potential = [opp for opp in opportunities if opp['offset_potential'] > 10000]

        if high_potential:
            recommendations.append(f"{len(high_potential)} high-value netting opportunities identified (> $10K each).")

        recommendations.append(f"Total of {total_opportunities} netting opportunities across all currencies.")

        # Currency concentration
        currency_counts = {}
        for opp in opportunities:
            currency_counts[opp['currency']] = currency_counts.get(opp['currency'], 0) + 1

        top_currencies = sorted(currency_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_currencies:
            recommendations.append(f"Focus on currencies: {', '.join([f'{curr} ({count} opportunities)' for curr, count in top_currencies])}")

        return recommendations

    def _generate_timing_optimization_summary(self, optimizations: List[Dict]) -> str:
        """Generate summary of timing optimization results."""
        if not optimizations:
            return "No payments available for timing optimization."

        beneficial_delays = [opt for opt in optimizations if opt['recommended_delay'] > 0]
        total_benefit = sum(opt['estimated_fx_benefit'] for opt in beneficial_delays)

        if beneficial_delays:
            return f"{len(beneficial_delays)} payments can benefit from timing optimization, " \
                   f"potentially saving ${total_benefit:.0f} in FX costs."
        else:
            return "No beneficial payment timing adjustments identified - current schedule is optimal."


def create_netting_optimizer(ap_data: pd.DataFrame, ar_data: pd.DataFrame) -> NettingOptimizer:
    """Factory function to create netting optimizer instance."""
    return NettingOptimizer(ap_data, ar_data)


if __name__ == "__main__":
    # Example usage and testing
    from data_ingest import get_data_loader

    # Load data
    loader = get_data_loader()
    ap_data = loader.load_ap_data()
    ar_data = loader.load_ar_data()

    # Create optimizer
    optimizer = NettingOptimizer(ap_data, ar_data)

    print("Netting Optimization Analysis:")
    print("=" * 40)

    # Test netting analysis (using historical data for demonstration)
    # Since our mock data is from 2023, we'll analyze a past period
    analysis_start = datetime(2023, 6, 1)
    analysis_end = datetime(2023, 7, 1)

    # Filter data for the analysis period
    period_ap = ap_data[
        (ap_data['due_date'] >= analysis_start) &
        (ap_data['due_date'] <= analysis_end)
    ]
    period_ar = ar_data[
        (ar_data['expected_payment_date'] >= analysis_start) &
        (ar_data['expected_payment_date'] <= analysis_end)
    ]

    if len(period_ap) > 0 and len(period_ar) > 0:
        # Create temporary optimizer for this period
        period_optimizer = NettingOptimizer(period_ap, period_ar)

        netting_analysis = period_optimizer.analyze_netting_opportunities(
            time_window_days=30, currency_focus=None
        )

        print(f"Netting Efficiency: {netting_analysis['netting_efficiency']:.1%}")
        print(f"Total Offset Potential: ${netting_analysis['total_offset_potential']:,.0f}")
        print(f"Total Residual Exposure: ${netting_analysis['total_residual_exposure']:,.0f}")

        if netting_analysis['recommendations']:
            print(f"Key Recommendations: {netting_analysis['recommendations'][0]}")

        # Test cross-currency opportunities
        print("\nCross-Currency Opportunities:")
        print("=" * 35)

        cross_currency = period_optimizer.identify_cross_currency_opportunities(30)
        print(f"Total Opportunities: {cross_currency['total_opportunities']}")
        print(f"Total Offset Potential: ${cross_currency['total_offset_potential']:,.0f}")

        if cross_currency['recommendations']:
            print(f"Recommendation: {cross_currency['recommendations'][0]}")

        # Test payment timing optimization
        print("\nPayment Timing Optimization:")
        print("=" * 32)

        # Test for USD if available
        usd_ap = period_ap[period_ap['currency'] == 'USD']
        if len(usd_ap) > 0:
            timing_opt = period_optimizer.optimize_payment_timing('USD', 30)
            if 'error' not in timing_opt:
                print(f"Analysis for USD payments: {timing_opt['summary']}")
                if timing_opt['timing_recommendations']:
                    top_rec = timing_opt['timing_recommendations'][0]
                    print(f"Top recommendation: Delay payment {top_rec['invoice_id']} by {top_rec['recommended_delay']} days")
            else:
                print(f"USD timing analysis: {timing_opt['error']}")
        else:
            print("No USD payments found in analysis period")
    else:
        print("Insufficient data in analysis period for netting optimization demo")