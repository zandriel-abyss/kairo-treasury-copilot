"""
Behavioral Forecasting Module for Kairo Treasury Optimization

Analyzes historical payment and collection patterns to forecast
future cash flow timing and provide behavioral insights.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class BehaviorForecastError(Exception):
    """Custom exception for behavior forecasting errors"""
    pass


class BehaviorForecaster:
    """
    Analyzes payment and collection behavior patterns to forecast cash flows.

    Provides insights into payment timing, collection patterns, and cash flow
    predictability for treasury decision making.
    """

    def __init__(self, ap_data: pd.DataFrame, ar_data: pd.DataFrame):
        """
        Initialize the behavior forecaster.

        Args:
            ap_data: Accounts Payable data (from data_ingest)
            ar_data: Accounts Receivable data (from data_ingest)
        """
        self.ap_data = ap_data.copy()
        self.ar_data = ar_data.copy()

        # Validate data structure
        self._validate_data()

        # Cache for computed metrics
        self._vendor_payment_patterns = None
        self._customer_collection_patterns = None
        self._seasonal_patterns = None

    def _validate_data(self):
        """Validate that required columns exist in the data."""
        ap_required = ['vendor', 'issue_date', 'due_date', 'actual_payment_date', 'payment_lag_days', 'amount', 'currency']
        ar_required = ['customer', 'issue_date', 'expected_payment_date', 'actual_payment_date', 'collection_lag_days', 'amount', 'currency']

        ap_missing = [col for col in ap_required if col not in self.ap_data.columns]
        ar_missing = [col for col in ar_required if col not in self.ar_data.columns]

        if ap_missing:
            raise BehaviorForecastError(f"AP data missing required columns: {ap_missing}")
        if ar_missing:
            raise BehaviorForecastError(f"AR data missing required columns: {ar_missing}")

    def analyze_payment_patterns(self, vendor: Optional[str] = None,
                               currency: Optional[str] = None) -> Dict:
        """
        Analyze payment behavior patterns for vendors.

        Args:
            vendor: Specific vendor to analyze (optional)
            currency: Specific currency to analyze (optional)

        Returns:
            Dictionary with payment pattern analysis
        """
        df = self.ap_data.copy()

        # Apply filters
        if vendor:
            df = df[df['vendor'] == vendor]
        if currency:
            df = df[df['currency'] == currency]

        if len(df) < 10:
            return {'error': f'Insufficient data for analysis: {len(df)} transactions'}

        # Basic statistics
        total_payments = len(df)
        avg_lag = df['payment_lag_days'].mean()
        std_lag = df['payment_lag_days'].std()
        early_payment_rate = (df['payment_lag_days'] < 0).mean()
        late_payment_rate = (df['payment_lag_days'] > 0).mean()
        on_time_rate = (df['payment_lag_days'] == 0).mean()

        # Payment consistency (lower is more consistent)
        consistency_score = std_lag / max(abs(avg_lag), 1)

        # Seasonal patterns by month
        df['payment_month'] = df['actual_payment_date'].dt.month
        monthly_patterns = df.groupby('payment_month')['payment_lag_days'].agg(['mean', 'std', 'count'])

        # Recent vs historical behavior (last 3 months vs prior)
        recent_cutoff = df['actual_payment_date'].max() - pd.DateOffset(months=3)
        recent_data = df[df['actual_payment_date'] >= recent_cutoff]
        historical_data = df[df['actual_payment_date'] < recent_cutoff]

        recent_avg_lag = recent_data['payment_lag_days'].mean() if len(recent_data) > 0 else avg_lag
        historical_avg_lag = historical_data['payment_lag_days'].mean() if len(historical_data) > 0 else avg_lag

        behavior_trend = recent_avg_lag - historical_avg_lag

        # Predictability score (0-1, higher is more predictable)
        predictability = max(0, 1 - (std_lag / 30))  # Assuming 30 days is highly variable

        return {
            'analysis_type': 'payment_patterns',
            'entity': vendor or 'all_vendors',
            'currency': currency or 'all_currencies',
            'total_payments': total_payments,
            'avg_payment_lag_days': avg_lag,
            'std_payment_lag_days': std_lag,
            'early_payment_rate': early_payment_rate,
            'late_payment_rate': late_payment_rate,
            'on_time_rate': on_time_rate,
            'consistency_score': consistency_score,
            'predictability_score': predictability,
            'recent_avg_lag': recent_avg_lag,
            'behavior_trend': behavior_trend,
            'monthly_patterns': monthly_patterns.to_dict(),
            'insights': self._generate_payment_insights(avg_lag, consistency_score,
                                                       predictability, behavior_trend)
        }

    def analyze_collection_patterns(self, customer: Optional[str] = None,
                                  currency: Optional[str] = None) -> Dict:
        """
        Analyze collection behavior patterns for customers.

        Args:
            customer: Specific customer to analyze (optional)
            currency: Specific currency to analyze (optional)

        Returns:
            Dictionary with collection pattern analysis
        """
        df = self.ar_data.copy()

        # Apply filters
        if customer:
            df = df[df['customer'] == customer]
        if currency:
            df = df[df['currency'] == currency]

        if len(df) < 10:
            return {'error': f'Insufficient data for analysis: {len(df)} transactions'}

        # Basic statistics
        total_collections = len(df)
        avg_lag = df['collection_lag_days'].mean()
        std_lag = df['collection_lag_days'].std()
        early_collection_rate = (df['collection_lag_days'] < 0).mean()
        late_collection_rate = (df['collection_lag_days'] > 0).mean()
        on_time_rate = (df['collection_lag_days'] == 0).mean()

        # Collection consistency
        consistency_score = std_lag / max(abs(avg_lag), 1)

        # Seasonal patterns by month
        df['collection_month'] = df['actual_payment_date'].dt.month
        monthly_patterns = df.groupby('collection_month')['collection_lag_days'].agg(['mean', 'std', 'count'])

        # Recent vs historical behavior
        recent_cutoff = df['actual_payment_date'].max() - pd.DateOffset(months=3)
        recent_data = df[df['actual_payment_date'] >= recent_cutoff]
        historical_data = df[df['actual_payment_date'] < recent_cutoff]

        recent_avg_lag = recent_data['collection_lag_days'].mean() if len(recent_data) > 0 else avg_lag
        historical_avg_lag = historical_data['collection_lag_days'].mean() if len(historical_data) > 0 else avg_lag

        behavior_trend = recent_avg_lag - historical_avg_lag

        # Predictability score
        predictability = max(0, 1 - (std_lag / 45))  # Collections often have longer lags

        return {
            'analysis_type': 'collection_patterns',
            'entity': customer or 'all_customers',
            'currency': currency or 'all_currencies',
            'total_collections': total_collections,
            'avg_collection_lag_days': avg_lag,
            'std_collection_lag_days': std_lag,
            'early_collection_rate': early_collection_rate,
            'late_collection_rate': late_collection_rate,
            'on_time_rate': on_time_rate,
            'consistency_score': consistency_score,
            'predictability_score': predictability,
            'recent_avg_lag': recent_avg_lag,
            'behavior_trend': behavior_trend,
            'monthly_patterns': monthly_patterns.to_dict(),
            'insights': self._generate_collection_insights(avg_lag, consistency_score,
                                                         predictability, behavior_trend)
        }

    def forecast_payment_timing(self, vendor: str, invoice_amount: float,
                              due_date: datetime, currency: str) -> Dict:
        """
        Forecast when a specific payment is likely to be made.

        Args:
            vendor: Vendor name
            invoice_amount: Invoice amount
            due_date: Invoice due date
            currency: Transaction currency

        Returns:
            Dictionary with payment timing forecast
        """
        # Get vendor's payment pattern
        vendor_pattern = self.analyze_payment_patterns(vendor=vendor, currency=currency)

        if 'error' in vendor_pattern:
            # Fall back to general patterns
            vendor_pattern = self.analyze_payment_patterns(currency=currency)

        avg_lag = vendor_pattern['avg_payment_lag_days']
        std_lag = vendor_pattern['std_payment_lag_days']
        predictability = vendor_pattern['predictability_score']

        # Forecast payment date
        expected_payment_date = due_date + timedelta(days=avg_lag)

        # Confidence intervals (using normal distribution assumption)
        ci_80_lower = due_date + timedelta(days=avg_lag - 1.28 * std_lag)
        ci_80_upper = due_date + timedelta(days=avg_lag + 1.28 * std_lag)
        ci_95_lower = due_date + timedelta(days=avg_lag - 1.96 * std_lag)
        ci_95_upper = due_date + timedelta(days=avg_lag + 1.96 * std_lag)

        # Probability of early payment
        early_payment_prob = vendor_pattern['early_payment_rate']

        # Cash flow impact timing
        days_to_payment = (expected_payment_date - datetime.now()).days
        cash_flow_window = self._classify_cash_flow_window(days_to_payment)

        return {
            'forecast_type': 'payment_timing',
            'vendor': vendor,
            'invoice_amount': invoice_amount,
            'due_date': due_date,
            'currency': currency,
            'expected_payment_date': expected_payment_date,
            'avg_historical_lag': avg_lag,
            'confidence_80pct_lower': ci_80_lower,
            'confidence_80pct_upper': ci_80_upper,
            'confidence_95pct_lower': ci_95_lower,
            'confidence_95pct_upper': ci_95_upper,
            'early_payment_probability': early_payment_prob,
            'predictability_score': predictability,
            'cash_flow_window': cash_flow_window,
            'days_to_payment': days_to_payment,
            'recommendations': self._generate_payment_recommendations(
                expected_payment_date, predictability, cash_flow_window)
        }

    def forecast_collection_timing(self, customer: str, invoice_amount: float,
                                 expected_date: datetime, currency: str) -> Dict:
        """
        Forecast when a specific collection is likely to be received.

        Args:
            customer: Customer name
            invoice_amount: Invoice amount
            expected_date: Expected collection date
            currency: Transaction currency

        Returns:
            Dictionary with collection timing forecast
        """
        # Get customer's collection pattern
        customer_pattern = self.analyze_collection_patterns(customer=customer, currency=currency)

        if 'error' in customer_pattern:
            # Fall back to general patterns
            customer_pattern = self.analyze_collection_patterns(currency=currency)

        avg_lag = customer_pattern['avg_collection_lag_days']
        std_lag = customer_pattern['std_collection_lag_days']
        predictability = customer_pattern['predictability_score']

        # Forecast collection date
        expected_collection_date = expected_date + timedelta(days=avg_lag)

        # Confidence intervals
        ci_80_lower = expected_date + timedelta(days=avg_lag - 1.28 * std_lag)
        ci_80_upper = expected_date + timedelta(days=avg_lag + 1.28 * std_lag)
        ci_95_lower = expected_date + timedelta(days=avg_lag - 1.96 * std_lag)
        ci_95_upper = expected_date + timedelta(days=avg_lag + 1.96 * std_lag)

        # Probability of early collection
        early_collection_prob = customer_pattern['early_collection_rate']

        # Cash flow impact timing
        days_to_collection = (expected_collection_date - datetime.now()).days
        cash_flow_window = self._classify_cash_flow_window(days_to_collection)

        return {
            'forecast_type': 'collection_timing',
            'customer': customer,
            'invoice_amount': invoice_amount,
            'expected_date': expected_date,
            'currency': currency,
            'expected_collection_date': expected_collection_date,
            'avg_historical_lag': avg_lag,
            'confidence_80pct_lower': ci_80_lower,
            'confidence_80pct_upper': ci_80_upper,
            'confidence_95pct_lower': ci_95_lower,
            'confidence_95pct_upper': ci_95_upper,
            'early_collection_probability': early_collection_prob,
            'predictability_score': predictability,
            'cash_flow_window': cash_flow_window,
            'days_to_collection': days_to_collection,
            'recommendations': self._generate_collection_recommendations(
                expected_collection_date, predictability, cash_flow_window)
        }

    def _classify_cash_flow_window(self, days: int) -> str:
        """Classify timing into cash flow windows."""
        if days <= 7:
            return 'immediate'
        elif days <= 30:
            return 'short_term'
        elif days <= 90:
            return 'medium_term'
        else:
            return 'long_term'

    def _generate_payment_insights(self, avg_lag: float, consistency: float,
                                 predictability: float, trend: float) -> List[str]:
        """Generate human-readable insights about payment behavior."""
        insights = []

        if avg_lag < -2:
            insights.append("Typically pays early, providing good cash flow visibility.")
        elif avg_lag > 5:
            insights.append("Typically pays late, requiring careful cash flow planning.")
        else:
            insights.append("Generally pays on or near due dates.")

        if consistency < 0.5:
            insights.append("Payment timing is relatively consistent.")
        else:
            insights.append("Payment timing varies significantly.")

        if predictability > 0.7:
            insights.append("Payment behavior is highly predictable.")
        elif predictability > 0.4:
            insights.append("Payment behavior is moderately predictable.")
        else:
            insights.append("Payment behavior is unpredictable - monitor closely.")

        if abs(trend) > 2:
            direction = "earlier" if trend < 0 else "later"
            insights.append(f"Recent payments tend to be {direction} than historical average.")

        return insights

    def _generate_collection_insights(self, avg_lag: float, consistency: float,
                                    predictability: float, trend: float) -> List[str]:
        """Generate human-readable insights about collection behavior."""
        insights = []

        if avg_lag < -2:
            insights.append("Typically collects early, improving cash flow.")
        elif avg_lag > 7:
            insights.append("Typically collects late, creating cash flow uncertainty.")
        else:
            insights.append("Generally collects on or near expected dates.")

        if consistency < 0.5:
            insights.append("Collection timing is relatively consistent.")
        else:
            insights.append("Collection timing varies significantly.")

        if predictability > 0.7:
            insights.append("Collection behavior is highly predictable.")
        elif predictability > 0.4:
            insights.append("Collection behavior is moderately predictable.")
        else:
            insights.append("Collection behavior is unpredictable - consider credit terms.")

        if abs(trend) > 2:
            direction = "earlier" if trend < 0 else "later"
            insights.append(f"Recent collections tend to be {direction} than historical average.")

        return insights

    def _generate_payment_recommendations(self, expected_date: datetime,
                                        predictability: float, window: str) -> List[str]:
        """Generate payment timing recommendations."""
        recommendations = []

        if window == 'immediate':
            recommendations.append("Payment expected within one week - prepare cash flow accordingly.")
        elif window == 'short_term':
            recommendations.append("Payment expected within one month - monitor working capital needs.")
        elif window == 'medium_term':
            recommendations.append("Payment expected in 1-3 months - consider early payment discounts if available.")

        if predictability < 0.5:
            recommendations.append("Low predictability - maintain buffer liquidity for uncertainty.")
        else:
            recommendations.append("High predictability - can plan cash flows with confidence.")

        return recommendations

    def _generate_collection_recommendations(self, expected_date: datetime,
                                           predictability: float, window: str) -> List[str]:
        """Generate collection timing recommendations."""
        recommendations = []

        if window == 'immediate':
            recommendations.append("Collection expected within one week - can rely on near-term cash inflow.")
        elif window == 'short_term':
            recommendations.append("Collection expected within one month - factor into short-term planning.")
        elif window == 'medium_term':
            recommendations.append("Collection expected in 1-3 months - consider factoring or credit insurance.")

        if predictability < 0.5:
            recommendations.append("Low predictability - diversify collection risk and maintain liquidity buffers.")
        else:
            recommendations.append("High predictability - can optimize working capital based on expected inflows.")

        return recommendations

    def get_portfolio_payment_forecast(self, days_ahead: int = 90) -> Dict:
        """
        Forecast aggregate payment outflows over the specified period.

        Args:
            days_ahead: Number of days to forecast

        Returns:
            Dictionary with portfolio-level payment forecast
        """
        cutoff_date = datetime.now() + timedelta(days=days_ahead)
        upcoming_payments = self.ap_data[
            (self.ap_data['due_date'] >= datetime.now()) &
            (self.ap_data['due_date'] <= cutoff_date)
        ].copy()

        # Apply behavioral forecasts to each payment
        forecasted_payments = []
        total_forecasted_amount = 0

        for _, payment in upcoming_payments.iterrows():
            forecast = self.forecast_payment_timing(
                vendor=payment['vendor'],
                invoice_amount=payment['amount'],
                due_date=payment['due_date'],
                currency=payment['currency']
            )

            forecasted_payments.append({
                'invoice_id': payment['invoice_id'],
                'vendor': payment['vendor'],
                'due_date': payment['due_date'],
                'expected_payment_date': forecast['expected_payment_date'],
                'amount': payment['amount'],
                'currency': payment['currency'],
                'confidence_score': forecast['predictability_score']
            })

            total_forecasted_amount += payment['amount']

        # Aggregate by week
        forecast_df = pd.DataFrame(forecasted_payments)
        if len(forecast_df) > 0:
            forecast_df['week'] = forecast_df['expected_payment_date'].dt.to_period('W')
            weekly_forecast = forecast_df.groupby('week')['amount'].sum().reset_index()
            weekly_forecast['week_start'] = weekly_forecast['week'].dt.start_time
        else:
            weekly_forecast = pd.DataFrame()

        return {
            'forecast_period_days': days_ahead,
            'total_upcoming_payments': len(upcoming_payments),
            'total_forecasted_amount': total_forecasted_amount,
            'avg_confidence_score': np.mean([p['confidence_score'] for p in forecasted_payments]) if forecasted_payments else 0,
            'weekly_breakdown': weekly_forecast.to_dict('records') if len(weekly_forecast) > 0 else [],
            'top_vendors_by_amount': upcoming_payments.groupby('vendor')['amount'].sum().nlargest(5).to_dict()
        }

    def get_portfolio_collection_forecast(self, days_ahead: int = 90) -> Dict:
        """
        Forecast aggregate collection inflows over the specified period.

        Args:
            days_ahead: Number of days to forecast

        Returns:
            Dictionary with portfolio-level collection forecast
        """
        cutoff_date = datetime.now() + timedelta(days=days_ahead)
        upcoming_collections = self.ar_data[
            (self.ar_data['expected_payment_date'] >= datetime.now()) &
            (self.ar_data['expected_payment_date'] <= cutoff_date)
        ].copy()

        # Apply behavioral forecasts to each collection
        forecasted_collections = []
        total_forecasted_amount = 0

        for _, collection in upcoming_collections.iterrows():
            forecast = self.forecast_collection_timing(
                customer=collection['customer'],
                invoice_amount=collection['amount'],
                expected_date=collection['expected_payment_date'],
                currency=collection['currency']
            )

            forecasted_collections.append({
                'invoice_id': collection['invoice_id'],
                'customer': collection['customer'],
                'expected_date': collection['expected_payment_date'],
                'expected_collection_date': forecast['expected_collection_date'],
                'amount': collection['amount'],
                'currency': collection['currency'],
                'confidence_score': forecast['predictability_score']
            })

            total_forecasted_amount += collection['amount']

        # Aggregate by week
        forecast_df = pd.DataFrame(forecasted_collections)
        if len(forecast_df) > 0:
            forecast_df['week'] = forecast_df['expected_collection_date'].dt.to_period('W')
            weekly_forecast = forecast_df.groupby('week')['amount'].sum().reset_index()
            weekly_forecast['week_start'] = weekly_forecast['week'].dt.start_time
        else:
            weekly_forecast = pd.DataFrame()

        return {
            'forecast_period_days': days_ahead,
            'total_upcoming_collections': len(upcoming_collections),
            'total_forecasted_amount': total_forecasted_amount,
            'avg_confidence_score': np.mean([c['confidence_score'] for c in forecasted_collections]) if forecasted_collections else 0,
            'weekly_breakdown': weekly_forecast.to_dict('records') if len(weekly_forecast) > 0 else [],
            'top_customers_by_amount': upcoming_collections.groupby('customer')['amount'].sum().nlargest(5).to_dict()
        }


def create_behavior_forecaster(ap_data: pd.DataFrame, ar_data: pd.DataFrame) -> BehaviorForecaster:
    """Factory function to create behavior forecaster instance."""
    return BehaviorForecaster(ap_data, ar_data)


if __name__ == "__main__":
    # Example usage and testing
    from data_ingest import get_data_loader

    # Load data
    loader = get_data_loader()
    ap_data = loader.load_ap_data()
    ar_data = loader.load_ar_data()

    # Create forecaster
    forecaster = BehaviorForecaster(ap_data, ar_data)

    # Test payment pattern analysis
    print("Payment Behavior Analysis:")
    print("=" * 40)

    payment_patterns = forecaster.analyze_payment_patterns()
    print(f"Average payment lag: {payment_patterns['avg_payment_lag_days']:.1f} days")
    print(f"Early payment rate: {payment_patterns['early_payment_rate']:.1%}")
    print(f"Late payment rate: {payment_patterns['late_payment_rate']:.1%}")
    print(f"Predictability score: {payment_patterns['predictability_score']:.1%}")
    print(f"Key insights: {'; '.join(payment_patterns['insights'][:2])}")

    # Test collection pattern analysis
    print("\nCollection Behavior Analysis:")
    print("=" * 40)

    collection_patterns = forecaster.analyze_collection_patterns()
    print(f"Average collection lag: {collection_patterns['avg_collection_lag_days']:.1f} days")
    print(f"Early collection rate: {collection_patterns['early_collection_rate']:.1%}")
    print(f"Late collection rate: {collection_patterns['late_collection_rate']:.1%}")
    print(f"Predictability score: {collection_patterns['predictability_score']:.1%}")
    print(f"Key insights: {'; '.join(collection_patterns['insights'][:2])}")

    # Test portfolio forecasts
    print("\nPortfolio Forecasts (Next 30 Days):")
    print("=" * 40)

    payment_forecast = forecaster.get_portfolio_payment_forecast(30)
    collection_forecast = forecaster.get_portfolio_collection_forecast(30)

    print(f"Upcoming payments: {payment_forecast['total_upcoming_payments']} totaling ${payment_forecast['total_forecasted_amount']:,.0f}")
    print(f"Upcoming collections: {collection_forecast['total_upcoming_collections']} totaling ${collection_forecast['total_forecasted_amount']:,.0f}")
    print(".1f")
    print(".1f")

    # Test specific vendor forecast
    print("\nSpecific Vendor Forecast:")
    print("=" * 30)

    # Get a sample upcoming payment
    upcoming = ap_data[ap_data['due_date'] > datetime.now()].head(1)
    if len(upcoming) > 0:
        payment = upcoming.iloc[0]
        forecast = forecaster.forecast_payment_timing(
            vendor=payment['vendor'],
            invoice_amount=payment['amount'],
            due_date=payment['due_date'],
            currency=payment['currency']
        )

        print(f"Vendor: {forecast['vendor']}")
        print(f"Due date: {forecast['due_date'].strftime('%Y-%m-%d')}")
        print(f"Expected payment: {forecast['expected_payment_date'].strftime('%Y-%m-%d')}")
        print(f"Confidence: {forecast['predictability_score']:.1%}")
        print(f"Recommendation: {forecast['recommendations'][0]}")