"""
Data Ingestion Module for Kairo Treasury Optimization

Loads and processes ERP-style data including:
- Accounts Payable (AP) transactions
- Accounts Receivable (AR) transactions
- Historical FX rates
- Bank balance data (future extension)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class DataIngestionError(Exception):
    """Custom exception for data ingestion errors"""
    pass


class TreasuryDataLoader:
    """
    Loads and processes treasury-related data from CSV files.

    Handles data validation, type conversion, and basic preprocessing
    for AP, AR, and FX rate data.
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.

        Args:
            data_dir: Path to the data directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        self._validate_data_directory()

        # Cache for loaded data
        self._ap_data = None
        self._ar_data = None
        self._fx_data = None
        self._bank_balances = None

    def _validate_data_directory(self):
        """Validate that the data directory exists and contains required files."""
        if not self.data_dir.exists():
            raise DataIngestionError(f"Data directory {self.data_dir} does not exist")

        required_files = ['ap_data.csv', 'ar_data.csv', 'fx_rates.csv']
        missing_files = []

        for file in required_files:
            if not (self.data_dir / file).exists():
                missing_files.append(file)

        if missing_files:
            raise DataIngestionError(f"Missing required data files: {missing_files}")

    def load_ap_data(self, reload: bool = False) -> pd.DataFrame:
        """
        Load and process Accounts Payable data.

        Args:
            reload: Force reload data from disk

        Returns:
            Processed AP DataFrame with additional derived columns
        """
        if self._ap_data is None or reload:
            file_path = self.data_dir / 'ap_data.csv'

            try:
                df = pd.read_csv(file_path, parse_dates=['issue_date', 'due_date', 'actual_payment_date'])

                # Validate required columns
                required_cols = ['invoice_id', 'vendor', 'issue_date', 'due_date',
                               'amount', 'currency', 'actual_payment_date']
                self._validate_columns(df, required_cols, 'AP data')

                # Add derived columns
                df['payment_lag_days'] = (df['actual_payment_date'] - df['due_date']).dt.days
                df['is_early_payment'] = df['payment_lag_days'] < 0
                df['is_late_payment'] = df['payment_lag_days'] > 0
                df['days_to_due'] = (df['due_date'] - df['issue_date']).dt.days

                # Sort by due date for processing
                df = df.sort_values('due_date').reset_index(drop=True)

                self._ap_data = df

            except Exception as e:
                raise DataIngestionError(f"Failed to load AP data: {str(e)}")

        return self._ap_data.copy()

    def load_ar_data(self, reload: bool = False) -> pd.DataFrame:
        """
        Load and process Accounts Receivable data.

        Args:
            reload: Force reload data from disk

        Returns:
            Processed AR DataFrame with additional derived columns
        """
        if self._ar_data is None or reload:
            file_path = self.data_dir / 'ar_data.csv'

            try:
                df = pd.read_csv(file_path, parse_dates=['issue_date', 'expected_payment_date', 'actual_payment_date'])

                # Validate required columns
                required_cols = ['invoice_id', 'customer', 'issue_date', 'expected_payment_date',
                               'amount', 'currency', 'actual_payment_date']
                self._validate_columns(df, required_cols, 'AR data')

                # Add derived columns
                df['collection_lag_days'] = (df['actual_payment_date'] - df['expected_payment_date']).dt.days
                df['is_early_collection'] = df['collection_lag_days'] < 0
                df['is_late_collection'] = df['collection_lag_days'] > 0
                df['days_to_expected'] = (df['expected_payment_date'] - df['issue_date']).dt.days

                # Sort by expected payment date for processing
                df = df.sort_values('expected_payment_date').reset_index(drop=True)

                self._ar_data = df

            except Exception as e:
                raise DataIngestionError(f"Failed to load AR data: {str(e)}")

        return self._ar_data.copy()

    def load_fx_data(self, reload: bool = False) -> pd.DataFrame:
        """
        Load and process FX rates data.

        Args:
            reload: Force reload data from disk

        Returns:
            Processed FX DataFrame with additional derived columns
        """
        if self._fx_data is None or reload:
            file_path = self.data_dir / 'fx_rates.csv'

            try:
                df = pd.read_csv(file_path, parse_dates=['date'])

                # Validate required columns
                required_cols = ['date', 'currency_pair', 'rate']
                self._validate_columns(df, required_cols, 'FX data')

                # Add derived columns
                df['base_currency'] = df['currency_pair'].str.split('/').str[0]
                df['quote_currency'] = df['currency_pair'].str.split('/').str[1]

                # Calculate daily returns for volatility analysis
                df = df.sort_values(['currency_pair', 'date'])
                df['daily_return'] = df.groupby('currency_pair')['rate'].pct_change()

                # Calculate rolling volatility (30-day window)
                df['volatility_30d'] = df.groupby('currency_pair')['daily_return'].rolling(30).std().reset_index(0, drop=True)

                self._fx_data = df

            except Exception as e:
                raise DataIngestionError(f"Failed to load FX data: {str(e)}")

        return self._fx_data.copy()

    def _validate_columns(self, df: pd.DataFrame, required_cols: List[str], data_type: str):
        """Validate that required columns exist in the DataFrame."""
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise DataIngestionError(f"{data_type} missing required columns: {missing_cols}")

    def get_ap_summary(self, currency: Optional[str] = None) -> Dict:
        """
        Get summary statistics for AP data.

        Args:
            currency: Filter by specific currency (optional)

        Returns:
            Dictionary with AP summary statistics
        """
        df = self.load_ap_data()

        if currency:
            df = df[df['currency'] == currency]

        return {
            'total_invoices': len(df),
            'total_amount': df['amount'].sum(),
            'avg_payment_lag': df['payment_lag_days'].mean(),
            'early_payment_rate': df['is_early_payment'].mean(),
            'late_payment_rate': df['is_late_payment'].mean(),
            'currencies': df['currency'].value_counts().to_dict(),
            'top_vendors': df.groupby('vendor')['amount'].sum().nlargest(5).to_dict()
        }

    def get_ar_summary(self, currency: Optional[str] = None) -> Dict:
        """
        Get summary statistics for AR data.

        Args:
            currency: Filter by specific currency (optional)

        Returns:
            Dictionary with AR summary statistics
        """
        df = self.load_ar_data()

        if currency:
            df = df[df['currency'] == currency]

        return {
            'total_invoices': len(df),
            'total_amount': df['amount'].sum(),
            'avg_collection_lag': df['collection_lag_days'].mean(),
            'early_collection_rate': df['is_early_collection'].mean(),
            'late_collection_rate': df['is_late_collection'].mean(),
            'currencies': df['currency'].value_counts().to_dict(),
            'top_customers': df.groupby('customer')['amount'].sum().nlargest(5).to_dict()
        }

    def get_upcoming_payments(self, days_ahead: int = 30,
                            currency: Optional[str] = None) -> pd.DataFrame:
        """
        Get upcoming payments within the specified time window.

        Args:
            days_ahead: Number of days to look ahead
            currency: Filter by specific currency (optional)

        Returns:
            DataFrame with upcoming payments
        """
        df = self.load_ap_data()
        cutoff_date = datetime.now() + timedelta(days=days_ahead)

        upcoming = df[df['due_date'] <= cutoff_date]

        if currency:
            upcoming = upcoming[upcoming['currency'] == currency]

        return upcoming.sort_values('due_date')

    def get_expected_receipts(self, days_ahead: int = 30,
                            currency: Optional[str] = None) -> pd.DataFrame:
        """
        Get expected receipts within the specified time window.

        Args:
            days_ahead: Number of days to look ahead
            currency: Filter by specific currency (optional)

        Returns:
            DataFrame with expected receipts
        """
        df = self.load_ar_data()
        cutoff_date = datetime.now() + timedelta(days=days_ahead)

        expected = df[df['expected_payment_date'] <= cutoff_date]

        if currency:
            expected = expected[expected['currency'] == currency]

        return expected.sort_values('expected_payment_date')

    def get_fx_rate(self, currency_pair: str, date: datetime) -> Optional[float]:
        """
        Get the FX rate for a specific currency pair on a specific date.

        Args:
            currency_pair: Currency pair (e.g., 'USD/EUR')
            date: Date for the rate

        Returns:
            FX rate or None if not found
        """
        df = self.load_fx_data()

        # Find the closest date (for forward-looking dates, use latest available)
        rate_data = df[(df['currency_pair'] == currency_pair) &
                      (df['date'] <= date)]

        if len(rate_data) == 0:
            return None

        # Return the most recent rate
        return rate_data.sort_values('date').iloc[-1]['rate']

    def get_currency_pairs(self) -> List[str]:
        """Get list of available currency pairs."""
        df = self.load_fx_data()
        return df['currency_pair'].unique().tolist()


# Convenience functions for easy access
def load_all_data(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all treasury data at once.

    Returns:
        Tuple of (ap_data, ar_data, fx_data)
    """
    loader = TreasuryDataLoader(data_dir)
    return loader.load_ap_data(), loader.load_ar_data(), loader.load_fx_data()


def get_data_loader(data_dir: str = "data") -> TreasuryDataLoader:
    """Get a configured data loader instance."""
    return TreasuryDataLoader(data_dir)


if __name__ == "__main__":
    # Example usage and testing
    loader = TreasuryDataLoader()

    print("Loading AP data...")
    ap_data = loader.load_ap_data()
    print(f"AP Data shape: {ap_data.shape}")
    print(f"AP Summary: {loader.get_ap_summary()}")

    print("\nLoading AR data...")
    ar_data = loader.load_ar_data()
    print(f"AR Data shape: {ar_data.shape}")
    print(f"AR Summary: {loader.get_ar_summary()}")

    print("\nLoading FX data...")
    fx_data = loader.load_fx_data()
    print(f"FX Data shape: {fx_data.shape}")
    print(f"Available currency pairs: {loader.get_currency_pairs()}")

    print("\nUpcoming payments (next 30 days):")
    upcoming = loader.get_upcoming_payments(30)
    print(upcoming[['invoice_id', 'vendor', 'due_date', 'amount', 'currency']].head())