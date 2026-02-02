"""
FX Prediction Model for Kairo Treasury Optimization

Provides short-term FX rate predictions using time series models
with confidence bands and delay favorability analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import shap
import warnings
warnings.filterwarnings('ignore')


class FXPredictionError(Exception):
    """Custom exception for FX prediction errors"""
    pass


class FXPredictor:
    """
    FX rate prediction model using time series analysis.

    Supports ARIMA and Exponential Smoothing models with confidence bands
    and delay favorability analysis for treasury decision making.
    """

    def __init__(self, fx_data: pd.DataFrame):
        """
        Initialize the FX predictor with historical data.

        Args:
            fx_data: DataFrame with FX rates (from data_ingest)
        """
        self.fx_data = fx_data.copy()
        self.models = {}  # Cache for fitted models
        self.predictions = {}  # Cache for predictions
        self.ml_models = {}  # Cache for ML models with SHAP
        self.shap_explainer = {}  # Cache for SHAP explainers
        self.scalers = {}  # Cache for feature scalers

        # Validate data structure
        required_cols = ['date', 'currency_pair', 'rate', 'volatility_30d']
        missing_cols = [col for col in required_cols if col not in self.fx_data.columns]
        if missing_cols:
            raise FXPredictionError(f"FX data missing required columns: {missing_cols}")

        # Ensure data is sorted
        self.fx_data = self.fx_data.sort_values(['currency_pair', 'date'])

    def _get_currency_pair_data(self, currency_pair: str) -> pd.DataFrame:
        """Get historical data for a specific currency pair."""
        pair_data = self.fx_data[self.fx_data['currency_pair'] == currency_pair].copy()

        if len(pair_data) < 30:
            raise FXPredictionError(f"Insufficient data for {currency_pair}: {len(pair_data)} observations")

        # Set date as index for time series modeling
        pair_data = pair_data.set_index('date')
        pair_data = pair_data.asfreq('D')  # Daily frequency

        # Forward fill missing values (weekends, holidays)
        pair_data['rate'] = pair_data['rate'].ffill()
        pair_data['volatility_30d'] = pair_data['volatility_30d'].ffill()

        return pair_data

    def _fit_arima_model(self, series: pd.Series, order: Tuple[int, int, int] = (1, 1, 1)) -> ARIMA:
        """Fit ARIMA model to the time series."""
        try:
            model = ARIMA(series, order=order)
            fitted_model = model.fit()
            return fitted_model
        except Exception as e:
            # Fallback to simpler model
            try:
                model = ARIMA(series, order=(0, 1, 0))  # Random walk
                fitted_model = model.fit()
                return fitted_model
            except Exception as e2:
                raise FXPredictionError(f"Failed to fit ARIMA model: {str(e2)}")

    def _fit_exponential_smoothing(self, series: pd.Series) -> ExponentialSmoothing:
        """Fit Exponential Smoothing model."""
        try:
            # Try Holt-Winters with additive seasonality
            model = ExponentialSmoothing(series, seasonal_periods=7, trend='add', seasonal='add')
            fitted_model = model.fit()
            return fitted_model
        except Exception:
            try:
                # Fallback to simple exponential smoothing
                model = ExponentialSmoothing(series, trend='add')
                fitted_model = model.fit()
                return fitted_model
            except Exception as e:
                raise FXPredictionError(f"Failed to fit Exponential Smoothing model: {str(e)}")

    def predict_fx_rate(self, currency_pair: str, prediction_date: datetime,
                       days_ahead: int = 7, confidence_level: float = 0.95,
                       model_type: str = 'auto', include_shap: bool = False) -> Dict:
        """
        Predict FX rate for a future date with confidence bands.

        Args:
            currency_pair: Currency pair to predict (e.g., 'USD/EUR')
            prediction_date: Date for which to make prediction
            days_ahead: Number of days to predict ahead (1-30)
            confidence_level: Confidence level for prediction intervals (0.80-0.99)
            model_type: Model to use ('arima', 'exp_smoothing', 'auto')

        Returns:
            Dictionary with prediction results
        """
        if days_ahead < 1 or days_ahead > 30:
            raise FXPredictionError("days_ahead must be between 1 and 30")

        if not 0.80 <= confidence_level <= 0.99:
            raise FXPredictionError("confidence_level must be between 0.80 and 0.99")

        # Get historical data up to prediction date
        pair_data = self._get_currency_pair_data(currency_pair)

        # Filter data up to prediction date
        historical_data = pair_data[pair_data.index <= prediction_date]

        if len(historical_data) < 30:
            raise FXPredictionError(f"Insufficient historical data for {currency_pair} up to {prediction_date}")

        current_rate = historical_data['rate'].iloc[-1]
        current_volatility = historical_data['volatility_30d'].iloc[-1] if not pd.isna(historical_data['volatility_30d'].iloc[-1]) else 0.01

        # Choose and fit model
        if model_type == 'auto':
            # Use Augmented Dickey-Fuller test to check stationarity
            try:
                adf_result = adfuller(historical_data['rate'])
                is_stationary = adf_result[1] < 0.05  # p-value < 0.05

                if is_stationary:
                    model_type = 'exp_smoothing'
                else:
                    model_type = 'arima'
            except:
                model_type = 'exp_smoothing'  # Default fallback

        # Fit the chosen model
        cache_key = f"{currency_pair}_{model_type}_{prediction_date.date()}"
        if cache_key not in self.models:
            if model_type == 'arima':
                self.models[cache_key] = self._fit_arima_model(historical_data['rate'])
            else:  # exp_smoothing
                self.models[cache_key] = self._fit_exponential_smoothing(historical_data['rate'])

        fitted_model = self.models[cache_key]

        # Make prediction
        try:
            if hasattr(fitted_model, 'forecast'):
                # ARIMA-style forecast
                forecast = fitted_model.forecast(steps=days_ahead)
                predicted_rate = forecast.iloc[-1] if hasattr(forecast, 'iloc') else forecast[-1]
            else:
                # Exponential smoothing forecast
                forecast = fitted_model.forecast(days_ahead)
                predicted_rate = forecast.iloc[-1]

            # Calculate confidence intervals using historical volatility
            # For treasury applications, we use a simplified approach based on rolling volatility
            z_score = {0.80: 1.28, 0.90: 1.645, 0.95: 1.96, 0.99: 2.576}[round(confidence_level, 2)]

            # Adjust volatility based on prediction horizon (increases with time)
            horizon_adjustment = np.sqrt(days_ahead / 30)  # Scale to monthly equivalent
            adjusted_volatility = current_volatility * horizon_adjustment

            confidence_interval = z_score * adjusted_volatility * current_rate

            lower_bound = predicted_rate - confidence_interval
            upper_bound = predicted_rate + confidence_interval

            # Calculate prediction quality metrics
            historical_errors = self._calculate_prediction_errors(historical_data, fitted_model)
            prediction_confidence = self._calculate_prediction_confidence(historical_errors, adjusted_volatility)

            # Determine if delay is favorable (for the base currency perspective)
            # If predicted rate is higher than current, delaying payment in base currency is favorable
            # (you get more base currency units per quote currency unit)
            rate_change_pct = (predicted_rate - current_rate) / current_rate * 100
            is_delay_favorable = rate_change_pct > 0.1  # More than 0.1% improvement

            # Calculate expected value of delay
            expected_improvement = rate_change_pct if is_delay_favorable else 0

            result = {
                'currency_pair': currency_pair,
                'prediction_date': prediction_date,
                'days_ahead': days_ahead,
                'current_rate': current_rate,
                'predicted_rate': predicted_rate,
                'rate_change_pct': rate_change_pct,
                'confidence_level': confidence_level,
                'confidence_interval': confidence_interval,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'is_delay_favorable': is_delay_favorable,
                'expected_improvement_pct': expected_improvement,
                'prediction_confidence': prediction_confidence,
                'model_type': model_type,
                'historical_volatility': current_volatility,
                'reasoning': self._generate_reasoning(is_delay_favorable, rate_change_pct,
                                                    prediction_confidence, days_ahead)
            }

            # Add SHAP analysis if requested
            if include_shap:
                try:
                    shap_result = self.predict_with_shap(currency_pair, prediction_date,
                                                       days_ahead, 'random_forest')
                    if 'error' not in shap_result:
                        result['shap_analysis'] = {
                            'shap_explanation': shap_result['shap_explanation'],
                            'top_features': shap_result['top_features'],
                            'shap_confidence': shap_result['confidence_score'],
                            'feature_importance': shap_result['feature_importance']
                        }
                    else:
                        result['shap_analysis'] = {'error': shap_result['error']}
                except Exception as e:
                    result['shap_analysis'] = {'error': f'SHAP analysis failed: {str(e)}'}

            return result

        except Exception as e:
            raise FXPredictionError(f"Prediction failed for {currency_pair}: {str(e)}")

    def _calculate_prediction_errors(self, historical_data: pd.DataFrame, fitted_model) -> float:
        """Calculate historical prediction errors for confidence assessment."""
        try:
            # Use a rolling forecast approach to estimate prediction accuracy
            series = historical_data['rate']
            errors = []

            # Use last 30 days for error calculation
            test_size = min(30, len(series) // 2)

            for i in range(test_size, len(series)):
                train = series[:i]
                actual = series.iloc[i]

                # Simple one-step ahead prediction
                if len(train) > 10:
                    try:
                        if hasattr(fitted_model, 'forecast'):
                            pred = fitted_model.apply(train).forecast(1).iloc[0]
                        else:
                            # Re-fit for each point (simplified)
                            temp_model = ExponentialSmoothing(train, trend='add').fit()
                            pred = temp_model.forecast(1).iloc[0]

                        errors.append(abs(pred - actual) / actual)
                    except:
                        continue

            return np.mean(errors) if errors else 0.02  # Default 2% error
        except:
            return 0.02  # Default fallback

    def _calculate_prediction_confidence(self, historical_errors: float, volatility: float) -> float:
        """Calculate overall prediction confidence score (0-1)."""
        # Combine historical accuracy and current volatility
        accuracy_score = max(0, 1 - historical_errors * 10)  # Scale errors to 0-1
        volatility_score = max(0, 1 - volatility * 50)  # Scale volatility to 0-1

        # Weighted average
        confidence = 0.6 * accuracy_score + 0.4 * volatility_score
        return min(1.0, max(0.0, confidence))

    def _generate_reasoning(self, is_delay_favorable: bool, rate_change_pct: float,
                          confidence: float, days_ahead: int) -> str:
        """Generate human-readable reasoning for the prediction."""
        direction = "appreciation" if rate_change_pct > 0 else "depreciation"

        if is_delay_favorable:
            reasoning = f"Delaying payment is favorable. Expected {abs(rate_change_pct):.2f}% {direction} "
            reasoning += f"in {days_ahead} days could improve FX rate."
        else:
            reasoning = f"Delaying payment is not recommended. Expected {abs(rate_change_pct):.2f}% {direction} "
            reasoning += f"in {days_ahead} days would worsen FX rate."

        confidence_desc = "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"
        reasoning += f" Prediction confidence: {confidence_desc} ({confidence:.1%})."

        return reasoning

    def get_multiple_predictions(self, currency_pairs: List[str], prediction_date: datetime,
                               days_ahead: int = 7) -> Dict[str, Dict]:
        """
        Get predictions for multiple currency pairs.

        Args:
            currency_pairs: List of currency pairs to predict
            prediction_date: Date for predictions
            days_ahead: Days ahead to predict

        Returns:
            Dictionary mapping currency pairs to prediction results
        """
        results = {}
        for pair in currency_pairs:
            try:
                results[pair] = self.predict_fx_rate(pair, prediction_date, days_ahead)
            except FXPredictionError as e:
                results[pair] = {'error': str(e)}

        return results

    def get_historical_performance(self, currency_pair: str, days_ahead: int = 7) -> Dict:
        """
        Analyze historical prediction performance for model validation.

        Args:
            currency_pair: Currency pair to analyze
            days_ahead: Prediction horizon to analyze

        Returns:
            Dictionary with performance metrics
        """
        try:
            pair_data = self._get_currency_pair_data(currency_pair)
            series = pair_data['rate']

            if len(series) < days_ahead + 30:
                return {'error': 'Insufficient data for performance analysis'}

            # Calculate hit rate for directional predictions
            correct_predictions = 0
            total_predictions = 0

            for i in range(30, len(series) - days_ahead):
                current_rate = series.iloc[i]
                future_rate = series.iloc[i + days_ahead]

                # Simple momentum-based prediction (for comparison)
                predicted_direction = 1 if current_rate > series.iloc[i-1] else -1
                actual_direction = 1 if future_rate > current_rate else -1

                if predicted_direction == actual_direction:
                    correct_predictions += 1
                total_predictions += 1

            hit_rate = correct_predictions / total_predictions if total_predictions > 0 else 0

            return {
                'currency_pair': currency_pair,
                'days_ahead': days_ahead,
                'hit_rate': hit_rate,
                'total_predictions': total_predictions,
                'avg_volatility': pair_data['volatility_30d'].mean()
            }

        except Exception as e:
            return {'error': str(e)}

    def _create_ml_features(self, series: pd.Series, prediction_horizon: int = 7) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create ML features from time series data for SHAP analysis.

        Args:
            series: Time series of FX rates
            prediction_horizon: Days ahead to predict

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        df = pd.DataFrame()

        # Basic lag features
        for lag in [1, 2, 3, 5, 7, 14, 21, 30]:
            df[f'lag_{lag}d'] = series.shift(lag)

        # Rolling statistics
        df['rolling_mean_7d'] = series.rolling(7).mean()
        df['rolling_mean_30d'] = series.rolling(30).mean()
        df['rolling_std_7d'] = series.rolling(7).std()
        df['rolling_std_30d'] = series.rolling(30).std()

        # Rate of change
        df['pct_change_1d'] = series.pct_change(1)
        df['pct_change_7d'] = series.pct_change(7)
        df['pct_change_30d'] = series.pct_change(30)

        # Momentum indicators
        df['momentum_7d'] = (series - series.shift(7)) / series.shift(7)
        df['momentum_30d'] = (series - series.shift(30)) / series.shift(30)

        # Volatility measures
        df['volatility_7d'] = series.rolling(7).std() / series.rolling(7).mean()
        df['volatility_30d'] = series.rolling(30).std() / series.rolling(30).mean()

        # Target: future rate change
        target = series.shift(-prediction_horizon) / series - 1  # Percentage change

        # Remove NaN values
        valid_idx = df.dropna().index
        df_clean = df.loc[valid_idx]
        target_clean = target.loc[valid_idx]

        return df_clean, target_clean

    def _train_ml_model(self, currency_pair: str, model_type: str = 'random_forest') -> bool:
        """
        Train an ML model for SHAP analysis.

        Args:
            currency_pair: Currency pair to train model for
            model_type: Type of ML model ('random_forest' or 'gradient_boosting')

        Returns:
            Success status
        """
        try:
            # Get historical data
            pair_data = self._get_currency_pair_data(currency_pair)
            series = pair_data['rate']

            if len(series) < 60:  # Need sufficient data
                return False

            # Create features and target
            X, y = self._create_ml_features(series, prediction_horizon=7)

            if len(X) < 30:  # Need sufficient training data
                return False

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )

            # Scale features
            scaler_key = f"{currency_pair}_scaler"
            if scaler_key not in self.scalers:
                self.scalers[scaler_key] = StandardScaler()

            X_train_scaled = self.scalers[scaler_key].fit_transform(X_train)
            X_test_scaled = self.scalers[scaler_key].transform(X_test)

            # Train model
            model_key = f"{currency_pair}_{model_type}"
            if model_type == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            model.fit(X_train_scaled, y_train)
            self.ml_models[model_key] = model

            # Create SHAP explainer
            explainer_key = f"{currency_pair}_{model_type}_explainer"
            self.shap_explainer[explainer_key] = shap.TreeExplainer(model)

            return True

        except Exception as e:
            print(f"Error training ML model for {currency_pair}: {str(e)}")
            return False

    def predict_with_shap(self, currency_pair: str, prediction_date: datetime,
                         days_ahead: int = 7, model_type: str = 'random_forest') -> Dict:
        """
        Make FX prediction with SHAP explainability.

        Args:
            currency_pair: Currency pair to predict
            prediction_date: Date for prediction
            days_ahead: Days ahead to predict
            model_type: ML model type for SHAP

        Returns:
            Prediction with SHAP explanations
        """
        try:
            # Ensure model is trained
            model_key = f"{currency_pair}_{model_type}"
            if model_key not in self.ml_models:
                if not self._train_ml_model(currency_pair, model_type):
                    return {'error': f'Could not train ML model for {currency_pair}'}

            # Get historical data up to prediction date
            pair_data = self._get_currency_pair_data(currency_pair)
            historical_data = pair_data[pair_data.index <= prediction_date]

            if len(historical_data) < 30:
                return {'error': 'Insufficient historical data'}

            # Create features for the prediction point
            series = historical_data['rate']
            X, _ = self._create_ml_features(series, prediction_horizon=days_ahead)

            if len(X) == 0:
                return {'error': 'Could not create features for prediction'}

            # Get the most recent feature vector
            latest_features = X.iloc[-1:].copy()

            # Scale features
            scaler_key = f"{currency_pair}_scaler"
            if scaler_key in self.scalers:
                latest_features_scaled = self.scalers[scaler_key].transform(latest_features)
            else:
                return {'error': 'Feature scaler not found'}

            # Make prediction
            model = self.ml_models[model_key]
            predicted_change = model.predict(latest_features_scaled)[0]

            # Current rate
            current_rate = series.iloc[-1]

            # Calculate predicted rate
            predicted_rate = current_rate * (1 + predicted_change)

            # SHAP explanation
            explainer_key = f"{currency_pair}_{model_type}_explainer"
            explainer = self.shap_explainer[explainer_key]

            # Get SHAP values for this prediction
            shap_values = explainer.shap_values(latest_features_scaled)

            # For multi-class or binary, shap_values might be a list
            if isinstance(shap_values, list):
                shap_values = shap_values[0] if len(shap_values) > 0 else shap_values

            # Get feature importance
            feature_importance = dict(zip(latest_features.columns, shap_values[0]))

            # Sort by absolute importance
            sorted_features = sorted(feature_importance.items(),
                                   key=lambda x: abs(x[1]), reverse=True)

            # Generate explanation
            explanation = self._generate_shap_explanation(sorted_features, predicted_change)

            # Determine if delay is favorable (based on predicted direction)
            is_delay_favorable = predicted_change < 0  # Negative change means rate goes down, favorable for buying

            result = {
                'currency_pair': currency_pair,
                'prediction_date': prediction_date,
                'days_ahead': days_ahead,
                'current_rate': current_rate,
                'predicted_rate': predicted_rate,
                'predicted_change_pct': predicted_change * 100,
                'is_delay_favorable': is_delay_favorable,
                'model_type': model_type,
                'shap_explanation': explanation,
                'top_features': sorted_features[:5],  # Top 5 most important features
                'feature_importance': feature_importance,
                'confidence_score': self._calculate_shap_confidence(feature_importance)
            }

            return result

        except Exception as e:
            return {'error': f'SHAP prediction failed: {str(e)}'}

    def _generate_shap_explanation(self, sorted_features: List[Tuple[str, float]],
                                 predicted_change: float) -> str:
        """Generate human-readable SHAP explanation."""
        if not sorted_features:
            return "No feature importance data available."

        explanation_parts = []

        # Overall direction
        direction = "increase" if predicted_change > 0 else "decrease"
        change_pct = abs(predicted_change * 100)
        explanation_parts.append(f"The model predicts a {change_pct:.2f}% {direction} in the FX rate.")

        # Key drivers
        explanation_parts.append("Key factors influencing this prediction:")

        for feature_name, shap_value in sorted_features[:3]:
            impact_direction = "pushing the prediction" if shap_value > 0 else "pulling the prediction"
            strength = "significantly" if abs(shap_value) > 0.01 else "moderately"

            # Human-readable feature names
            readable_name = self._get_readable_feature_name(feature_name)
            explanation_parts.append(f"• {readable_name} is {strength} {impact_direction} {'upward' if shap_value > 0 else 'downward'}")

        return " ".join(explanation_parts)

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

    def _calculate_shap_confidence(self, feature_importance: Dict[str, float]) -> float:
        """Calculate confidence score based on SHAP feature importance distribution."""
        if not feature_importance:
            return 0.5

        # Calculate the concentration of importance in top features
        values = list(feature_importance.values())
        abs_values = [abs(v) for v in values]

        if not abs_values:
            return 0.5

        # Sort by absolute importance
        sorted_importance = sorted(abs_values, reverse=True)

        # Calculate Gini coefficient-like concentration
        n = len(sorted_importance)
        if n <= 1:
            return 0.5

        # Concentration in top 3 features
        top_3_sum = sum(sorted_importance[:3])
        total_sum = sum(sorted_importance)

        if total_sum == 0:
            return 0.5

        concentration_ratio = top_3_sum / total_sum

        # Convert to confidence score (higher concentration = lower confidence in single factors)
        confidence = 1 - (concentration_ratio - 0.3) / 0.7  # Scale to 0-1
        confidence = max(0.1, min(0.9, confidence))  # Bound between 0.1 and 0.9

        return confidence

    def get_shap_feature_importance(self, currency_pair: str,
                                   model_type: str = 'random_forest') -> Dict:
        """
        Get global feature importance for SHAP analysis.

        Args:
            currency_pair: Currency pair
            model_type: ML model type

        Returns:
            Feature importance dictionary
        """
        try:
            model_key = f"{currency_pair}_{model_type}"
            if model_key not in self.ml_models:
                if not self._train_ml_model(currency_pair, model_type):
                    return {'error': f'Could not train ML model for {currency_pair}'}

            model = self.ml_models[model_key]

            # Get feature names from scaler
            scaler_key = f"{currency_pair}_scaler"
            if scaler_key not in self.scalers:
                return {'error': 'Feature scaler not found'}

            scaler = self.scalers[scaler_key]

            # Create dummy feature names if not available
            if not hasattr(scaler, 'feature_names_in_'):
                # Create features from a sample
                pair_data = self._get_currency_pair_data(currency_pair)
                series = pair_data['rate']
                X, _ = self._create_ml_features(series, prediction_horizon=7)
                feature_names = list(X.columns)
            else:
                feature_names = scaler.feature_names_in_

            # Get feature importance from the model
            if hasattr(model, 'feature_importances_'):
                importance_values = model.feature_importances_
            else:
                # Fallback for models without feature_importances_
                importance_values = np.ones(len(feature_names)) / len(feature_names)

            # Create importance dictionary
            importance_dict = dict(zip(feature_names, importance_values))

            # Sort by importance
            sorted_importance = sorted(importance_dict.items(),
                                     key=lambda x: x[1], reverse=True)

            return {
                'currency_pair': currency_pair,
                'model_type': model_type,
                'feature_importance': importance_dict,
                'sorted_features': sorted_importance,
                'top_features': sorted_importance[:10]
            }

        except Exception as e:
            return {'error': f'SHAP feature importance failed: {str(e)}'}


def create_fx_predictor(fx_data: pd.DataFrame) -> FXPredictor:
    """Factory function to create FX predictor instance."""
    return FXPredictor(fx_data)


if __name__ == "__main__":
    # Example usage and testing
    from data_ingest import get_data_loader

    # Load data
    loader = get_data_loader()
    fx_data = loader.load_fx_data()

    # Create predictor
    predictor = FXPredictor(fx_data)

    # Test prediction for next week
    prediction_date = datetime(2024, 1, 15)  # Future date for testing
    currency_pairs = ['USD/EUR', 'USD/GBP', 'USD/JPY']

    print("FX Rate Predictions for Next 7 Days:")
    print("=" * 50)

    for pair in currency_pairs:
        try:
            result = predictor.predict_fx_rate(pair, prediction_date, days_ahead=7)

            print(f"\n{pair}:")
            print(f"  Current rate: {result['current_rate']:.4f}")
            print(f"  Predicted rate: {result['predicted_rate']:.4f}")
            print(f"  Expected change: {result['rate_change_pct']:+.2f}%")
            print(f"  Delay favorable: {result['is_delay_favorable']}")
            print(f"  Confidence: {result['prediction_confidence']:.1%}")
            print(f"  Reasoning: {result['reasoning']}")

        except Exception as e:
            print(f"  Error predicting {pair}: {str(e)}")

    # Test historical performance
    print("\nHistorical Performance Analysis:")
    print("=" * 40)

    for pair in currency_pairs:
        perf = predictor.get_historical_performance(pair, 7)
        if 'error' not in perf:
            print(f"{pair}: {perf['hit_rate']:.1%} hit rate over {perf['total_predictions']} predictions")
        else:
            print(f"{pair}: {perf['error']}")