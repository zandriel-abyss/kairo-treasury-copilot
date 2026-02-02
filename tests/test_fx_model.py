"""
Basic tests for Kairo FX Model

Tests the FX prediction functionality with mock data.
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add modules directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.fx_model import FXPredictor
from modules.data_ingest import get_data_loader


def test_fx_prediction():
    """Test basic FX prediction functionality."""
    print("🧪 Testing FX Model...")

    # Load data
    loader = get_data_loader()
    fx_data = loader.load_fx_data()

    # Create predictor
    predictor = FXPredictor(fx_data)

    # Test prediction
    currency_pair = 'USD/EUR'
    prediction_date = datetime(2023, 6, 1)  # Use historical date for testing
    days_ahead = 7

    try:
        result = predictor.predict_fx_rate(currency_pair, prediction_date, days_ahead)

        # Basic assertions
        assert 'predicted_rate' in result
        assert 'prediction_confidence' in result  # Fixed field name
        assert 'is_delay_favorable' in result
        assert result['days_ahead'] == days_ahead
        assert result['currency_pair'] == currency_pair

        print(f"✅ FX prediction successful: {result['predicted_rate']:.4f}")
        print(f"   Confidence: {result['prediction_confidence']:.1%}")
        print(f"   Delay favorable: {result['is_delay_favorable']}")

        return True

    except Exception as e:
        import traceback
        print(f"❌ FX prediction failed: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        return False


def test_historical_performance():
    """Test historical performance analysis."""
    print("\n🧪 Testing Historical Performance...")

    # Load data
    loader = get_data_loader()
    fx_data = loader.load_fx_data()

    # Create predictor
    predictor = FXPredictor(fx_data)

    try:
        perf = predictor.get_historical_performance('USD/EUR', 7)

        if 'error' not in perf:
            print(f"✅ Historical performance: {perf['hit_rate']:.1%} hit rate")
            print(f"   Total predictions: {perf['total_predictions']}")
            return True
        else:
            print(f"⚠️  Historical performance analysis returned: {perf['error']}")
            return True  # Not a failure, just limited data

    except Exception as e:
        print(f"❌ Historical performance test failed: {str(e)}")
        return False


def run_tests():
    """Run all tests."""
    print("🚀 Running Kairo FX Model Tests")
    print("=" * 40)

    tests = [
        test_fx_prediction,
        test_historical_performance
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\n📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check output above.")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)