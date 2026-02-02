# Kairo - AI Treasury Optimization Copilot

An AI-powered treasury assistant that helps mid-market CFOs optimize payment timing, predict FX exposure, and suggest natural hedging opportunities using behavioral AR/AP data.

## 🎯 MVP Goal

Deliver a working CLI simulation that can:
- Ingest ERP-style data (AP/AR transactions, FX rates)
- Predict FX trends with confidence ranges
- Forecast AR/AP timing behavior
- Recommend whether to pay now or delay
- Quantify savings vs risk

## 🏗️ Architecture

### Core Modules

1. **`data_ingest.py`** - Loads and processes ERP data
   - Accounts Payable: invoice dates, due dates, vendors, currencies, amounts
   - Accounts Receivable: invoice dates, payment dates, customers, currencies, amounts
   - Bank balance data (multi-currency)

2. **`fx_model.py`** - Short-term FX prediction
   - ARIMA/Exponential Smoothing for level prediction
   - Confidence bands using rolling volatility
   - Outputs: predicted level + "is delay favorable?" boolean

3. **`behavior_forecast.py`** - Payment timing analysis
   - AP: actual vs due date analysis
   - AR: expected vs received date analysis
   - Outputs: estimated cash flow arrival windows

4. **`netting_optimizer.py`** - Natural hedging identification
   - Matches inflows (AR) and outflows (AP) by currency and timing
   - Identifies offsettable flows and residual risk

5. **`recommendation_engine.py`** - Integrated decision engine
   - Combines FX predictions, behavior analysis, and netting opportunities
   - Generates explainable recommendations with confidence scores

6. **`interface_cli.py`** - Command-line interface
   - Displays payment recommendations, FX risk projections, offset opportunities
   - Includes override options and reason explanations

## 📁 Project Structure

```
kairo/
├── data/
│   ├── fx_rates.csv          # Historical FX rates
│   ├── ar_data.csv           # Accounts Receivable transactions
│   └── ap_data.csv           # Accounts Payable transactions
├── modules/
│   ├── data_ingest.py        # Data loading and preprocessing
│   ├── fx_model.py           # FX prediction models
│   ├── behavior_forecast.py  # Payment timing analysis
│   ├── netting_optimizer.py  # Natural hedging optimization
│   └── recommendation_engine.py # Recommendation generation
├── interface/
│   ├── interface_cli.py      # CLI interface
│   └── dashboard.py          # Future Streamlit dashboard
├── tests/
│   └── test_fx_model.py      # Unit tests
└── README.md
```

## 🔄 Key Principles

- **Modular Design**: Each component can be developed and tested independently
- **Explainability**: All recommendations include confidence scores, historical examples, and reasoning
- **Behavioral Focus**: Uses actual AR/AP payment patterns, not just due dates
- **Natural Hedging**: Identifies opportunities to offset FX exposure through timing
- **Copilot Approach**: Provides actionable recommendations with override capabilities

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Required packages: pandas, numpy, statsmodels, scikit-learn

### Installation

```bash
# Clone or navigate to the kairo directory
cd kairo

# Install dependencies
pip install pandas numpy statsmodels scikit-learn

# Run tests to verify installation
python tests/test_fx_model.py
```

### Usage

#### Interactive Dashboard (Enhanced User Experience)
```bash
# Start the Streamlit dashboard
streamlit run interface/dashboard.py
```

The dashboard provides:
- 🏠 **Dashboard Overview** - Key metrics, portfolio health, and cash flow charts
- 💱 **FX Rate Analysis** - Interactive charts, predictions, and model performance
- 👥 **Behavior Insights** - Payment/collection pattern analysis with monthly trends
- 🔄 **Netting Opportunities** - Natural hedging identification and currency optimization
- 💡 **Recommendations** - AI-powered payment timing with confidence scores and export
- 🎲 **Treasury Simulation** - Comprehensive scenario comparison (AI vs. Current vs. Conservative strategies)

#### CLI Interface (Programmatic Access)
```bash
# Start the interactive CLI
python interface/interface_cli.py
```

The CLI provides:
- 📄 **Analyze Specific Payment** - Get recommendations for individual invoices
- 📊 **Portfolio Recommendations** - Bulk analysis of upcoming payments
- 💱 **FX Rate Analysis** - Direct FX prediction queries
- 👥 **Payment Behavior Analysis** - Historical pattern insights
- 🔄 **Netting Opportunities** - Natural hedging identification
- 💾 **Export Recommendations** - CSV export for integration

#### Direct Module Usage
```python
from modules.data_ingest import get_data_loader
from modules.recommendation_engine import create_recommendation_engine, PaymentScenario
from datetime import datetime, timedelta

# Load data
loader = get_data_loader()
ap_data = loader.load_ap_data()
ar_data = loader.load_ar_data()
fx_data = loader.load_fx_data()

# Create recommendation engine
engine = create_recommendation_engine(fx_data, ap_data, ar_data)

# Analyze a payment scenario
scenario = PaymentScenario(
    invoice_id="INV-001",
    vendor="ABC Corp",
    amount=50000,
    currency="EUR",
    due_date=datetime.now() + timedelta(days=7),
    payment_options=[
        datetime.now() + timedelta(days=7),  # Due date
        datetime.now() + timedelta(days=10), # 3-day delay
        datetime.now() + timedelta(days=14), # 1-week delay
    ]
)

recommendation = engine.analyze_payment_scenario(scenario)
print(f"Recommendation: {recommendation.recommended_action}")
print(f"Confidence: {recommendation.confidence_score:.1%}")
```

## 📊 Sample Data Structure

### AP Data (ap_data.csv)
- invoice_id: Unique invoice identifier
- vendor: Vendor name
- issue_date: Invoice issue date
- due_date: Payment due date
- amount: Invoice amount
- currency: Transaction currency
- actual_payment_date: When payment was actually made

### AR Data (ar_data.csv)
- invoice_id: Unique invoice identifier
- customer: Customer name
- issue_date: Invoice issue date
- expected_payment_date: Expected payment date
- amount: Invoice amount
- currency: Transaction currency
- actual_payment_date: When payment was actually received

### FX Rates (fx_rates.csv)
- date: Date of rate
- currency_pair: e.g., "USD/EUR", "USD/JPY"
- rate: Exchange rate
- timestamp: Time of day (optional)

## 🎛️ Future Enhancements

- Real ERP API integrations
- Streamlit dashboard
- PAIN protocol integration
- Advanced ML models with SHAP explainability
- Multi-entity treasury management

## ✅ MVP Achievements

This implementation delivers a comprehensive treasury optimization system with:

- **✅ Modular Architecture** - 7 independent modules that can be developed/tested separately
- **✅ Interactive Dashboard** - Streamlit-based visualization with charts and scenario analysis
- **✅ Advanced Simulation Engine** - What-if analysis comparing payment strategies and hedging approaches
- **✅ Explainable AI** - All recommendations include confidence scores, reasoning, and historical context
- **✅ Behavioral Analysis** - Uses actual payment patterns, not just due dates
- **✅ Natural Hedging** - Identifies timing opportunities to reduce FX exposure
- **✅ Interactive CLI** - User-friendly interface with override capabilities
- **✅ Mock Data Pipeline** - Realistic test data for development and validation
- **✅ PAIN Protocol Ready** - Structured output for future payment rail optimization

### Key Features Demonstrated
- **SHAP Explainability**: AI model transparency showing why predictions are made
- FX rate prediction with confidence bands (ARIMA/Exponential Smoothing + ML models)
- Payment/collection behavior forecasting with predictability scoring
- Netting optimization for natural hedge identification
- Integrated recommendation engine with multi-factor scoring
- Comprehensive simulation engine for strategy comparison
- Interactive dashboard with data visualization and export capabilities
- Historical performance validation and scenario analysis

## 🔄 Future Enhancements

### ✅ Recently Completed
- **SHAP Explainability** - Feature importance analysis for FX model predictions with interactive visualizations
- **Interactive Dashboard** - Streamlit-based visualization with comprehensive charts
- **Advanced Simulation Engine** - Multi-scenario comparison with risk/cost analysis
- **Enhanced Data Visualization** - FX trends, payment patterns, and portfolio insights

### ✅ Recently Completed (Phase 2)
- **SHAP Explainability** - Feature importance analysis for FX model predictions with interactive visualizations
- **Interactive Dashboard** - Streamlit-based visualization with comprehensive charts
- **Advanced Simulation Engine** - Multi-scenario comparison with risk/cost analysis
- **Enhanced Data Visualization** - FX trends, payment patterns, and portfolio insights
- **Comprehensive Demo Script** - Automated showcase of all system capabilities

### Phase 3: Advanced Models
- LSTM neural networks for improved FX prediction
- Advanced behavioral clustering (vendor/customer segmentation)
- Real-time FX data integration
- Multi-currency portfolio optimization

### Phase 3: Production Integration
- **PAIN Protocol Integration** - Programmable payment rail optimization
- REST API for enterprise integration
- Real ERP connectors (SAP, Oracle, QuickBooks)
- Compliance reporting and audit trails
- Automated payment scheduling

### Phase 4: Advanced Features
- Integration with treasury management systems
- Multi-entity optimization
- Regulatory reporting automation
- Real-time alerting system

## 🚀 Deployment & Production Use

### Automated Deployment
```bash
# Run automated deployment (creates virtual environment, installs dependencies, validates system)
python3 deploy.py

# Or deploy with custom options
python3 deploy.py --skip-tests  # Skip integration tests
python3 deploy.py --skip-demo   # Skip verification demo
```

### Manual Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run system validation
python3 tests/test_system_integration.py

# 3. Run feature demo
python3 demo.py
```

### Production Startup
```bash
# CLI Interface (Command Line)
python3 interface/interface_cli.py

# Dashboard Interface (Web UI)
streamlit run interface/dashboard.py

# Run Tests
python3 tests/test_system_integration.py
```

### System Architecture
```
kairo/
├── deploy.py              # 🚀 Automated deployment script
├── demo.py               # 🎬 Interactive feature demonstration
├── interface/
│   ├── dashboard.py      # 📊 Streamlit web dashboard
│   └── interface_cli.py  # 💻 Command-line interface
├── modules/              # 🧠 Core AI modules
├── tests/               # ✅ Comprehensive test suite
└── data/                # 📁 Mock ERP data
```

## 🤝 Contributing

This is an MVP built for explainability and modularity. Each module should be independently testable and well-documented.

### Development Guidelines
- Maintain explainability - every recommendation needs reasoning
- Keep modules independent for testing and maintenance
- Include confidence scores for all predictions
- Use behavioral data over static assumptions
- Document all major functions with examples

### Testing
```bash
# Run FX model tests
python tests/test_fx_model.py

# Add more test files following the same pattern
# tests/test_[module_name].py
```

## 📄 License

This project is part of the Kairo Treasury Optimization initiative. See project documentation for licensing details.