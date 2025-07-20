# 🔋 Battery Arbitrage Analysis App

A comprehensive Streamlit application for analyzing battery arbitrage opportunities using realistic QLD electricity price data.

## 🚀 Features

- **Realistic Price Data Generation**: Simulates QLD electricity prices based on historical market patterns
- **Battery Arbitrage Simulation**: Calculates profit from buying low and selling high
- **Parameter Testing**: Test different battery configurations on the same dataset
- **Interactive Charts**: Visualize price patterns, daily profits, and cumulative returns
- **Export Functionality**: Download results as CSV for further analysis
- **Educational Content**: Detailed explanations of calculations and market patterns

## 📊 Key Capabilities

### Price Data Generation

- **Seasonal Adjustments**: Summer/winter higher, spring/autumn lower
- **Time-of-Day Pricing**: Peak hours (6-9 AM, 5-8 PM) vs off-peak
- **Weekend Adjustments**: Lower demand on weekends
- **Realistic Volatility**: ±15% random variation
- **Price Bounds**: $20 - $300/MWh

### Battery Simulation

- **Power Constraints**: Limited by battery rated power (MW)
- **Energy Constraints**: Limited by battery capacity (MWh)
- **Efficiency Modeling**: Round-trip efficiency losses
- **Trading Windows**: Configurable charge/discharge periods

### Profit Calculation

- **Daily Analysis**: Charge costs vs discharge revenue
- **Cumulative Tracking**: Profit growth over time
- **Key Metrics**: Total, average, max, and min daily profits
- **Export Results**: CSV download with detailed breakdown

## 🛠️ Installation

1. **Clone the repository**:

   ```bash
   git clone <your-repo-url>
   cd battery-arbitrage-analysis
   ```

2. **Create virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 📋 Requirements

- Python 3.8+
- Streamlit
- Pandas
- Plotly
- NumPy

## 🎯 How to Use

### First Time Setup

1. **Configure Battery**: Set power (MW) and capacity (MWh)
2. **Set Efficiency**: Choose round-trip efficiency percentage
3. **Define Trading Windows**: Set charge and discharge time periods
4. **Select Period**: Choose simulation duration (7-90 days)
5. **Run Initial Simulation**: Click "🚀 Run Initial Simulation" to start

### Parameter Testing

- **Adjust Settings**: Modify battery parameters in the sidebar
- **Test Parameters**: Click "⚡ Test New Parameters" to see results instantly
- **Fresh Data**: Click "🔄 Generate New Data & Run" for new price data

### Results Analysis

- **Daily Profit Chart**: Color-coded daily performance
- **Cumulative Profit**: Growth over time visualization
- **Price Patterns**: Sample data with market explanations
- **Export Data**: Download CSV for external analysis

## 📈 Key Calculations

### Profit Formula

```
Daily Profit = Total Discharge Revenue - Total Charge Cost
```

### Battery Constraints

- **Power**: Max discharge/charge rate = Battery Power (MW)
- **Energy**: Max storage = Battery Capacity (MWh)
- **Efficiency**: Available energy = Stored Energy × Efficiency

### Price Generation

```
Final Price = Base Price × Seasonal Factor × Time Factor × (1 + Volatility)
```

## 🔧 Configuration Options

### Battery Specifications

- **Power**: 1-1000 MW
- **Capacity**: 1-10000 MWh
- **Efficiency**: 70-95% round-trip

### Trading Windows

- **Charge Period**: Configurable hours (0-23)
- **Discharge Period**: Configurable hours (0-23)

### Simulation Period

- **Data Range**: 7, 14, 30, 60, or 90 days
- **Time Resolution**: 30-minute intervals

## 📊 Example Results

The app provides comprehensive analysis including:

- **Total Profit**: Sum of all daily profits
- **Average Daily Profit**: Mean daily performance
- **Maximum Daily Profit**: Best single day
- **Minimum Daily Profit**: Worst single day
- **Cumulative Growth**: Profit over time
- **Price Statistics**: Min, max, average prices

## 🎓 Educational Value

### Market Understanding

- **QLD Electricity Market**: Realistic price patterns
- **Arbitrage Strategy**: Buy low, sell high concept
- **Battery Economics**: Power vs energy constraints
- **Efficiency Impact**: How losses affect profitability

### Technical Insights

- **Data Generation**: How realistic prices are created
- **Calculation Methods**: Detailed profit formulas
- **Constraint Modeling**: Real-world battery limitations
- **Risk Factors**: Market volatility and technical issues

## 🔍 Use Cases

- **Energy Traders**: Analyze arbitrage opportunities
- **Battery Investors**: Evaluate project economics
- **Students**: Learn about energy markets and battery economics
- **Researchers**: Study arbitrage strategies and market dynamics

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For questions or issues, please open an issue on GitHub.

---

**Note**: This application uses simulated data for educational purposes. Real trading decisions should be based on actual market data and professional analysis.
