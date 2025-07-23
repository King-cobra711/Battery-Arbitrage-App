# 🔋 Battery Arbitrage Analysis App

A comprehensive Streamlit application for analyzing battery arbitrage opportunities using realistic QLD electricity price data.

## 🚀 Features

- **Realistic Price Data Generation**: Simulates QLD electricity prices based on historical market patterns
- **Real Market Data Upload**: Upload actual QLD electricity market data from Open Electricity Australia
- **Battery Arbitrage Simulation**: Calculates profit from buying low and selling high
- **Parameter Testing**: Test different battery configurations on the same dataset
- **Interactive Charts**: Visualize price patterns, daily profits, and cumulative returns
- **Export Functionality**: Download results as CSV for further analysis
- **Educational Content**: Detailed explanations of calculations and market patterns

## 📊 Key Capabilities

### Data Sources

#### Simulated Price Data Generation

- **Seasonal Adjustments**: Summer/winter higher, spring/autumn lower
- **Time-of-Day Pricing**: Peak hours (6-9 AM, 5-8 PM) vs off-peak
- **Weekend Adjustments**: Lower demand on weekends
- **Realistic Volatility**: ±15% random variation
- **Price Bounds**: $20 - $300/MWh

#### Real Market Data Upload

- **Open Electricity Australia Integration**: Upload actual QLD market data
- **CSV Format Support**: Compatible with Open Electricity Australia exports
- **Automatic Date Range Detection**: Extracts analysis period from uploaded data
- **Solar Intensity Estimation**: Estimates solar patterns for real market data
- **Real Price Statistics**: Actual min/max/average prices from market data

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

### Data Source Selection

The app supports two data sources:

#### 1. Simulated Data (Default)

- **Best for**: Testing scenarios, learning, parameter optimization
- **Features**: Solar crash modeling, duck curve effects, seasonal variations
- **Period**: 7-90 days (user selectable)

#### 2. Real Market Data Upload

- **Best for**: Real market analysis, actual arbitrage opportunities
- **Source**: [Open Electricity Australia](https://explore.openelectricity.org.au/energy/qld1/?range=7d&interval=30m&view=discrete-time&group=Detailed)
- **Format**: CSV with specific column requirements
- **Period**: Fixed 7-day analysis period

### Real Market Data Setup

#### Step 1: Download Data from Open Electricity Australia

1. Visit [Open Electricity Australia QLD](https://explore.openelectricity.org.au/energy/qld1/?range=7d&interval=30m&view=discrete-time&group=Detailed)
2. **Set Range**: Must be exactly 7 days
3. **Set Interval**: Must be 30 minutes
4. **View**: Discrete-time
5. **Group**: Detailed
6. **Export**: Download as CSV

#### Step 2: Upload to App

1. Select "Upload Real Market Data" in the sidebar
2. Click "Choose a CSV file" and select your downloaded file
3. Verify the file contains required columns:
   - `date`: Timestamp column
   - `Price - AUD/MWh`: Price column

#### Step 3: Run Analysis

1. Configure battery parameters (Power, Capacity, Efficiency)
2. Set trading windows (charge/discharge times)
3. Click "🚀 Run Analysis with Real Data"

### First Time Setup (Simulated Data)

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

### Real Market Data Processing

#### CSV Parsing

```python
# Expected CSV format from Open Electricity Australia
date,Price - AUD/MWh,other_columns...
2025-07-16 10:30,-0.01,...
2025-07-16 11:00,-12.28,...
```

#### Price Statistics Calculation

```python
# Calculated on ALL uploaded data points
Min Price = df['price'].min()  # Lowest price in dataset
Max Price = df['price'].max()  # Highest price in dataset
Avg Price = df['price'].mean() # Average of all prices
```

#### Solar Intensity Estimation

- **Real data doesn't include solar intensity**
- **App estimates based on time of day patterns**
- **Seasonal adjustments applied based on month**

### Simulated Data Processing

#### Price Generation Formula

```
Final Price = Base Price × Seasonal Factor × Time Factor × Solar Factor × (1 + Volatility)
```

#### Solar Crash Modeling

- **Midday price depression** (10 AM - 4 PM)
- **Negative prices** during peak solar (11:30 AM - 1:00 PM)
- **Duck curve effects** with evening peaks

## 🔧 Technical Details

### Real Market Data Requirements

#### CSV Format

- **Required Columns**: `date`, `Price - AUD/MWh`
- **Date Format**: YYYY-MM-DD HH:MM
- **Price Format**: Numeric values in AUD/MWh
- **Interval**: 30-minute intervals
- **Period**: 7 days (336 data points)

#### Data Source

- **Platform**: [Open Electricity Australia](https://explore.openelectricity.org.au/energy/qld1/?range=7d&interval=30m&view=discrete-time&group=Detailed)
- **Region**: QLD (Queensland)
- **Market**: NEM (National Electricity Market)
- **Frequency**: Real-time 30-minute settlement prices

### Error Handling

#### Common Issues

- **Missing Columns**: CSV must contain `date` and `Price - AUD/MWh`
- **Wrong Format**: Dates must be parseable timestamps
- **Empty Data**: File must contain valid price data
- **Wrong Period**: Must be 7-day data for optimal analysis

#### Validation

- **Column Check**: Verifies required columns exist
- **Data Type Check**: Ensures prices are numeric
- **Date Range Check**: Confirms 7-day period
- **Price Range Check**: Validates realistic price bounds

## 📊 Analysis Features

### Real Market Data Analysis

- **Actual Price Statistics**: Real min/max/average from market
- **True Arbitrage Opportunities**: Based on actual price spreads
- **Market Pattern Recognition**: Identifies real duck curve effects
- **OTC Recommendations**: Based on actual historical patterns

### Simulated Data Analysis

- **Solar Crash Modeling**: Simulated midday price depression
- **Seasonal Variations**: Summer/winter/spring/autumn effects
- **Volatility Modeling**: ±15% random variation
- **Educational Content**: Detailed pattern explanations

## 🎯 Use Cases

### Real Market Data

- **Actual arbitrage analysis** for QLD market
- **Historical pattern recognition** in price data
- **Real OTC contract planning** based on actual prices
- **Market research** for battery investment decisions

### Simulated Data

- **Scenario testing** with different parameters
- **Educational purposes** to understand market dynamics
- **Parameter optimization** without real data constraints
- **Learning solar crash effects** and duck curve patterns

## 📈 Example Results

### Real Market Data (7-day QLD data)

```
Min Price: -$16.43/MWh (solar crash periods)
Max Price: $288.80/MWh (evening peak periods)
Avg Price: $95.23/MWh (overall average)
Analysis Period: 2025-07-16 to 2025-07-22
```

### Simulated Data (30-day period)

```
Min Price: $20.00/MWh (simulated bounds)
Max Price: $300.00/MWh (simulated bounds)
Avg Price: $85.50/MWh (with solar modeling)
Analysis Period: 30 days with seasonal effects
```

## 🔗 Data Sources

- **Real Market Data**: [Open Electricity Australia](https://explore.openelectricity.org.au/energy/qld1/?range=7d&interval=30m&view=discrete-time&group=Detailed)
- **Market Information**: NEM (National Electricity Market)
- **Geographic Focus**: QLD (Queensland) electricity market
- **Data Frequency**: 30-minute settlement prices
