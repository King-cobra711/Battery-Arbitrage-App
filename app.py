import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import io

# Page configuration
st.set_page_config(
    page_title="Battery Arbitrage Analysis",
    page_icon="🔋",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def generate_qld_price_data(start_date, end_date):
    """
    Generate realistic QLD electricity price data based on historical market patterns
    """
    try:
        st.info("📊 Generating realistic QLD electricity price data...")
        
        # Create date range with 30-minute intervals
        date_range = pd.date_range(start=start_date, end=end_date, freq='30min')
        
        # Base price patterns for QLD (realistic ranges based on historical data)
        base_price = 80  # Base price in $/MWh
        
        prices = []
        for timestamp in date_range:
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            month = timestamp.month
            
            # Seasonal adjustment (higher in summer/winter, lower in spring/autumn)
            seasonal_factor = 1.0
            if month in [12, 1, 2]:  # Summer
                seasonal_factor = 1.2
            elif month in [6, 7, 8]:  # Winter
                seasonal_factor = 1.15
            elif month in [3, 4, 5]:  # Autumn
                seasonal_factor = 0.9
            else:  # Spring
                seasonal_factor = 0.85
            
            # Time-of-day pricing (realistic QLD patterns)
            if 6 <= hour <= 9:  # Morning peak
                time_factor = 1.8
            elif 17 <= hour <= 20:  # Evening peak
                time_factor = 2.0
            elif 22 <= hour or hour <= 5:  # Off-peak night
                time_factor = 0.6
            else:  # Daytime
                time_factor = 1.2
            
            # Weekend adjustment (lower demand)
            if day_of_week >= 5:  # Weekend
                time_factor *= 0.8
            
            # Add realistic volatility
            volatility = np.random.normal(0, 0.15)
            
            # Calculate final price
            price = base_price * seasonal_factor * time_factor * (1 + volatility)
            
            # Ensure price stays within realistic bounds
            price = max(20, min(300, price))
            
            prices.append({
                'timestamp': timestamp,
                'price': price
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(prices)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        st.success(f"✅ Successfully generated {len(df)} realistic QLD price points")
        st.write(f"📈 Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}/MWh")
        st.write(f"📊 Average price: ${df['price'].mean():.2f}/MWh")
        st.write(f"📅 Date range: {start_date} to {end_date}")
        st.info("💡 Using realistic QLD electricity price patterns based on historical market data")
        
        return df
            
    except Exception as e:
        st.error(f"❌ Error generating price data: {e}")
        return None

def simulate_arbitrage(df, battery_size_mw, battery_capacity_mwh, 
                      charge_start, charge_end, discharge_start, discharge_end,
                      round_trip_efficiency):
    """
    Simulate battery arbitrage trading
    """
    if df is None or df.empty:
        return None
    
    # Convert efficiency to decimal
    efficiency = round_trip_efficiency / 100
    
    # Initialize results
    results = []
    daily_profits = {}
    
    # Group by date for daily analysis
    df['date'] = df.index.date
    df['hour'] = df.index.hour
    
    for date, day_data in df.groupby('date'):
        day_data = day_data.sort_index()
        
        # Find charge and discharge periods
        charge_mask = (
            (day_data['hour'] >= charge_start) & 
            (day_data['hour'] < charge_end)
        )
        
        discharge_mask = (
            (day_data['hour'] >= discharge_start) & 
            (day_data['hour'] < discharge_end)
        )
        
        charge_prices = day_data[charge_mask]['price']
        discharge_prices = day_data[discharge_mask]['price']
        
        if len(charge_prices) == 0 or len(discharge_prices) == 0:
            continue
        
        # Calculate optimal arbitrage
        total_charge_cost = 0
        total_discharge_revenue = 0
        energy_stored = 0
        
        # Charge during low price periods
        for _, row in day_data[charge_mask].iterrows():
            if energy_stored < battery_capacity_mwh:
                charge_amount = min(battery_size_mw * 0.5, battery_capacity_mwh - energy_stored)
                charge_cost = charge_amount * row['price']
                total_charge_cost += charge_cost
                energy_stored += charge_amount
        
        # Discharge during high price periods
        for _, row in day_data[discharge_mask].iterrows():
            if energy_stored > 0:
                discharge_amount = min(battery_size_mw * 0.5, energy_stored * efficiency)
                discharge_revenue = discharge_amount * row['price']
                total_discharge_revenue += discharge_revenue
                energy_stored -= discharge_amount / efficiency
        
        daily_profit = total_discharge_revenue - total_charge_cost
        
        results.append({
            'date': date,
            'charge_cost': total_charge_cost,
            'discharge_revenue': total_discharge_revenue,
            'profit': daily_profit,
            'energy_stored': energy_stored
        })
        
        daily_profits[date] = daily_profit
    
    return pd.DataFrame(results), daily_profits

def main():
    # Initialize session state for storing data
    if 'price_data' not in st.session_state:
        st.session_state.price_data = None
    if 'current_start_date' not in st.session_state:
        st.session_state.current_start_date = None
    if 'current_end_date' not in st.session_state:
        st.session_state.current_end_date = None
    if 'has_run_initial_simulation' not in st.session_state:
        st.session_state.has_run_initial_simulation = False
    
    # Header
    st.markdown('<h1 class="main-header">🔋 Battery Arbitrage Analysis</h1>', unsafe_allow_html=True)
    st.markdown("Analyze battery arbitrage opportunities using realistic QLD electricity price data")
    
    # Sidebar for user inputs
    st.sidebar.markdown('<div class="sidebar-header">⚙️ Configuration</div>', unsafe_allow_html=True)
    
    # Battery specifications
    st.sidebar.subheader("Battery Specifications")
    battery_size_mw = st.sidebar.number_input(
        "Battery Power (MW)", 
        min_value=1.0, 
        max_value=1000.0, 
        value=100.0, 
        step=10.0
    )
    
    battery_capacity_mwh = st.sidebar.number_input(
        "Battery Capacity (MWh)", 
        min_value=1.0, 
        max_value=10000.0, 
        value=400.0, 
        step=50.0
    )
    
    round_trip_efficiency = st.sidebar.slider(
        "Round-trip Efficiency (%)", 
        min_value=70, 
        max_value=95, 
        value=85, 
        step=5
    )
    
    # Trading windows
    st.sidebar.subheader("Trading Windows")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        charge_start = st.number_input("Charge Start (Hour)", 0, 23, 10)
        charge_end = st.number_input("Charge End (Hour)", 0, 23, 14)
    
    with col2:
        discharge_start = st.number_input("Discharge Start (Hour)", 0, 23, 17)
        discharge_end = st.number_input("Discharge End (Hour)", 0, 23, 21)
    
    # Simulation period
    st.sidebar.subheader("Simulation Period")
    days_back = st.sidebar.selectbox(
        "Data Period", 
        [7, 14, 30, 60, 90], 
        index=2
    )
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    st.sidebar.info(f"Analyzing data from {start_date} to {end_date}")
    
    # Check if we have existing data
    has_existing_data = st.session_state.price_data is not None
    
    # Show data status
    if has_existing_data:
        st.sidebar.success(f"✅ Using existing price data from {st.session_state.current_start_date} to {st.session_state.current_end_date}")
        st.sidebar.info("💡 Adjust parameters and click 'Test New Parameters' to see results")
    else:
        st.sidebar.info("�� No data available. Set your parameters and click 'Run Initial Simulation' to start analysis.")
    
    # Simulation buttons
    st.sidebar.subheader("Simulation Controls")
    
    # Initial simulation button (only show when no data exists)
    if not has_existing_data:
        if st.sidebar.button("🚀 Run Initial Simulation", type="primary"):
            with st.spinner("Generating realistic electricity price data and running simulation..."):
                # Generate new data
                df = generate_qld_price_data(start_date, end_date)
                
                if df is not None:
                    # Store data in session state
                    st.session_state.price_data = df
                    st.session_state.current_start_date = start_date
                    st.session_state.current_end_date = end_date
                    
                    # Run simulation
                    result = simulate_arbitrage(
                        df, battery_size_mw, battery_capacity_mwh,
                        charge_start, charge_end, discharge_start, discharge_end,
                        round_trip_efficiency
                    )
                    
                    if result is not None:
                        results_df, daily_profits = result
                        # Store results in session state for display
                        st.session_state.current_results = (results_df, daily_profits)
                        st.rerun()
    
    # Manual control buttons (only show when data exists)
    if has_existing_data:
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🔄 Generate New Data & Run", type="primary"):
                with st.spinner("Generating new price data and running simulation..."):
                    # Generate new data
                    df = generate_qld_price_data(start_date, end_date)
                    
                    if df is not None:
                        # Store data in session state
                        st.session_state.price_data = df
                        st.session_state.current_start_date = start_date
                        st.session_state.current_end_date = end_date
                        
                        # Run simulation with current parameters
                        result = simulate_arbitrage(
                            df, battery_size_mw, battery_capacity_mwh,
                            charge_start, charge_end, discharge_start, discharge_end,
                            round_trip_efficiency
                        )
                        
                        if result is not None:
                            results_df, daily_profits = result
                            # Store results in session state for display
                            st.session_state.current_results = (results_df, daily_profits)
                            st.rerun()
        
        with col2:
            if st.button("⚡ Test New Parameters", type="secondary"):
                with st.spinner("Running simulation with new parameters..."):
                    # Use existing data
                    df = st.session_state.price_data
                    
                    # Run simulation
                    result = simulate_arbitrage(
                        df, battery_size_mw, battery_capacity_mwh,
                        charge_start, charge_end, discharge_start, discharge_end,
                        round_trip_efficiency
                    )
                    
                    if result is not None:
                        results_df, daily_profits = result
                        # Store results in session state for display
                        st.session_state.current_results = (results_df, daily_profits)
                        st.rerun()
    
    # Display results if we have them
    if 'current_results' in st.session_state and st.session_state.current_results is not None:
        results_df, daily_profits = st.session_state.current_results
        
        # Get the current price data for display
        df = st.session_state.price_data
        
        if df is not None:
            # Show sample of the generated price data for demo purposes
            st.markdown("## 📊 Generated Price Data (Demo)")
            
            # Create a sample of the price data for display
            sample_df = df.head(48)  # Show first 24 hours (48 x 30-min intervals)
            
            # Format for display
            display_price_df = sample_df.reset_index()
            display_price_df['timestamp'] = display_price_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            display_price_df['price'] = display_price_df['price'].round(2)
            
            # Rename columns properly
            display_price_df = display_price_df[['timestamp', 'price']]
            display_price_df.columns = ['Timestamp', 'Price ($/MWh)']
            
            # Show price statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Min Price", f"${df['price'].min():.2f}/MWh")
            with col2:
                st.metric("Max Price", f"${df['price'].max():.2f}/MWh")
            with col3:
                st.metric("Avg Price", f"${df['price'].mean():.2f}/MWh")
            
            # Price chart
            fig_price = px.line(
                sample_df.reset_index(),
                x='timestamp',
                y='price',
                title="Sample QLD Electricity Prices (24 Hours)",
                labels={'price': 'Price ($/MWh)', 'timestamp': 'Time'},
                markers=True
            )
            fig_price.update_layout(height=300)
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Price data table
            st.markdown("### 📋 Sample Price Data")
            st.dataframe(display_price_df, use_container_width=True)
            
            # Show price patterns explanation
            with st.expander("ℹ️ Price Pattern Explanation"):
                st.markdown("""
                **QLD Electricity Price Patterns:**
                
                - **Peak Hours (6-9 AM, 5-8 PM)**: Higher prices due to increased demand
                - **Off-Peak Hours (10 PM - 6 AM)**: Lower prices due to reduced demand
                - **Weekend Prices**: Generally lower due to reduced commercial activity
                - **Seasonal Variations**: Summer/winter higher, spring/autumn lower
                - **Realistic Volatility**: ±15% random variation to simulate market uncertainty
                
                These patterns are based on actual QLD wholesale electricity market behavior.
                """)
            
            st.markdown("---")
            
            # Display results
            st.markdown("## 📊 Simulation Results")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_profit = results_df['profit'].sum()
            avg_daily_profit = results_df['profit'].mean()
            max_daily_profit = results_df['profit'].max()
            min_daily_profit = results_df['profit'].min()
            
            with col1:
                st.metric("Total Profit", f"${total_profit:,.2f}")
            
            with col2:
                st.metric("Avg Daily Profit", f"${avg_daily_profit:,.2f}")
            
            with col3:
                st.metric("Max Daily Profit", f"${max_daily_profit:,.2f}")
            
            with col4:
                st.metric("Min Daily Profit", f"${min_daily_profit:,.2f}")
            
            # Charts
            st.markdown("### 📈 Profit Analysis")
            
            # Cumulative profit chart
            results_df['cumulative_profit'] = results_df['profit'].cumsum()
            
            fig_cumulative = px.line(
                results_df, 
                x='date', 
                y='cumulative_profit',
                title="Cumulative Profit Over Time",
                labels={'cumulative_profit': 'Cumulative Profit ($)', 'date': 'Date'}
            )
            fig_cumulative.update_layout(height=400)
            st.plotly_chart(fig_cumulative, use_container_width=True)
            
            # Daily profit chart
            fig_daily = px.bar(
                results_df, 
                x='date', 
                y='profit',
                title="Daily Profit",
                labels={'profit': 'Daily Profit ($)', 'date': 'Date'},
                color='profit',
                color_continuous_scale='RdYlGn'
            )
            fig_daily.update_layout(height=400)
            st.plotly_chart(fig_daily, use_container_width=True)
            
            # Results table
            st.markdown("### 📋 Daily Results")
            
            # Format the results table
            display_df = results_df.copy()
            display_df['date'] = display_df['date'].astype(str)
            display_df['charge_cost'] = display_df['charge_cost'].round(2)
            display_df['discharge_revenue'] = display_df['discharge_revenue'].round(2)
            display_df['profit'] = display_df['profit'].round(2)
            display_df['cumulative_profit'] = display_df['cumulative_profit'].round(2)
            
            display_df.columns = ['Date', 'Charge Cost ($)', 'Discharge Revenue ($)', 
                               'Daily Profit ($)', 'Energy Stored (MWh)', 'Cumulative Profit ($)']
            
            st.dataframe(display_df, use_container_width=True)
            
            # Export functionality
            st.markdown("### 💾 Export Results")
            
            # Create CSV for download
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            csv_str = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download CSV",
                data=csv_str,
                file_name=f"battery_arbitrage_results_{st.session_state.current_start_date}_{st.session_state.current_end_date}.csv",
                mime="text/csv"
            )
            
            # Configuration summary
            st.markdown("### ⚙️ Configuration Summary")
            config_data = {
                'Parameter': [
                    'Battery Power (MW)',
                    'Battery Capacity (MWh)',
                    'Round-trip Efficiency (%)',
                    'Charge Window',
                    'Discharge Window',
                    'Simulation Period'
                ],
                'Value': [
                    f"{battery_size_mw} MW",
                    f"{battery_capacity_mwh} MWh",
                    f"{round_trip_efficiency}%",
                    f"{charge_start}:00 - {charge_end}:00",
                    f"{discharge_start}:00 - {discharge_end}:00",
                    f"{days_back} days"
                ]
            }
            
            config_df = pd.DataFrame(config_data)
            st.dataframe(config_df, use_container_width=True)
    
    # Instructions (always show when no simulation has been run)
    st.markdown("""
    ## 🎯 How to Use This App
    
    1. **Configure Battery**: Set your battery power (MW) and capacity (MWh)
    2. **Set Efficiency**: Choose the round-trip efficiency percentage
    3. **Define Trading Windows**: Set charge and discharge time windows
    4. **Select Period**: Choose how many days of historical data to analyze
    5. **Run Simulation**: Click the button to analyze arbitrage opportunities
    
    ## 📊 What You'll Get
    
    - **Daily profit analysis** with charge costs and discharge revenue
    - **Cumulative profit chart** showing growth over time
    - **Daily profit breakdown** with color-coded performance
    - **CSV export** for further analysis
    
    ## 🔋 Arbitrage Strategy
    
    The simulation charges the battery during low-price periods and discharges during high-price periods, 
    accounting for round-trip efficiency losses to calculate net profit.
    """)
    
    # Calculations Section
    with st.expander("🧮 View All Calculations & Formulas"):
        st.markdown("""
        ## 📊 Price Data Generation
        
        **Base Price**: $80/MWh (QLD wholesale average)
        
        **Seasonal Adjustments**:
        - Summer (Dec-Feb): Base Price × 1.2
        - Winter (Jun-Aug): Base Price × 1.15  
        - Autumn (Mar-May): Base Price × 0.9
        - Spring (Sep-Nov): Base Price × 0.85
        
        **Time-of-Day Factors**:
        - Morning Peak (6-9 AM): Base Price × 1.8
        - Evening Peak (5-8 PM): Base Price × 2.0
        - Off-Peak Night (10 PM-6 AM): Base Price × 0.6
        - Daytime (Other): Base Price × 1.2
        
        **Final Price Formula**:
        ```
        Final Price = Base Price × Seasonal Factor × Time Factor × (1 + Volatility)
        Where: Volatility = ±15% random variation
        Price Bounds: $20 - $300/MWh
        ```
        
        ## ⚡ Battery Specifications
        
        **Power to Energy Conversion (30-min intervals)**:
        ```
        Max Discharge per 30min = Battery Power (MW) × 0.5 hours
        Max Charge per 30min = Battery Power (MW) × 0.5 hours
        ```
        
        **Efficiency Conversion**:
        ```
        Efficiency Decimal = Round-trip Efficiency (%) ÷ 100
        Example: 85% efficiency = 0.85
        ```
        
        ## 💰 Charge Calculations
        
        **Charge Amount (per 30-min interval)**:
        ```
        Charge Amount = min(Battery Power × 0.5, Available Capacity)
        Where: Available Capacity = Battery Capacity - Current Energy Stored
        ```
        
        **Charge Cost**:
        ```
        Charge Cost = Charge Amount (MWh) × Electricity Price ($/MWh)
        ```
        
        **Energy Stored After Charge**:
        ```
        New Energy Stored = Current Energy Stored + Charge Amount
        ```
        
        ## 🔋 Discharge Calculations
        
        **Discharge Amount (per 30-min interval)**:
        ```
        Discharge Amount = min(Battery Power × 0.5, Energy Stored × Efficiency)
        ```
        
        **Discharge Revenue**:
        ```
        Discharge Revenue = Discharge Amount (MWh) × Electricity Price ($/MWh)
        ```
        
        **Energy Removed from Battery**:
        ```
        Energy Removed = Discharge Amount ÷ Efficiency
        ```
        
        ## 📈 Profit Calculations
        
        **Daily Profit**:
        ```
        Daily Profit = Total Discharge Revenue - Total Charge Cost
        ```
        
        **Cumulative Profit**:
        ```
        Cumulative Profit = Previous Cumulative Profit + Daily Profit
        ```
        
        **Profit Margin**:
        ```
        Profit Margin (%) = (Daily Profit ÷ Total Charge Cost) × 100
        ```
        
        ## 📊 Example Calculation
        
        **Scenario**: 100 MW / 400 MWh Battery, 85% Efficiency
        
        **Charge Period (10 AM - 2 PM, $60/MWh)**:
        ```
        Charge Amount = min(100 × 0.5, 400 - 0) = 50 MWh
        Charge Cost = 50 MWh × $60/MWh = $3,000
        Energy Stored = 0 + 50 = 50 MWh
        ```
        
        **Discharge Period (5 PM - 9 PM, $150/MWh)**:
        ```
        Discharge Amount = min(100 × 0.5, 50 × 0.85) = min(50, 42.5) = 42.5 MWh
        Discharge Revenue = 42.5 MWh × $150/MWh = $6,375
        Energy Removed = 42.5 ÷ 0.85 = 50 MWh
        Energy Stored = 50 - 50 = 0 MWh
        ```
        
        **Daily Profit**:
        ```
        Daily Profit = $6,375 - $3,000 = $3,375
        ```
        
        ## 🔧 Key Constraints
        
        **Power Constraints**:
        - Max Discharge Rate = Battery Power (MW)
        - Max Charge Rate = Battery Power (MW)
        
        **Energy Constraints**:
        - Max Energy Stored = Battery Capacity (MWh)
        - Min Energy Stored = 0 MWh
        
        **Efficiency Constraints**:
        - Available Energy = Stored Energy × Efficiency
        - Energy Loss = Energy × (1 - Efficiency)
        
        ## 📋 Data Processing
        
        **30-Minute Intervals**:
        - Intervals per Day = 24 hours × 2 = 48 intervals
        - Intervals per Hour = 2 intervals
        
        **Price Data Structure**:
        - Timestamp: YYYY-MM-DD HH:MM
        - Price: $/MWh (2 decimal places)
        
        ## 🎯 Key Insights
        
        **Profit Drivers**:
        1. Price Spread: Higher difference between charge and discharge prices
        2. Battery Size: Larger capacity = more energy arbitrage
        3. Efficiency: Higher efficiency = less energy loss
        4. Trading Windows: Optimal timing maximizes profit
        
        **Risk Factors**:
        1. Price Volatility: Unpredictable price movements
        2. Battery Degradation: Reduced capacity over time
        3. Market Changes: Regulatory or structural changes
        4. Technical Issues: Battery or grid failures
        """)

if __name__ == "__main__":
    main() 