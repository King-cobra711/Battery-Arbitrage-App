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
    including solar crash effects and duck curve patterns
    """
    try:
        st.info("📊 Generating realistic QLD electricity price data with solar crash modeling...")
        
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
            
            # Solar generation patterns (QLD has high solar penetration)
            solar_factor = 1.0
            solar_intensity = 0.0
            
            # Solar generation curve with specific time patterns
            if 6 <= hour <= 18:  # Daylight hours
                # More realistic solar curve based on actual patterns
                if hour < 10:  # Early morning ramp
                    solar_intensity = (hour - 6) / 4  # 0 to 1 over 4 hours
                elif 10 <= hour <= 11.5:  # Ramping up to peak
                    solar_intensity = 0.8 + (hour - 10) * 0.13  # 0.8 to 1.0
                elif 11.5 <= hour <= 13:  # Peak solar (11:30 AM - 1:00 PM)
                    solar_intensity = 1.0
                elif 13 < hour <= 14.5:  # Starting to ramp down
                    solar_intensity = 1.0 - (hour - 13) * 0.13  # 1.0 to 0.8
                else:  # Afternoon decline
                    solar_intensity = max(0, 0.8 - (hour - 14.5) * 0.2)
                
                # Seasonal solar intensity (stronger in summer, weaker in winter)
                seasonal_solar = 1.0
                if month in [12, 1, 2]:  # Summer - strongest solar
                    seasonal_solar = 1.3
                elif month in [6, 7, 8]:  # Winter - weakest solar
                    seasonal_solar = 0.6
                elif month in [3, 4, 5]:  # Autumn
                    seasonal_solar = 0.8
                else:  # Spring
                    seasonal_solar = 1.1
                
                solar_intensity *= seasonal_solar
                
                # Solar crash effect on prices based on specific time patterns
                if 10 <= hour < 11.5:  # 10:00 AM - 11:30 AM: Solar ramping up
                    solar_crash = solar_intensity * 0.3  # Up to 30% price reduction
                    solar_factor = 1.0 - solar_crash
                elif 11.5 <= hour <= 13:  # 11:30 AM - 1:00 PM: Maximum solar penetration
                    solar_crash = solar_intensity * 0.8  # Up to 80% price reduction (can go negative)
                    solar_factor = 1.0 - solar_crash
                elif 13 < hour <= 14.5:  # 1:00 PM - 2:30 PM: Solar ramping down
                    solar_crash = solar_intensity * 0.5  # Up to 50% price reduction
                    solar_factor = 1.0 - solar_crash
                elif 7 <= hour <= 9:  # Morning ramp (before solar peak)
                    # Higher prices as demand rises but solar not yet strong
                    solar_factor = 1.1
                elif 17 <= hour <= 19:  # Evening ramp (after solar drops)
                    # Highest prices as solar drops and demand peaks
                    solar_factor = 1.8
            
            # Time-of-day pricing (modified for solar effects)
            if 6 <= hour <= 9:  # Morning peak (before solar)
                time_factor = 1.6
            elif 10 <= hour <= 16:  # Solar peak hours (reduced due to solar)
                time_factor = 0.8  # Lower due to solar generation
            elif 17 <= hour <= 20:  # Evening peak (after solar drops)
                time_factor = 2.2  # Higher due to solar drop + demand
            elif 22 <= hour or hour <= 5:  # Off-peak night
                time_factor = 0.6
            else:  # Other daytime hours
                time_factor = 1.0
            
            # Weekend adjustment (lower demand, but solar still affects prices)
            if day_of_week >= 5:  # Weekend
                time_factor *= 0.8
                solar_factor = 1.0 - (solar_intensity * 0.3)  # Reduced solar crash on weekends
            
            # Add realistic volatility
            volatility = np.random.normal(0, 0.15)
            
            # Calculate final price with solar effects
            price = base_price * seasonal_factor * time_factor * solar_factor * (1 + volatility)
            
            # Ensure price stays within realistic bounds (including negative prices during solar peak)
            if 11.5 <= hour <= 13 and solar_intensity > 0.8:  # Maximum solar penetration
                # Allow negative prices during peak solar (realistic for high solar penetration)
                price = max(-50, min(300, price))
            else:
                # Normal bounds for other times
                price = max(20, min(300, price))
            
            prices.append({
                'timestamp': timestamp,
                'price': price,
                'solar_intensity': solar_intensity  # For debugging/analysis
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(prices)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        st.success(f"✅ Successfully generated {len(df)} realistic QLD price points with solar crash modeling")
        st.write(f"📈 Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}/MWh")
        st.write(f"📊 Average price: ${df['price'].mean():.2f}/MWh")
        st.write(f"📅 Date range: {start_date} to {end_date}")
        st.info("💡 Using realistic QLD electricity price patterns including solar crash effects and duck curve dynamics")
        
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

def recommend_optimal_windows(df, battery_size_mw, battery_capacity_mwh, round_trip_efficiency):
    """
    Analyze price patterns and recommend optimal 2-hour charge/discharge windows for OTC hedging
    """
    if df is None or df.empty:
        return None
    
    # Convert efficiency to decimal
    efficiency = round_trip_efficiency / 100
    
    # Group by hour and calculate average prices
    df['hour'] = df.index.hour
    hourly_prices = df.groupby('hour')['price'].agg(['mean', 'std', 'min', 'max']).reset_index()
    
    # Find optimal 2-hour charge blocks (lowest prices)
    best_charge_blocks = []
    
    # Get all possible 2-hour consecutive blocks
    for i in range(24):
        hour1 = i
        hour2 = (i + 1) % 24  # Handle wrap-around from 23 to 0
        
        # Get the two hours' average prices
        price1 = hourly_prices[hourly_prices['hour'] == hour1]['mean'].iloc[0]
        price2 = hourly_prices[hourly_prices['hour'] == hour2]['mean'].iloc[0]
        
        avg_price = (price1 + price2) / 2
        
        best_charge_blocks.append({
            'start_hour': hour1,
            'end_hour': hour2,
            'avg_price': avg_price,
            'hour1_price': price1,
            'hour2_price': price2
        })
    
    # Sort by average price (lowest first for charging)
    best_charge_blocks.sort(key=lambda x: x['avg_price'])
    
    # Find optimal 2-hour discharge blocks (highest prices)
    best_discharge_blocks = []
    
    # Use the same blocks but sort by highest price
    for block in best_charge_blocks:
        best_discharge_blocks.append({
            'start_hour': block['start_hour'],
            'end_hour': block['end_hour'],
            'avg_price': block['avg_price'],
            'hour1_price': block['hour1_price'],
            'hour2_price': block['hour2_price']
        })
    
    # Sort by average price (highest first for discharging)
    best_discharge_blocks.sort(key=lambda x: x['avg_price'], reverse=True)
    
    # Calculate potential arbitrage opportunities
    arbitrage_opportunities = []
    for charge_block in best_charge_blocks[:3]:  # Top 3 charge blocks
        for discharge_block in best_discharge_blocks[:3]:  # Top 3 discharge blocks
            if charge_block['start_hour'] != discharge_block['start_hour']:  # Avoid same time
                price_spread = discharge_block['avg_price'] - charge_block['avg_price']
                potential_profit = price_spread * battery_size_mw * 2 * efficiency  # 2 hours
                
                arbitrage_opportunities.append({
                    'charge_window': f"{int(charge_block['start_hour']):02d}:00-{int(charge_block['end_hour']):02d}:00",
                    'discharge_window': f"{int(discharge_block['start_hour']):02d}:00-{int(discharge_block['end_hour']):02d}:00",
                    'charge_price': charge_block['avg_price'],
                    'discharge_price': discharge_block['avg_price'],
                    'price_spread': price_spread,
                    'potential_daily_profit': potential_profit,
                    'charge_rank': best_charge_blocks.index(charge_block) + 1,
                    'discharge_rank': best_discharge_blocks.index(discharge_block) + 1
                })
    
    # Sort by potential profit
    arbitrage_opportunities.sort(key=lambda x: x['potential_daily_profit'], reverse=True)
    
    return {
        'hourly_prices': hourly_prices,
        'best_charge_blocks': best_charge_blocks[:5],  # Top 5
        'best_discharge_blocks': best_discharge_blocks[:5],  # Top 5
        'arbitrage_opportunities': arbitrage_opportunities[:5]  # Top 5
    }

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
                **QLD Electricity Price Patterns (with Solar Crash Modeling):**
                
                - **Morning Peak (6-9 AM)**: Higher prices due to increased demand before solar generation
                - **Solar Ramp Up (10-11:30 AM)**: Prices begin falling as solar ramps up
                - **Maximum Solar (11:30 AM-1:00 PM)**: Prices often <$50/MWh, can be $0 or negative
                - **Solar Ramp Down (1:00-2:30 PM)**: Prices begin rising as solar ramps down
                - **Evening Peak (5-8 PM)**: Highest prices due to solar drop + peak demand (duck curve)
                - **Off-Peak Night (10 PM-6 AM)**: Lower prices due to reduced demand
                - **Weekend Prices**: Generally lower due to reduced commercial activity
                - **Seasonal Variations**: Summer/winter higher, spring/autumn lower
                - **Solar Intensity**: Stronger in summer, weaker in winter
                - **Realistic Volatility**: ±15% random variation to simulate market uncertainty
                
                **Solar Crash Effects:**
                - **Duck Curve**: Low midday prices, high evening prices
                - **Solar Penetration**: QLD has high rooftop solar adoption (~30% of households)
                - **Seasonal Variation**: Summer strongest solar, winter weakest
                - **Negative Prices**: Can occur during maximum solar penetration (11:30 AM-1:00 PM)
                - **Price Depression**: Up to 80% price reduction during peak solar hours
                
                **Specific Time Patterns:**
                - **10:00 AM**: Rooftop & utility solar ramping up, prices begin to fall
                - **11:30 AM**: Solar near maximum output, prices often <$50/MWh
                - **12:00-1:00 PM**: Maximum solar penetration, prices can be $0 or negative
                - **2:00-2:30 PM**: Solar starts ramping down, prices begin rising again
                
                These patterns reflect modern QLD electricity market dynamics with high solar penetration.
                """)
            
            # Solar analysis section
            if 'solar_intensity' in df.columns:
                st.markdown("### ☀️ Solar Crash Analysis")
                
                # Create solar analysis
                solar_df = df.copy()
                solar_df['hour'] = solar_df.index.hour
                solar_df['date'] = solar_df.index.date
                
                # Average solar intensity by hour
                hourly_solar = solar_df.groupby('hour')['solar_intensity'].mean().reset_index()
                hourly_solar['hour'] = hourly_solar['hour'].astype(str) + ':00'
                hourly_solar.columns = ['Hour', 'Avg Solar Intensity']
                hourly_solar['Avg Solar Intensity'] = hourly_solar['Avg Solar Intensity'].round(3)
                
                # Show solar intensity chart
                fig_solar = px.line(
                    hourly_solar,
                    x='Hour',
                    y='Avg Solar Intensity',
                    title="Average Solar Generation by Hour (QLD)",
                    labels={'Avg Solar Intensity': 'Solar Intensity (0-1)', 'Hour': 'Time of Day'},
                    markers=True
                )
                fig_solar.update_layout(height=300)
                st.plotly_chart(fig_solar, use_container_width=True)
                
                # Solar vs price analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📊 Solar Intensity by Hour")
                    st.dataframe(hourly_solar, use_container_width=True)
                
                with col2:
                    # Price impact during solar hours
                    solar_hours = solar_df[(solar_df['hour'] >= 10) & (solar_df['hour'] <= 16)]
                    non_solar_hours = solar_df[(solar_df['hour'] < 10) | (solar_df['hour'] > 16)]
                    
                    solar_avg = solar_hours['price'].mean()
                    non_solar_avg = non_solar_hours['price'].mean()
                    price_reduction = ((non_solar_avg - solar_avg) / non_solar_avg) * 100
                    
                    st.markdown("#### 💰 Solar Crash Impact")
                    impact_data = {
                        'Metric': ['Avg Price (Solar Hours)', 'Avg Price (Non-Solar)', 'Price Reduction'],
                        'Value': [f"${solar_avg:.2f}/MWh", f"${non_solar_avg:.2f}/MWh", f"{price_reduction:.1f}%"]
                    }
                    impact_df = pd.DataFrame(impact_data)
                    st.dataframe(impact_df, use_container_width=True)
                
                # Duck curve explanation
                with st.expander("ℹ️ Duck Curve & Solar Crash Explanation"):
                    st.markdown("""
                    **Duck Curve Pattern:**
                    
                    The "duck curve" shows how solar generation affects electricity prices with specific time patterns:
                    
                    **Morning (6-9 AM)**: High prices as demand rises but solar not yet strong
                    **10:00 AM**: Rooftop & utility solar ramping up, prices begin to fall
                    **11:30 AM**: Solar near maximum output, prices often <$50/MWh
                    **12:00-1:00 PM**: Maximum solar penetration, prices can be $0 or negative
                    **2:00-2:30 PM**: Solar starts ramping down, prices begin rising again
                    **Evening (5-8 PM)**: Highest prices as solar drops and demand peaks
                    **Night (10 PM-6 AM)**: Lower prices due to reduced demand
                    
                    **Solar Crash Effects:**
                    - **Price Depression**: Solar generation reduces wholesale prices
                    - **Negative Prices**: Can occur during maximum solar penetration
                    - **Geographic Impact**: QLD has high solar penetration (~30% of households)
                    - **Seasonal Variation**: Summer strongest solar, winter weakest
                    - **Network Constraints**: Solar export limitations can affect prices
                    
                    **Battery Arbitrage Opportunities:**
                    - **Charge during solar crash**: Buy low during midday solar peak (including negative prices)
                    - **Discharge during evening peak**: Sell high when solar drops
                    - **Seasonal strategies**: Adjust for solar intensity variations
                    - **Weekend patterns**: Different solar/demand dynamics
                    - **Negative price opportunities**: Charge when prices are negative (paid to consume)
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
            
            # OTC Hedging Recommendations
            st.markdown("### 🎯 OTC Hedging Recommendations")
            
            # Get recommendations
            recommendations = recommend_optimal_windows(df, battery_size_mw, battery_capacity_mwh, round_trip_efficiency)
            
            if recommendations:
                # Show hourly price analysis
                st.markdown("#### 📊 Hourly Price Analysis")
                
                hourly_df = recommendations['hourly_prices'].copy()
                hourly_df['hour'] = hourly_df['hour'].astype(str) + ':00'
                hourly_df.columns = ['Hour', 'Avg Price ($/MWh)', 'Std Dev', 'Min Price', 'Max Price']
                hourly_df['Avg Price ($/MWh)'] = hourly_df['Avg Price ($/MWh)'].round(2)
                hourly_df['Std Dev'] = hourly_df['Std Dev'].round(2)
                hourly_df['Min Price'] = hourly_df['Min Price'].round(2)
                hourly_df['Max Price'] = hourly_df['Max Price'].round(2)
                
                st.dataframe(hourly_df, use_container_width=True)
                
                # Show best charge blocks
                st.markdown("#### 🔋 Best 2-Hour Charge Windows (Lowest Prices)")
                charge_blocks = recommendations['best_charge_blocks']
                if charge_blocks:
                    charge_data = []
                    for i, block in enumerate(charge_blocks, 1):
                        charge_data.append({
                            'Rank': i,
                            'Window': f"{int(block['start_hour']):02d}:00-{int(block['end_hour']):02d}:00",
                            'Avg Price ($/MWh)': f"${block['avg_price']:.2f}",
                            'Hour 1 Price': f"${block['hour1_price']:.2f}",
                            'Hour 2 Price': f"${block['hour2_price']:.2f}"
                        })
                    
                    charge_df = pd.DataFrame(charge_data)
                    st.dataframe(charge_df, use_container_width=True)
                
                # Show best discharge blocks
                st.markdown("#### ⚡ Best 2-Hour Discharge Windows (Highest Prices)")
                discharge_blocks = recommendations['best_discharge_blocks']
                if discharge_blocks:
                    discharge_data = []
                    for i, block in enumerate(discharge_blocks, 1):
                        discharge_data.append({
                            'Rank': i,
                            'Window': f"{int(block['start_hour']):02d}:00-{int(block['end_hour']):02d}:00",
                            'Avg Price ($/MWh)': f"${block['avg_price']:.2f}",
                            'Hour 1 Price': f"${block['hour1_price']:.2f}",
                            'Hour 2 Price': f"${block['hour2_price']:.2f}"
                        })
                    
                    discharge_df = pd.DataFrame(discharge_data)
                    st.dataframe(discharge_df, use_container_width=True)
                
                # Show arbitrage opportunities
                st.markdown("#### 💰 Top Arbitrage Opportunities (2-Hour Blocks)")
                opportunities = recommendations['arbitrage_opportunities']
                if opportunities:
                    opp_data = []
                    for i, opp in enumerate(opportunities, 1):
                        opp_data.append({
                            'Rank': i,
                            'Charge Window': opp['charge_window'],
                            'Discharge Window': opp['discharge_window'],
                            'Charge Price ($/MWh)': f"${opp['charge_price']:.2f}",
                            'Discharge Price ($/MWh)': f"${opp['discharge_price']:.2f}",
                            'Price Spread ($/MWh)': f"${opp['price_spread']:.2f}",
                            'Potential Daily Profit ($)': f"${opp['potential_daily_profit']:,.2f}"
                        })
                    
                    opp_df = pd.DataFrame(opp_data)
                    st.dataframe(opp_df, use_container_width=True)
                    
                    # Highlight the best opportunity
                    if opportunities:
                        best_opp = opportunities[0]
                        st.success(f"**🎯 Recommended OTC Strategy:** Charge {best_opp['charge_window']}, Discharge {best_opp['discharge_window']}")
                        st.info(f"**Expected Daily Profit:** ${best_opp['potential_daily_profit']:,.2f} | **Price Spread:** ${best_opp['price_spread']:.2f}/MWh")
                
                # Show explanation
                with st.expander("ℹ️ OTC Hedging Strategy Explanation"):
                    st.markdown("""
                    **OTC (Over-The-Counter) Hedging Strategy:**
                    
                    This analysis identifies the most profitable 2-hour blocks for battery arbitrage based on historical price patterns including solar crash effects:
                    
                    **Charge Strategy (Solar Crash Era):**
                    - Target the lowest average price 2-hour periods
                    - Focus on midday solar crash periods (10 AM-4 PM)
                    - Consider overnight/off-peak periods
                    - Account for seasonal solar intensity variations
                    
                    **Discharge Strategy (Duck Curve Era):**
                    - Target the highest average price 2-hour periods
                    - Focus on evening peak periods (5-8 PM) when solar drops
                    - Consider morning ramp periods (6-9 AM) before solar peaks
                    - Maximize energy sales revenue during high-demand periods
                    
                    **Solar Crash Arbitrage Opportunities:**
                    - **Charge during solar peak**: Buy low during midday solar generation
                    - **Discharge during evening peak**: Sell high when solar drops and demand peaks
                    - **Seasonal adjustments**: Stronger solar effects in summer
                    - **Weekend patterns**: Different solar/demand dynamics
                    
                    **Arbitrage Opportunities:**
                    - Ranked by potential daily profit
                    - Calculated using: (Discharge Price - Charge Price) × Power × Hours × Efficiency
                    - Considers battery power constraints and efficiency losses
                    - Accounts for solar crash price depression effects
                    
                    **OTC Contract Structure:**
                    - Fixed 2-hour blocks for simplicity
                    - Contiguous time windows for operational ease
                    - Based on average historical prices including solar effects
                    - Suitable for forward contract negotiations
                    - Adapts to duck curve dynamics
                    """)
    
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
        ## 📊 Price Data Generation (with Solar Crash Modeling)
        
        **Base Price**: $80/MWh (QLD wholesale average)
        
        **Seasonal Adjustments**:
        - Summer (Dec-Feb): Base Price × 1.2
        - Winter (Jun-Aug): Base Price × 1.15  
        - Autumn (Mar-May): Base Price × 0.9
        - Spring (Sep-Nov): Base Price × 0.85
        
        **Solar Generation Patterns**:
        - Early Morning Ramp (6-10 AM): Solar intensity = (hour - 6) / 4
        - Ramp Up (10-11:30 AM): Solar intensity = 0.8 + (hour - 10) × 0.13
        - Peak Solar (11:30 AM-1:00 PM): Solar intensity = 1.0
        - Ramp Down (1:00-2:30 PM): Solar intensity = 1.0 - (hour - 13) × 0.13
        - Afternoon Decline (2:30-6 PM): Solar intensity = max(0, 0.8 - (hour - 14.5) × 0.2)
        
        **Seasonal Solar Intensity**:
        - Summer (Dec-Feb): Solar intensity × 1.3 (strongest)
        - Winter (Jun-Aug): Solar intensity × 0.6 (weakest)
        - Autumn (Mar-May): Solar intensity × 0.8
        - Spring (Sep-Nov): Solar intensity × 1.1
        
        **Solar Crash Price Effects**:
        - 10-11:30 AM: Solar crash = Solar intensity × 0.3 (up to 30% price reduction)
        - 11:30 AM-1:00 PM: Solar crash = Solar intensity × 0.8 (up to 80% price reduction, can go negative)
        - 1:00-2:30 PM: Solar crash = Solar intensity × 0.5 (up to 50% price reduction)
        - Morning Ramp (7-9 AM): Solar factor = 1.1 (higher prices before solar)
        - Evening Ramp (17-19): Solar factor = 1.8 (highest prices after solar drops)
        
        **Time-of-Day Factors (Modified for Solar)**:
        - Morning Peak (6-9 AM): Base Price × 1.6 (before solar)
        - Solar Peak (10-16): Base Price × 0.8 (reduced due to solar)
        - Evening Peak (17-20): Base Price × 2.2 (highest due to solar drop + demand)
        - Off-Peak Night (22-6): Base Price × 0.6
        - Other Daytime: Base Price × 1.0
        
        **Solar Crash Time Patterns**:
        - 10:00 AM: Rooftop & utility solar ramping up, prices begin to fall
        - 11:30 AM: Solar near maximum output, prices often <$50/MWh
        - 12:00-1:00 PM: Maximum solar penetration, prices can be $0 or negative
        - 2:00-2:30 PM: Solar starts ramping down, prices begin rising again
        
        **Final Price Formula (with Solar)**:
        ```
        Final Price = Base Price × Seasonal Factor × Time Factor × Solar Factor × (1 + Volatility)
        Where: Volatility = ±15% random variation
        Price Bounds: 
        - Normal times: $20 - $300/MWh
        - Peak solar (11:30 AM-1:00 PM): -$50 to $300/MWh
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
        Note: Can be negative during peak solar (paid to consume)
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
        Note: Charge cost can be negative during solar crash periods
        ```
        
        **Cumulative Profit**:
        ```
        Cumulative Profit = Previous Cumulative Profit + Daily Profit
        ```
        
        **Profit Margin**:
        ```
        Profit Margin (%) = (Daily Profit ÷ Total Charge Cost) × 100
        ```
        
        ## 📊 Example Calculation (with Solar Crash)
        
        **Scenario**: 100 MW / 400 MWh Battery, 85% Efficiency
        
        **Charge Period (11:30 AM-1:00 PM, -$10/MWh during solar crash)**:
        ```
        Charge Amount = min(100 × 0.5, 400 - 0) = 50 MWh
        Charge Cost = 50 MWh × -$10/MWh = -$500 (paid to consume)
        Energy Stored = 0 + 50 = 50 MWh
        ```
        
        **Discharge Period (5 PM-7 PM, $180/MWh after solar drops)**:
        ```
        Discharge Amount = min(100 × 0.5, 50 × 0.85) = min(50, 42.5) = 42.5 MWh
        Discharge Revenue = 42.5 MWh × $180/MWh = $7,650
        Energy Removed = 42.5 ÷ 0.85 = 50 MWh
        Energy Stored = 50 - 50 = 0 MWh
        ```
        
        **Daily Profit**:
        ```
        Daily Profit = $7,650 - (-$500) = $8,150
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
        
        **Solar Constraints**:
        - Solar generation only during daylight hours (6 AM-6 PM)
        - Seasonal solar intensity variations
        - Geographic solar penetration factors
        
        ## 📋 Data Processing
        
        **30-Minute Intervals**:
        - Intervals per Day = 24 hours × 2 = 48 intervals
        - Intervals per Hour = 2 intervals
        
        **Price Data Structure**:
        - Timestamp: YYYY-MM-DD HH:MM
        - Price: $/MWh (can be negative during solar peak)
        - Solar Intensity: 0-1 scale for solar generation
        
        ## 🎯 Key Insights
        
        **Profit Drivers**:
        1. Price Spread: Higher difference between charge and discharge prices
        2. Solar Crash Opportunities: Negative prices during peak solar
        3. Battery Size: Larger capacity = more energy arbitrage
        4. Efficiency: Higher efficiency = less energy loss
        5. Trading Windows: Optimal timing maximizes profit
        6. Duck Curve Dynamics: Evening peak after solar drop
        
        **Risk Factors**:
        1. Price Volatility: Unpredictable price movements
        2. Solar Generation Variability: Weather-dependent solar output
        3. Battery Degradation: Reduced capacity over time
        4. Market Changes: Regulatory or structural changes
        5. Technical Issues: Battery or grid failures
        6. Solar Penetration Changes: Increasing solar adoption
        
        **Solar Arbitrage Opportunities**:
        - Charge during solar peak: Buy low during midday solar generation
        - Discharge during evening peak: Sell high when solar drops and demand peaks
        - Negative price opportunities: Charge when prices are negative (paid to consume)
        - Seasonal adjustments: Stronger solar effects in summer
        """)

if __name__ == "__main__":
    main() 