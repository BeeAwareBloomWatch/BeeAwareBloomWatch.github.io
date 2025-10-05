import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("2026 SUPERBLOOM PREDICTION")
print("Based on Historical Data (2020 - September 2025)")
print("="*60)

# Load the trained model
print("\nLoading trained model...")
model_artifacts = joblib.load('superbloom_model.pkl')
rf_model = model_artifacts['model']
feature_columns = model_artifacts['feature_columns']
climate_params = model_artifacts['climate_params']
location_encoding = model_artifacts['location_encoding']

print(f"Model loaded successfully!")
print(f"Training R²: {model_artifacts['train_metrics']['r2']:.3f}")
print(f"Training RMSE: {model_artifacts['train_metrics']['rmse']:.3f}")

# Load climate data and bloom scores
print("\nLoading historical data...")
bloom_data = pd.read_csv('bloom_score_dataset_2020-2025.csv')
climate_cp = pd.read_csv('POWER_CP_Monthly.csv')
climate_av = pd.read_csv('POWER_AV_Monthly.csv')

# Function to pivot climate data from wide to long format
def pivot_climate_data(df, location_name):
    """Convert wide-format monthly climate data to long format"""
    records = []
    
    for _, row in df.iterrows():
        parameter = row['PARAMETER']
        year = row['YEAR']
        
        month_cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                      'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        
        for month_idx, month_col in enumerate(month_cols, start=1):
            records.append({
                'Location': location_name,
                'Year': year,
                'Month': month_idx,
                'Parameter': parameter,
                'Value': row[month_col]
            })
    
    return pd.DataFrame(records)

# Process climate data
climate_cp_long = pivot_climate_data(climate_cp, 'Carrizo Plain')
climate_av_long = pivot_climate_data(climate_av, 'Antelope Valley')
climate_combined = pd.concat([climate_cp_long, climate_av_long], ignore_index=True)

# Pivot to get parameters as columns
climate_wide = climate_combined.pivot_table(
    index=['Location', 'Year', 'Month'],
    columns='Parameter',
    values='Value'
).reset_index()
climate_wide.columns.name = None

# Sort for proper feature engineering
climate_wide = climate_wide.sort_values(['Location', 'Year', 'Month']).reset_index(drop=True)

print(f"Climate data shape: {climate_wide.shape}")
print(f"Years available: {sorted(climate_wide['Year'].unique())}")

# Feature Engineering Function
def engineer_features(df, climate_params):
    """Apply the same feature engineering as training"""
    
    # Create lagged features by location
    for location in df['Location'].unique():
        mask = df['Location'] == location
        
        for param in climate_params:
            if param in df.columns:
                for lag in [1, 2, 3, 6]:
                    df.loc[mask, f'{param}_lag{lag}'] = df.loc[mask, param].shift(lag)
        
        # Create rolling averages
        for param in climate_params:
            if param in df.columns:
                df.loc[mask, f'{param}_roll3'] = df.loc[mask, param].rolling(window=3, min_periods=1).mean()
                df.loc[mask, f'{param}_roll6'] = df.loc[mask, param].rolling(window=6, min_periods=1).mean()
    
    # Create interaction features
    if 'PRECTOTCORR' in df.columns and 'T2M' in df.columns:
        df['precip_temp_interaction'] = df['PRECTOTCORR'] * df['T2M']
    
    # Add cyclical month encoding
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    # Add year normalization (using same range as training: 2020-2025)
    df['Year_normalized'] = (df['Year'] - 2020) / (2025 - 2020)
    
    # Location encoding
    df['Location_encoded'] = df['Location'].map(location_encoding)
    
    return df

# Apply feature engineering
print("\nEngineering features...")
df_engineered = engineer_features(climate_wide.copy(), climate_params)

# Drop rows with NaN (from lagging)
df_complete = df_engineered.dropna().copy()

print(f"Complete data after feature engineering: {len(df_complete)} records")

# Prepare features for prediction
X_all = df_complete[feature_columns]

# Make predictions on all available data
predictions = rf_model.predict(X_all)
df_complete['Predicted_Bloom_Score'] = predictions

# Categorize bloom potential
def categorize_bloom(score):
    if score >= 0.7:
        return "🌸 SUPERBLOOM LIKELY"
    elif score >= 0.5:
        return "🌼 GOOD BLOOM EXPECTED"
    elif score >= 0.3:
        return "🌱 MODERATE BLOOM POSSIBLE"
    else:
        return "🍂 MINIMAL BLOOM EXPECTED"

df_complete['Bloom_Category'] = df_complete['Predicted_Bloom_Score'].apply(categorize_bloom)

# Month names for display
month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
               7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

# SECTION 1: Current State Analysis (Most Recent Data)
print("\n" + "="*60)
print("CURRENT STATE ANALYSIS")
print("="*60)

# Find the most recent month with data for each location
for location in ['Carrizo Plain', 'Antelope Valley']:
    loc_data = df_complete[df_complete['Location'] == location].sort_values(['Year', 'Month'])
    
    if len(loc_data) > 0:
        # Get most recent record
        latest = loc_data.iloc[-1]
        latest_month = month_names[latest['Month']]
        latest_year = latest['Year']
        
        print(f"\n📍 {location} - {latest_month} {latest_year}")
        print(f"   {'─'*50}")
        print(f"   Current Bloom Score: {latest['Predicted_Bloom_Score']:.3f}")
        print(f"   Status: {latest['Bloom_Category']}")
        
        # Current climate conditions
        print(f"\n   Climate Conditions:")
        if 'GWETROOT' in latest:
            soil_status = "Good" if latest['GWETROOT'] > 0.3 else "Moderate" if latest['GWETROOT'] > 0.2 else "Low"
            print(f"   • Soil Moisture (GWETROOT): {latest['GWETROOT']:.3f} [{soil_status}]")
        if 'T2M' in latest:
            print(f"   • Temperature (T2M): {latest['T2M']:.1f}°C ({latest['T2M']*9/5 + 32:.1f}°F)")
        if 'PRECTOTCORR' in latest:
            print(f"   • Precipitation: {latest['PRECTOTCORR']:.2f} mm/day")
        if 'ALLSKY_SFC_SW_DWN' in latest:
            print(f"   • Solar Radiation: {latest['ALLSKY_SFC_SW_DWN']:.2f} kW-hr/m²/day")
        if 'WS2M' in latest:
            print(f"   • Wind Speed: {latest['WS2M']:.2f} m/s")
        
        # Compare to historical average for this month
        historical_same_month = df_complete[
            (df_complete['Location'] == location) &
            (df_complete['Month'] == latest['Month']) &
            (df_complete['Year'] < latest_year)
        ]
        
        if len(historical_same_month) > 0:
            hist_avg_score = historical_same_month['Predicted_Bloom_Score'].mean()
            deviation = latest['Predicted_Bloom_Score'] - hist_avg_score
            
            print(f"\n   Historical Comparison ({latest_month}):")
            print(f"   • Historical Average: {hist_avg_score:.3f}")
            print(f"   • Current vs. Average: {deviation:+.3f}", end="")
            
            if abs(deviation) > 0.1:
                print(" [SIGNIFICANT DEVIATION]")
            elif abs(deviation) > 0.05:
                print(" [Moderate deviation]")
            else:
                print(" [Normal]")
        
        # Recent 3-month trend
        recent_3mo = loc_data.tail(3)
        if len(recent_3mo) >= 2:
            trend_direction = "↑ Improving" if recent_3mo['Predicted_Bloom_Score'].is_monotonic_increasing else \
                            "↓ Declining" if recent_3mo['Predicted_Bloom_Score'].is_monotonic_decreasing else \
                            "→ Stable"
            print(f"\n   Recent 3-Month Trend: {trend_direction}")
            print(f"   • Range: {recent_3mo['Predicted_Bloom_Score'].min():.3f} to {recent_3mo['Predicted_Bloom_Score'].max():.3f}")

# SECTION 2: Historical Spring Pattern Analysis
print("\n" + "="*60)
print("HISTORICAL SPRING PATTERN ANALYSIS (2020-2025)")
print("="*60)

spring_months = [3, 4, 5]
historical_springs = df_complete[df_complete['Month'].isin(spring_months)].copy()

print("\nYear-by-Year Spring Bloom Performance:")
print("-" * 60)

for location in ['Carrizo Plain', 'Antelope Valley']:
    loc_springs = historical_springs[historical_springs['Location'] == location]
    
    print(f"\n📍 {location}:")
    
    # Group by year
    yearly_stats = loc_springs.groupby('Year')['Predicted_Bloom_Score'].agg(['mean', 'max', 'min', 'std'])
    
    for year in sorted(yearly_stats.index):
        avg_score = yearly_stats.loc[year, 'mean']
        max_score = yearly_stats.loc[year, 'max']
        min_score = yearly_stats.loc[year, 'min']
        std_score = yearly_stats.loc[year, 'std']
        category = categorize_bloom(avg_score)
        
        # Determine if superbloom year
        is_superbloom = "SUPERBLOOM" if avg_score >= 0.7 else ""
        
        print(f"   Spring {year}:")
        print(f"      Average: {avg_score:.3f} | Peak: {max_score:.3f} | Range: {min_score:.3f}-{max_score:.3f}")
        print(f"      {category} {is_superbloom}")
    
    # Overall statistics
    print(f"\n   Overall Statistics (All Springs):")
    print(f"      Mean Score: {loc_springs['Predicted_Bloom_Score'].mean():.3f}")
    print(f"      Std Dev: {loc_springs['Predicted_Bloom_Score'].std():.3f}")
    print(f"      Best Year: {yearly_stats['mean'].idxmax()} ({yearly_stats['mean'].max():.3f})")
    print(f"      Weakest Year: {yearly_stats['mean'].idxmin()} ({yearly_stats['mean'].min():.3f})")

# Monthly breakdown within spring season
print("\n" + "-" * 60)
print("Spring Season Monthly Patterns:")
print("-" * 60)

for location in ['Carrizo Plain', 'Antelope Valley']:
    loc_springs = historical_springs[historical_springs['Location'] == location]
    
    print(f"\n📍 {location}:")
    
    for month in spring_months:
        month_data = loc_springs[loc_springs['Month'] == month]
        if len(month_data) > 0:
            month_avg = month_data['Predicted_Bloom_Score'].mean()
            month_name_str = month_names[month]
            
            # Peak bloom month indicator
            peak_indicator = "🌸 PEAK" if month == loc_springs.groupby('Month')['Predicted_Bloom_Score'].mean().idxmax() else ""
            
            print(f"   {month_name_str}: Average {month_avg:.3f} {peak_indicator}")

# SECTION 3: Spring 2026 Superbloom Forecast
print("\n" + "="*60)
print("SPRING 2026 SUPERBLOOM FORECAST")
print("="*60)

print("\nBased on Historical Patterns & Current Conditions")
print("-" * 60)

spring_months = [3, 4, 5]
historical_springs = df_complete[df_complete['Month'].isin(spring_months)].copy()

for location in ['Carrizo Plain', 'Antelope Valley']:
    loc_springs = historical_springs[historical_springs['Location'] == location]
    
    # Calculate recent trend (last 3 years)
    recent_years = [2023, 2024, 2025]
    recent_springs = loc_springs[loc_springs['Year'].isin(recent_years)]
    
    if len(recent_springs) > 0:
        recent_avg = recent_springs['Predicted_Bloom_Score'].mean()
        overall_avg = loc_springs['Predicted_Bloom_Score'].mean()
        historical_max = loc_springs['Predicted_Bloom_Score'].max()
        
        # Get most recent fall data (critical for spring prediction)
        most_recent_year = df_complete['Year'].max()
        fall_current = df_complete[
            (df_complete['Location'] == location) & 
            (df_complete['Year'] == most_recent_year) & 
            (df_complete['Month'] >= 9)
        ]
        
        # Calculate soil moisture anomaly if available
        moisture_anomaly = 0
        moisture_status = "Data pending"
        
        if len(fall_current) > 0 and 'GWETROOT' in fall_current.columns:
            current_soil_moisture = fall_current['GWETROOT'].mean()
            
            # Historical fall moisture
            historical_fall_moisture = df_complete[
                (df_complete['Location'] == location) & 
                (df_complete['Month'] >= 9) & 
                (df_complete['Month'] <= 12) &
                (df_complete['Year'] < most_recent_year)
            ]['GWETROOT'].mean()
            
            moisture_anomaly = current_soil_moisture - historical_fall_moisture
            
            if current_soil_moisture > 0.30:
                moisture_status = "EXCELLENT (> 0.30)"
            elif current_soil_moisture > 0.25:
                moisture_status = "GOOD (0.25-0.30)"
            elif current_soil_moisture > 0.20:
                moisture_status = "MODERATE (0.20-0.25)"
            else:
                moisture_status = "LOW (< 0.20)"
        
        # Calculate forecast score
        # Base: recent 3-year trend
        # Adjustment: soil moisture anomaly (weighted by its 40.5% importance)
        forecast_score = recent_avg + (moisture_anomaly * 0.405)
        forecast_score = np.clip(forecast_score, 0, 1)  # Keep in valid range
        
        # Calculate confidence level
        trend_stability = recent_springs['Predicted_Bloom_Score'].std()
        if trend_stability < 0.05:
            confidence_level = "HIGH"
            confidence_explanation = "Recent years show consistent pattern"
        elif trend_stability < 0.10:
            confidence_level = "MODERATE"
            confidence_explanation = "Some variability in recent years"
        else:
            confidence_level = "LOW"
            confidence_explanation = "High variability in recent years"
        
        # Determine probability of superbloom (score >= 0.70)
        if forecast_score >= 0.80:
            probability = "VERY HIGH (85-95%)"
            recommendation = "🌸🌸 Exceptional superbloom expected - Send Bees there immediately"
        elif forecast_score >= 0.70:
            probability = "HIGH (70-85%)"
            recommendation = "🌸 Superbloom likely - Send Bees!"
        elif forecast_score >= 0.60:
            probability = "MODERATE (50-70%)"
            recommendation = "🌼 Good bloom expected - Worth sending the bees"
        elif forecast_score >= 0.50:
            probability = "MODERATE-LOW (30-50%)"
            recommendation = "🌼 Decent bloom possible - monitor conditions"
        elif forecast_score >= 0.40:
            probability = "LOW (15-30%)"
            recommendation = "🌱 Below average bloom - consider waiting for updates"
        else:
            probability = "VERY LOW (<15%)"
            recommendation = "🍂 Minimal bloom expected - not recommended"
        
        print(f"\n{'='*60}")
        print(f"📍 {location.upper()}")
        print(f"{'='*60}")
        
        print(f"\nFORECAST METRICS:")
        print(f"   2026 Spring Forecast Score: {forecast_score:.3f}")
        print(f"   Prediction Category: {categorize_bloom(forecast_score)}")
        print(f"   Superbloom Probability: {probability}")
        
        print(f"\nSUPPORTING DATA:")
        print(f"   Historical Average (2020-2025): {overall_avg:.3f}")
        print(f"   Recent 3-Year Trend (2023-2025): {recent_avg:.3f}")
        print(f"   Historical Best Performance: {historical_max:.3f}")
        print(f"   Current Soil Moisture Status: {moisture_status}")
        
        if moisture_anomaly != 0:
            anomaly_direction = "ABOVE" if moisture_anomaly > 0 else "BELOW"
            print(f"   Fall Soil Moisture Anomaly: {anomaly_direction} average ({moisture_anomaly:+.3f})")
        
        print(f"\nFORECAST CONFIDENCE: {confidence_level}")
        print(f"   Basis: {confidence_explanation}")
        
        print(f"\nRECOMMENDATION:")
        print(f"   {recommendation}")
        
        # Comparison to known superbloom year (2023)
        superbloom_2023 = loc_springs[loc_springs['Year'] == 2023]['Predicted_Bloom_Score'].mean()
        comparison_to_2023 = (forecast_score / superbloom_2023 * 100) if superbloom_2023 > 0 else 0
        
        print(f"\nCOMPARISON TO 2023 SUPERBLOOM:")
        print(f"   2023 Spring Score: {superbloom_2023:.3f}")
        print(f"   2026 vs 2023: {comparison_to_2023:.1f}% of superbloom intensity")
        
        if comparison_to_2023 >= 95:
            print(f"   Assessment: Similar to or better than 2023! 🌸🌸")
        elif comparison_to_2023 >= 85:
            print(f"   Assessment: Approaching 2023 levels 🌸")
        elif comparison_to_2023 >= 70:
            print(f"   Assessment: Good bloom but below 2023 🌼")
        else:
            print(f"   Assessment: Significantly below 2023 superbloom 🌱")
        
        # Key factors that could change the forecast
        print(f"\nFORECAST COULD IMPROVE IF:")
        print(f"   • October-December 2025 precipitation exceeds 2.0 mm/day average")
        print(f"   • Soil moisture rises above 0.35 by January 2026")
        print(f"   • Multiple well-distributed rain events occur (3+ storms)")
        
        print(f"\nFORECAST COULD DECLINE IF:")
        print(f"   • Warm, dry winter (temps >18°C, precip <1.0 mm/day)")
        print(f"   • Soil moisture drops below 0.20 in December-January")
        print(f"   • Single large storm followed by prolonged drought")

# Summary comparison
print(f"\n{'='*60}")
print(f"COMPARATIVE FORECAST SUMMARY")
print(f"{'='*60}")

print(f"\nWhich location is predicted to have the better bloom?")

cp_forecast = None
av_forecast = None

for location in ['Carrizo Plain', 'Antelope Valley']:
    loc_springs = historical_springs[historical_springs['Location'] == location]
    recent_springs = loc_springs[loc_springs['Year'].isin([2023, 2024, 2025])]
    
    if len(recent_springs) > 0:
        recent_avg = recent_springs['Predicted_Bloom_Score'].mean()
        
        most_recent_year = df_complete['Year'].max()
        fall_current = df_complete[
            (df_complete['Location'] == location) & 
            (df_complete['Year'] == most_recent_year) & 
            (df_complete['Month'] >= 9)
        ]
        
        moisture_anomaly = 0
        if len(fall_current) > 0 and 'GWETROOT' in fall_current.columns:
            current_soil_moisture = fall_current['GWETROOT'].mean()
            historical_fall_moisture = df_complete[
                (df_complete['Location'] == location) & 
                (df_complete['Month'] >= 9) & 
                (df_complete['Month'] <= 12) &
                (df_complete['Year'] < most_recent_year)
            ]['GWETROOT'].mean()
            moisture_anomaly = current_soil_moisture - historical_fall_moisture
        
        forecast_score = recent_avg + (moisture_anomaly * 0.405)
        forecast_score = np.clip(forecast_score, 0, 1)
        
        if location == 'Carrizo Plain':
            cp_forecast = forecast_score
        else:
            av_forecast = forecast_score

if cp_forecast is not None and av_forecast is not None:
    if abs(cp_forecast - av_forecast) < 0.05:
        print(f"\n   SIMILAR: Both locations show comparable bloom potential")
        print(f"      Carrizo Plain: {cp_forecast:.3f}")
        print(f"      Antelope Valley: {av_forecast:.3f}")
        print(f"      Difference: {abs(cp_forecast - av_forecast):.3f} (negligible)")
    elif cp_forecast > av_forecast:
        print(f"\n   CARRIZO PLAIN is predicted to have the superior bloom")
        print(f"      Carrizo Plain: {cp_forecast:.3f}")
        print(f"      Antelope Valley: {av_forecast:.3f}")
        print(f"      Advantage: {(cp_forecast - av_forecast):.3f} points")
    else:
        print(f"\n   ANTELOPE VALLEY is predicted to have the superior bloom")
        print(f"      Antelope Valley: {av_forecast:.3f}")
        print(f"      Carrizo Plain: {cp_forecast:.3f}")
        print(f"      Advantage: {(av_forecast - cp_forecast):.3f} points")

print(f"\nBoth preserves typically bloom around the same time (March-April)")
print(f"   Consider visiting both if scores are within 0.10 of each other!")


# SECTION 5: Monitoring Guidance
print("\n" + "="*60)
print("COMPREHENSIVE MONITORING GUIDANCE FOR 2026")
print("="*60)

print("\nPRIORITY MONITORING INDICATORS")
print("="*60)

# Get feature importance from model
feature_importance_df = model_artifacts['feature_importance']
top_features = feature_importance_df.head(10)

print("\nTop 10 Most Predictive Factors (from model):")
for idx, row in top_features.iterrows():
    feature_name = row['Feature']
    importance = row['Importance']
    
    # Simplify feature names for readability
    if 'GWETROOT' in feature_name:
        readable = f"Soil Moisture ({feature_name})"
    elif 'T2M' in feature_name:
        readable = f"Temperature ({feature_name})"
    elif 'PRECTOTCORR' in feature_name:
        readable = f"Precipitation ({feature_name})"
    elif 'ALLSKY' in feature_name:
        readable = f"Solar Radiation ({feature_name})"
    elif 'WS2M' in feature_name:
        readable = f"Wind Speed ({feature_name})"
    else:
        readable = feature_name
    
    importance_pct = importance * 100
    bar_length = int(importance_pct / 2)
    bar = "█" * bar_length
    
    print(f"   {importance_pct:5.1f}% {bar} {readable}")

print("\n" + "="*60)
print("\nCRITICAL MONITORING PERIODS FOR SPRING 2026 SUPERBLOOM")
print("="*60)

monitoring_periods = [
    {
        'period': 'October - November 2025',
        'priority': 'CRITICAL',
        'parameters': ['Precipitation', 'Soil Moisture'],
        'targets': {
            'PRECTOTCORR': '> 1.5 mm/day',
            'GWETROOT': '> 0.25'
        },
        'rationale': 'Early winter precipitation primes soil for spring bloom. Soil moisture lag features are top predictors.'
    },
    {
        'period': 'December 2025 - January 2026',
        'priority': 'CRITICAL',
        'parameters': ['Precipitation', 'Soil Moisture', 'Temperature'],
        'targets': {
            'PRECTOTCORR': '> 2.0 mm/day',
            'GWETROOT': '> 0.30',
            'T2M': '8-15°C'
        },
        'rationale': 'Peak winter moisture period. Multiple storm events needed. Soil moisture 1-month lag is #1 predictor (40.5% importance).'
    },
    {
        'period': 'February 2026',
        'priority': 'HIGH',
        'parameters': ['Soil Moisture', 'Temperature', 'Solar Radiation'],
        'targets': {
            'GWETROOT': '> 0.28',
            'T2M': '10-16°C',
            'ALLSKY_SFC_SW_DWN': '> 4.0 kW-hr/m²/day'
        },
        'rationale': 'Final moisture boost before spring. Temperature begins warming for germination.'
    },
    {
        'period': 'March - May 2026',
        'priority': 'MODERATE',
        'parameters': ['Temperature', 'Solar Radiation'],
        'targets': {
            'T2M': '12-20°C',
            'ALLSKY_SFC_SW_DWN': '> 5.0 kW-hr/m²/day'
        },
        'rationale': 'Bloom season. Monitor for excessive heat (>25°C) which can end bloom early.'
    }
]

for period_info in monitoring_periods:
    print(f"\n{period_info['priority']} {period_info['period']}")
    print(f"   Parameters to Monitor: {', '.join(period_info['parameters'])}")
    print(f"   Targets:")
    for param, target in period_info['targets'].items():
        param_name = {
            'PRECTOTCORR': 'Precipitation',
            'GWETROOT': 'Soil Moisture',
            'T2M': 'Temperature',
            'ALLSKY_SFC_SW_DWN': 'Solar Radiation'
        }.get(param, param)
        print(f"      • {param_name}: {target}")
    print(f"   Why: {period_info['rationale']}")

print("\n" + "="*60)
print("\n⚠️ WARNING SIGNALS - Indicators of Poor Bloom Potential")
print("="*60)

warning_signals = [
    {
        'signal': 'Soil Moisture < 0.20 in Dec-Jan',
        'impact': 'Severely reduces superbloom probability',
        'action': 'Hope for late winter storms in Feb'
    },
    {
        'signal': 'No significant rain events (>15mm) Oct-Dec',
        'impact': 'Insufficient moisture accumulation',
        'action': 'Monitor December-January precipitation closely'
    },
    {
        'signal': 'Temperature > 18°C average in Dec-Jan',
        'impact': 'Warm winter reduces vernalization',
        'action': 'Lower bloom expectations, focus on microclimates'
    },
    {
        'signal': 'Single large storm followed by drought',
        'impact': 'Seeds germinate then die from lack of moisture',
        'action': 'Better to have distributed rainfall'
    },
    {
        'signal': 'Soil moisture decline in Feb-Mar',
        'impact': 'Early season drying shortens bloom window',
        'action': 'Visit preserves early (late March vs. April)'
    }
]

for warning in warning_signals:
    print(f"\n⚠️ {warning['signal']}")
    print(f"   Impact: {warning['impact']}")
    print(f"   Action: {warning['action']}")

print("\n" + "="*60)
print("\n POSITIVE INDICATORS - Superbloom Signals")
print("="*60)

positive_signals = [
    {
        'signal': 'Soil Moisture > 0.35 by January',
        'impact': 'Strong superbloom potential',
        'probability': '+25% bloom score'
    },
    {
        'signal': '3+ significant rain events (>20mm) Oct-Jan',
        'impact': 'Well-distributed moisture ideal for germination',
        'probability': '+20% bloom score'
    },
    {
        'signal': 'Cool winter temps (10-13°C avg)',
        'impact': 'Optimal for seed dormancy break',
        'probability': '+15% bloom score'
    },
    {
        'signal': 'Steady soil moisture (not spiking/crashing)',
        'impact': 'Consistent moisture better than boom-bust',
        'probability': '+10% bloom score'
    }
]

for signal in positive_signals:
    print(f"\n✓ {signal['signal']}")
    print(f"   Impact: {signal['impact']}")
    print(f"   Forecast Boost: {signal['probability']}")

print("\n" + "="*60)
print("\n DATA COLLECTION RECOMMENDATIONS")
print("="*60)

print("\n1. AUTOMATED MONITORING:")
print("   • Set up NASA POWER API calls to download monthly data")
print("   • Update prediction model monthly (Oct 2025 - Feb 2026)")
print("   • Create alerts when soil moisture crosses thresholds")

print("\n2. MANUAL CHECKS:")
print("   • Weekly: Review NOAA precipitation forecasts")
print("   • Bi-weekly: Check soil moisture trends")
print("   • Monthly: Re-run prediction script with updated data")

print("\n3. FIELD OBSERVATIONS (if possible):")
print("   • Early March: Check for germination")
print("   • Mid-March: Assess early bloom density")
print("   • Late March - April: Peak bloom monitoring")

print("\n4. COMPLEMENTARY DATA SOURCES:")
print("   • NOAA Climate Prediction Center seasonal forecasts")
print("   • USDA SNOTEL snow water equivalent (higher elevations)")
print("   • Local weather station precipitation data")
print("   • Satellite vegetation indices (NDVI) starting Feb 2026")

print("\n" + "="*60)
print("\n DECISION THRESHOLDS")
print("="*60)

print("\nBased on model predictions, use these thresholds:")
print("\n   Forecast Score ≥ 0.80  → 🌸🌸 EXCEPTIONAL SUPERBLOOM")
print("                            Send Bees there immediately")
print("\n   Forecast Score ≥ 0.70  → 🌸 SUPERBLOOM LIKELY")
print("                            Send Bees!")
print("\n   Forecast Score ≥ 0.50  → 🌼 GOOD BLOOM EXPECTED")
print("                            Worth sending the bees")
print("\n   Forecast Score ≥ 0.30  → 🌱 MODERATE BLOOM POSSIBLE")
print("                            Patchy displays")
print("\n   Forecast Score < 0.30  → 🍂 MINIMAL BLOOM")
print("                            Focus on microclimates only")

print("\n Update forecast monthly as new data arrives!")
print("   Re-run this script in: October, November, December, January, February")

# SECTION 4: Seasonal Anomaly Detection
print("\n" + "="*60)
print("SEASONAL ANOMALY DETECTION")
print("="*60)

# Define seasons
seasons = {
    'Winter': [12, 1, 2],
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Fall': [9, 10, 11]
}

# Get the most recent complete year
most_recent_year = df_complete['Year'].max()
baseline_years = [y for y in df_complete['Year'].unique() if y < most_recent_year]

print(f"\nComparing {most_recent_year} to Historical Baseline ({min(baseline_years)}-{max(baseline_years)}):")
print("-" * 60)

for location in ['Carrizo Plain', 'Antelope Valley']:
    print(f"\n📍 {location}:")
    
    anomalies_detected = []
    
    for season_name, months in seasons.items():
        # Historical baseline (all years except most recent)
        hist_season = df_complete[
            (df_complete['Location'] == location) &
            (df_complete['Year'].isin(baseline_years)) &
            (df_complete['Month'].isin(months))
        ]
        
        # Current year season
        current_season = df_complete[
            (df_complete['Location'] == location) &
            (df_complete['Year'] == most_recent_year) &
            (df_complete['Month'].isin(months))
        ]
        
        if len(hist_season) > 0 and len(current_season) > 0:
            # Calculate statistics
            hist_mean = hist_season['Predicted_Bloom_Score'].mean()
            hist_std = hist_season['Predicted_Bloom_Score'].std()
            current_mean = current_season['Predicted_Bloom_Score'].mean()
            
            anomaly = current_mean - hist_mean
            z_score = (current_mean - hist_mean) / hist_std if hist_std > 0 else 0
            
            # Classify anomaly severity
            if abs(z_score) > 2:
                severity = "🚨 EXTREME ANOMALY"
                anomalies_detected.append((season_name, anomaly, "extreme"))
            elif abs(z_score) > 1:
                severity = "⚠️ SIGNIFICANT ANOMALY"
                anomalies_detected.append((season_name, anomaly, "significant"))
            elif abs(anomaly) > 0.05:
                severity = "⚡ Minor anomaly"
                anomalies_detected.append((season_name, anomaly, "minor"))
            else:
                severity = "✓ Normal"
            
            direction = "above" if anomaly > 0 else "below"
            
            # Climate driver analysis for anomalous seasons
            climate_drivers = ""
            if abs(z_score) > 1:
                # Identify which climate parameter shows the biggest deviation
                for param in ['GWETROOT', 'T2M', 'PRECTOTCORR']:
                    if param in current_season.columns and param in hist_season.columns:
                        param_hist = hist_season[param].mean()
                        param_curr = current_season[param].mean()
                        param_change = ((param_curr - param_hist) / param_hist * 100) if param_hist != 0 else 0
                        
                        if abs(param_change) > 15:
                            param_name = {'GWETROOT': 'Soil Moisture', 'T2M': 'Temperature', 'PRECTOTCORR': 'Precipitation'}[param]
                            climate_drivers += f"\n      → Driven by {param_name}: {param_change:+.1f}%"
            
            print(f"\n   {season_name}:")
            print(f"      {most_recent_year}: {current_mean:.3f} | Historical: {hist_mean:.3f} ({anomaly:+.3f})")
            print(f"      Status: {severity} ({direction} average){climate_drivers}")
    
    # Summary of anomalies
    if anomalies_detected:
        print(f"\n   ⚡ Anomaly Summary for {location}:")
        extreme_count = sum(1 for _, _, sev in anomalies_detected if sev == "extreme")
        significant_count = sum(1 for _, _, sev in anomalies_detected if sev == "significant")
        minor_count = sum(1 for _, _, sev in anomalies_detected if sev == "minor")
        
        if extreme_count > 0:
            print(f"      • {extreme_count} extreme anomaly/anomalies")
        if significant_count > 0:
            print(f"      • {significant_count} significant anomaly/anomalies")
        if minor_count > 0:
            print(f"      • {minor_count} minor anomaly/anomalies")
    else:
        print(f"\n   ✓ No significant anomalies detected for {location}")

# Year-over-year change analysis
print("\n" + "-" * 60)
print("Year-over-Year Change Analysis:")
print("-" * 60)

for location in ['Carrizo Plain', 'Antelope Valley']:
    loc_data = df_complete[df_complete['Location'] == location]
    
    # Get annual averages
    annual_avg = loc_data.groupby('Year')['Predicted_Bloom_Score'].mean()
    
    if len(annual_avg) > 1:
        print(f"\n📍 {location}:")
        
        for i in range(1, len(annual_avg)):
            year = annual_avg.index[i]
            prev_year = annual_avg.index[i-1]
            change = annual_avg.iloc[i] - annual_avg.iloc[i-1]
            pct_change = (change / annual_avg.iloc[i-1] * 100) if annual_avg.iloc[i-1] != 0 else 0
            
            trend_icon = "📈" if change > 0 else "📉" if change < 0 else "→"
            
            print(f"   {prev_year} → {year}: {change:+.3f} ({pct_change:+.1f}%) {trend_icon}")

# Save comprehensive predictions
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Save all predictions
output_cols = ['Location', 'Year', 'Month', 'Predicted_Bloom_Score', 'Bloom_Category']
df_complete[output_cols].to_csv('complete_bloom_predictions_2020-2025.csv', index=False)
print(" Full predictions saved to 'complete_bloom_predictions_2020-2025.csv'")

# Save spring-only predictions
spring_predictions = df_complete[df_complete['Month'].isin(spring_months)][output_cols]
spring_predictions.to_csv('spring_bloom_predictions_2020-2025.csv', index=False)
print(" Spring predictions saved to 'spring_bloom_predictions_2020-2025.csv'")

print("\n" + "="*60)
print("PREDICTION ANALYSIS COMPLETE")
print("="*60)
print("\nNext Steps:")
print("   1. Monitor October-December 2025 precipitation and soil moisture")
print("   2. Update climate data as winter 2025-2026 progresses")
print("   3. Re-run predictions in January 2026 for refined forecast")
print("   4. Watch for winter storm patterns - multiple events better than single storms")