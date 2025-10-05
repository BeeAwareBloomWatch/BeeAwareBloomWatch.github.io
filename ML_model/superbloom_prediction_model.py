import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the datasets
print("Loading datasets...")
bloom_data = pd.read_csv('bloom_score_dataset_2020-2025.csv')
climate_cp = pd.read_csv('POWER_CP_Monthly.csv')
climate_av = pd.read_csv('POWER_AV_Monthly.csv')

print(f"Bloom data shape: {bloom_data.shape}")
print(f"Climate CP shape: {climate_cp.shape}")
print(f"Climate AV shape: {climate_av.shape}")

# Function to pivot climate data from wide to long format
def pivot_climate_data(df, location_name):
    """Convert wide-format monthly climate data to long format"""
    records = []
    
    for _, row in df.iterrows():
        parameter = row['PARAMETER']
        year = row['YEAR']
        
        # Process each month
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

# Process both climate datasets
print("\nProcessing climate data...")
climate_cp_long = pivot_climate_data(climate_cp, 'Carrizo Plain')
climate_av_long = pivot_climate_data(climate_av, 'Antelope Valley')

# Combine both climate datasets
climate_combined = pd.concat([climate_cp_long, climate_av_long], ignore_index=True)

# Pivot to get parameters as columns
climate_wide = climate_combined.pivot_table(
    index=['Location', 'Year', 'Month'],
    columns='Parameter',
    values='Value'
).reset_index()

# Clean column names
climate_wide.columns.name = None

print(f"\nClimate data pivoted shape: {climate_wide.shape}")
print(f"Climate parameters: {[col for col in climate_wide.columns if col not in ['Location', 'Year', 'Month']]}")

# Merge climate data with bloom scores
print("\nMerging climate and bloom data...")
merged_data = bloom_data.merge(
    climate_wide,
    on=['Location', 'Year', 'Month'],
    how='left'
)

print(f"Merged data shape: {merged_data.shape}")
print(f"\nMissing values:\n{merged_data.isnull().sum()}")

# Feature Engineering
print("\nEngineering features...")

# Sort data for proper lag calculations
merged_data = merged_data.sort_values(['Location', 'Year', 'Month']).reset_index(drop=True)

# Create a copy for feature engineering
df = merged_data.copy()

# Get climate parameter columns (excluding metadata columns)
climate_params = [col for col in df.columns if col not in ['Location', 'Year', 'Month', 'Estimated_Score', 'Notes']]

# Function to create lagged features
def create_lag_features(df, location, params, lags=[1, 2, 3, 6]):
    """Create lagged features for each climate parameter"""
    df_loc = df[df['Location'] == location].copy()
    
    for param in params:
        for lag in lags:
            df_loc[f'{param}_lag{lag}'] = df_loc[param].shift(lag)
    
    return df_loc

# Create lagged features for each location
df_cp = create_lag_features(df, 'Carrizo Plain', climate_params)
df_av = create_lag_features(df, 'Antelope Valley', climate_params)

# Combine back
df_engineered = pd.concat([df_cp, df_av], ignore_index=True)

# Create seasonal aggregates (e.g., winter precipitation)
# Winter = Dec (prev year), Jan, Feb
# We'll create rolling windows for key bloom-prediction periods

for param in climate_params:
    # 3-month rolling average
    df_engineered[f'{param}_roll3'] = df_engineered.groupby('Location')[param].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # 6-month rolling average
    df_engineered[f'{param}_roll6'] = df_engineered.groupby('Location')[param].transform(
        lambda x: x.rolling(window=6, min_periods=1).mean()
    )

# Create interaction features for key relationships
if 'PRECTOTCORR' in climate_params and 'T2M' in climate_params:
    df_engineered['precip_temp_interaction'] = df_engineered['PRECTOTCORR'] * df_engineered['T2M']

# Add month as categorical feature (for seasonality)
df_engineered['Month_sin'] = np.sin(2 * np.pi * df_engineered['Month'] / 12)
df_engineered['Month_cos'] = np.cos(2 * np.pi * df_engineered['Month'] / 12)

# Add year trend
df_engineered['Year_normalized'] = (df_engineered['Year'] - df_engineered['Year'].min()) / (df_engineered['Year'].max() - df_engineered['Year'].min())

# Location encoding
df_engineered['Location_encoded'] = df_engineered['Location'].map({
    'Carrizo Plain': 0,
    'Antelope Valley': 1
})

print(f"Engineered features shape: {df_engineered.shape}")

# Prepare data for modeling
print("\nPreparing data for modeling...")

# Drop rows with NaN values (from lagging)
df_model = df_engineered.dropna().copy()

print(f"Data shape after dropping NaN: {df_model.shape}")

# Define features and target
feature_cols = [col for col in df_model.columns if col not in ['Location', 'Year', 'Month', 'Estimated_Score', 'Notes']]
X = df_model[feature_cols]
y = df_model['Estimated_Score']

print(f"\nFeature count: {len(feature_cols)}")
print(f"Sample count: {len(X)}")

# Train-test split (temporal split to avoid data leakage)
# Use data up to 2024 for training, 2025 for testing
train_mask = df_model['Year'] < 2025
test_mask = df_model['Year'] == 2025

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Train Random Forest model
print("\nTraining Random Forest model...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Model evaluation
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

# Training performance
y_train_pred = rf_model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print(f"\nTraining Performance:")
print(f"  RMSE: {train_rmse:.3f}")
print(f"  MAE:  {train_mae:.3f}")
print(f"  R²:   {train_r2:.3f}")

# Test performance (if test data available)
if len(X_test) > 0:
    y_test_pred = rf_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nTest Performance (2025 data):")
    print(f"  RMSE: {test_rmse:.3f}")
    print(f"  MAE:  {test_mae:.3f}")
    print(f"  R²:   {test_r2:.3f}")

# Cross-validation score
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores.mean())
print(f"\nCross-Validation RMSE: {cv_rmse:.3f} (+/- {cv_scores.std():.3f})")

# Feature importance analysis
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importance.head(20).to_string(index=False))

# Save the model and feature information
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

model_artifacts = {
    'model': rf_model,
    'feature_columns': feature_cols,
    'feature_importance': feature_importance,
    'climate_params': climate_params,
    'train_metrics': {
        'rmse': train_rmse,
        'mae': train_mae,
        'r2': train_r2
    },
    'location_encoding': {
        'Carrizo Plain': 0,
        'Antelope Valley': 1
    }
}

joblib.dump(model_artifacts, 'superbloom_model.pkl')
print("\nModel saved as 'superbloom_model.pkl'")

# Save feature importance to CSV
feature_importance.to_csv('feature_importance.csv', index=False)
print("Feature importance saved as 'feature_importance.csv'")

# Save processed dataset for reference
df_model[['Location', 'Year', 'Month', 'Estimated_Score'] + feature_cols].to_csv(
    'processed_training_data.csv', index=False
)
print("Processed training data saved as 'processed_training_data.csv'")

print("\n" + "="*60)
print("MODEL TRAINING COMPLETE")
print("="*60)
print("\nKey Insights:")
print(f"1. Model trained on {len(X_train)} samples from 2020-2024")
print(f"2. Model uses {len(feature_cols)} engineered features")
print(f"3. Training R² score: {train_r2:.3f}")
print(f"4. Cross-validation RMSE: {cv_rmse:.3f}")
print(f"\nThe model is now ready to predict 2026 superbloom potential!")