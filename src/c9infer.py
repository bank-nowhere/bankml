import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt

# Load the trained model
print("Loading trained LSTM model...")
model = load_model('/opt/bankml/src/c9model.keras', compile=False)
# Recompile with correct loss function
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
print("âœ“ Model loaded successfully")

# Load preprocessing objects
print("\nLoading preprocessing objects...")
with open('/opt/bankml/src/preprocessing_objects.pkl', 'rb') as f:
    preprocessing_objects = pickle.load(f)

scaler = preprocessing_objects['scaler']
label_encoder = preprocessing_objects['label_encoder']
feature_columns = preprocessing_objects['feature_columns']
LOOKBACK = preprocessing_objects['lookback']
FORECAST_DAYS = preprocessing_objects['forecast_days']

print("âœ“ Preprocessing objects loaded successfully")
print(f"  - Lookback period: {LOOKBACK} days")
print(f"  - Forecast horizon: {FORECAST_DAYS} days ahead")
print(f"  - Feature columns: {feature_columns}")

# Load the data
print("\n" + "="*50)
print("Loading Data for Inference")
print("="*50)
df = pd.read_csv('/opt/bankml/data/c9data.csv')
print(f"Dataset shape: {df.shape}")

# Data preprocessing (same as training)
print("\nPreprocessing data...")
df['datesold'] = pd.to_datetime(df['datesold'])
df = df.sort_values('datesold').reset_index(drop=True)

# Extract time-based features
df['year'] = df['datesold'].dt.year
df['month'] = df['datesold'].dt.month
df['day'] = df['datesold'].dt.day
df['dayofweek'] = df['datesold'].dt.dayofweek
df['dayofyear'] = df['datesold'].dt.dayofyear

# Encode propertyType
df['propertyType_encoded'] = label_encoder.transform(df['propertyType'])

# Select features
df_clean = df[feature_columns].dropna()
print(f"Cleaned dataset shape: {df_clean.shape}")

# Normalize the data
scaled_data = scaler.transform(df_clean.values)

# Create sequences for inference
def create_sequences_inference(data, lookback, forecast_days):
    X, y, indices = [], [], []
    for i in range(lookback, len(data) - forecast_days):
        X.append(data[i-lookback:i])
        y.append(data[i + forecast_days, -1])  # actual price
        indices.append(i + forecast_days)  # index of the prediction
    return np.array(X), np.array(y), indices

print(f"\nCreating sequences for inference...")
X_inference, y_actual, prediction_indices = create_sequences_inference(scaled_data, LOOKBACK, FORECAST_DAYS)

print(f"Total sequences available for inference: {X_inference.shape[0]}")

# Make predictions on all available data
print("\n" + "="*50)
print("Running Inference")
print("="*50)

print("\nMaking predictions...")
y_pred = model.predict(X_inference, verbose=1)

# Inverse transform to get actual prices
dummy_array_actual = np.zeros((len(y_actual), scaled_data.shape[1]))
dummy_array_actual[:, -1] = y_actual
y_actual_prices = scaler.inverse_transform(dummy_array_actual)[:, -1]

dummy_array_pred = np.zeros((len(y_pred), scaled_data.shape[1]))
dummy_array_pred[:, -1] = y_pred.flatten()
y_pred_prices = scaler.inverse_transform(dummy_array_pred)[:, -1]

# Calculate metrics
mae = np.mean(np.abs(y_pred_prices - y_actual_prices))
rmse = np.sqrt(np.mean((y_pred_prices - y_actual_prices)**2))
mape = np.mean(np.abs((y_actual_prices - y_pred_prices) / y_actual_prices)) * 100

print("\n" + "="*50)
print("Inference Results")
print("="*50)
print(f"\nOverall Performance Metrics:")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Display sample predictions with dates
print(f"\n{'-'*80}")
print(f"Sample Predictions (First 30 samples)")
print(f"{'-'*80}")
print(f"{'Date':12} {'Actual Price':>15} {'Predicted Price':>15} {'Difference':>15} {'Error %':>10}")
print(f"{'-'*80}")

for i in range(min(30, len(y_actual_prices))):
    idx = prediction_indices[i]
    date = df.iloc[idx]['datesold'].strftime('%Y-%m-%d')
    diff = y_pred_prices[i] - y_actual_prices[i]
    error_pct = (diff / y_actual_prices[i]) * 100
    print(f"{date:12} ${y_actual_prices[i]:>13,.2f} ${y_pred_prices[i]:>13,.2f} ${diff:>13,.2f} {error_pct:>9.2f}%")

# Show recent predictions (last 30)
print(f"\n{'-'*80}")
print(f"Recent Predictions (Last 30 samples)")
print(f"{'-'*80}")
print(f"{'Date':12} {'Actual Price':>15} {'Predicted Price':>15} {'Difference':>15} {'Error %':>10}")
print(f"{'-'*80}")

start_idx = max(0, len(y_actual_prices) - 30)
for i in range(start_idx, len(y_actual_prices)):
    idx = prediction_indices[i]
    date = df.iloc[idx]['datesold'].strftime('%Y-%m-%d')
    diff = y_pred_prices[i] - y_actual_prices[i]
    error_pct = (diff / y_actual_prices[i]) * 100
    print(f"{date:12} ${y_actual_prices[i]:>13,.2f} ${y_pred_prices[i]:>13,.2f} ${diff:>13,.2f} {error_pct:>9.2f}%")

# Statistical analysis
print(f"\n{'-'*80}")
print(f"Statistical Analysis")
print(f"{'-'*80}")
print(f"Total predictions made: {len(y_pred_prices)}")
print(f"Average actual price: ${np.mean(y_actual_prices):,.2f}")
print(f"Average predicted price: ${np.mean(y_pred_prices):,.2f}")
print(f"Median actual price: ${np.median(y_actual_prices):,.2f}")
print(f"Median predicted price: ${np.median(y_pred_prices):,.2f}")
print(f"Max actual price: ${np.max(y_actual_prices):,.2f}")
print(f"Max predicted price: ${np.max(y_pred_prices):,.2f}")
print(f"Min actual price: ${np.min(y_actual_prices):,.2f}")
print(f"Min predicted price: ${np.min(y_pred_prices):,.2f}")

# Accuracy within different thresholds
thresholds = [50000, 100000, 150000, 200000]
print(f"\nPrediction Accuracy within Thresholds:")
for threshold in thresholds:
    within_threshold = np.sum(np.abs(y_pred_prices - y_actual_prices) <= threshold)
    percentage = (within_threshold / len(y_pred_prices)) * 100
    print(f"  Within ${threshold:,}: {within_threshold}/{len(y_pred_prices)} ({percentage:.2f}%)")

# Visualizations
print("\n" + "="*50)
print("Generating Visualizations")
print("="*50)

# Plot 1: Actual vs Predicted (scatter plot)
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_actual_prices, y_pred_prices, alpha=0.5, s=10)
plt.plot([y_actual_prices.min(), y_actual_prices.max()],
         [y_actual_prices.min(), y_actual_prices.max()],
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Actual vs Predicted Prices (5-day ahead forecast)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
errors = y_pred_prices - y_actual_prices
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Prediction Error ($)')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('inference_scatter_plot.png', dpi=300, bbox_inches='tight')
print("âœ“ Scatter plot saved to 'inference_scatter_plot.png'")

# Plot 2: Time series of predictions
plt.figure(figsize=(16, 6))

# Plot subset for clarity (last 500 samples)
plot_start = max(0, len(y_actual_prices) - 500)
plot_indices = range(plot_start, len(y_actual_prices))

plt.plot(plot_indices, y_actual_prices[plot_start:],
         label='Actual Price', marker='o', markersize=2, alpha=0.7, linewidth=1)
plt.plot(plot_indices, y_pred_prices[plot_start:],
         label='Predicted Price (5 days ahead)', marker='x', markersize=2, alpha=0.7, linewidth=1)
plt.xlabel('Sample Index')
plt.ylabel('Price ($)')
plt.title('Price Prediction Time Series (Last 500 samples)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('inference_time_series.png', dpi=300, bbox_inches='tight')
print("âœ“ Time series plot saved to 'inference_time_series.png'")

# Plot 3: Error analysis over time
plt.figure(figsize=(16, 6))

plt.subplot(2, 1, 1)
plt.plot(errors, alpha=0.6, linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.ylabel('Error ($)')
plt.title('Prediction Errors Over Time')
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
error_pct = (errors / y_actual_prices) * 100
plt.plot(error_pct, alpha=0.6, linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Sample Index')
plt.ylabel('Error (%)')
plt.title('Prediction Errors (%) Over Time')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('inference_error_analysis.png', dpi=300, bbox_inches='tight')
print("âœ“ Error analysis plot saved to 'inference_error_analysis.png'")

# Save predictions to CSV
print("\n" + "="*50)
print("Saving Predictions")
print("="*50)

results_df = pd.DataFrame({
    'date': [df.iloc[idx]['datesold'] for idx in prediction_indices],
    'actual_price': y_actual_prices,
    'predicted_price': y_pred_prices,
    'difference': y_pred_prices - y_actual_prices,
    'error_percentage': ((y_pred_prices - y_actual_prices) / y_actual_prices) * 100,
    'bedrooms': [df.iloc[idx]['bedrooms'] for idx in prediction_indices],
    'propertyType': [df.iloc[idx]['propertyType'] for idx in prediction_indices]
})

results_df.to_csv('inference_results.csv', index=False)
print("âœ“ Detailed predictions saved to 'inference_results.csv'")

# Function to predict price for new data
def predict_price_for_new_data(input_data):
    """
    Predict price for new property data.

    Parameters:
    -----------
    input_data : dict
        Dictionary containing recent 30-day historical data with keys:
        'datesold', 'propertyType', 'bedrooms', 'price'
        Each value should be a list of 30 elements

    Returns:
    --------
    predicted_price : float
        Predicted price 5 days ahead
    """
    # Create DataFrame from input
    df_new = pd.DataFrame(input_data)
    df_new['datesold'] = pd.to_datetime(df_new['datesold'])

    # Extract features
    df_new['year'] = df_new['datesold'].dt.year
    df_new['month'] = df_new['datesold'].dt.month
    df_new['day'] = df_new['datesold'].dt.day
    df_new['dayofweek'] = df_new['datesold'].dt.dayofweek
    df_new['dayofyear'] = df_new['datesold'].dt.dayofyear
    df_new['propertyType_encoded'] = label_encoder.transform(df_new['propertyType'])

    # Select features and scale
    features = df_new[feature_columns].values
    scaled_features = scaler.transform(features)

    # Reshape for LSTM
    X_new = scaled_features.reshape(1, LOOKBACK, -1)

    # Predict
    y_pred_scaled = model.predict(X_new, verbose=0)

    # Inverse transform
    dummy = np.zeros((1, scaled_features.shape[1]))
    dummy[:, -1] = y_pred_scaled.flatten()
    predicted_price = scaler.inverse_transform(dummy)[:, -1][0]

    return predicted_price

print("\n" + "="*50)
print("Inference Complete!")
print("="*50)
print(f"\nSummary:")
print(f"- Total predictions: {len(y_pred_prices)}")
print(f"- Mean Absolute Error: ${mae:,.2f}")
print(f"- Root Mean Squared Error: ${rmse:,.2f}")
print(f"- Mean Absolute Percentage Error: {mape:.2f}%")
print(f"\nFiles created:")
print(f"- inference_results.csv (detailed predictions)")
print(f"- inference_scatter_plot.png (actual vs predicted)")
print(f"- inference_time_series.png (time series visualization)")
print(f"- inference_error_analysis.png (error analysis)")
print(f"\nThe model is ready for real-time predictions using predict_price_for_new_data()")