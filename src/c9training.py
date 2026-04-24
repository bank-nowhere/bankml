import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load the data
print("Loading data...")
df = pd.read_csv('/opt/bankml/data/c9data.csv')

# Display basic information
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head(10))
print(f"\nData types:")
print(df.dtypes)
print(f"\nMissing values:")
print(df.isnull().sum())
print(f"\nStatistical summary:")
print(df.describe())

# Data preprocessing
print("\n" + "="*50)
print("Data Preprocessing")
print("="*50)

# Convert datesold to datetime
df['datesold'] = pd.to_datetime(df['datesold'])

# Sort by date
df = df.sort_values('datesold').reset_index(drop=True)

# Extract time-based features
df['year'] = df['datesold'].dt.year
df['month'] = df['datesold'].dt.month
df['day'] = df['datesold'].dt.day
df['dayofweek'] = df['datesold'].dt.dayofweek
df['dayofyear'] = df['datesold'].dt.dayofyear

# Encode propertyType
le = LabelEncoder()
df['propertyType_encoded'] = le.fit_transform(df['propertyType'])
print(f"\nProperty types: {le.classes_}")

# Select features for training
feature_columns = ['year', 'month', 'day', 'dayofweek', 'dayofyear',
                   'propertyType_encoded', 'bedrooms', 'price']

# Remove any rows with missing values in selected features
df_clean = df[feature_columns].dropna()
print(f"\nCleaned dataset shape: {df_clean.shape}")

# Prepare data for LSTM
# We'll use a sequence of past data to predict price 5 days ahead
LOOKBACK = 30  # Use 30 days of historical data
FORECAST_DAYS = 5  # Predict 5 days ahead

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_clean.values)

# Create sequences for LSTM
def create_sequences(data, lookback, forecast_days):
    X, y = [], []
    for i in range(lookback, len(data) - forecast_days):
        # Use lookback period as input
        X.append(data[i-lookback:i])
        # Predict price (last column) 5 days ahead
        y.append(data[i + forecast_days, -1])  # price is the last column
    return np.array(X), np.array(y)

print(f"\nCreating sequences with lookback={LOOKBACK} days and forecast={FORECAST_DAYS} days ahead...")
X, y = create_sequences(scaled_data, LOOKBACK, FORECAST_DAYS)

print(f"X shape: {X.shape}")  # (samples, timesteps, features)
print(f"y shape: {y.shape}")  # (samples,)

# Split data for training and testing
# Use 80% for training and 20% for testing
split_ratio = 0.8
split_index = int(len(X) * split_ratio)

X_train = X[:split_index]
y_train = y[:split_index]
X_test = X[split_index:]
y_test = y[split_index:]

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Build LSTM model
print("\n" + "="*50)
print("Building LSTM Model")
print("="*50)

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

print("\nModel Summary:")
model.summary()

# Train the model
print("\n" + "="*50)
print("Training Model")
print("="*50)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate on test set
print("\n" + "="*50)
print("Model Evaluation")
print("="*50)

test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss (MSE): {test_loss:.6f}")
print(f"Test MAE: {test_mae:.6f}")

# Make predictions
print("\n" + "="*50)
print("Making Predictions (Inference Phase)")
print("="*50)

y_pred = model.predict(X_test)

# Inverse transform to get actual prices
# Create dummy array with same shape as original data
dummy_array = np.zeros((len(y_test), scaled_data.shape[1]))
dummy_array[:, -1] = y_test
y_test_actual = scaler.inverse_transform(dummy_array)[:, -1]

dummy_array[:, -1] = y_pred.flatten()
y_pred_actual = scaler.inverse_transform(dummy_array)[:, -1]

# Calculate metrics
mae = np.mean(np.abs(y_pred_actual - y_test_actual))
rmse = np.sqrt(np.mean((y_pred_actual - y_test_actual)**2))
mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100

print(f"\nPrediction Results (Actual Prices):")
print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"Root Mean Squared Error: ${rmse:,.2f}")
print(f"Mean Absolute Percentage Error: {mape:.2f}%")

# Display sample predictions
print(f"\nSample Predictions (5 days ahead forecast):")
print(f"{'Actual Price':>15} {'Predicted Price':>15} {'Difference':>15}")
print("-" * 50)
for i in range(min(20, len(y_test_actual))):
    diff = y_pred_actual[i] - y_test_actual[i]
    print(f"${y_test_actual[i]:>13,.2f} ${y_pred_actual[i]:>13,.2f} ${diff:>13,.2f}")

# Plot training history
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['mean_absolute_error'], label='Training MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model MAE During Training')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("\nTraining history saved to 'training_history.png'")

# Plot predictions vs actual
plt.figure(figsize=(14, 6))

# Plot subset for clarity
plot_samples = min(200, len(y_test_actual))
plt.plot(range(plot_samples), y_test_actual[:plot_samples],
         label='Actual Price', marker='o', markersize=3, alpha=0.7)
plt.plot(range(plot_samples), y_pred_actual[:plot_samples],
         label='Predicted Price (5 days ahead)', marker='x', markersize=3, alpha=0.7)
plt.title('Price Prediction: Actual vs Predicted (Test Set - First 200 samples)')
plt.xlabel('Sample Index')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
print("Prediction comparison saved to 'predictions_vs_actual.png'")

# Save the model
model.save('c9model.keras')
print("\nModel saved to 'c9model.keras'")

# Save preprocessing objects
import pickle

preprocessing_objects = {
    'scaler': scaler,
    'label_encoder': le,
    'feature_columns': feature_columns,
    'lookback': LOOKBACK,
    'forecast_days': FORECAST_DAYS
}

with open('preprocessing_objects.pkl', 'wb') as f:
    pickle.dump(preprocessing_objects, f)
print("Preprocessing objects saved to 'preprocessing_objects.pkl'")

print("\n" + "="*50)
print("Training Complete!")
print("="*50)
print(f"\nModel Summary:")
print(f"- Architecture: LSTM with 3 layers (128, 64, 32 units)")
print(f"- Lookback period: {LOOKBACK} days")
print(f"- Forecast horizon: {FORECAST_DAYS} days ahead")
print(f"- Training samples: {X_train.shape[0]}")
print(f"- Test samples: {X_test.shape[0]}")
print(f"- Test MAE: ${mae:,.2f}")
print(f"- Test RMSE: ${rmse:,.2f}")
print(f"- Test MAPE: {mape:.2f}%")
print("\nFiles created:")
print("- c9data.csv (trained model)")
print("- preprocessing_objects.pkl (scalers and encoders)")
print("- training_history.png (loss/MAE plots)")
print("- predictions_vs_actual.png (prediction visualization)")