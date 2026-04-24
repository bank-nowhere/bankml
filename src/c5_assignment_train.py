import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os
import joblib

# Configuration
DATA_PATH = 'data/c5data.csv'
MODEL_SAVE_PATH = 'examples/c5_assignment_model.h5'
SCALER_SAVE_PATH = 'examples/c5_scaler.pkl'
ENCODER_SAVE_PATH = 'examples/c5_encoder.pkl'
FEATURES_SAVE_PATH = 'examples/c5_features.pkl'

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    weight = alpha * y_true * tf.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma)
    return tf.reduce_mean(weight * cross_entropy)

def preprocess_data(df):
    data = df.copy()
    continuous_cols = ['Age', 'DurationOfPitch', 'NumberOfFollowups', 'PreferredPropertyStar',
                       'NumberOfTrips', 'PitchSatisfactionScore', 'NumberOfChildrenVisiting', 'MonthlyIncome']
    for col in continuous_cols:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
    categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 
                        'MaritalStatus', 'Designation']
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].mode()[0])
    if 'Gender' in data.columns:
        data['Gender'] = data['Gender'].replace('Fe Male', 'Female')
    return data

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df_clean = preprocess_data(df)
    
    X = df_clean.drop('ProdTaken', axis=1)
    y = df_clean['ProdTaken']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    feature_cols = X_encoded.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs('examples', exist_ok=True)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    joblib.dump(label_encoder, ENCODER_SAVE_PATH)
    joblib.dump(feature_cols, FEATURES_SAVE_PATH)
    
    # SMOTE to balance the classes
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    
    print("Building Model...")
    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(X_train_smote.shape[1],)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=focal_loss,
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    print("Starting Training...")
    history = model.fit(
        X_train_smote, y_train_smote,
        epochs=150,
        batch_size=64,
        validation_data=(X_test_scaled, y_test),
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
        ]
    )
    
    test_loss, test_accuracy, test_auc = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} (Must be > 0.93)")
    print(f"Test AUC: {test_auc:.4f}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('examples/training_history.png', dpi=150)
    model.save(MODEL_SAVE_PATH)
    
    y_pred_proba = model.predict(X_test_scaled)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
