import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Configuration
DATA_PATH = 'data/c5data.csv'
MODEL_PATH = 'examples/c5_assignment_model.h5'
SCALER_PATH = 'examples/c5_scaler.pkl'
ENCODER_PATH = 'examples/c5_encoder.pkl'
FEATURES_PATH = 'examples/c5_features.pkl'
OUT_DIR = 'examples'

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    weight = alpha * y_true * tf.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma)
    return tf.reduce_mean(weight * cross_entropy)

def preprocess_data(df):
    """Same preprocessing as training"""
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

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

def main():
    print("=" * 60)
    print("Assignment 5: Inference Script")
    print("=" * 60)
    
    # Check if necessary files exist
    files_needed = [MODEL_PATH, SCALER_PATH, ENCODER_PATH, FEATURES_PATH]
    for f in files_needed:
        if not os.path.exists(f):
            print(f"Error: Required file {f} not found. Train the model first.")
            return

    # Load artifacts
    model = keras.models.load_model(MODEL_PATH, custom_objects={'focal_loss': focal_loss}, compile=False)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    
    print(f"Loaded model and artifacts.")
    
    # Load and clean Data
    df = pd.read_csv(DATA_PATH)
    df_clean = preprocess_data(df)
    
    X = df_clean.drop('ProdTaken', axis=1)
    y = df_clean['ProdTaken']
    y_encoded = label_encoder.transform(y)
    
    # One-hot encode using the exact same columns from training
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    
    # Ensure all training feature columns are present and in the same order
    for col in feature_cols:
        if col not in X_encoded.columns:
            # If a categorical level is missing in testing data but was present in training, fill with 0
            # However this script loads entire c5data so it should be fine.
            X_encoded[col] = 0
            print(f"Adding missing column {col}")
            
    # Keep only the features used in training, in the exact same order
    X_encoded = X_encoded[feature_cols]
    
    # Scale
    X_scaled = scaler.transform(X_encoded)
    
    # Predict
    print("\nGenerating predictions...")
    y_pred_proba = model.predict(X_scaled)
    # Using 0.5 threshold as standard
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Generate confusion matrix
    cm = confusion_matrix(y_encoded, y_pred)
    
    # Plot & Save
    cm_path = os.path.join(OUT_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(cm, label_encoder.classes_, cm_path)
    
    # Evaluate Accuracy using sklearn
    accuracy = accuracy_score(y_encoded, y_pred)
    
    # Save text evaluation
    metrics_path = os.path.join(OUT_DIR, 'evaluation_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Model Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Test Dataset Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_encoded, y_pred, target_names=["No", "Yes"]))
        f.write("\nConfusion Matrix:\n")
        f.write(f"{cm}\n")
        
    print(f"\nFinal evaluation saved to: {metrics_path}")
    print(f"Overall Accuracy: {accuracy*100:.2f}% (must exceed 93%)")

if __name__ == "__main__":
    main()
