import pandas as pd
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create output directory if it doesn't exist
os.makedirs('data/output', exist_ok=True)

# Define custom loss functions (must match the ones used in training)
def combined_bce_l1_weights_loss(model, alpha=0.01, beta=0.001):
    """
    Combined loss: alpha * binary_crossentropy + beta * L1_regularization on model weights
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        bce_loss = tf.reduce_mean(bce)
        l1_reg = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in model.trainable_weights])
        total_loss = alpha * bce_loss + beta * l1_reg
        return total_loss
    return loss


def combined_bce_l1_loss(alpha=1.0, beta=0.01):
    """
    Combined loss function: alpha * binary_crossentropy + beta * L1_regularization
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        bce_loss = tf.reduce_mean(bce)
        l1_reg = tf.reduce_mean(tf.abs(y_pred))
        total_loss = alpha * bce_loss + beta * l1_reg
        return total_loss
    return loss


# Load the trained model with custom loss function
print("Loading trained model...")
# Load the model without compiling first
model = keras.models.load_model('examples/bank_classification_model5.h5', compile=False)

# Recompile with the custom loss that was used during training
# Match the loss function and parameters used in class4training.py
custom_loss = combined_bce_l1_weights_loss(model, alpha=1.0, beta=0.001)
model.compile(
    optimizer='adam',
    loss=custom_loss,
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

print("Model loaded successfully!")

# Display model summary
model.summary()

# Load the test data
print("\nLoading test data...")
df_test = pd.read_csv('data/bank.csv', sep=';')

print(f"Test dataset shape: {df_test.shape}")
print(f"\nTarget distribution:\n{df_test['y'].value_counts()}")

# Separate features and target
X_test = df_test.drop('y', axis=1)
y_test = df_test['y']

# Encode target variable (yes -> 1, no -> 0)
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)

# Handle categorical variables using one-hot encoding
# Must match the training data preprocessing
categorical_columns = X_test.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns: {categorical_columns}")

# One-hot encode categorical features
X_test_encoded = pd.get_dummies(X_test, columns=categorical_columns, drop_first=True)

print(f"Features after encoding: {X_test_encoded.shape[1]}")

# Standardize the features
# Note: In production, you should save and load the scaler from training
# For now, we'll fit a new scaler (ideally should use the training scaler)
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test_encoded)

# Make predictions
print("\nMaking predictions...")
y_pred_proba = model.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_proba > 0.47).astype(int).flatten()

# Calculate metrics
print("\n" + "="*60)
print("INFERENCE RESULTS")
print("="*60)

accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

try:
    auc_score = roc_auc_score(y_test_encoded, y_pred_proba)
    print(f"AUC Score: {auc_score:.4f}")
except:
    print("AUC Score: Could not calculate")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))

# Calculate confusion matrix
cm = confusion_matrix(y_test_encoded, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Bank Classification Model', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

# Add additional statistics as text
tn, fp, fn, tp = cm.ravel()
stats_text = f'True Negatives: {tn}\nFalse Positives: {fp}\nFalse Negatives: {fn}\nTrue Positives: {tp}'
stats_text += f'\n\nAccuracy: {accuracy:.4f}'
plt.text(2.5, 0.5, stats_text, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         verticalalignment='center')

plt.tight_layout()

# Save the figure
output_path = 'data/output/confusion_matrix6.jpg'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nConfusion matrix plot saved to: {output_path}")

# Also create a normalized confusion matrix
plt.figure(figsize=(10, 8))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            cbar_kws={'label': 'Percentage'})
plt.title('Normalized Confusion Matrix - Bank Classification Model', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
output_path_normalized = 'data/output/confusion_matrix_normalized3.jpg'
plt.savefig(output_path_normalized, dpi=300, bbox_inches='tight')
print(f"Normalized confusion matrix plot saved to: {output_path_normalized}")

# Save predictions to CSV
results_df = df_test.copy()
results_df['predicted_label'] = label_encoder.inverse_transform(y_pred)
results_df['prediction_probability'] = y_pred_proba.flatten()
results_df['true_label'] = y_test
results_df['correct_prediction'] = (results_df['true_label'] == results_df['predicted_label'])

output_csv_path = 'data/output/predictions.csv'
results_df.to_csv(output_csv_path, index=False, sep=';')
print(f"Predictions saved to: {output_csv_path}")

print("\n" + "="*60)
print("Inference completed successfully!")
print("="*60)