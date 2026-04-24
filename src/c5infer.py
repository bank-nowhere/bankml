"""
Image Classification Inference Script
Loads trained model and evaluates on test data with confusion matrix
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import glob

# Configuration (matching training script)
TEST_DATA_DIR = 'data/archive/test'
MODEL_DIR = 'examples'
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

def load_latest_model(model_dir='examples', model_pattern='cats_dogs_model_*.h5'):
    """
    Load the most recently trained model from the examples directory

    Args:
        model_dir: Directory containing saved models
        model_pattern: Pattern to match model files

    Returns:
        Loaded Keras model
    """
    # Try to find timestamped models first
    model_files = glob.glob(os.path.join(model_dir, model_pattern))

    if model_files:
        # Sort by modification time and get the most recent
        latest_model = max(model_files, key=os.path.getmtime)
        print(f"Loading latest timestamped model: {latest_model}")
    else:
        # Fall back to final model
        latest_model = os.path.join(model_dir, 'cats_dogs_model_final.h5')
        if not os.path.exists(latest_model):
            raise FileNotFoundError(
                f"No model found in {model_dir}. "
                f"Please train a model first using class5training.py"
            )
        print(f"Loading final model: {latest_model}")

    model = keras.models.load_model(latest_model)
    return model, latest_model

def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plot and save confusion matrix

    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))

    # Create heatmap
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
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()

def plot_normalized_confusion_matrix(cm, class_names, save_path):
    """
    Plot and save normalized confusion matrix (percentages)

    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Path to save the plot
    """
    # Normalize by true labels (rows)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage'}
    )

    plt.title('Normalized Confusion Matrix (Percentages)', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Normalized confusion matrix saved to: {save_path}")
    plt.close()

def plot_sample_predictions(model, test_generator, class_names, save_path, num_samples=16):
    """
    Plot sample predictions with images

    Args:
        model: Trained model
        test_generator: Test data generator
        class_names: List of class names
        save_path: Path to save the plot
        num_samples: Number of samples to display
    """
    # Get a batch of test images
    test_generator.reset()
    images, true_labels = next(test_generator)

    # Limit to num_samples
    images = images[:num_samples]
    true_labels = true_labels[:num_samples]

    # Get predictions
    predictions = model.predict(images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(true_labels, axis=1)

    # Plot
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    axes = axes.flatten()

    for idx in range(min(num_samples, len(images))):
        ax = axes[idx]

        # Display image
        ax.imshow(images[idx])

        # Get prediction confidence
        confidence = predictions[idx][predicted_classes[idx]] * 100

        # Set title with prediction info
        true_class = class_names[true_classes[idx]]
        pred_class = class_names[predicted_classes[idx]]

        # Color: green if correct, red if wrong
        color = 'green' if true_classes[idx] == predicted_classes[idx] else 'red'

        title = f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.1f}%"
        ax.set_title(title, color=color, fontsize=10)
        ax.axis('off')

    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Sample predictions saved to: {save_path}")
    plt.close()

def calculate_metrics(cm):
    """
    Calculate per-class and overall metrics from confusion matrix

    Args:
        cm: Confusion matrix

    Returns:
        Dictionary of metrics
    """
    # For binary classification
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    }

def main():
    print("=" * 70)
    print("Image Classification Inference Script")
    print("=" * 70)

    # Load trained model
    print("\nLoading trained model...")
    try:
        model, model_path = load_latest_model(MODEL_DIR)
        print(f"Model loaded successfully from: {model_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Print model summary
    print("\nModel Architecture:")
    model.summary()

    # Prepare test data generator (no augmentation, only rescaling)
    print(f"\nLoading test data from: {TEST_DATA_DIR}")
    print(f"Image size: {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"Batch size: {BATCH_SIZE}")

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        TEST_DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # Important: don't shuffle for confusion matrix
    )

    print(f"\nNumber of test samples: {test_generator.samples}")
    print(f"Number of classes: {test_generator.num_classes}")
    print(f"Class indices: {test_generator.class_indices}")

    # Get class names
    class_names = list(test_generator.class_indices.keys())
    print(f"Class names: {class_names}")

    # Evaluate model
    print("\n" + "=" * 70)
    print("Evaluating model on test set...")
    print("=" * 70)

    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # Generate predictions
    print("\nGenerating predictions...")
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)

    # Get true labels
    true_classes = test_generator.classes

    # Generate confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(true_classes, predicted_classes)

    print("\nConfusion Matrix:")
    print(cm)

    # Calculate and display metrics
    metrics = calculate_metrics(cm)

    print("\n" + "=" * 70)
    print("Detailed Metrics:")
    print("=" * 70)
    print(f"Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision:    {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:       {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1-Score:     {metrics['f1_score']:.4f}")
    print(f"Specificity:  {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
    print("\nConfusion Matrix Breakdown:")
    print(f"True Positives:  {metrics['true_positives']}")
    print(f"True Negatives:  {metrics['true_negatives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")

    # Classification report
    print("\n" + "=" * 70)
    print("Classification Report:")
    print("=" * 70)
    print(classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names,
        digits=4
    ))

    # Save visualizations
    print("\n" + "=" * 70)
    print("Saving visualizations...")
    print("=" * 70)

    # Plot confusion matrix
    cm_path = os.path.join(MODEL_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, cm_path)

    # Plot normalized confusion matrix
    cm_norm_path = os.path.join(MODEL_DIR, 'confusion_matrix_normalized.png')
    plot_normalized_confusion_matrix(cm, class_names, cm_norm_path)

    # Plot sample predictions
    samples_path = os.path.join(MODEL_DIR, 'sample_predictions.png')
    plot_sample_predictions(model, test_generator, class_names, samples_path)

    # Save metrics to file
    metrics_path = os.path.join(MODEL_DIR, 'evaluation_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Model Evaluation Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test Data: {TEST_DATA_DIR}\n")
        f.write(f"Number of test samples: {test_generator.samples}\n\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n\n")
        f.write("=" * 70 + "\n")
        f.write("Detailed Metrics:\n")
        f.write("=" * 70 + "\n")
        f.write(f"Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"Precision:    {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)\n")
        f.write(f"Recall:       {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)\n")
        f.write(f"F1-Score:     {metrics['f1_score']:.4f}\n")
        f.write(f"Specificity:  {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)\n\n")
        f.write("Confusion Matrix Breakdown:\n")
        f.write(f"True Positives:  {metrics['true_positives']}\n")
        f.write(f"True Negatives:  {metrics['true_negatives']}\n")
        f.write(f"False Positives: {metrics['false_positives']}\n")
        f.write(f"False Negatives: {metrics['false_negatives']}\n\n")
        f.write("=" * 70 + "\n")
        f.write("Confusion Matrix:\n")
        f.write("=" * 70 + "\n")
        f.write(str(cm) + "\n\n")
        f.write("=" * 70 + "\n")
        f.write("Classification Report:\n")
        f.write("=" * 70 + "\n")
        f.write(classification_report(
            true_classes,
            predicted_classes,
            target_names=class_names,
            digits=4
        ))

    print(f"Evaluation metrics saved to: {metrics_path}")

    print("\n" + "=" * 70)
    print("Inference Complete!")
    print("=" * 70)
    print(f"\nGenerated files in {MODEL_DIR}:")
    print(f"  - confusion_matrix.png")
    print(f"  - confusion_matrix_normalized.png")
    print(f"  - sample_predictions.png")
    print(f"  - evaluation_metrics.txt")
    print("=" * 70)

    return model, predictions, metrics

if __name__ == "__main__":
    model, predictions, metrics = main()