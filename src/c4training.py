import pandas as pd
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Load the data
print("Loading data...")
df = pd.read_csv('data/bank-full.csv', sep=';')

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nTarget distribution:\n{df['y'].value_counts()}")

# Separate features and target
X = df.drop('y', axis=1)
y = df['y']

# Encode target variable (yes -> 1, no -> 0)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Handle categorical variables using one-hot encoding
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns: {categorical_columns}")

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

print(f"Features after encoding: {X_encoded.shape[1]}")

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train_scaled.shape[0]}")
print(f"Testing set size: {X_test_scaled.shape[0]}")

# Build a simple 3-layer neural network
print("\nBuilding neural network model...")
model = keras.Sequential([
    # Input layer + First hidden layer
    keras.layers.Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    keras.layers.Dropout(0.3),

    # Second hidden layer
    keras.layers.Dense(8, activation='tanh'),
    # keras.layers.Dropout(0.3),
    
    # thrid hidden layer
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dropout(0.3),

    # Output layer
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

# Display model summary
model.summary()


def compile_model_with_custom_loss(model, loss_function, optimizer='adam', metrics=None):
    """
    Compile the model with a custom loss function.

    Parameters:
    -----------
    model : keras.Model
        The model to compile
    loss_function : str or callable
        Either a string for built-in loss functions (e.g., 'binary_crossentropy')
        or a custom loss function that takes (y_true, y_pred) as arguments
    optimizer : str or keras.optimizers.Optimizer
        Optimizer to use (default: 'adam')
    metrics : list
        List of metrics to track (default: ['accuracy', AUC])

    Returns:
    --------
    model : keras.Model
        The compiled model

    Example:
    --------
    # Using built-in loss
    compile_model_with_custom_loss(model, 'binary_crossentropy')

    # Using custom loss
    def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        bce_exp = tf.exp(-bce)
        focal_loss = alpha * tf.pow(1 - bce_exp, gamma) * bce
        return tf.reduce_mean(focal_loss)

    compile_model_with_custom_loss(model, focal_loss)
    """
    if metrics is None:
        metrics = ['accuracy', keras.metrics.AUC(name='auc')]

    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metrics
    )

    return model


def weighted_binary_crossentropy(y_true, y_pred, pos_weight=0.001):
    """
    Weighted binary crossentropy loss for imbalanced datasets.
    Applies higher weight to positive class errors.
    """
    y_true = tf.cast(y_true, tf.float32)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    weights = y_true * (pos_weight - 1.0) + 1.0
    weighted_bce = bce * weights
    return tf.reduce_mean(weighted_bce)


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal loss for handling class imbalance.
    Focuses on hard-to-classify examples.

    Parameters:
    -----------
    alpha : float
        Weighting factor (default: 0.25)
    gamma : float
        Focusing parameter (default: 2.0)
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    # Calculate focal loss
    cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    weight = alpha * y_true * tf.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma)

    focal_loss_value = weight * cross_entropy
    return tf.reduce_mean(focal_loss_value)


def combined_bce_l1_weights_loss(model, alpha=1.0, beta=0.01):
    """
    Combined loss: alpha * binary_crossentropy + beta * L1_regularization on model weights

    This applies L1 regularization to the model's trainable weights instead of predictions.
    This is more commonly used for feature selection and model sparsity.

    Parameters:
    -----------
    model : keras.Model
        The model whose weights will be regularized
    alpha : float
        Weight for binary cross-entropy term (default: 1.0)
    beta : float
        Weight for L1 regularization on weights (default: 0.01)

    Returns:
    --------
    loss_function : callable
        A loss function that takes (y_true, y_pred) as arguments

    Example:
    --------
    custom_loss = combined_bce_l1_weights_loss(model, alpha=1.0, beta=0.001)
    compile_model_with_custom_loss(model, custom_loss)
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        # Binary cross-entropy component
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        bce_loss = tf.reduce_mean(bce)

        # L1 regularization on model weights
        l1_reg = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in model.trainable_weights])

        # Combined loss
        total_loss = alpha * bce_loss + beta * l1_reg

        return total_loss

    return loss
compile_model_with_custom_loss(model, combined_bce_l1_weights_loss(model, alpha=1.0, beta=0.001))

# Compile with default loss (or use custom loss by calling the function above)
# To use custom loss, uncomment one of these:
# compile_model_with_custom_loss(model, weighted_binary_crossentropy)
compile_model_with_custom_loss(model, focal_loss)
# compile_model_with_custom_loss(model, lambda y_true, y_pred: focal_loss(y_true, y_pred, alpha=0.3, gamma=2.5))

# Combined BCE + L1 loss examples:
# compile_model_with_custom_loss(model, combined_bce_l1_loss(alpha=1.0, beta=0.01))

# Train the model
print("\nTraining the model...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=300,
    batch_size=16,
    validation_split=0.2,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
)

# Evaluate the model on test set
print("\nEvaluating the model...")
test_loss, test_accuracy, test_auc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# Make predictions
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.6).astype(int).flatten()

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model
model.save('examples/bank_classification_model5.h5')
print("\nModel saved to examples/bank_classification_model5.h5")