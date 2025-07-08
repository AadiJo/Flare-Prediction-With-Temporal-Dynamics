import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, TimeDistributed, LSTM, Dense, Dropout
import os
from datetime import datetime
import json

def load_latest_model(models_dir="models"):
    """Load the most recent model from the models directory."""
    if not os.path.exists(models_dir):
        print(f"Models directory '{models_dir}' not found.")
        return None
    
    # List all model files
    model_files = [f for f in os.listdir(models_dir) 
                  if os.path.isfile(os.path.join(models_dir, f)) and 
                  f.startswith("solar_flare_model_") and 
                  (f.endswith(".keras") or f.endswith(".h5"))]
    
    if not model_files:
        print("No saved models found.")
        return None
    
    # Find the most recent model file
    latest_model_file = max(model_files)
    model_path = os.path.join(models_dir, latest_model_file)
    
    print(f"Loading model from: {model_path}")
    return load_model(model_path)

print("Loading pre-processed data...")

# Loads pre-processed solar data from a compressed NumPy archive file (x is input features, y is the target labels)
with np.load('processed_solar_data.npz') as data:
    X = data['X']
    y = data['y']

print(f"Data loaded successfully. Shapes: X={X.shape}, y={y.shape}")

# Checks if the data is in the expected format (and that we have data for 2 diff classifications)
unique_classes = np.unique(y)
print(f"Found {len(unique_classes)} unique classes in the labels: {unique_classes}")

# Checks, and fixes, if the data is imbalanced
if len(unique_classes) > 1:
    print("Balancing the dataset...")
    X_reshaped = X.reshape(X.shape[0], -1)
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_reshaped, y)
    X_balanced = X_resampled.reshape(-1, X.shape[1], X.shape[2], X.shape[3], X.shape[4])
    print(f"Dataset balanced. New shapes: X={X_balanced.shape}, y={y_resampled.shape}")

    print("Splitting data into training, validation, and test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_balanced, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Training set size:   {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size:       {len(X_test)}")

else:
    print("\nWARNING: Only one class found in the dataset. Skipping balancing and splitting.")
    print("Proceeding with the full dataset for a preliminary architecture test.\n")
    X_train, y_train = X, y
    X_val, y_val = np.array([]), np.array([])
    X_test, y_test = np.array([]), np.array([])

print("-" * 40)

model_input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4])

# Function to create a CNN-LSTM model for solar flare prediction
# The model uses TimeDistributed layers to apply CNNs to each time step of the input sequence
def create_cnnlstm_model(input_shape):
    model = Sequential(name="Solar_Flare_Predictor_v1")
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))
    # Flatten the output of the CNN layers before passing to LSTM
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.5))
    # Output layer for binary classification (solar flare or not)
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
# TRAINING THE MODEL
print("Creating the CNN-LSTM model...")
model = create_cnnlstm_model(model_input_shape)
print("Model Architecture Summary:")
model.summary()
print("-" * 40)

print("Starting model training...")

validation_data = (X_val, y_val) if X_val.size > 0 else None

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=8,
    validation_data=validation_data,
    verbose=1
)

print("\nModel training completed!")
print("-" * 40)

# Save the trained model and training history
# Create models directory if it doesn't exist
models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)

# Generate a timestamp for the model name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"solar_flare_model_{timestamp}"

# Add the .keras extension to the model path (recommended format)
model_path = os.path.join(models_dir, f"{model_name}.keras")

# Save the entire model (architecture + weights + optimizer state)
model.save(model_path)
print(f"Model saved to: {model_path}")

# Create a folder for additional model artifacts
model_artifacts_dir = os.path.join(models_dir, model_name)
os.makedirs(model_artifacts_dir, exist_ok=True)

# Save training history in the artifacts directory
history_dict = history.history
with open(os.path.join(model_artifacts_dir, 'training_history.json'), 'w') as f:
    json.dump(history_dict, f)
print(f"Training history saved to: {os.path.join(model_artifacts_dir, 'training_history.json')}")

# Save model weights separately (optional) - with correct file extension
weights_path = os.path.join(model_artifacts_dir, 'model.weights.h5')
model.save_weights(weights_path)
print(f"Model weights saved to: {weights_path}")

# TESTING THE MODEL
if X_test.size > 0:
    print("Evaluating model performance on the unseen test set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Set Loss: {loss:.4f}")
else:
    print("No test set to evaluate. Evaluation skipped.")