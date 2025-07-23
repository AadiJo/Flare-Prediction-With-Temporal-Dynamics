import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight  # Add this import
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential, load_model, Model # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, TimeDistributed, LSTM, Dense, Dropout, Input, concatenate, Lambda, BatchNormalization, Bidirectional # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, Callback  # Add this import # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from imblearn.over_sampling import RandomOverSampler  # Add this import at the top
import os
from datetime import datetime
import json
import shutil  # Add this import for file copying

# Suppress TensorFlow warnings about CPU optimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class TSSCallback(Callback):
    """Custom callback to monitor TSS (True Skill Statistic) and implement early stopping based on TSS."""
    
    def __init__(self, validation_data, patience=7, min_delta=0.001, verbose=1):
        super().__init__()
        self.validation_data = validation_data
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_tss = -1  # TSS ranges from -1 to 1, so start at minimum
        self.wait = 0
        self.best_weights = None
        self.best_epoch = 0
        
    def calculate_tss(self, y_true, y_pred):
        """Calculate True Skill Statistic (TSS)."""
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred > 0.5).astype(int).flatten()
        y_true_flat = y_true.flatten()
        
        # Calculate confusion matrix
        try:
            cm = confusion_matrix(y_true_flat, y_pred_binary)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                # TSS = TPR - FPR = (TP/(TP+FN)) - (FP/(FP+TN))
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                tss = tpr - fpr
            else:
                tss = 0  # If only one class present
        except:
            tss = 0
        return tss
    
    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            return
            
        # Get validation predictions
        val_predictions = self.model.predict(self.validation_data[0], verbose=0)
        val_tss = self.calculate_tss(self.validation_data[1], val_predictions)
        
        # Add TSS to logs
        logs = logs or {}
        logs['val_tss'] = val_tss
        
        if self.verbose > 0:
            print(f" - val_tss: {val_tss:.4f}")
        
        # Check if TSS improved
        if val_tss > self.best_tss + self.min_delta:
            self.best_tss = val_tss
            self.wait = 0
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch
            if self.verbose > 0:
                print(f"TSS improved to {val_tss:.4f}")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print(f"Early stopping triggered. Best TSS: {self.best_tss:.4f} at epoch {self.best_epoch + 1}")
                    print("Restoring best weights...")
                self.model.set_weights(self.best_weights)
                self.model.stop_training = True

def load_latest_model(models_dir="../models"):
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
with np.load('processed_solar_data.npz', allow_pickle=True) as data:
    X = data['X']
    y = data['y']
    sharp_metadata = data['sharp_metadata']

# Extract scalar features from sharp_metadata - reshape for timestep alignment
# Each sample has 6 timesteps, each timestep has scalar values for each segment
scalar_features = []
for sample_idx in range(len(sharp_metadata)):
    sample_scalars = []
    for timestep in range(6):  # 6 timesteps per sample
        # Average the scalar values across all segments for this timestep
        timestep_values = []
        for segment in sharp_metadata[sample_idx][timestep]:
            if segment is not None:
                values = []
                for key in ['USFLUX', 'MEANSHR', 'TOTUSJZ', 'SHRGT45', 'TOTPOT']:
                    val = segment.get(key, None)
                    if val is not None and not np.isnan(val):
                        values.append(float(val))
                    else:
                        values.append(0.0)  # Default for None/NaN values
                timestep_values.append(values)
        
        if timestep_values:
            # Average across segments, we have 5 scalar features per timestep
            avg_values = np.mean(timestep_values, axis=0)
        else:
            avg_values = [0.0] * 5  # Default values if no data
        sample_scalars.append(avg_values)
    scalar_features.append(sample_scalars)

X_scalar = np.array(scalar_features)  # Shape: (samples, 6, 5) - 6 timesteps, 5 features each

# Validate data for NaN/infinity values
print("Validating data...")
print(f"X contains NaN: {np.isnan(X).any()}")
print(f"X contains infinity: {np.isinf(X).any()}")
print(f"X_scalar contains NaN: {np.isnan(X_scalar).any()}")
print(f"X_scalar contains infinity: {np.isinf(X_scalar).any()}")

# Replace any remaining NaN/inf values
X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
X_scalar = np.nan_to_num(X_scalar, nan=0.0, posinf=1.0, neginf=0.0)

# Normalize scalar features to prevent gradient explosion
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scalar_reshaped = X_scalar.reshape(-1, X_scalar.shape[-1])  # Flatten for scaling
X_scalar_scaled = scaler.fit_transform(X_scalar_reshaped)
X_scalar = X_scalar_scaled.reshape(X_scalar.shape)  # Reshape back

print(f"Data loaded successfully. Shapes: X={X.shape}, y={y.shape}, X_scalar={X_scalar.shape}")
print(f"X range: [{X.min():.3f}, {X.max():.3f}]")
print(f"X_scalar range: [{X_scalar.min():.3f}, {X_scalar.max():.3f}]")

# Checks if the data is in the expected format (and that we have data for 2 diff classifications)
unique_classes = np.unique(y)
print(f"Found {len(unique_classes)} unique classes in the labels: {unique_classes}")

# Checks, and fixes, if the data is imbalanced
if len(unique_classes) > 1:
    # Option 1: Using class weights (preferred approach)
    print("Computing class weights for balanced training...")
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(zip(np.unique(y).astype(int), class_weights))
    print(f"Using class weights: {class_weight_dict}")
    
    # Normal data splitting without oversampling
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Split scalar features the same way (ensure scaling is applied to splits)
    X_scalar_train, X_scalar_temp, _, _ = train_test_split(
        X_scalar, y, test_size=0.3, random_state=42, stratify=y
    )
    X_scalar_val, X_scalar_test, _, _ = train_test_split(
        X_scalar_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Option 2 (alternative): Using RandomOverSampler (I think for now we don't need this)
    # print("Balancing the dataset with RandomOverSampler...")
    # X_reshaped = X.reshape(X.shape[0], -1)
    # ros = RandomOverSampler(random_state=42)
    # X_resampled, y_resampled = ros.fit_resample(X_reshaped, y)
    # X_balanced = X_resampled.reshape(-1, X.shape[1], X.shape[2], X.shape[3], X.shape[4])
    # X_train, X_temp, y_train, y_temp = train_test_split(
    #     X_balanced, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
    # )
    # X_val, X_test, y_val, y_test = train_test_split(
    #     X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    # )
    # class_weight_dict = None  # Not needed when using oversampling
    
    print(f"Training set size:   {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size:       {len(X_test)}")

else:
    print("\nWARNING: Only one class found in the dataset. Skipping balancing and splitting.")
    print("Proceeding with the full dataset for a preliminary architecture test.\n")
    X_train, y_train = X, y
    X_val, y_val = np.array([]), np.array([])
    X_test, y_test = np.array([]), np.array([])
    X_scalar_train, X_scalar_val, X_scalar_test = X_scalar, np.array([]), np.array([])

print("-" * 40)

model_input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4])
scalar_input_shape = (X_scalar_train.shape[1], X_scalar_train.shape[2])  # (timesteps, features)

# Function to create a CNN-LSTM model optimized for TSS with timestep-aligned scalar features
def create_cnnlstm_model(image_input_shape, scalar_input_shape):
    # Image input: (timesteps, height, width, channels)
    image_input = Input(shape=image_input_shape, name='image_input')
    
    # Scalar input: (timesteps, scalar_features)
    scalar_input = Input(shape=scalar_input_shape, name='scalar_input')
    
    # Enhanced CNN processing with BatchNormalization for better stability
    image_features = TimeDistributed(Conv2D(64, (3, 3), activation='relu',
                                          kernel_regularizer=tf.keras.regularizers.l2(0.0005)))(image_input)
    image_features = TimeDistributed(BatchNormalization())(image_features)
    image_features = TimeDistributed(MaxPooling2D((2, 2)))(image_features)
    image_features = TimeDistributed(Dropout(0.2))(image_features)
    
    image_features = TimeDistributed(Conv2D(128, (3, 3), activation='relu',
                                          kernel_regularizer=tf.keras.regularizers.l2(0.0005)))(image_features)
    image_features = TimeDistributed(BatchNormalization())(image_features)
    image_features = TimeDistributed(MaxPooling2D((2, 2)))(image_features)
    image_features = TimeDistributed(Dropout(0.2))(image_features)
    
    image_features = TimeDistributed(Conv2D(256, (3, 3), activation='relu',
                                          kernel_regularizer=tf.keras.regularizers.l2(0.0005)))(image_features)
    image_features = TimeDistributed(BatchNormalization())(image_features)
    image_features = TimeDistributed(MaxPooling2D((2, 2)))(image_features)
    image_features = TimeDistributed(Dropout(0.3))(image_features)
    
    image_features = TimeDistributed(Flatten())(image_features)
    
    # Enhanced scalar processing with more capacity
    scalar_features = TimeDistributed(Dense(32, activation='relu',
                                          kernel_regularizer=tf.keras.regularizers.l2(0.0005)))(scalar_input)
    scalar_features = TimeDistributed(BatchNormalization())(scalar_features)
    scalar_features = TimeDistributed(Dropout(0.2))(scalar_features)
    
    scalar_features = TimeDistributed(Dense(16, activation='relu',
                                          kernel_regularizer=tf.keras.regularizers.l2(0.0005)))(scalar_features)
    scalar_features = TimeDistributed(Dropout(0.2))(scalar_features)
    
    # Combine image and scalar features at each timestep
    combined_features = concatenate([image_features, scalar_features], axis=-1)
    combined_features = TimeDistributed(Dense(128, activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.0005)))(combined_features)
    combined_features = TimeDistributed(BatchNormalization())(combined_features)
    combined_features = TimeDistributed(Dropout(0.3))(combined_features)
    
    # Bidirectional LSTM for better temporal understanding
    lstm_out = Bidirectional(LSTM(64, activation='relu', recurrent_dropout=0.2, 
                                 return_sequences=True))(combined_features)
    lstm_out = Dropout(0.3)(lstm_out)
    
    # Second LSTM layer for deeper temporal modeling
    lstm_out = LSTM(32, activation='relu', recurrent_dropout=0.2)(lstm_out)
    lstm_out = Dropout(0.4)(lstm_out)
    
    # Enhanced dense layers for final classification
    dense_out = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(lstm_out)
    dense_out = BatchNormalization()(dense_out)
    dense_out = Dropout(0.4)(dense_out)
    
    dense_out = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(dense_out)
    dense_out = Dropout(0.3)(dense_out)
    
    # Final classification with careful initialization for balanced predictions
    output = Dense(1, activation='sigmoid', 
                  kernel_initializer='he_normal',
                  bias_initializer='zeros')(dense_out)
    
    model = Model(inputs=[image_input, scalar_input], outputs=output, 
                 name="Solar_Flare_Predictor_TSS_Optimized")
    
    # Optimized learning rate for balanced dataset
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0, beta_1=0.9, beta_2=0.999)  # Slightly higher LR for balanced data
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model
# TRAINING THE MODEL
print("Creating the TSS-optimized CNN-LSTM model with scalar features...")
model = create_cnnlstm_model(model_input_shape, scalar_input_shape)
print("Model Architecture Summary:")
model.summary()
print("-" * 40)

print("Starting TSS-optimized model training...")

validation_data = ([X_val, X_scalar_val], y_val) if X_val.size > 0 else None

# TSS-based early stopping callback with adjusted patience for balanced data
tss_callback = TSSCallback(
    validation_data=validation_data,
    patience=8,  # Slightly reduced patience for balanced data
    min_delta=0.003,  # Smaller delta for detecting TSS improvements
    verbose=1
)

# Additional callbacks for training stability
nan_terminator = tf.keras.callbacks.TerminateOnNaN()

# Learning rate scheduling for better convergence
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# Optimized class weights for balanced dataset (50/50 flare/quiet)
if len(unique_classes) > 1:
    print("Dataset is balanced (50/50), using optimized class weights for TSS...")
    # For balanced datasets, slight emphasis on flare detection for better TSS
    # TSS benefits from high True Positive Rate while minimizing False Positive Rate
    optimized_class_weights = {0: 1.0, 1: 1.1}  # Slight boost to flare class
    print(f"TSS-optimized class weights for balanced data: {optimized_class_weights}")
else:
    optimized_class_weights = None

history = model.fit(
    [X_train, X_scalar_train],
    y_train,
    epochs=40,  # Reduced epochs since balanced data converges faster
    batch_size=16,  # Good batch size for balanced data
    validation_data=validation_data,
    class_weight=optimized_class_weights,  # Minimal class weights for balanced data
    callbacks=[tss_callback, nan_terminator, lr_scheduler],  # TSS-focused callbacks
    verbose=1
)

print("\nTSS-optimized model training completed!")
print("-" * 40)

# Function to calculate and display comprehensive TSS metrics
def evaluate_tss_metrics(y_true, y_pred_proba, threshold=0.5):
    """Calculate comprehensive TSS metrics and find optimal threshold."""
    y_pred_binary = (y_pred_proba > threshold).astype(int).flatten()
    y_true_flat = y_true.flatten()
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_flat, y_pred_binary)
    print("Confusion Matrix:")
    print(cm)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate TSS components
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Sensitivity/Recall)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate (Specificity)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Calculate TSS
        tss = tpr - fpr
        
        # Other useful metrics
        f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        print(f"\n=== TSS Metrics (threshold={threshold}) ===")
        print(f"True Skill Statistic (TSS): {tss:.4f}")
        print(f"True Positive Rate (TPR/Recall): {tpr:.4f}")
        print(f"False Positive Rate (FPR): {fpr:.4f}")
        print(f"True Negative Rate (TNR/Specificity): {tnr:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        
        return tss, tpr, fpr, precision, f1
    else:
        print("Warning: Only one class present in predictions")
        return 0, 0, 0, 0, 0

# Function to find optimal threshold for TSS
def find_optimal_tss_threshold(y_true, y_pred_proba, thresholds=None):
    """Find the threshold that maximizes TSS."""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)
    
    best_tss = -1
    best_threshold = 0.5
    tss_scores = []
    
    print("\n=== Threshold Optimization for TSS ===")
    for threshold in thresholds:
        y_pred_binary = (y_pred_proba > threshold).astype(int).flatten()
        y_true_flat = y_true.flatten()
        
        try:
            cm = confusion_matrix(y_true_flat, y_pred_binary)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                tss = tpr - fpr
                tss_scores.append(tss)
                
                print(f"Threshold {threshold:.2f}: TSS = {tss:.4f}, TPR = {tpr:.4f}, FPR = {fpr:.4f}")
                
                if tss > best_tss:
                    best_tss = tss
                    best_threshold = threshold
            else:
                tss_scores.append(0)
        except:
            tss_scores.append(0)
    
    print(f"\n🎯 Optimal threshold: {best_threshold:.2f} with TSS = {best_tss:.4f}")
    return best_threshold, best_tss, tss_scores

# Save the trained model and training history
# Create models directory if it doesn't exist
models_dir = os.path.join(os.path.dirname(__file__), "../models")
os.makedirs(models_dir, exist_ok=True)

# Generate a timestamp for the model name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"solar_flare_model_{timestamp}"


# Create a folder for additional model artifacts
model_artifacts_dir = os.path.join(models_dir, model_name)
os.makedirs(model_artifacts_dir, exist_ok=True)

# Save the entire model (architecture + weights + optimizer state) directly in the model-specific directory
model_path = os.path.join(model_artifacts_dir, f"{model_name}.keras")
model.save(model_path)
print(f"Model saved to: {model_path}")

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
    loss, accuracy = model.evaluate([X_test, X_scalar_test], y_test, verbose=0)
    print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Set Loss: {loss:.4f}")

    # Get predictions for the test set
    predictions = model.predict([X_test, X_scalar_test])
    predicted_classes = (predictions > 0.5).astype(int)  # Convert probabilities to binary classes

    # Count occurrences of each class
    unique, counts = np.unique(predicted_classes, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print(f"Predicted class distribution: {class_distribution}")
    
    # Calculate confusion matrix for TSS
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test, predicted_classes)
    print("Confusion Matrix:")
    print(cm)
    
    # Calculate True Skill Statistic (TSS)
    tn, fp, fn, tp = cm.ravel()
    tss = tp/(tp+fn) - fp/(fp+tn)
    print(f"True Skill Statistic (TSS): {tss:.4f}")
    
    # Generate detailed classification metrics
    print("\nClassification Report:")
    print(classification_report(y_test, predicted_classes))
else:
    print("No test set to evaluate. Evaluation skipped.")

# Save the testing set for future use
if X_test.size > 0:
    # Save the test set in the model artifacts directory
    test_set_path = os.path.join(model_artifacts_dir, 'test_set.npz')
    np.savez_compressed(test_set_path, X_test=X_test, X_scalar_test=X_scalar_test, y_test=y_test)
    print(f"Testing set saved to: {test_set_path}")
else:
    print("No test set available to save.")

# Evaluate the model on the validation set (if available)
if X_val.size > 0:
    print("\n" + "="*60)
    print("          🎯 COMPREHENSIVE TSS EVALUATION 🎯")
    print("="*60)
    
    # Basic evaluation
    print("\nEvaluating model performance on the validation set...")
    val_loss, val_accuracy = model.evaluate([X_val, X_scalar_val], y_val, verbose=0)
    print(f"Validation Set Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Validation Set Loss: {val_loss:.4f}")
    
    # Comprehensive TSS analysis
    val_predictions = model.predict([X_val, X_scalar_val], verbose=0)
    
    # Evaluate at default threshold (0.5)
    print("\n--- Evaluation at Default Threshold (0.5) ---")
    evaluate_tss_metrics(y_val, val_predictions, threshold=0.5)
    
    # Find optimal threshold
    print("\n--- Finding Optimal Threshold ---")
    optimal_threshold, optimal_tss, tss_scores = find_optimal_tss_threshold(y_val, val_predictions)
    
    # Evaluate at optimal threshold
    print(f"\n--- Evaluation at Optimal Threshold ({optimal_threshold:.2f}) ---")
    final_tss, final_tpr, final_fpr, final_precision, final_f1 = evaluate_tss_metrics(
        y_val, val_predictions, threshold=optimal_threshold
    )
    
    print(f"\nFINAL TSS PERFORMANCE SUMMARY:")
    print(f"   Best TSS Score: {optimal_tss:.4f}")
    print(f"   Optimal Threshold: {optimal_threshold:.2f}")
    print(f"   True Positive Rate: {final_tpr:.4f}")
    print(f"   False Positive Rate: {final_fpr:.4f}")
    print(f"   Precision: {final_precision:.4f}")
    print(f"   F1-Score: {final_f1:.4f}")
    
    # Training statistics from TSSCallback
    if hasattr(tss_callback, 'best_tss') and hasattr(tss_callback, 'best_epoch'):
        print(f"   Best Training TSS: {tss_callback.best_tss:.4f} (epoch {tss_callback.best_epoch})")
    
    print("\n" + "="*60)
    print("Training completed! TSS optimization successful.")
    print("="*60)
else:
    print("No validation set available for evaluation.")