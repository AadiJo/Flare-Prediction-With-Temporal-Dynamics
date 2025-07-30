
import os
import json
import gc
import psutil  # For memory monitoring
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, TimeDistributed, LSTM, Dense, Dropout, 
                                     Input, concatenate, BatchNormalization, Bidirectional, 
                                     GlobalAveragePooling2D)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

# --- Print quiet/flare split at the very top ---
try:
    with np.load('processed_HED_data.npz', mmap_mode='r', allow_pickle=True) as data_map:
        y = data_map['y']
        unique, counts = np.unique(y, return_counts=True)
        split_dict = dict(zip(unique.astype(str), counts))
        print(f"Class distribution (quiet/flare) in full dataset: {split_dict}")
except Exception as e:
    print(f"Could not print class distribution at top: {e}")

# --- Configuration ---
BATCH_SIZE = 16
CHUNK_SIZE = 500
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Additional GPU memory setting

# --- GPU Setup ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Enabled memory growth for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU memory configuration failed: {e}")

def extract_scalar_features(metadata_chunk, scalar_keys):
    """Extract scalar features from metadata chunk."""
    batch_size = metadata_chunk.shape[0]
    timesteps = metadata_chunk.shape[1]
    
    # Initialize result array: (batch_size, timesteps, num_features)
    result = np.zeros((batch_size, timesteps, len(scalar_keys)), dtype=np.float32)
    
    for i in range(batch_size):
        for t in range(timesteps):
            timestep_data = metadata_chunk[i, t]  # Shape: (4,) array of dicts
            
            # Average across the 4 segments for this timestep
            values_for_timestep = []
            for segment_dict in timestep_data:
                if isinstance(segment_dict, dict):
                    segment_values = [float(segment_dict.get(key, 0.0) or 0.0) for key in scalar_keys]
                    values_for_timestep.append(segment_values)
            
            if values_for_timestep:
                result[i, t, :] = np.mean(values_for_timestep, axis=0)
            # else: already initialized to zeros
    
    return result

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def load_data_chunk(x_map, metadata_map, y_map, indices, scalar_keys, scaler=None):
    """Load a chunk of data and return processed X_images, X_scalars, y."""
    print(f"Loading chunk of {len(indices)} samples... Memory: {get_memory_usage():.1f}GB", end=" -> ")
    
    # Load raw data
    X_images = x_map[indices]  # Shape: (chunk_size, 12, 128, 128, 4)
    metadata = metadata_map[indices]  # Shape: (chunk_size, 12, 4)
    y = y_map[indices]  # Shape: (chunk_size,)
    
    # Extract scalar features
    X_scalars = extract_scalar_features(metadata, scalar_keys)  # Shape: (chunk_size, 12, 5)
    
    # Apply scaling if provided
    if scaler is not None:
        original_shape = X_scalars.shape
        X_scalars_flat = X_scalars.reshape(-1, X_scalars.shape[-1])
        X_scalars_scaled = scaler.transform(X_scalars_flat)
        X_scalars = X_scalars_scaled.reshape(original_shape)
    
    print(f"{get_memory_usage():.1f}GB")
    return X_images, X_scalars, y

# --- TSS Metric ---
@tf.keras.utils.register_keras_serializable()
class TSSMetric(tf.keras.metrics.Metric):
    def __init__(self, name='tss', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred_binary = tf.cast(tf.greater_equal(y_pred, self.threshold), tf.float32)
        
        self.true_positives.assign_add(tf.reduce_sum(y_true * y_pred_binary))
        self.false_negatives.assign_add(tf.reduce_sum(y_true * (1 - y_pred_binary)))
        self.false_positives.assign_add(tf.reduce_sum((1 - y_true) * y_pred_binary))
        self.true_negatives.assign_add(tf.reduce_sum((1 - y_true) * (1 - y_pred_binary)))

    def result(self):
        tpr = tf.math.divide_no_nan(self.true_positives, self.true_positives + self.false_negatives)
        fpr = tf.math.divide_no_nan(self.false_positives, self.false_positives + self.true_negatives)
        return tpr - fpr

    def reset_state(self):
        for var in self.variables:
            var.assign(0.0)
            
    def get_config(self):
        config = super().get_config()
        config.update({'threshold': self.threshold})
        return config

def create_cnnlstm_model(image_shape, scalar_shape):
    """Build CNN-LSTM model with proper input shapes."""
    print(f"Building model with input shapes:")
    print(f"  Images: {image_shape}")  # Should be (12, 128, 128, 4)
    print(f"  Scalars: {scalar_shape}")  # Should be (12, 5)
    
    # CNN for processing individual frames (removes timestep dimension)
    single_frame_shape = image_shape[1:]  # (128, 128, 4)
    
    cnn = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=single_frame_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        GlobalAveragePooling2D(),
        Dense(64, activation='relu'),
        Dropout(0.3)
    ], name="cnn_feature_extractor")

    # Model inputs
    image_input = Input(shape=image_shape, name="image_input")  # (12, 128, 128, 4)
    scalar_input = Input(shape=scalar_shape, name="scalar_input")  # (12, 5)

    # Process images through CNN for each timestep
    x = TimeDistributed(cnn, name="time_distributed_cnn")(image_input)
    
    # Process scalar features
    y = TimeDistributed(Dense(32, activation='relu'), name="scalar_dense1")(scalar_input)
    y = TimeDistributed(Dropout(0.2), name="scalar_dropout")(y)
    y = TimeDistributed(Dense(16, activation='relu'), name="scalar_dense2")(y)

    # Combine features
    combined = concatenate([x, y], axis=-1, name="feature_concatenation")
    combined = TimeDistributed(Dense(64, activation='relu'), name="combined_dense")(combined)
    combined = TimeDistributed(Dropout(0.3), name="combined_dropout")(combined)
    
    # LSTM processing
    lstm_out = Bidirectional(LSTM(32, return_sequences=False, dropout=0.3, recurrent_dropout=0.2), 
                            name="bidirectional_lstm")(combined)
    
    # Final layers
    z = Dense(32, activation='relu', name="final_dense1")(lstm_out)
    z = Dropout(0.4, name="final_dropout")(z)
    z = Dense(16, activation='relu', name="final_dense2")(z)
    output = Dense(1, activation='sigmoid', name="output")(z)
    
    return Model(inputs=[image_input, scalar_input], outputs=output)

def train_in_chunks(model, x_map, metadata_map, y_map, train_indices, val_indices, 
                    scalar_keys, scaler, class_weight_dict, classification_threshold, callbacks):
    """Train model by loading data in chunks and validating manually."""
    
    print(f"Training on {len(train_indices)} samples in chunks of {CHUNK_SIZE}")
    print(f"Validation will use a classification threshold of {classification_threshold}")

    history = {'loss': [], 'accuracy': [], 'tss': [], 'val_loss': [], 'val_accuracy': [], 'val_tss': []}
    best_val_tss = -1.0
    patience_counter = 0
    patience = 15 # Corresponds to EarlyStopping patience

    for epoch in range(50):  # Max epochs
        print(f"\nEpoch {epoch + 1}/50")
        
        # --- Training Phase ---
        np.random.shuffle(train_indices)
        epoch_train_losses = []
        epoch_train_accuracies = []
        epoch_train_tss = []
        
        for chunk_start in range(0, len(train_indices), CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(train_indices))
            chunk_indices = train_indices[chunk_start:chunk_end]
            
            chunk_X_images, chunk_X_scalars, chunk_y = load_data_chunk(
                x_map, metadata_map, y_map, chunk_indices, scalar_keys, scaler
            )
            
            chunk_history = model.fit(
                [chunk_X_images, chunk_X_scalars], chunk_y,
                batch_size=BATCH_SIZE, epochs=1, verbose=0, class_weight=class_weight_dict
            )
            
            epoch_train_losses.append(chunk_history.history['loss'][0])
            epoch_train_accuracies.append(chunk_history.history['accuracy'][0])
            epoch_train_tss.append(chunk_history.history['tss'][0])
            
            del chunk_X_images, chunk_X_scalars, chunk_y, chunk_history
            gc.collect()
        
        # --- Validation Phase ---
        val_preds, val_labels, val_losses = [], [], []
        for val_chunk_start in range(0, len(val_indices), CHUNK_SIZE):
            val_chunk_indices = val_indices[val_chunk_start:val_chunk_start + CHUNK_SIZE]
            
            val_chunk_X_images, val_chunk_X_scalars, val_chunk_y = load_data_chunk(
                x_map, metadata_map, y_map, val_chunk_indices, scalar_keys, scaler
            )
            
            preds = model.predict([val_chunk_X_images, val_chunk_X_scalars], batch_size=BATCH_SIZE, verbose=0)
            val_preds.append(preds)
            val_labels.append(val_chunk_y)

            # Get loss on the validation chunk
            val_loss = model.evaluate([val_chunk_X_images, val_chunk_X_scalars], val_chunk_y, batch_size=BATCH_SIZE, verbose=0)[0]
            val_losses.append(val_loss)

            del val_chunk_X_images, val_chunk_X_scalars, val_chunk_y, preds
            gc.collect()

        # --- Calculate Metrics for the Full Validation Set ---
        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        val_pred_classes = (val_preds > classification_threshold).astype(int)
        
        cm = confusion_matrix(val_labels, val_pred_classes)
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        current_val_tss = tpr - fpr
        current_val_accuracy = (tp + tn) / (tp + tn + fp + fn)

        # --- Log Metrics ---
        history['loss'].append(np.mean(epoch_train_losses))
        history['accuracy'].append(np.mean(epoch_train_accuracies))
        history['tss'].append(np.mean(epoch_train_tss)) # Train TSS from Keras metric
        history['val_loss'].append(np.mean(val_losses))
        history['val_accuracy'].append(current_val_accuracy) # Val accuracy from manual calculation
        history['val_tss'].append(current_val_tss) # Val TSS from manual calculation
        
        print(f"Train - Loss: {history['loss'][-1]:.4f}, Acc: {history['accuracy'][-1]:.4f}, TSS: {history['tss'][-1]:.4f}")
        print(f"Val   - Loss: {history['val_loss'][-1]:.4f}, Acc: {history['val_accuracy'][-1]:.4f}, TSS: {history['val_tss'][-1]:.4f} (Threshold: {classification_threshold})")
        print("Validation Confusion Matrix:\n", cm)

        # --- Checkpointing and Early Stopping ---
        if current_val_tss > best_val_tss:
            print(f"✅ New best model found! val_tss improved from {best_val_tss:.4f} to {current_val_tss:.4f}. Saving model...")
            best_val_tss = current_val_tss
            model.save('best_model.keras')
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"val_tss did not improve. Best was {best_val_tss:.4f}. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered after {patience} epochs with no improvement.")
            break
            
    # Load the best performing model before returning
    print(f"\nTraining finished. Loading best model with val_tss: {best_val_tss:.4f}")
    model = tf.keras.models.load_model('best_model.keras', custom_objects={'TSSMetric': TSSMetric})
    return history, model

def main():
    print("Loading data...")
    data_map = np.load('processed_HED_data.npz', mmap_mode='r', allow_pickle=True)
    x_map = data_map['X']  # (2857, 12, 128, 128, 4)
    y = data_map['y']      # (2857,)
    metadata_map = data_map['metadata']  # (2857, 12, 4)
    
    print(f"Data shapes: X={x_map.shape}, y={y.shape}, metadata={metadata_map.shape}")
    
    scalar_keys = ['USFLUX', 'MEANSHR', 'TOTUSJZ', 'SHRGT45', 'TOTPOT']
    
    # Split indices chronologically
    print("Splitting data chronologically...")

    # This assumes your data in 'processed_HED_data.npz' is already sorted by time.
    # If not, you must sort X, y, and metadata by their timestamps before this step.
    n_samples = len(y)
    all_indices = np.arange(n_samples)

    # Define split proportions (e.g., 70% train, 15% validation, 15% test)
    train_end = int(n_samples * 0.7)
    val_end = int(n_samples * 0.85)

    train_indices = all_indices[:train_end]
    val_indices = all_indices[train_end:val_end]
    test_indices = all_indices[val_end:]

    # Do NOT shuffle the indices.
    print(f"Chronological Split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")
    
    # Fit scaler on sample
    print("Fitting scaler...")
    sample_size = min(200, len(train_indices))  # Even smaller sample for scaler
    sample_indices = np.random.choice(train_indices, size=sample_size, replace=False)
    sample_X_images, sample_X_scalars, _ = load_data_chunk(
        x_map, metadata_map, y, sample_indices, scalar_keys, scaler=None
    )
    
    # Fit scaler on flattened scalar features
    scaler = StandardScaler()
    scaler.fit(sample_X_scalars.reshape(-1, sample_X_scalars.shape[-1]))
    print(f"Scaler fitted on {sample_X_scalars.shape[0] * sample_X_scalars.shape[1]} samples")
    
    # Clean up sample data immediately
    del sample_X_images, sample_X_scalars
    gc.collect()
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y[train_indices])
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")
    
    # --- ADD a single source of truth for the threshold ---
    CLASSIFICATION_THRESHOLD = 0.3
    
    # Build model - use known shapes
    image_shape = (12, 128, 128, 4)  # timesteps, height, width, channels
    scalar_shape = (12, 5)           # timesteps, features
    
    model = create_cnnlstm_model(image_shape, scalar_shape)
    
    # --- UPDATE model.compile() ---
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', TSSMetric(name='tss', threshold=CLASSIFICATION_THRESHOLD)]
    )
    model.summary()
    
    # --- REMOVE old callbacks ---
    # The new train_in_chunks function handles this logic manually.
    callbacks = [] 
    
    # --- UPDATE the call to train_in_chunks ---
    print("\nStarting training...")
    history, model = train_in_chunks(model, x_map, metadata_map, y, train_indices, val_indices,
                                     scalar_keys, scaler, class_weight_dict, 
                                     CLASSIFICATION_THRESHOLD, callbacks)
    
    # --- UPDATE test evaluation for clarity ---
    print("\nEvaluating on test set with the best model...")
    all_test_preds = []
    all_test_labels = []

    for test_chunk_start in range(0, len(test_indices), CHUNK_SIZE):
        test_chunk_indices = test_indices[test_chunk_start:test_chunk_start + CHUNK_SIZE]
        test_X_images, test_X_scalars, test_y = load_data_chunk(
            x_map, metadata_map, y, test_chunk_indices, scalar_keys, scaler
        )
        preds = model.predict([test_X_images, test_X_scalars], batch_size=BATCH_SIZE, verbose=0)
        all_test_preds.append(preds)
        all_test_labels.append(test_y)
        del test_X_images, test_X_scalars, test_y, preds
        gc.collect()

    all_test_preds = np.concatenate(all_test_preds)
    all_test_labels = np.concatenate(all_test_labels)
    predicted_classes = (all_test_preds > CLASSIFICATION_THRESHOLD).astype(int)

    print(f"\n--- Final Test Results (Threshold = {CLASSIFICATION_THRESHOLD}) ---")
    cm = confusion_matrix(all_test_labels, predicted_classes)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(all_test_labels, predicted_classes, digits=4))

    # Manually calculate final TSS from the confusion matrix for final reporting
    tn, fp, fn, tp = cm.ravel()
    final_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    final_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    final_tss = final_tpr - final_fpr
    print(f"\nFinal Test TSS: {final_tss:.4f}")

    # Save history
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("\nTraining completed!")

if __name__ == "__main__":
    main()