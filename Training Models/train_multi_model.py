import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, TimeDistributed, LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import os
from datetime import datetime
import json
import shutil
<<<<<<< HEAD
=======
import random
>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11

# Define channel names corresponding to the 4 magnetic field components
CHANNEL_NAMES = ['Bp', 'Br', 'Bt', 'continuum']

<<<<<<< HEAD
=======
# Add a global flag to toggle reproducibility
enable_reproducibility = True

if enable_reproducibility:
    # Set all random seeds
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # For TensorFlow 2.19.0, use tf.config.experimental instead of environment variable
    tf.config.experimental.enable_op_determinism()
    
    print("Reproducibility enabled with fixed seeds.")
else:
    print("Reproducibility is disabled. Randomness may vary.")

>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11
def create_cnnlstm_model(input_shape, model_name):
    """Create a CNN-LSTM model for a specific channel."""
    model = Sequential(name=f"Solar_Flare_Predictor_{model_name}")
    
    # Add L2 regularization to Conv2D layers
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)), 
                             input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))
    
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001))))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))
    
    # Flatten the output of the CNN layers before passing to LSTM
    model.add(TimeDistributed(Flatten()))
    
    # Add recurrent_dropout to LSTM
    model.add(LSTM(50, activation='relu', recurrent_dropout=0.2))
    model.add(Dropout(0.5))
    
    # Output layer for binary classification (solar flare or not)
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_channel_model(X_channel, y, channel_name, channel_idx):
    """Train a model for a specific channel."""
    print(f"\n{'='*60}")
    print(f"Training model for channel {channel_idx}: {channel_name}")
    print(f"{'='*60}")
    
    # Check if we have multiple classes
    unique_classes = np.unique(y)
    print(f"Found {len(unique_classes)} unique classes in the labels: {unique_classes}")
    
    if len(unique_classes) <= 1:
        print(f"WARNING: Only one class found for {channel_name}. Skipping training.")
        return None, None, None
    
    # Compute class weights for balanced training
    print("Computing class weights for balanced training...")
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(zip(np.unique(y).astype(int), class_weights))
    print(f"Using class weights: {class_weight_dict}")
    
    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_channel, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Training set size:   {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size:       {len(X_test)}")
    
    # Create the model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)  # Single channel
    model = create_cnnlstm_model(input_shape, channel_name)
    
    print(f"\nModel Architecture Summary for {channel_name}:")
    model.summary()
    
    # Set up early stopping
    early_stopping = EarlyStopping(
<<<<<<< HEAD
        monitor='val_loss',
        patience=5,
=======
        monitor='val_tss',
        patience=8,
>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model
    print(f"\nStarting training for {channel_name}...")
    history = model.fit(
        X_train,
        y_train,
<<<<<<< HEAD
        epochs=10,
        batch_size=8,
=======
        epochs=20,
        batch_size=16,
>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate on test set
    if len(X_test) > 0:
        print(f"\nEvaluating {channel_name} model on test set...")
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
        print(f"Test Set Loss: {loss:.4f}")
        
        # Get predictions for TSS calculation
        predictions = model.predict(X_test)
        predicted_classes = (predictions > 0.5).astype(int)
        
        # Calculate TSS
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, predicted_classes)
        print("Confusion Matrix:")
        print(cm)
        
        if cm.shape == (2, 2):  # Only calculate TSS for binary classification
            tn, fp, fn, tp = cm.ravel()
            tss = tp/(tp+fn) - fp/(fp+tn)
            print(f"True Skill Statistic (TSS): {tss:.4f}")
    
    return model, history, (X_test, y_test)

def train_channel_model_with_splits(X_train, X_val, X_test, y_train, y_val, y_test, channel_name, channel_idx):
    """Train a model for a specific channel using pre-split data."""
    print(f"\n{'='*60}")
    print(f"Training model for channel {channel_idx}: {channel_name}")
    print(f"{'='*60}")
    
    # Check if we have multiple classes
    unique_classes = np.unique(y_train)
    print(f"Found {len(unique_classes)} unique classes in the labels: {unique_classes}")
    
    if len(unique_classes) <= 1:
        print(f"WARNING: Only one class found for {channel_name}. Skipping training.")
        return None, None, None
    
    # Compute class weights for balanced training
    print("Computing class weights for balanced training...")
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train).astype(int), class_weights))
    print(f"Using class weights: {class_weight_dict}")
    
    print(f"Training set size:   {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size:       {len(X_test)}")
    
    # Create the model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)  # Single channel
    model = create_cnnlstm_model(input_shape, channel_name)
    
    print(f"\nModel Architecture Summary for {channel_name}:")
    model.summary()
    
<<<<<<< HEAD
    # Set up early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
=======
    # Custom callback to monitor val_tss and stop training when it does not improve
    class ValTSSMonitor(tf.keras.callbacks.Callback):
        def __init__(self, validation_data, patience=5):
            super().__init__()
            self.validation_data = validation_data
            self.patience = patience
            self.best_tss = -np.inf
            self.wait = 0
            self.best_weights = None
            self.best_epoch = 0

        def on_epoch_end(self, epoch, logs=None):
            val_pred = self.model.predict(self.validation_data[0], verbose=0)
            val_pred_classes = (val_pred > 0.5).astype(int)
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(self.validation_data[1], val_pred_classes)
            tss = None
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                tss = tp/(tp+fn) - fp/(fp+tn)
            else:
                tss = 0.0
            logs = logs if logs is not None else {}
            logs['val_tss'] = tss
            print(f"Epoch {epoch+1}: val_accuracy={logs.get('val_accuracy', 'N/A'):.4f}, val_tss={tss}")

            # Early stopping logic
            if tss > self.best_tss:
                self.best_tss = tss
                self.wait = 0
                self.best_weights = self.model.get_weights()
                self.best_epoch = epoch+1
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"Epoch {epoch+1}: early stopping (no improvement in val_tss for {self.patience} epochs)")
                    print(f"Restoring model weights from the end of the best epoch: {self.best_epoch}.")
                    self.model.stop_training = True
                    self.model.set_weights(self.best_weights)

    val_tss_monitor = ValTSSMonitor((X_val, y_val), patience=5)

>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11
    # Train the model
    print(f"\nStarting training for {channel_name}...")
    history = model.fit(
        X_train,
        y_train,
<<<<<<< HEAD
        epochs=10,
        batch_size=8,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=[early_stopping],
=======
        epochs=30,
        batch_size=8,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=[val_tss_monitor],
>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11
        verbose=1
    )
    
    # Evaluate on test set
    test_accuracy = None
    if len(X_test) > 0:
        print(f"\nEvaluating {channel_name} model on test set...")
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        test_accuracy = accuracy
        print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
        print(f"Test Set Loss: {loss:.4f}")
        
        # Get predictions for TSS calculation
        predictions = model.predict(X_test)
        predicted_classes = (predictions > 0.5).astype(int)
        
        # Calculate TSS
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, predicted_classes)
        print("Confusion Matrix:")
        print(cm)
        
        if cm.shape == (2, 2):  # Only calculate TSS for binary classification
            tn, fp, fn, tp = cm.ravel()
            tss = tp/(tp+fn) - fp/(fp+tn)
            print(f"True Skill Statistic (TSS): {tss:.4f}")
    
    return model, history, test_accuracy

def save_model_artifacts(model, history, test_data, channel_name, channel_idx, ensemble_dir):
    """Save model and related artifacts in the ensemble directory."""
    model_name = f"solar_flare_model_{channel_name}"
    
    # Save the model file in the ensemble directory
    model_path = os.path.join(ensemble_dir, f"{model_name}.keras")
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Save training history
    history_dict = history.history
    history_path = os.path.join(ensemble_dir, f"training_history_{channel_name}.json")
    with open(history_path, 'w') as f:
        json.dump(history_dict, f)
    
    # Save model weights separately
    weights_path = os.path.join(ensemble_dir, f"model_{channel_name}.weights.h5")
    model.save_weights(weights_path)
    
    # Save model configuration
    config = {
        'channel_name': channel_name,
        'channel_index': channel_idx,
        'model_name': model_name,
        'input_shape': model.input_shape,
        'total_params': model.count_params()
    }
    
    config_path = os.path.join(ensemble_dir, f"config_{channel_name}.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Model artifacts saved to: {ensemble_dir}")
    return model_path

def main():
    print("Multi-Channel Solar Flare Prediction Model Training")
    print("="*60)
    
    # Load pre-processed data
    print("Loading pre-processed data...")
<<<<<<< HEAD
    with np.load('processed_solar_data.npz') as data:
=======
    with np.load('processed_HED_data.npz') as data:
>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11
        X = data['X']  # Shape: (samples, timesteps, height, width, channels)
        y = data['y']  # Shape: (samples,)
    
    print(f"Data loaded successfully. Shapes: X={X.shape}, y={y.shape}")
    print(f"Number of channels: {X.shape[-1]}")
    
    # Verify we have the expected number of channels
    channel_names = CHANNEL_NAMES.copy()  # Create a local copy
    if X.shape[-1] != len(channel_names):
        print(f"WARNING: Expected {len(channel_names)} channels, but got {X.shape[-1]}")
        print("Adjusting channel names...")
        channel_names = [f"Channel_{i}" for i in range(X.shape[-1])]
    
    # Create ensemble directory
<<<<<<< HEAD
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
=======
    models_dir = os.path.join(os.path.dirname(__file__), "../models")
    os.makedirs(models_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11
    ensemble_dir = os.path.join(models_dir, f"ensemble_{timestamp}")
    os.makedirs(ensemble_dir, exist_ok=True)
    
    print(f"Creating ensemble directory: {ensemble_dir}")
    
    # Split data once for all models (shared test set)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Data split - Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    # Save shared test set
    test_set_path = os.path.join(ensemble_dir, 'test_set.npz')
    np.savez_compressed(test_set_path, X_test=X_test, y_test=y_test)
    print(f"Shared test set saved to: {test_set_path}")
    
    # Store model information for ensemble
    trained_models = []
    model_info = {}
<<<<<<< HEAD
    
    # Train a separate model for each channel
    for channel_idx in range(X.shape[-1]):
        channel_name = channel_names[channel_idx]
        
=======
    tss_scores = {}
    # Train a separate model for each channel
    for channel_idx in range(X.shape[-1]):
        channel_name = channel_names[channel_idx]
>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11
        # Extract single channel data from the split datasets
        X_train_channel = X_train[:, :, :, :, channel_idx:channel_idx+1]
        X_val_channel = X_val[:, :, :, :, channel_idx:channel_idx+1]
        X_test_channel = X_test[:, :, :, :, channel_idx:channel_idx+1]
<<<<<<< HEAD
        
        print(f"\nPreparing data for {channel_name}...")
        print(f"Channel data shapes - Train: {X_train_channel.shape}, Val: {X_val_channel.shape}, Test: {X_test_channel.shape}")
        
=======
        print(f"\nPreparing data for {channel_name}...")
        print(f"Channel data shapes - Train: {X_train_channel.shape}, Val: {X_val_channel.shape}, Test: {X_test_channel.shape}")
>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11
        # Train the model using the pre-split data
        model, history, test_accuracy = train_channel_model_with_splits(
            X_train_channel, X_val_channel, X_test_channel, 
            y_train, y_val, y_test, channel_name, channel_idx
        )
<<<<<<< HEAD
        
        if model is not None:
            # Save model artifacts in ensemble directory
            model_path = save_model_artifacts(model, history, None, channel_name, channel_idx, ensemble_dir)
            
=======
        # Calculate TSS for test set
        tss_score = None
        if model is not None and test_accuracy is not None:
            predictions = model.predict(X_test_channel)
            predicted_classes = (predictions > 0.5).astype(int)
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, predicted_classes)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                tss_score = tp/(tp+fn) - fp/(fp+tn)
            else:
                tss_score = None
        if model is not None:
            # Save model artifacts in ensemble directory
            model_path = save_model_artifacts(model, history, None, channel_name, channel_idx, ensemble_dir)
>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11
            # Store model information
            model_info[channel_name] = {
                'model_path': model_path,
                'channel_index': channel_idx,
                'test_accuracy': test_accuracy
            }
<<<<<<< HEAD
            
            trained_models.append((channel_name, model_path))
=======
            trained_models.append((channel_name, model_path))
            tss_scores[channel_name] = tss_score
>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11
    
    # Save ensemble configuration in the ensemble directory
    ensemble_config = {
        'timestamp': timestamp,
        'ensemble_dir': ensemble_dir,
        'channel_names': channel_names,
        'models': model_info,
        'data_shape': X.shape,
        'total_samples': len(X),
        'test_set_path': test_set_path,
        'class_distribution': {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    }
    
    ensemble_config_path = os.path.join(ensemble_dir, "ensemble_config.json")
    with open(ensemble_config_path, 'w') as f:
        json.dump(ensemble_config, f, indent=2)
    
    # Remove redundant saving in the main models directory
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully trained {len(trained_models)} models:")
    for channel_name, model_path in trained_models:
        accuracy = model_info[channel_name].get('test_accuracy', 'N/A')
<<<<<<< HEAD
        if accuracy != 'N/A':
            accuracy = f"{accuracy*100:.2f}%"
        print(f"  - {channel_name}: {accuracy}")
    
=======
        tss = tss_scores.get(channel_name, 'N/A')
        if accuracy != 'N/A':
            accuracy = f"{accuracy*100:.2f}%"
        if tss != 'N/A' and tss is not None:
            tss = f"{tss:.4f}"
        print(f"  - {channel_name}: Accuracy = {accuracy}, TSS = {tss}")
>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11
    print(f"\nEnsemble directory: {ensemble_dir}")
    print(f"Ensemble configuration saved to: {ensemble_config_path}")
    print("You can now use 'ensemble_predict.py' to make predictions with all models.")

if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11
