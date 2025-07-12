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

# Define channel names corresponding to the 4 magnetic field components
CHANNEL_NAMES = ['Bp', 'Br', 'Bt', 'continuum']

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
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model
    print(f"\nStarting training for {channel_name}...")
    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=16,
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
    
    # Set up early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model
    print(f"\nStarting training for {channel_name}...")
    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=8,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=[early_stopping],
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
    with np.load('processed_solar_data.npz') as data:
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
    models_dir = os.path.join(os.path.dirname(__file__), "../models")
    os.makedirs(models_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    
    # Train a separate model for each channel
    for channel_idx in range(X.shape[-1]):
        channel_name = channel_names[channel_idx]
        
        # Extract single channel data from the split datasets
        X_train_channel = X_train[:, :, :, :, channel_idx:channel_idx+1]
        X_val_channel = X_val[:, :, :, :, channel_idx:channel_idx+1]
        X_test_channel = X_test[:, :, :, :, channel_idx:channel_idx+1]
        
        print(f"\nPreparing data for {channel_name}...")
        print(f"Channel data shapes - Train: {X_train_channel.shape}, Val: {X_val_channel.shape}, Test: {X_test_channel.shape}")
        
        # Train the model using the pre-split data
        model, history, test_accuracy = train_channel_model_with_splits(
            X_train_channel, X_val_channel, X_test_channel, 
            y_train, y_val, y_test, channel_name, channel_idx
        )
        
        if model is not None:
            # Save model artifacts in ensemble directory
            model_path = save_model_artifacts(model, history, None, channel_name, channel_idx, ensemble_dir)
            
            # Store model information
            model_info[channel_name] = {
                'model_path': model_path,
                'channel_index': channel_idx,
                'test_accuracy': test_accuracy
            }
            
            trained_models.append((channel_name, model_path))
    
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
        if accuracy != 'N/A':
            accuracy = f"{accuracy*100:.2f}%"
        print(f"  - {channel_name}: {accuracy}")
    
    print(f"\nEnsemble directory: {ensemble_dir}")
    print(f"Ensemble configuration saved to: {ensemble_config_path}")
    print("You can now use 'ensemble_predict.py' to make predictions with all models.")

if __name__ == "__main__":
    main()
