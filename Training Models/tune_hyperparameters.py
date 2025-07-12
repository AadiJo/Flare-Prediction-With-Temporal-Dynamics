import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, TimeDistributed, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import os
import json
import itertools
from datetime import datetime

# Import channel names from the original script
from train_multi_model import CHANNEL_NAMES

def create_tuned_cnnlstm_model(input_shape, model_name, lstm_units=50, dropout_rate=0.25, 
                              learning_rate=0.001, l2_reg=0.001, recurrent_dropout=0.2):
    """Create a CNN-LSTM model with tunable hyperparameters focused on preventing overfitting."""
    model = Sequential(name=f"Solar_Flare_Predictor_{model_name}")
    
    # CNN layers with tunable dropout and L2 regularization
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg)), 
                             input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(dropout_rate)))
    
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg))))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(dropout_rate)))
    
    # Flatten and LSTM with tunable units and recurrent dropout
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(lstm_units, activation='relu', recurrent_dropout=recurrent_dropout))
    model.add(Dropout(min(dropout_rate * 1.5, 0.8)))  # Higher dropout before output, but capped at 0.8
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile with tunable learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_single_channel_tuned(X_train, X_val, X_test, y_train, y_val, y_test, 
                              channel_name, hyperparams):
    """Train a single channel with given hyperparameters."""
    
    # Check for multiple classes
    if len(np.unique(y_train)) <= 1:
        print(f"WARNING: Only one class found for {channel_name}. Skipping.")
        return None, 0.0
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train).astype(int), class_weights))
    
    # Create model with hyperparameters
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)
    model = create_tuned_cnnlstm_model(
        input_shape, 
        channel_name,
        lstm_units=hyperparams['lstm_units'],
        dropout_rate=hyperparams['dropout_rate'],
        learning_rate=hyperparams['learning_rate'],
        l2_reg=hyperparams['l2_reg'],
        recurrent_dropout=hyperparams['recurrent_dropout']
    )
    
    # Early stopping with more aggressive patience for overfitting prevention
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=0)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=15,  # Reduced for tuning
        batch_size=hyperparams['batch_size'],
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=[early_stopping],
        verbose=0  # Quiet training
    )
    
    # Evaluate on test set
    if len(X_test) > 0:
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        return model, accuracy
    
    return model, 0.0

def main():
    print("Hyperparameter Tuning for Multi-Channel Solar Flare Prediction")
    print("=" * 70)
    
    # Load data
    print("Loading data...")
    with np.load('processed_solar_data.npz') as data:
        X = data['X']
        y = data['y']
    
    print(f"Data loaded: X={X.shape}, y={y.shape}")
    
    # Define hyperparameter search space focused on preventing overfitting
    hyperparams_grid = {
        'learning_rate': [0.0001, 0.0005, 0.001],  # Lower learning rates to prevent overfitting
        'lstm_units': [32, 48, 64],  # Slightly adjusted range
        'dropout_rate': [0.3, 0.4, 0.5],  # Higher dropout rates to combat overfitting
        'l2_reg': [0.001, 0.01, 0.1],  # More aggressive L2 regularization options
        'recurrent_dropout': [0.2, 0.3, 0.4],  # Higher recurrent dropout for LSTM
        'batch_size': [8, 16]  # Keep batch size options the same
    }
    
    print(f"Hyperparameter grid:")
    for param, values in hyperparams_grid.items():
        print(f"  {param}: {values}")
    
    # Calculate total combinations
    total_combinations = 1
    for values in hyperparams_grid.values():
        total_combinations *= len(values)
    print(f"\nTotal combinations to test: {total_combinations}")
    print(f"Estimated time: ~{int(total_combinations * 0.75)} minutes (assuming 45 seconds per combination)")
    print("NOTE: Focused on anti-overfitting hyperparameters")
    
    # Split data once
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Results storage
    results = []
    best_score = 0
    best_params = None
    
    # Generate all combinations
    param_names = list(hyperparams_grid.keys())
    param_values = list(hyperparams_grid.values())
    
    combination_count = 0
    
    # Grid search
    for combination in itertools.product(*param_values):
        combination_count += 1
        hyperparams = dict(zip(param_names, combination))
        
        print(f"\n[{combination_count}/{total_combinations}] Testing: {hyperparams}")
        
        # Test on first channel only for speed (you can change this)
        channel_idx = 0  # Test on Bp channel
        channel_name = CHANNEL_NAMES[channel_idx]
        
        # Extract channel data
        X_train_channel = X_train[:, :, :, :, channel_idx:channel_idx+1]
        X_val_channel = X_val[:, :, :, :, channel_idx:channel_idx+1]
        X_test_channel = X_test[:, :, :, :, channel_idx:channel_idx+1]
        
        # Train and evaluate
        model, accuracy = train_single_channel_tuned(
            X_train_channel, X_val_channel, X_test_channel,
            y_train, y_val, y_test,
            channel_name, hyperparams
        )
        
        if model is not None:
            print(f"  Accuracy: {accuracy*100:.2f}%")
            
            # Store results
            result = {
                'hyperparams': hyperparams,
                'accuracy': accuracy,
                'channel': channel_name
            }
            results.append(result)
            
            # Track best
            if accuracy > best_score:
                best_score = accuracy
                best_params = hyperparams.copy()
                print(f"  *** New best score! ***")
        else:
            print("  Skipped (insufficient data)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"hyperparameter_tuning_results_{timestamp}.json"
    
    tuning_summary = {
        'best_params': best_params,
        'best_accuracy': best_score,
        'all_results': results,
        'search_space': hyperparams_grid,
        'total_combinations_tested': len(results)
    }
    
    with open(results_file, 'w') as f:
        json.dump(tuning_summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print("HYPERPARAMETER TUNING SUMMARY")
    print(f"{'='*70}")
    print(f"Best accuracy: {best_score*100:.2f}%")
    print(f"Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\nTop 5 configurations:")
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)[:5]
    for i, result in enumerate(sorted_results, 1):
        print(f"{i}. Accuracy: {result['accuracy']*100:.2f}% - {result['hyperparams']}")
    
    print(f"\nResults saved to: {results_file}")
    print(f"Use the best parameters in your train_multi_model.py script!")

if __name__ == "__main__":
    main()
