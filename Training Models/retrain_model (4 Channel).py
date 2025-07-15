import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime

<<<<<<< HEAD
def load_latest_model(models_dir="models"):
=======
def load_latest_model(models_dir="../models"):
>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11
    """Load the most recent model from the models directory."""
    if not os.path.exists(models_dir):
        print(f"Models directory '{models_dir}' not found.")
        return None
    
    # List all model directories
    model_dirs = [d for d in os.listdir(models_dir) 
                 if os.path.isdir(os.path.join(models_dir, d)) and 
                 d.startswith("solar_flare_model_")]
    
    if not model_dirs:
        print("No saved models found.")
        return None
    
    # Find the most recent model
    latest_model_dir = max(model_dirs)
    model_path = os.path.join(models_dir, latest_model_dir)
    
    print(f"Loading model from: {model_path}")
    return load_model(model_path)

def retrain_model(model, X_train, y_train, X_val=None, y_val=None, epochs=5, batch_size=8):
    """Retrain the model with new data."""
    validation_data = (X_val, y_val) if X_val is not None and X_val.size > 0 else None
    
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        verbose=1
    )
    
    return history, model

def save_model(model, prefix="retrained"):
    """Save the retrained model."""
    # Create models directory if it doesn't exist
<<<<<<< HEAD
    models_dir = "models"
=======
    models_dir = "../models"
>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11
    os.makedirs(models_dir, exist_ok=True)

    # Generate a timestamp for the model name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"solar_flare_model_{prefix}_{timestamp}"
    model_path = os.path.join(models_dir, model_name)

    # Save the entire model
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    # Also save just the weights
    weights_path = os.path.join(models_dir, f"{model_name}_weights.h5")
    model.save_weights(weights_path)
    print(f"Model weights saved to: {weights_path}")
    
    return model_path

if __name__ == "__main__":
    print("Solar Flare Model Retraining Tool")
    print("-" * 40)
    
    # Load the latest model
    model = load_latest_model()
    
    if model is None:
        print("No model available to retrain. Please train a model first.")
        exit()
    
    # Load data for retraining
    try:
        print("Loading data for retraining...")
        with np.load('processed_solar_data.npz') as data:
            X = data['X']
            y = data['y']
            
        # Here you would normally split the data, but for this example we'll use all data
        # In a real scenario, you'd want to use new data or properly split data
        
        print(f"Loaded training data with shape: {X.shape}")
        
        # Retrain the model (using a smaller number of epochs for example)
        print("\nRetraining model...")
        history, retrained_model = retrain_model(
            model, 
            X, 
            y,
            epochs=3,  # Fewer epochs for retraining
            batch_size=8
        )
        
        # Save the retrained model
        model_path = save_model(retrained_model)
        
        print("\nModel retraining completed!")
        
    except Exception as e:
        print(f"Error during retraining: {e}")