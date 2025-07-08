import os
import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import load_model


def load_latest_model(models_dir="models"):
    """Load the most recent model from the models directory."""
    if not os.path.exists(models_dir):
        print(f"Models directory '{models_dir}' not found.")
        return None
    
    # List all model FILES (not directories) with .keras or .h5 endings
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

def load_specific_model(model_name, models_dir="models"):
    """Load a specific model by name from the models directory."""
    # Check if model_name already has an extension
    if not (model_name.endswith(".keras") or model_name.endswith(".h5")):
        # Try .keras first
        keras_path = os.path.join(models_dir, f"{model_name}.keras")
        if os.path.exists(keras_path):
            model_path = keras_path
        else:
            # Try with .h5 extension
            h5_path = os.path.join(models_dir, f"{model_name}.h5")
            if os.path.exists(h5_path):
                model_path = h5_path
            else:
                print(f"Model '{model_name}' not found in {models_dir}")
                return None
    else:
        # Model name already has an extension
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
            print(f"Model '{model_name}' not found at {model_path}")
            return None
    
    print(f"Loading model from: {model_path}")
    return load_model(model_path)

def predict_solar_flare(model, input_data):
    """Make predictions using the loaded model."""
    predictions = model.predict(input_data)
    return predictions

if __name__ == "__main__":
    # Example usage
    print("Solar Flare Prediction Tool")
    print("-" * 40)
    
    # Load the model
    model = load_latest_model()
    
    if model is None:
        print("No model available. Please train a model first.")
        exit()
    
    # Load some data to predict on
    try:
        with np.load('processed_solar_data.npz') as data:
            X = data['X']
            y_true = data['y']
            
        print(f"Loaded sample data with shape: {X.shape}")
        
        # Make predictions on a few samples
        sample_size = min(5, X.shape[0])
        sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
        
        X_samples = X[sample_indices]
        y_true_samples = y_true[sample_indices]
        
        predictions = predict_solar_flare(model, X_samples)
        
        # Print results
        print("\nPrediction Results:")
        print("-" * 40)
        for i in range(sample_size):
            print(f"Sample {i+1}: Prediction = {(predictions[i][0]-.0008):.4f}, Actual = {y_true_samples[i]}")
            
    except Exception as e:
        print(f"Error loading or predicting data: {e}")
        print("You can modify this script to load your own data for predictions.")