import os
import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report


def list_model_subdirectories(models_dir="models"):
    """List all subdirectories in the models directory."""
    if not os.path.exists(models_dir):
        print(f"Models directory '{models_dir}' not found.")
        return []
    
    subdirectories = [d for d in os.listdir(models_dir) 
                      if os.path.isdir(os.path.join(models_dir, d))]
    return subdirectories


def load_specific_model_from_subdir(subdir_name, models_dir="models"):
    """Load a specific model from a subdirectory."""
    subdir_path = os.path.join(models_dir, subdir_name)
    if not os.path.exists(subdir_path):
        print(f"Subdirectory '{subdir_name}' not found in {models_dir}")
        return None
    
    # Search for .keras files in the subdirectory
    model_files = [f for f in os.listdir(subdir_path) 
                   if f.endswith(".keras") and os.path.isfile(os.path.join(subdir_path, f))]
    
    if not model_files:
        print(f"No .keras files found in subdirectory '{subdir_name}'")
        return None
    
    # Let the user choose a model file if multiple exist
    if len(model_files) > 1:
        print(f"Multiple models found in '{subdir_name}':")
        for i, model_file in enumerate(model_files, 1):
            print(f"{i}. {model_file}")
        choice = int(input("Enter the number of the model to load: ")) - 1
        model_file = model_files[choice]
    else:
        model_file = model_files[0]
    
    model_path = os.path.join(subdir_path, model_file)
    print(f"Loading model from: {model_path}")
    return load_model(model_path), subdir_name


def load_test_data_from_subdir(subdir_name, models_dir="models"):
    """Load the test data from a specific subdirectory."""
    subdir_path = os.path.join(models_dir, subdir_name)
    test_data_path = os.path.join(subdir_path, 'test_set.npz')
    
    if not os.path.exists(test_data_path):
        print(f"Test data not found at {test_data_path}")
        return None, None
    
    with np.load(test_data_path) as data:
        X_test = data['X_test']
        y_test = data['y_test']
    
    print(f"Loaded test data with shape: X={X_test.shape}, y={y_test.shape}")
    return X_test, y_test


if __name__ == "__main__":
    print("Solar Flare Prediction Tool")
    print("-" * 40)
    
    # List available subdirectories
    subdirectories = list_model_subdirectories()
    if not subdirectories:
        print("No model subdirectories found. Please ensure models are organized correctly.")
        exit()
    
    print("Available model subdirectories:")
    for i, subdir in enumerate(subdirectories, 1):
        print(f"{i}. {subdir}")
    
    choice = int(input("Enter the number of the subdirectory to use: ")) - 1
    selected_subdir = subdirectories[choice]
    
    # Load the model from the selected subdirectory
    result = load_specific_model_from_subdir(selected_subdir)
    if result is None:
        print("Failed to load model. Exiting.")
        exit()
    
    model, subdir_name = result
    
    print("\n1. Test on saved test data")
    print("2. Test on sample data from processed_solar_data.npz")
    choice = input("Enter your choice (1/2): ")
    
    if choice == '1':
        # Evaluate on the saved test data
        X_test, y_test = load_test_data_from_subdir(subdir_name)
        if X_test is not None and y_test is not None:
            print("\nEvaluating model on saved test data...")
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
            print(f"Test Set Loss: {loss:.4f}")
            
            predictions = model.predict(X_test)
            predicted_classes = (predictions > 0.5).astype(int)
            
            cm = confusion_matrix(y_test, predicted_classes)
            print("\nConfusion Matrix:")
            print(cm)
            
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                tss = tp/(tp+fn) - fp/(fp+tn)
                print(f"\nTrue Skill Statistic (TSS): {tss:.4f}")
            
            print("\nClassification Report:")
            print(classification_report(y_test, predicted_classes))
        else:
            print("Could not load test data.")
    
    elif choice == '2':
        try:
            with np.load('processed_solar_data.npz') as data:
                X = data['X']
                y_true = data['y']
            
            print(f"Loaded sample data with shape: {X.shape}")
            
            sample_size = min(5, X.shape[0])
            sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
            
            X_samples = X[sample_indices]
            y_true_samples = y_true[sample_indices]
            
            predictions = model.predict(X_samples)
            
            print("\nPrediction Results:")
            print("-" * 40)
            for i in range(sample_size):
                print(f"Sample {i+1}: Prediction = {predictions[i][0]:.4f}, Actual = {y_true_samples[i]}")
        except Exception as e:
            print(f"Error loading or predicting data: {e}")
    else:
        print("Invalid choice. Please run again and select 1 or 2.")