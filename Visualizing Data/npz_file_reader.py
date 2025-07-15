import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.gridspec import GridSpec

def find_latest_test_set():
    """Find the most recent test_set.npz file in the models directory structure."""
    # Look for test_set.npz files in the models directory and its subdirectories
<<<<<<< HEAD
    models_dir = os.path.join(os.path.dirname(__file__), "models")
=======
    models_dir = os.path.join(os.path.dirname(__file__), "../models")
>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11
    if not os.path.exists(models_dir):
        print(f"Models directory '{models_dir}' not found.")
        return None
    
    npz_files = glob.glob(os.path.join(models_dir, "**", "test_set.npz"), recursive=True)
    
    if not npz_files:
        print("No test_set.npz files found.")
        return None
    
    # Get the most recent file based on modification time
    latest_file = max(npz_files, key=os.path.getmtime)
    print(f"Found latest test set: {latest_file}")
    return latest_file

def load_and_examine_test_set(file_path=None):
    """Load and examine a test_set.npz file."""
    # If no file path is provided, find the latest one
    if file_path is None:
        file_path = find_latest_test_set()
        if file_path is None:
            return
    
    try:
        # Load the test set data
        with np.load(file_path) as data:
            X_test = data['X_test']
            y_test = data['y_test']
            
        print(f"\nTest Set Details:")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        # Display information about the data dimensions
        if len(X_test.shape) == 5:  # Expected shape for CNN-LSTM: (samples, time_steps, height, width, channels)
            samples, time_steps, height, width, channels = X_test.shape
            print(f"\nData structure:")
            print(f"- Samples: {samples}")
            print(f"- Time steps: {time_steps}")
            print(f"- Image height: {height}")
            print(f"- Image width: {width}")
            print(f"- Channels: {channels}")
            
            # Display class distribution
            unique_labels, counts = np.unique(y_test, return_counts=True)
            print(f"\nClass distribution:")
            for label, count in zip(unique_labels, counts):
                print(f"- Class {label}: {count} samples ({count/len(y_test)*100:.1f}%)")
            
            return X_test, y_test
        else:
            print(f"Unexpected data shape: {X_test.shape}")
            return None, None
            
    except Exception as e:
        print(f"Error loading test set: {e}")
        return None, None

def visualize_sample(X_test, y_test, sample_idx=0):
    """Visualize a single sample from the test set."""
    if X_test is None or sample_idx >= len(X_test):
        print("Invalid sample index or no data available.")
        return
    
    sample = X_test[sample_idx]
    label = y_test[sample_idx]
    
    time_steps, height, width, channels = sample.shape
    
    # Create a figure with a grid layout
    plt.figure(figsize=(15, 8))
    fig_title = f"Sample {sample_idx} - Class: {label}"
    plt.suptitle(fig_title, fontsize=16)
    
    # Determine the layout based on the number of time steps and channels
    if time_steps <= 5:
        rows, cols = time_steps, channels
    else:
        rows = min(3, time_steps)
        cols = min(5, (time_steps * channels + rows - 1) // rows)
    
    # Create subplots for each time step and channel
    plot_idx = 1
    for t in range(time_steps):
        for c in range(channels):
            if plot_idx <= rows * cols:
                plt.subplot(rows, cols, plot_idx)
                plt.imshow(sample[t, :, :, c], cmap='viridis')
                plt.title(f"T{t}, C{c}")
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.axis('off')
                plot_idx += 1
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def browse_samples(X_test, y_test):
    """Interactive browser for the test set samples."""
    if X_test is None or len(X_test) == 0:
        print("No data available to browse.")
        return
    
    current_idx = 0
    max_idx = len(X_test) - 1
    
    print("\nSample Browser")
    print("Controls:")
    print("  n - next sample")
    print("  p - previous sample")
    print("  j - jump to specific sample index")
    print("  c - show samples of specific class")
    print("  q - quit browser")
    
    visualize_sample(X_test, y_test, current_idx)
    
    while True:
        cmd = input(f"\nSample {current_idx}/{max_idx} (Class {y_test[current_idx]}) > ").lower()
        
        if cmd == 'q':
            break
        elif cmd == 'n':
            current_idx = min(current_idx + 1, max_idx)
            visualize_sample(X_test, y_test, current_idx)
        elif cmd == 'p':
            current_idx = max(current_idx - 1, 0)
            visualize_sample(X_test, y_test, current_idx)
        elif cmd == 'j':
            try:
                idx = int(input("Enter sample index: "))
                if 0 <= idx <= max_idx:
                    current_idx = idx
                    visualize_sample(X_test, y_test, current_idx)
                else:
                    print(f"Index must be between 0 and {max_idx}")
            except ValueError:
                print("Please enter a valid number")
        elif cmd == 'c':
            try:
                class_val = int(input("Enter class value to view (0 or 1): "))
                class_indices = np.where(y_test == class_val)[0]
                if len(class_indices) > 0:
                    print(f"Found {len(class_indices)} samples of class {class_val}")
                    current_idx = class_indices[0]
                    visualize_sample(X_test, y_test, current_idx)
                else:
                    print(f"No samples found for class {class_val}")
            except ValueError:
                print("Please enter a valid class (0 or 1)")
        else:
            print("Unknown command")

def compare_samples(X_test, y_test):
    """Compare positive and negative samples side by side."""
    if X_test is None or len(X_test) == 0:
        print("No data available for comparison.")
        return
    
    # Find indices of positive and negative samples
    positive_indices = np.where(y_test == 1)[0]
    negative_indices = np.where(y_test == 0)[0]
    
    if len(positive_indices) == 0 or len(negative_indices) == 0:
        print("Need both positive and negative samples to compare.")
        return
    
    # Get a random positive and negative sample
    pos_idx = np.random.choice(positive_indices)
    neg_idx = np.random.choice(negative_indices)
    
    pos_sample = X_test[pos_idx]
    neg_sample = X_test[neg_idx]
    
    time_steps, height, width, channels = pos_sample.shape
    
    # Display the samples side by side
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, time_steps, figure=fig)
    
    plt.suptitle(f"Comparison: Positive (top) vs Negative (bottom)", fontsize=16)
    
    # Show each time step for the positive sample (top row)
    for t in range(time_steps):
        ax = fig.add_subplot(gs[0, t])
        # For multi-channel data, show the first channel or a combination
        if channels > 1:
            img = np.mean(pos_sample[t], axis=2)  # Average across channels
            ax.imshow(img, cmap='viridis')
        else:
            ax.imshow(pos_sample[t, :, :, 0], cmap='viridis')
        ax.set_title(f"Pos T{t}")
        ax.set_axis_off()
    
    # Show each time step for the negative sample (bottom row)
    for t in range(time_steps):
        ax = fig.add_subplot(gs[1, t])
        if channels > 1:
            img = np.mean(neg_sample[t], axis=2)  # Average across channels
            ax.imshow(img, cmap='viridis')
        else:
            ax.imshow(neg_sample[t, :, :, 0], cmap='viridis')
        ax.set_title(f"Neg T{t}")
        ax.set_axis_off()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def main():
    """Main function to execute the script."""
    print("NPZ Test Set Viewer")
    print("=================")
    
    # First, try to find and load the latest test set
    X_test, y_test = load_and_examine_test_set()
    
    if X_test is None:
        # If no file was found automatically, ask the user to provide a path
        custom_path = input("Enter the path to a test_set.npz file: ")
        if custom_path:
            X_test, y_test = load_and_examine_test_set(custom_path)
    
    if X_test is not None:
        while True:
            print("\nOptions:")
            print("1. Browse samples")
            print("2. Compare positive and negative samples")
            print("3. View a specific sample")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ")
            
            if choice == '1':
                browse_samples(X_test, y_test)
            elif choice == '2':
                compare_samples(X_test, y_test)
            elif choice == '3':
                try:
                    idx = int(input("Enter sample index: "))
                    visualize_sample(X_test, y_test, idx)
                except ValueError:
                    print("Please enter a valid number")
            elif choice == '4':
                break
            else:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()