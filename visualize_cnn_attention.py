import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import cv2
from sklearn.preprocessing import MinMaxScaler
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import warnings
warnings.filterwarnings('ignore')

console = Console()

class CNNVisualizationTool:
    def __init__(self, ensemble_config_path=None):
        self.models = {}
        self.channel_names = []
        self.ensemble_dir = None
        
        config_path = ensemble_config_path or self.find_latest_ensemble_config()
        if config_path and os.path.exists(config_path):
            self.load_ensemble_config(config_path)
        else:
            console.print("[red]No ensemble configuration found. Please train models first.[/red]")
    
    def find_latest_ensemble_config(self):
        """Find the latest ensemble configuration file"""
        models_dir = "models"
        latest_config_path = os.path.join(models_dir, "latest_ensemble_config.json")
        if os.path.exists(latest_config_path):
            return latest_config_path
        
        ensemble_dirs = [d for d in os.listdir(models_dir) 
                        if os.path.isdir(os.path.join(models_dir, d)) and d.startswith('ensemble_')]
        
        if ensemble_dirs:
            latest_ensemble_dir = max(ensemble_dirs)
            config_path = os.path.join(models_dir, latest_ensemble_dir, "ensemble_config.json")
            return config_path if os.path.exists(config_path) else None
        return None
    
    def load_ensemble_config(self, config_path):
        """Load ensemble configuration and models"""
        console.print(f"[green]Loading ensemble configuration from:[/green] {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.channel_names = config['channel_names']
        model_info = config['models']
        self.ensemble_dir = config.get('ensemble_dir', os.path.dirname(config_path))
        
        console.print(f"[cyan]Loading {len(model_info)} models...[/cyan]")
        
        for channel_name, info in model_info.items():
            model_path = os.path.join(self.ensemble_dir, f"solar_flare_model_{channel_name}.keras")
            
            if os.path.exists(model_path):
                try:
                    self.models[channel_name] = load_model(model_path)
                    console.print(f"  ✓ Loaded {channel_name} model")
                except Exception as e:
                    console.print(f"  ✗ Failed to load {channel_name} model: {e}")
            else:
                console.print(f"  ✗ Model file not found: {model_path}")
    
    def create_gradcam_heatmap(self, model, img_array, last_conv_layer_name=None, pred_index=None):
        """Generate Grad-CAM heatmap using a simpler approach for CNN-LSTM models"""
        # Convert numpy array to tensor if needed
        if isinstance(img_array, np.ndarray):
            img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        # For CNN-LSTM models, let's use a different approach
        # We'll create activation maps from the input gradients
        with tf.GradientTape() as tape:
            tape.watch(img_array)
            predictions = model(img_array)
            # Use the prediction score for the positive class
            score = predictions[0][0]
        
        # Compute gradients
        grads = tape.gradient(score, img_array)
        
        if grads is None:
            console.print("[yellow]Warning: Could not compute gradients[/yellow]")
            return None
        
        # Create a simple activation map by taking the mean absolute gradient
        # across the time and channel dimensions
        if len(grads.shape) == 5:  # (batch, time, height, width, channels)
            # Take mean across time and channels
            activation_map = tf.reduce_mean(tf.abs(grads[0]), axis=[0, 3])  # Average over time and channels
        else:
            activation_map = tf.reduce_mean(tf.abs(grads[0]), axis=-1)  # Average over channels
        
        # Normalize to 0-1
        activation_map = activation_map - tf.reduce_min(activation_map)
        activation_map = activation_map / tf.reduce_max(activation_map)
        
        console.print(f"[green]Created simple activation map with shape: {activation_map.shape}[/green]")
        return activation_map.numpy()
    
    def create_saliency_map(self, model, img_array):
        """Generate saliency map using gradients"""
        # Convert numpy array to tensor if needed
        if isinstance(img_array, np.ndarray):
            img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(img_array)
            predictions = model(img_array)
            loss = predictions[0][0]  # Assuming binary classification
        
        # Get the gradients of the loss w.r.t to the input image
        gradients = tape.gradient(loss, img_array)
        
        # Get the absolute values and max across color channels (if any)
        gradients = tf.math.abs(gradients)
        
        # Average across the time dimension for CNN-LSTM models
        if len(gradients.shape) == 5:  # (batch, time, height, width, channels)
            gradients = tf.reduce_mean(gradients, axis=1)
        
        # Take the maximum across channels if multiple channels
        if gradients.shape[-1] > 1:
            gradients = tf.reduce_max(gradients, axis=-1)
        else:
            gradients = tf.squeeze(gradients, axis=-1)
        
        # Normalize
        gradients = gradients / tf.reduce_max(gradients)
        return gradients[0].numpy()
    
    def visualize_sample(self, X_sample, y_true=None, channel_name=None, sample_idx=0):
        """Visualize what the CNN focuses on for a single sample"""
        if not self.models:
            console.print("[red]No models loaded![/red]")
            return
        
        # If no specific channel is requested, use the first available model
        if channel_name is None:
            channel_name = list(self.models.keys())[0]
        
        if channel_name not in self.models:
            console.print(f"[red]Model for channel '{channel_name}' not found![/red]")
            return
        
        model = self.models[channel_name]
        channel_idx = self.channel_names.index(channel_name)
        
        # Extract channel data
        channel_data = X_sample[:, :, :, channel_idx:channel_idx+1]
        channel_data_batch = np.expand_dims(channel_data, axis=0)
        
        # Convert to tensor for TensorFlow operations
        channel_data_tensor = tf.convert_to_tensor(channel_data_batch, dtype=tf.float32)
        
        # Make prediction
        prediction = model.predict(channel_data_batch, verbose=0)[0][0]
        predicted_class = "Flare" if prediction > 0.5 else "No Flare"
        
        console.print(f"\n[bold]Visualizing {channel_name} model predictions[/bold]")
        console.print(f"Prediction: {prediction:.4f} ({predicted_class})")
        if y_true is not None:
            actual_class = "Flare" if y_true == 1 else "No Flare"
            console.print(f"Actual: {actual_class}")
        
        # Create visualizations
        try:
            # Generate Grad-CAM heatmap
            heatmap = self.create_gradcam_heatmap(model, channel_data_tensor)
            
            # Generate saliency map
            saliency = self.create_saliency_map(model, channel_data_tensor)
            
            # Create the visualization plot
            self.plot_visualization(channel_data, heatmap, saliency, channel_name, 
                                  prediction, y_true, sample_idx)
            
        except Exception as e:
            console.print(f"[red]Error creating visualizations: {e}[/red]")
            # Fallback: just show the original images
            self.plot_simple_visualization(channel_data, channel_name, prediction, y_true, sample_idx)
    
    def plot_visualization(self, channel_data, heatmap, saliency, channel_name, 
                          prediction, y_true, sample_idx):
        """Plot the comprehensive visualization"""
        n_timesteps = channel_data.shape[0]
        
        # Simple approach: divide total timesteps by 6 and show one at each interval
        n_images_to_show = 6
        if n_timesteps <= n_images_to_show:
            timesteps_to_show = list(range(n_timesteps))
        else:
            # Divide the timesteps into equal intervals
            interval = n_timesteps // n_images_to_show
            timesteps_to_show = [i * interval for i in range(n_images_to_show)]
            # Make sure we don't exceed the array bounds
            timesteps_to_show = [min(t, n_timesteps - 1) for t in timesteps_to_show]
        
        n_cols = len(timesteps_to_show)
        n_rows = 4  # Original, Saliency, Grad-CAM, Overlay
        
        # Create figure with subplots
        fig = plt.figure(figsize=(3*n_cols, 12))
        
        # Calculate actual time spacing
        if len(timesteps_to_show) > 1:
            time_interval = (timesteps_to_show[1] - timesteps_to_show[0]) * 12  # 12 minutes per timestep
        else:
            time_interval = 12
        
        # Set up the main title
        title_text = f"CNN Attention Visualization - {channel_name}\n"
        title_text += f"Prediction: {prediction:.4f} ({'Flare' if prediction > 0.5 else 'No Flare'})"
        if y_true is not None:
            title_text += f" | Actual: {'Flare' if y_true == 1 else 'No Flare'}"
        title_text += f"\nTimesteps: {timesteps_to_show} (every ~{time_interval} minutes)"
        
        fig.suptitle(title_text, fontsize=12, fontweight='bold')
        
        for col_idx, t in enumerate(timesteps_to_show):
            # Original image
            ax1 = plt.subplot(n_rows, n_cols, col_idx + 1)
            img = channel_data[t, :, :, 0]
            im1 = ax1.imshow(img, cmap='gray')
            ax1.set_title(f'Original\nt={t} (~{t*12}min)')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, shrink=0.6)
            
            # Saliency map
            if saliency is not None:
                ax2 = plt.subplot(n_rows, n_cols, n_cols + col_idx + 1)
                # For time series data, we need to handle the saliency map appropriately
                if len(saliency.shape) == 3:  # (time, height, width)
                    saliency_t = saliency[t] if t < saliency.shape[0] else saliency[0]
                else:  # (height, width) - averaged across time
                    saliency_t = saliency
                im2 = ax2.imshow(saliency_t, cmap='hot')
                ax2.set_title(f'Saliency\nt={t}')
                ax2.axis('off')
                plt.colorbar(im2, ax=ax2, shrink=0.6)
            
            # Grad-CAM heatmap
            if heatmap is not None:
                ax3 = plt.subplot(n_rows, n_cols, 2*n_cols + col_idx + 1)
                # Resize heatmap to match image size
                heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                im3 = ax3.imshow(heatmap_resized, cmap='jet', alpha=0.8)
                ax3.set_title(f'Grad-CAM\nt={t}')
                ax3.axis('off')
                plt.colorbar(im3, ax=ax3, shrink=0.6)
                
                # Overlay on original
                ax4 = plt.subplot(n_rows, n_cols, 3*n_cols + col_idx + 1)
                ax4.imshow(img, cmap='gray')
                ax4.imshow(heatmap_resized, cmap='jet', alpha=0.4)
                ax4.set_title(f'Overlay\nt={t}')
                ax4.axis('off')
        
        plt.tight_layout()
        
        # Save the plot
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)
        filename = f"cnn_attention_{channel_name}_sample_{sample_idx}.png"
        filepath = os.path.join(plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        console.print(f"[green]Visualization saved to:[/green] {filepath}")
        
        plt.show()
    
    def plot_simple_visualization(self, channel_data, channel_name, prediction, y_true, sample_idx):
        """Simple fallback visualization showing just the original images"""
        n_timesteps = channel_data.shape[0]
        
        # Simple approach: divide total timesteps by 6 and show one at each interval
        n_images_to_show = 6
        if n_timesteps <= n_images_to_show:
            timesteps_to_show = list(range(n_timesteps))
        else:
            # Divide the timesteps into equal intervals
            interval = n_timesteps // n_images_to_show
            timesteps_to_show = [i * interval for i in range(n_images_to_show)]
            # Make sure we don't exceed the array bounds
            timesteps_to_show = [min(t, n_timesteps - 1) for t in timesteps_to_show]
        
        n_cols = min(len(timesteps_to_show), 3)
        n_rows = int(np.ceil(len(timesteps_to_show) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)
        
        # Calculate actual time spacing
        if len(timesteps_to_show) > 1:
            time_interval = (timesteps_to_show[1] - timesteps_to_show[0]) * 12  # 12 minutes per timestep
        else:
            time_interval = 12
        
        title_text = f"Time Series Data - {channel_name}\n"
        title_text += f"Prediction: {prediction:.4f} ({'Flare' if prediction > 0.5 else 'No Flare'})"
        if y_true is not None:
            title_text += f" | Actual: {'Flare' if y_true == 1 else 'No Flare'}"
        title_text += f"\nTimesteps: {timesteps_to_show} (every ~{time_interval} minutes)"
        
        fig.suptitle(title_text, fontsize=12, fontweight='bold')
        
        for idx, t in enumerate(timesteps_to_show):
            row = idx // n_cols
            col = idx % n_cols
            
            img = channel_data[t, :, :, 0]
            im = axes[row, col].imshow(img, cmap='gray')
            axes[row, col].set_title(f't={t} (~{t*12} min)')
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], shrink=0.6)
        
        # Hide empty subplots
        for idx in range(len(timesteps_to_show), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Save the plot
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)
        filename = f"simple_viz_{channel_name}_sample_{sample_idx}.png"
        filepath = os.path.join(plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        console.print(f"[green]Simple visualization saved to:[/green] {filepath}")
        
        plt.show()
    
    def visualize_multiple_samples(self, X_data, y_data=None, channel_name=None, n_samples=3):
        """Visualize multiple samples"""
        if len(X_data) < n_samples:
            n_samples = len(X_data)
        
        # Select random samples
        sample_indices = np.random.choice(len(X_data), n_samples, replace=False)
        
        console.print(f"\n[bold blue]Visualizing {n_samples} random samples[/bold blue]")
        
        for i, sample_idx in enumerate(sample_indices):
            console.print(f"\n[cyan]--- Sample {i+1}/{n_samples} (Index: {sample_idx}) ---[/cyan]")
            y_true = y_data[sample_idx] if y_data is not None else None
            self.visualize_sample(X_data[sample_idx], y_true, channel_name, sample_idx)
    
    def compare_channels_single_sample(self, X_sample, y_true=None, sample_idx=0):
        """Compare how different channel models focus on the same sample"""
        if not self.models:
            console.print("[red]No models loaded![/red]")
            return
        
        console.print(f"\n[bold blue]Comparing channel attention for sample {sample_idx}[/bold blue]")
        
        # Create a figure to compare all channels
        n_channels = len(self.models)
        fig, axes = plt.subplots(2, n_channels, figsize=(5*n_channels, 10))
        if n_channels == 1:
            axes = axes.reshape(2, 1)
        
        predictions = {}
        
        for idx, (channel_name, model) in enumerate(self.models.items()):
            channel_idx = self.channel_names.index(channel_name)
            channel_data = X_sample[:, :, :, channel_idx:channel_idx+1]
            channel_data_batch = np.expand_dims(channel_data, axis=0)
            
            # Convert to tensor for TensorFlow operations
            channel_data_tensor = tf.convert_to_tensor(channel_data_batch, dtype=tf.float32)
            
            # Make prediction
            prediction = model.predict(channel_data_batch, verbose=0)[0][0]
            predictions[channel_name] = prediction
            
            # Show first timestep of original data
            axes[0, idx].imshow(channel_data[0, :, :, 0], cmap='gray')
            axes[0, idx].set_title(f'{channel_name}\nPred: {prediction:.3f}')
            axes[0, idx].axis('off')
            
            # Generate and show saliency map
            try:
                saliency = self.create_saliency_map(model, channel_data_tensor)
                if len(saliency.shape) == 3:
                    saliency_show = saliency[0]
                else:
                    saliency_show = saliency
                axes[1, idx].imshow(saliency_show, cmap='hot')
                axes[1, idx].set_title(f'Saliency Map')
                axes[1, idx].axis('off')
            except Exception as e:
                axes[1, idx].text(0.5, 0.5, f'Error:\n{str(e)[:30]}...', 
                                 ha='center', va='center', transform=axes[1, idx].transAxes)
                axes[1, idx].axis('off')
        
        title_text = f"Channel Comparison - Sample {sample_idx}"
        if y_true is not None:
            title_text += f" | Actual: {'Flare' if y_true == 1 else 'No Flare'}"
        fig.suptitle(title_text, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save comparison plot
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)
        filename = f"channel_comparison_sample_{sample_idx}.png"
        filepath = os.path.join(plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        console.print(f"[green]Channel comparison saved to:[/green] {filepath}")
        
        plt.show()
        
        # Print prediction summary
        table = Table(title="Channel Predictions")
        table.add_column("Channel", style="cyan")
        table.add_column("Prediction", style="magenta")
        table.add_column("Class", style="yellow")
        
        for channel_name, pred in predictions.items():
            pred_class = "Flare" if pred > 0.5 else "No Flare"
            table.add_row(channel_name, f"{pred:.4f}", pred_class)
        
        console.print(table)

def load_sample_data():
    """Load sample data for visualization"""
    try:
        console.print("[cyan]Loading sample data...[/cyan]")
        with np.load('processed_solar_data.npz') as data:
            X = data['X']
            y = data['y']
        console.print(f"[green]Loaded data with shape:[/green] X={X.shape}, y={y.shape}")
        return X, y
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        return None, None

def main():
    console.clear()
    console.print(Panel.fit("[bold orange1]CNN Attention Visualization Tool[/bold orange1]", 
                           border_style="orange1"))
    
    # Initialize the visualization tool
    viz_tool = CNNVisualizationTool()
    
    if not viz_tool.models:
        console.print("[red]No models loaded. Exiting.[/red]")
        return
    
    # Load sample data
    X, y = load_sample_data()
    if X is None:
        return
    
    # Interactive menu
    while True:
        console.print("\n[bold cyan]Visualization Options:[/bold cyan]")
        console.print("1. Visualize single sample (specific channel)")
        console.print("2. Visualize multiple samples")
        console.print("3. Compare all channels on single sample")
        console.print("4. Exit")
        
        choice = console.input("\n[yellow]Select an option (1-4): [/yellow]").strip()
        
        if choice == '1':
            # Single sample visualization
            console.print(f"\n[cyan]Available channels:[/cyan] {list(viz_tool.models.keys())}")
            channel = console.input("[yellow]Enter channel name (or press Enter for first available): [/yellow]").strip()
            if not channel:
                channel = None
            
            sample_idx = console.input("[yellow]Enter sample index (or press Enter for random): [/yellow]").strip()
            if sample_idx:
                try:
                    sample_idx = int(sample_idx)
                    if sample_idx >= len(X):
                        console.print(f"[red]Invalid index. Max index is {len(X)-1}[/red]")
                        continue
                except ValueError:
                    console.print("[red]Invalid index. Using random sample.[/red]")
                    sample_idx = np.random.randint(len(X))
            else:
                sample_idx = np.random.randint(len(X))
            
            y_true = y[sample_idx] if y is not None else None
            viz_tool.visualize_sample(X[sample_idx], y_true, channel, sample_idx)
        
        elif choice == '2':
            # Multiple samples visualization
            channel = console.input("[yellow]Enter channel name (or press Enter for first available): [/yellow]").strip()
            if not channel:
                channel = None
            
            n_samples = console.input("[yellow]Number of samples to visualize (default 3): [/yellow]").strip()
            try:
                n_samples = int(n_samples) if n_samples else 3
            except ValueError:
                n_samples = 3
            
            viz_tool.visualize_multiple_samples(X, y, channel, n_samples)
        
        elif choice == '3':
            # Channel comparison
            sample_idx = console.input("[yellow]Enter sample index (or press Enter for random): [/yellow]").strip()
            if sample_idx:
                try:
                    sample_idx = int(sample_idx)
                    if sample_idx >= len(X):
                        console.print(f"[red]Invalid index. Max index is {len(X)-1}[/red]")
                        continue
                except ValueError:
                    console.print("[red]Invalid index. Using random sample.[/red]")
                    sample_idx = np.random.randint(len(X))
            else:
                sample_idx = np.random.randint(len(X))
            
            y_true = y[sample_idx] if y is not None else None
            viz_tool.compare_channels_single_sample(X[sample_idx], y_true, sample_idx)
        
        elif choice == '4':
            console.print("[green]Goodbye![/green]")
            break
        
        else:
            console.print("[red]Invalid choice. Please try again.[/red]")

if __name__ == "__main__":
    main()
