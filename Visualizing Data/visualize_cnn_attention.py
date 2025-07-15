import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import json, os, cv2, warnings
from sklearn.preprocessing import MinMaxScaler
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

warnings.filterwarnings('ignore')
console = Console()

class CNNVisualizationTool:
    def __init__(self, ensemble_config_path=None):
        self.models, self.channel_names, self.ensemble_dir = {}, [], None
        config_path = ensemble_config_path or self.find_latest_ensemble_config()
        if config_path and os.path.exists(config_path): self.load_ensemble_config(config_path)
        else: console.print("[red]No ensemble configuration found. Please train models first.[/red]")

    def find_latest_ensemble_config(self):
        models_dir = "models"
        latest_config_path = os.path.join(models_dir, "latest_ensemble_config.json")
        if os.path.exists(latest_config_path): return latest_config_path
        ensemble_dirs = sorted([d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) and d.startswith('ensemble_')], reverse=True)
        if not ensemble_dirs: return None
        config_path = os.path.join(models_dir, ensemble_dirs[0], "ensemble_config.json")
        return config_path if os.path.exists(config_path) else None

    def load_ensemble_config(self, config_path):
        console.print(f"[green]Loading ensemble configuration from:[/green] {config_path}")
        with open(config_path, 'r') as f: config = json.load(f)
        self.channel_names = config['channel_names']
        self.ensemble_dir = config.get('ensemble_dir', os.path.dirname(config_path))
        console.print(f"[cyan]Loading {len(config['models'])} models...[/cyan]")
        for channel_name in config['models']:
<<<<<<< HEAD
            model_path = os.path.join(self.ensemble_dir, f"solar_flare_model_{channel_name}.keras")
=======
            model_filename = f"solar_flare_model_{channel_name}.keras"
            model_path = os.path.normpath(os.path.join(self.ensemble_dir, model_filename))
>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11
            try:
                if not os.path.exists(model_path): raise FileNotFoundError(f"Model file not found: {model_path}")
                self.models[channel_name] = load_model(model_path)
                console.print(f"  ✓ Loaded {channel_name} model")
            except Exception as e: console.print(f"  ✗ Failed to load {channel_name} model: {e}")

    def create_gradcam_heatmap(self, model, img_array, last_conv_layer_name=None, pred_index=None):
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32) if isinstance(img_array, np.ndarray) else img_array
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            score = model(img_tensor)[0][0]
        grads = tape.gradient(score, img_tensor)
        if grads is None:
            console.print("[yellow]Warning: Could not compute gradients[/yellow]"); return None
        axis_to_reduce = [0, 3] if len(grads.shape) == 5 else -1
        activation_map = tf.reduce_mean(tf.abs(grads[0]), axis=axis_to_reduce)
        activation_map = (activation_map - tf.reduce_min(activation_map)) / (tf.reduce_max(activation_map) + 1e-8)
        console.print(f"[green]Created simple activation map with shape: {activation_map.shape}[/green]")
        return activation_map.numpy()

    def create_saliency_map(self, model, img_array):
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32) if isinstance(img_array, np.ndarray) else img_array
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            loss = model(img_tensor)[0][0]
        grads = tf.math.abs(tape.gradient(loss, img_tensor))
        if len(grads.shape) == 5: grads = tf.reduce_mean(grads, axis=1)
        grads = tf.reduce_max(grads, axis=-1) if grads.shape[-1] > 1 else tf.squeeze(grads, axis=-1)
        return (grads / (tf.reduce_max(grads) + 1e-8))[0].numpy()

    def visualize_sample(self, X_sample, y_true=None, channel_name=None, sample_idx=0):
        if not self.models: return console.print("[red]No models loaded![/red]")
        channel_name = channel_name or list(self.models.keys())[0]
        if channel_name not in self.models: return console.print(f"[red]Model for channel '{channel_name}' not found![/red]")
        model = self.models[channel_name]
        channel_idx = self.channel_names.index(channel_name)
        channel_data = X_sample[:, :, :, channel_idx:channel_idx+1]
        channel_data_batch = np.expand_dims(channel_data, axis=0)
        channel_data_tensor = tf.convert_to_tensor(channel_data_batch, dtype=tf.float32)
        prediction = model.predict(channel_data_batch, verbose=0)[0][0]
        predicted_class, actual_class = ("Flare" if prediction > 0.5 else "No Flare"), ("Flare" if y_true == 1 else "No Flare" if y_true is not None else "N/A")
        console.print(f"\n[bold]Visualizing {channel_name} model[/bold]\nPrediction: {prediction:.4f} ({predicted_class}) | Actual: {actual_class}")
        try:
            heatmap = self.create_gradcam_heatmap(model, channel_data_tensor)
            saliency = self.create_saliency_map(model, channel_data_tensor)
            self.plot_visualization(channel_data, heatmap, saliency, channel_name, prediction, y_true, sample_idx)
        except Exception as e:
            console.print(f"[red]Error creating visualizations: {e}[/red]")
            self.plot_simple_visualization(channel_data, channel_name, prediction, y_true, sample_idx)

    def _plot_setup(self, channel_data, channel_name, prediction, y_true, sample_idx):
        n_timesteps = channel_data.shape[0]
        n_images_to_show = 6
        # Always spread 6 images evenly across the full time sequence
        timesteps_to_show = np.unique(np.linspace(0, n_timesteps - 1, n_images_to_show, dtype=int))
        time_interval = (n_timesteps - 1) * 12 / (n_images_to_show - 1) if n_images_to_show > 1 else 12
        title_text = f"CNN Attention - {channel_name} | Sample {sample_idx}\n"
        title_text += f"Prediction: {prediction:.4f} ({'Flare' if prediction > 0.5 else 'No Flare'})"
        if y_true is not None: title_text += f" | Actual: {'Flare' if y_true == 1 else 'No Flare'}"
        title_text += f"\nTimesteps: {timesteps_to_show} (evenly spaced across {n_timesteps} total timesteps)"
        plots_dir = "plots"; os.makedirs(plots_dir, exist_ok=True)
        return timesteps_to_show, title_text, plots_dir

    def plot_visualization(self, channel_data, heatmap, saliency, channel_name, prediction, y_true, sample_idx):
        timesteps_to_show, title_text, plots_dir = self._plot_setup(channel_data, channel_name, prediction, y_true, sample_idx)
        n_cols, n_rows = len(timesteps_to_show), 4
        fig = plt.figure(figsize=(3 * n_cols, 12)); fig.suptitle(title_text, fontsize=12, fontweight='bold')
        for col_idx, t in enumerate(timesteps_to_show):
            img = channel_data[t, :, :, 0]
            # For saliency, use the timestep if it's 3D, otherwise use the full 2D map
            saliency_data = saliency[t] if saliency is not None and len(saliency.shape) == 3 else saliency
            for row_idx, (map_data, map_name, cmap, alpha) in enumerate([
                (img, f'Original\nt={t}', 'gray', 1.0),
                (saliency_data, 'Saliency', 'hot', 1.0),
                (cv2.resize(heatmap, (img.shape[1], img.shape[0])) if heatmap is not None else None, 'Grad-CAM', 'jet', 0.8),
                (cv2.resize(heatmap, (img.shape[1], img.shape[0])) if heatmap is not None else None, 'Overlay', 'jet', 0.4)
            ]):
                if map_data is None: continue
                ax = plt.subplot(n_rows, n_cols, row_idx * n_cols + col_idx + 1)
                if map_name == 'Overlay': ax.imshow(img, cmap='gray')
                im = ax.imshow(map_data, cmap=cmap, alpha=alpha); ax.set_title(map_name); ax.axis('off')
                if map_name != 'Overlay': plt.colorbar(im, ax=ax, shrink=0.6)
        filepath = os.path.join(plots_dir, f"cnn_attention_{channel_name}_sample_{sample_idx}.png")
        plt.tight_layout(); plt.savefig(filepath, dpi=300, bbox_inches='tight'); plt.show()
        console.print(f"[green]Visualization saved to:[/green] {filepath}")

    def plot_simple_visualization(self, channel_data, channel_name, prediction, y_true, sample_idx):
        timesteps_to_show, title_text, plots_dir = self._plot_setup(channel_data, channel_name, prediction, y_true, sample_idx)
        n_cols = min(len(timesteps_to_show), 3); n_rows = int(np.ceil(len(timesteps_to_show) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows)); fig.suptitle(title_text, fontsize=12, fontweight='bold')
        axes = np.array(axes).flatten()
        for idx, t in enumerate(timesteps_to_show):
            im = axes[idx].imshow(channel_data[t, :, :, 0], cmap='gray')
            axes[idx].set_title(f't={t}'); axes[idx].axis('off'); plt.colorbar(im, ax=axes[idx], shrink=0.6)
        for idx in range(len(timesteps_to_show), len(axes)): axes[idx].axis('off')
        filepath = os.path.join(plots_dir, f"simple_viz_{channel_name}_sample_{sample_idx}.png")
        plt.tight_layout(); plt.savefig(filepath, dpi=300, bbox_inches='tight'); plt.show()
        console.print(f"[green]Simple visualization saved to:[/green] {filepath}")

    def visualize_multiple_samples(self, X_data, y_data=None, channel_name=None, n_samples=3):
        n_samples = min(n_samples, len(X_data))
        sample_indices = np.random.choice(len(X_data), n_samples, replace=False)
        console.print(f"\n[bold blue]Visualizing {n_samples} random samples[/bold blue]")
        for i, sample_idx in enumerate(sample_indices):
            console.print(f"\n[cyan]--- Sample {i+1}/{n_samples} (Index: {sample_idx}) ---[/cyan]")
            self.visualize_sample(X_data[sample_idx], y_data[sample_idx] if y_data is not None else None, channel_name, sample_idx)

    def compare_channels_single_sample(self, X_sample, y_true=None, sample_idx=0):
        if not self.models: return console.print("[red]No models loaded![/red]")
        console.print(f"\n[bold blue]Comparing channel attention for sample {sample_idx}[/bold blue]")
        n_channels = len(self.models)
        fig, axes = plt.subplots(2, n_channels, figsize=(5 * n_channels, 10))
        if n_channels == 1: axes = axes.reshape(2, 1)
        predictions, table = {}, Table(title="Channel Predictions")
        table.add_column("Channel", style="cyan"); table.add_column("Prediction", style="magenta"); table.add_column("Class", style="yellow")
        for idx, (channel_name, model) in enumerate(self.models.items()):
            channel_idx = self.channel_names.index(channel_name)
            channel_data = X_sample[:, :, :, channel_idx:channel_idx+1]
            channel_data_tensor = tf.convert_to_tensor(np.expand_dims(channel_data, axis=0), dtype=tf.float32)
            prediction = model.predict(channel_data_tensor, verbose=0)[0][0]
            predictions[channel_name] = prediction
            axes[0, idx].imshow(channel_data[0, :, :, 0], cmap='gray'); axes[0, idx].set_title(f'{channel_name}\nPred: {prediction:.3f}'); axes[0, idx].axis('off')
            try:
                saliency = self.create_saliency_map(model, channel_data_tensor)
                saliency_show = saliency[0] if len(saliency.shape) == 3 else saliency
                axes[1, idx].imshow(saliency_show, cmap='hot'); axes[1, idx].set_title('Saliency Map'); axes[1, idx].axis('off')
            except Exception as e:
                axes[1, idx].text(0.5, 0.5, f'Error:\n{str(e)[:30]}...', ha='center', va='center', transform=axes[1, idx].transAxes); axes[1, idx].axis('off')
            table.add_row(channel_name, f"{prediction:.4f}", "Flare" if prediction > 0.5 else "No Flare")
        title = f"Channel Comparison - Sample {sample_idx}" + (f" | Actual: {'Flare' if y_true == 1 else 'No Flare'}" if y_true is not None else "")
        fig.suptitle(title, fontsize=16, fontweight='bold'); plt.tight_layout()
        plots_dir = "plots"; os.makedirs(plots_dir, exist_ok=True)
        filepath = os.path.join(plots_dir, f"channel_comparison_sample_{sample_idx}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight'); plt.show()
        console.print(f"[green]Channel comparison saved to:[/green] {filepath}")
        console.print(table)

def load_sample_data():
    try:
        console.print("[cyan]Loading sample data...[/cyan]")
<<<<<<< HEAD
        with np.load('processed_solar_data.npz') as data: X, y = data['X'], data['y']
=======
        with np.load('processed_HED_data.npz') as data: X, y = data['X'], data['y']
>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11
        console.print(f"[green]Loaded data with shape:[/green] X={X.shape}, y={y.shape}")
        return X, y
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]"); return None, None

def get_user_input(prompt, default=None, type_cast=str):
    val = console.input(prompt).strip()
    if not val and default is not None: return default
    try: return type_cast(val)
    except (ValueError, TypeError): return default

def main():
    console.clear()
    console.print(Panel.fit("[bold orange1]CNN Attention Visualization Tool[/bold orange1]", border_style="orange1"))
    viz_tool = CNNVisualizationTool()
    if not viz_tool.models: return console.print("[red]No models loaded. Exiting.[/red]")
    X, y = load_sample_data()
    if X is None: return
    while (choice := console.input("\n[bold cyan]Options: (1) Single Sample (2) Multiple Samples (3) Compare Channels (4) Exit [/bold cyan]\n[yellow]Select: [/yellow]")) != '4':
        if choice not in ['1', '2', '3']: console.print("[red]Invalid choice.[/red]"); continue
        try:
            idx_str = get_user_input(f"[yellow]Enter sample index (0-{len(X)-1}, Enter for random): [/yellow]", '')
            sample_idx = int(idx_str) if idx_str and 0 <= int(idx_str) < len(X) else np.random.randint(len(X))
            if idx_str and not (0 <= int(idx_str) < len(X)): console.print("[red]Index out of range. Using random.[/red]")
        except ValueError: sample_idx = np.random.randint(len(X)); console.print("[red]Invalid index. Using random.[/red]")
        y_true = y[sample_idx] if y is not None else None
        if choice == '1' or choice == '2':
            channel = get_user_input(f"[yellow]Channel {viz_tool.channel_names} (Enter for first): [/yellow]") or None
        if choice == '1': viz_tool.visualize_sample(X[sample_idx], y_true, channel, sample_idx)
        elif choice == '2':
            n_samples = get_user_input("[yellow]Number of samples (default 3): [/yellow]", 3, int)
            viz_tool.visualize_multiple_samples(X, y, channel, n_samples)
        elif choice == '3': viz_tool.compare_channels_single_sample(X[sample_idx], y_true, sample_idx)
    console.print("[green]Goodbye![/green]")

if __name__ == "__main__":
    main()