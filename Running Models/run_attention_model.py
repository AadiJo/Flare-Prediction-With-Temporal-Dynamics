import os
# Suppress TensorFlow warnings (they were starting to get annoying)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import json
import sys
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

<<<<<<< HEAD


=======
>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11
console = Console()

class SimpleAttentionPredictor:
    def __init__(self):
        self.attention_model = None
        self.base_models = {}
        self.channel_names = []
        self.config = None
        
    def load_models(self):
        """Load the attention fusion model and base models"""
        console.print("[cyan]Loading attention fusion model...[/cyan]")
        
        # Find the attention fusion config
        config_path = self.find_attention_config()
        if not config_path:
            console.print("[red]No attention fusion model found. Please train the model first.[/red]")
            return False
            
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # Load attention fusion model
        fusion_model_path = self.config['fusion_model_path']
        if os.path.exists(fusion_model_path):
            self.attention_model = tf.keras.models.load_model(fusion_model_path)
            console.print(f"[green]✓ Loaded attention fusion model from: {fusion_model_path}[/green]")
        else:
            console.print(f"[red]Attention fusion model not found at: {fusion_model_path}[/red]")
            return False
            
        # Load base models
        ensemble_config_path = self.find_ensemble_config()
        if ensemble_config_path:
            self.load_base_models(ensemble_config_path)
        else:
            console.print("[red]No ensemble configuration found.[/red]")
            return False
            
        return True
        
    def find_attention_config(self):
        """Find attention fusion configuration"""
<<<<<<< HEAD
        models_dir = "models"
=======
        models_dir = "../models"
>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11
        if not os.path.exists(models_dir):
            return None
            
        # Look for attention fusion config
        config_path = os.path.join(models_dir, "attention_fusion_config.json")
        if os.path.exists(config_path):
            return config_path
            
        # Look in subdirectories
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path):
                config_path = os.path.join(item_path, "attention_fusion_config.json")
                if os.path.exists(config_path):
                    return config_path
        return None
        
    def find_ensemble_config(self):
        """Find ensemble configuration"""
<<<<<<< HEAD
        models_dir = "models"
=======
        models_dir = "../models"
>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11
        if not os.path.exists(models_dir):
            return None
            
        # Look for latest ensemble config
        latest_config_path = os.path.join(models_dir, "latest_ensemble_config.json")
        if os.path.exists(latest_config_path):
            return latest_config_path
            
        # Look for ensemble directories
        ensemble_dirs = [d for d in os.listdir(models_dir) 
                        if os.path.isdir(os.path.join(models_dir, d)) and d.startswith('ensemble_')]
        
        if ensemble_dirs:
            latest_ensemble_dir = max(ensemble_dirs)
            config_path = os.path.join(models_dir, latest_ensemble_dir, "ensemble_config.json")
            return config_path if os.path.exists(config_path) else None
        return None
        
    def load_base_models(self, config_path):
        """Load base models for generating predictions"""
        console.print("[cyan]Loading base models...[/cyan]")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        self.channel_names = config['channel_names']
        model_info = config['models']
        ensemble_dir = config.get('ensemble_dir', os.path.dirname(config_path))
        
        for channel_name, info in model_info.items():
            model_path = os.path.join(ensemble_dir, f"solar_flare_model_{channel_name}.keras")
            
            if os.path.exists(model_path):
                try:
                    model = tf.keras.models.load_model(model_path)
                    self.base_models[channel_name] = model
                    console.print(f"[green]✓ Loaded {channel_name} base model[/green]")
                except Exception as e:
                    console.print(f"[red]✗ Failed to load {channel_name} model: {e}[/red]")
            else:
                console.print(f"[red]✗ Model file not found: {model_path}[/red]")
                
    def load_test_data(self):
        """Load test data"""
        console.print("[cyan]Loading test data...[/cyan]")
        
        # Try to find test_set.npz in the ensemble model directory
        ensemble_config_path = self.find_ensemble_config()
        if ensemble_config_path:
            with open(ensemble_config_path, 'r') as f:
                config = json.load(f)
            ensemble_dir = config.get('ensemble_dir', os.path.dirname(ensemble_config_path))
            test_data_path = os.path.join(ensemble_dir, 'test_set.npz')
        else:
            test_data_path = 'test_set.npz'
        
        try:
            with np.load(test_data_path) as data:
                X = data['X_test']
                y = data['y_test']
            console.print(f"[green]✓ Test data loaded from: {test_data_path}[/green]")
            console.print(f"[green]✓ Data shape: X shape: {X.shape}, y shape: {y.shape}[/green]")
            return X, y
        except FileNotFoundError:
            console.print(f"[red]test_set.npz not found at: {test_data_path}[/red]")
            return None, None
            
    def extract_base_predictions(self, X):
        """Extract predictions from base models"""
        console.print("[cyan]Extracting predictions from base models...[/cyan]")
        
        predictions = {}
        for channel_name, model in self.base_models.items():
            channel_idx = self.channel_names.index(channel_name)
            channel_data = X[:, :, :, :, channel_idx:channel_idx+1]
            
            channel_predictions = model.predict(channel_data, verbose=0)
            predictions[channel_name] = channel_predictions.flatten()
            console.print(f"[green]✓ {channel_name}: {len(channel_predictions)} predictions[/green]")
            
        # Stack predictions
        prediction_matrix = np.column_stack([predictions[channel] for channel in self.base_models.keys()])
        return prediction_matrix
        
    def predict(self, X, y=None):
        """Make predictions using the attention fusion model"""
        console.print("[bold blue]Making predictions...[/bold blue]")
        
        # Convert labels to binary if provided
        if y is not None:
            y_binary = (y > 0).astype(int)
        else:
            y_binary = None
            
        # Extract base model predictions
        base_predictions = self.extract_base_predictions(X)
        
        # Get attention fusion predictions
        fusion_predictions = self.attention_model.predict(base_predictions, verbose=0)
        fusion_classes = (fusion_predictions > 0.5).astype(int).flatten()
        
        # Get individual model predictions for comparison
        individual_predictions = {}
        for channel_name, model in self.base_models.items():
            channel_idx = self.channel_names.index(channel_name)
            channel_data = X[:, :, :, :, channel_idx:channel_idx+1]
            channel_pred = model.predict(channel_data, verbose=0)
            individual_predictions[channel_name] = (channel_pred > 0.5).astype(int).flatten()
            
        results = {
            'fusion_predictions': fusion_classes,
            'fusion_probabilities': fusion_predictions.flatten(),
            'individual_predictions': individual_predictions,
            'y_true': y_binary
        }
        
        return results
        
    def print_results(self, results):
        """Print prediction results"""
        console.print(f"\n[bold blue]{'='*60}[/bold blue]")
        console.print("[bold blue]ATTENTION FUSION MODEL RESULTS[/bold blue]")
        console.print(f"[bold blue]{'='*60}[/bold blue]")
        
        if results['y_true'] is not None:
            y_true = results['y_true']
            fusion_pred = results['fusion_predictions']
            
            # Create results table
            results_table = Table(title="Model Performance", box=box.ROUNDED)
            results_table.add_column("Model", style="cyan")
            results_table.add_column("Accuracy", style="green")
            results_table.add_column("No Flare Correct", style="yellow")
            results_table.add_column("Flare Correct", style="red")
            
            # Individual model results
            for channel_name, pred in results['individual_predictions'].items():
                accuracy = accuracy_score(y_true, pred)
                no_flare_correct = np.sum((y_true == 0) & (pred == 0))
                flare_correct = np.sum((y_true == 1) & (pred == 1))
                results_table.add_row(
                    f"{channel_name} (base)",
                    f"{accuracy:.3f}",
                    str(no_flare_correct),
                    str(flare_correct)
                )
            
            # Fusion model results
            fusion_accuracy = accuracy_score(y_true, fusion_pred)
            fusion_no_flare_correct = np.sum((y_true == 0) & (fusion_pred == 0))
            fusion_flare_correct = np.sum((y_true == 1) & (fusion_pred == 1))
            results_table.add_row(
                "Attention Fusion",
                f"{fusion_accuracy:.3f}",
                str(fusion_no_flare_correct),
                str(fusion_flare_correct)
            )
            
            console.print(results_table)
            
            # Confusion matrix for fusion model
            cm = confusion_matrix(y_true, fusion_pred)
            cm_table = Table(title="Attention Fusion Confusion Matrix", box=box.ROUNDED)
            cm_table.add_column("", style="cyan")
            cm_table.add_column("Pred No Flare", style="green")
            cm_table.add_column("Pred Flare", style="red")
            
            cm_table.add_row("True No Flare", str(cm[0, 0]), str(cm[0, 1]))
            cm_table.add_row("True Flare", str(cm[1, 0]), str(cm[1, 1]))
            
            console.print(cm_table)
            
        else:
            console.print("[yellow]No ground truth labels provided - showing predictions only[/yellow]")
            
        # Sample predictions
        fusion_pred = results['fusion_predictions']
        fusion_probs = results['fusion_probabilities']
        
        sample_table = Table(title="Sample Predictions", box=box.ROUNDED)
        sample_table.add_column("Sample", style="cyan")
        sample_table.add_column("Prediction", style="green")
        sample_table.add_column("Probability", style="yellow")
        
        for i in range(min(10, len(fusion_pred))):
            pred_text = "Flare" if fusion_pred[i] == 1 else "No Flare"
            sample_table.add_row(
                str(i + 1),
                pred_text,
                f"{fusion_probs[i]:.3f}"
            )
            
        console.print(sample_table)

def main():
    """Main function to run the attention fusion model"""
    try:
        console.clear()
        console.print(Panel.fit("[bold orange1]Solar Flare Prediction - Attention Fusion Model[/bold orange1]", 
                                border_style="orange1"))
        
        # Initialize predictor
        predictor = SimpleAttentionPredictor()
        
        # Load models
        if not predictor.load_models():
            console.print("[red]Failed to load models. Exiting.[/red]")
            return
            
        # Load test data
        X, y = predictor.load_test_data()
        if X is None:
            console.print("[red]Failed to load test data. Exiting.[/red]")
            return
            
        # Make predictions
        results = predictor.predict(X, y)
        
        # Print results
        predictor.print_results(results)
        
        console.print(f"\n[green]Prediction completed successfully![/green]")
        console.print(f"[green]Total samples processed: {len(results['fusion_predictions'])}[/green]")
        
    except KeyboardInterrupt:
        console.print("\n[red]Prediction cancelled by user.[/red]")
    except Exception as e:
        console.print(f"[red]Error during prediction: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
