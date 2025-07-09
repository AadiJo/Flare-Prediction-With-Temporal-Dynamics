import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os
import sys
import glob
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# Initialize Rich console
console = Console()

class EnsemblePredictor:
    def __init__(self, ensemble_config_path=None):
        """
        Initialize the ensemble predictor.
        
        Args:
            ensemble_config_path: Path to ensemble configuration file. 
                                If None, will look for the latest one.
        """
        self.models = {}
        self.channel_names = []
        self.weights = {}
        
        if ensemble_config_path is None:
            ensemble_config_path = self.find_latest_ensemble_config()
        
        if ensemble_config_path and os.path.exists(ensemble_config_path):
            self.load_ensemble_config(ensemble_config_path)
        else:
            print("No ensemble configuration found. Please train models first.")
    
    def find_latest_ensemble_config(self):
        """Find the latest ensemble configuration file."""
        models_dir = "models"
        
        # First check for latest_ensemble_config.json
        latest_config_path = os.path.join(models_dir, "latest_ensemble_config.json")
        if os.path.exists(latest_config_path):
            return latest_config_path
        
        # Otherwise look for ensemble directories
        ensemble_dirs = [d for d in os.listdir(models_dir) 
                        if os.path.isdir(os.path.join(models_dir, d)) and d.startswith('ensemble_')]
        
        if not ensemble_dirs:
            return None
        
        # Get the most recent ensemble directory
        latest_ensemble_dir = max(ensemble_dirs)
        config_path = os.path.join(models_dir, latest_ensemble_dir, "ensemble_config.json")
        
        if os.path.exists(config_path):
            return config_path
        
        return None
    
    def load_ensemble_config(self, config_path):
        """Load ensemble configuration and models."""
        print(f"Loading ensemble configuration from: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.channel_names = config['channel_names']
        model_info = config['models']
        ensemble_dir = config.get('ensemble_dir', os.path.dirname(config_path))
        
        print(f"Loading {len(model_info)} models from ensemble directory...")
        
        # Load each model
        for channel_name, info in model_info.items():
            # Construct model path relative to ensemble directory
            model_filename = f"solar_flare_model_{channel_name}.keras"
            model_path = os.path.join(ensemble_dir, model_filename)
            
            if os.path.exists(model_path):
                try:
                    model = load_model(model_path)
                    self.models[channel_name] = model
                    
                    # Initialize weights based on test accuracy
                    test_accuracy = info.get('test_accuracy', 0.5)
                    self.weights[channel_name] = test_accuracy if test_accuracy and test_accuracy > 0 else 0.5
                    
                    print(f"  ✓ Loaded {channel_name} model (accuracy: {test_accuracy*100:.2f}%)")
                except Exception as e:
                    print(f"  ✗ Failed to load {channel_name} model: {e}")
            else:
                print(f"  ✗ Model file not found: {model_path}")
        
        # Normalize weights so they sum to 1
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        print(f"Model weights: {self.weights}")
    
    def set_custom_weights(self, weights_dict):
        """
        Set custom weights for ensemble averaging.
        
        Args:
            weights_dict: Dictionary mapping channel names to weights
        """
        for channel_name in weights_dict:
            if channel_name in self.models:
                self.weights[channel_name] = weights_dict[channel_name]
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        print(f"Updated model weights: {self.weights}")
    
    def predict_single_sample(self, X_sample):
        """
        Predict for a single sample using ensemble averaging.
        
        Args:
            X_sample: Input data with shape (timesteps, height, width, channels)
            
        Returns:
            prediction: Ensemble prediction probability
            individual_predictions: Dictionary of individual model predictions
        """
        if not self.models:
            raise ValueError("No models loaded. Please load ensemble configuration first.")
        
        individual_predictions = {}
        weighted_sum = 0
        total_weight = 0
        
        # Get predictions from each model
        for channel_name, model in self.models.items():
            channel_idx = self.channel_names.index(channel_name)
            
            # Extract the specific channel data
            channel_data = X_sample[:, :, :, channel_idx:channel_idx+1]
            
            # Add batch dimension and predict
            channel_data_batch = np.expand_dims(channel_data, axis=0)
            prediction = model.predict(channel_data_batch, verbose=0)[0][0]
            
            individual_predictions[channel_name] = prediction
            
            # Add to weighted sum
            weight = self.weights.get(channel_name, 0)
            weighted_sum += prediction * weight
            total_weight += weight
        
        # Calculate ensemble prediction
        ensemble_prediction = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        return ensemble_prediction, individual_predictions
    
    def predict_batch(self, X_batch):
        """
        Predict for a batch of samples.
        
        Args:
            X_batch: Input data with shape (samples, timesteps, height, width, channels)
            
        Returns:
            predictions: Array of ensemble predictions
            all_individual_predictions: List of dictionaries with individual predictions
        """
        predictions = []
        all_individual_predictions = []
        
        for i in range(len(X_batch)):
            pred, individual = self.predict_single_sample(X_batch[i])
            predictions.append(pred)
            all_individual_predictions.append(individual)
        
        return np.array(predictions), all_individual_predictions
    
    def evaluate_ensemble(self, X_test, y_test, threshold=0.5):
        """
        Evaluate the ensemble model on test data.
        
        Args:
            X_test: Test input data
            y_test: Test labels
            threshold: Classification threshold
            
        Returns:
            results: Dictionary containing evaluation metrics
        """
        print("Evaluating ensemble model...")
        
        # Get ensemble predictions
        ensemble_predictions, individual_predictions = self.predict_batch(X_test)
        ensemble_classes = (ensemble_predictions > threshold).astype(int)
        
        # Calculate ensemble metrics
        cm = confusion_matrix(y_test, ensemble_classes)
        accuracy = np.mean(ensemble_classes == y_test)
        
        results = {
            'ensemble_accuracy': accuracy,
            'ensemble_predictions': ensemble_predictions,
            'ensemble_classes': ensemble_classes,
            'confusion_matrix': cm,
            'individual_predictions': individual_predictions
        }
        
        # Calculate TSS if we have binary classification
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            tss = tp/(tp+fn) - fp/(fp+tn) if (tp+fn) > 0 and (fp+tn) > 0 else 0
            results['tss'] = tss
        
        # Evaluate individual models
        individual_accuracies = {}
        for channel_name in self.models.keys():
            channel_preds = [pred[channel_name] for pred in individual_predictions]
            channel_classes = (np.array(channel_preds) > threshold).astype(int)
            individual_accuracies[channel_name] = np.mean(channel_classes == y_test)
        
        results['individual_accuracies'] = individual_accuracies
        
        return results
    
    def print_evaluation_results(self, results):
        """Print detailed evaluation results using Rich formatting."""
        console.print(f"\n[bold blue]{'='*60}[/bold blue]")
        console.print("[bold blue]ENSEMBLE EVALUATION RESULTS[/bold blue]")
        console.print(f"[bold blue]{'='*60}[/bold blue]")
        
        # Create main results table
        eval_table = Table(title="Ensemble Performance", box=box.ROUNDED)
        eval_table.add_column("Metric", style="cyan", no_wrap=True)
        eval_table.add_column("Value", style="magenta")
        eval_table.add_row("Ensemble Accuracy", f"{results['ensemble_accuracy']*100:.2f}%")
        if 'tss' in results:
            eval_table.add_row("True Skill Statistic (TSS)", f"{results['tss']:.4f}")
        console.print(eval_table)
        
        # Create confusion matrix table
        cm = results['confusion_matrix']
        cm_table = Table(title="Confusion Matrix", box=box.ROUNDED)
        cm_table.add_column("", style="cyan")
        cm_table.add_column("Predicted No Flare", style="green")
        cm_table.add_column("Predicted Flare", style="red")
        cm_table.add_row("Actual No Flare", str(cm[0,0]), str(cm[0,1]))
        cm_table.add_row("Actual Flare", str(cm[1,0]), str(cm[1,1]))
        console.print(cm_table)
        
        # Create individual model performance table
        individual_table = Table(title="Individual Model Performance", box=box.ROUNDED)
        individual_table.add_column("Model", style="cyan", no_wrap=True)
        individual_table.add_column("Accuracy", style="green")
        individual_table.add_column("Weight", style="yellow")
        
        for channel_name, accuracy in results['individual_accuracies'].items():
            weight = self.weights.get(channel_name, 0)
            individual_table.add_row(channel_name, f"{accuracy*100:.2f}%", f"{weight:.3f}")
        
        console.print(individual_table)

def predict_on_random_samples():
    """Predict on 10 random samples from the processed data."""
    try:
        with console.status("[bold green]Loading sample data..."):
            with np.load('processed_solar_data.npz') as data:
                X = data['X']
                y_true = data['y']
        
        console.print(f"[green]Loaded sample data with shape:[/green] {X.shape}")
        
        # Choose 10 random samples
        sample_size = min(10, X.shape[0])
        sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
        
        X_samples = X[sample_indices]
        y_true_samples = y_true[sample_indices]
        
        # Initialize ensemble predictor
        predictor = EnsemblePredictor()
        
        if not predictor.models:
            console.print("[red]No models loaded. Please train models first using train_multi_model.py[/red]")
            return
        
        with console.status("[bold green]Making ensemble predictions..."):
            predictions, individual_predictions = predictor.predict_batch(X_samples)
        
        # Calculate accuracy
        predicted_classes = (predictions > 0.5).astype(int)
        accuracy = np.mean(predicted_classes == y_true_samples) * 100
        
        # Create results table
        results_table = Table(title=f"Ensemble Prediction Results (Accuracy: {accuracy:.2f}%)", box=box.ROUNDED)
        results_table.add_column("Sample", style="cyan", no_wrap=True)
        results_table.add_column("Ensemble Score", style="magenta")
        results_table.add_column("Predicted Class", style="yellow")
        results_table.add_column("Actual Class", style="green")
        results_table.add_column("Correct", style="blue")
        
        for i in range(sample_size):
            predicted_class = "Flare" if predicted_classes[i] == 1 else "No Flare"
            actual_class = "Flare" if y_true_samples[i] == 1 else "No Flare"
            is_correct = "✓" if predicted_classes[i] == y_true_samples[i] else "✗"
            correct_style = "green" if predicted_classes[i] == y_true_samples[i] else "red"
            
            results_table.add_row(
                str(i+1),
                f"{predictions[i]:.4f}",
                predicted_class,
                actual_class,
                f"[{correct_style}]{is_correct}[/{correct_style}]"
            )
        
        console.print("\n")
        console.print(results_table)
        
        # Create individual predictions table for first few samples
        console.print("\n[bold]Individual Model Predictions (First 3 samples):[/bold]")
        for sample_idx in range(min(3, sample_size)):
            individual_table = Table(title=f"Sample {sample_idx + 1} - Individual Predictions", box=box.SIMPLE)
            individual_table.add_column("Model", style="cyan")
            individual_table.add_column("Prediction", style="magenta")
            individual_table.add_column("Weight", style="yellow")
            
            individual_pred = individual_predictions[sample_idx]
            for channel_name, pred_value in individual_pred.items():
                weight = predictor.weights.get(channel_name, 0)
                individual_table.add_row(channel_name, f"{pred_value:.4f}", f"{weight:.3f}")
            
            console.print(individual_table)
        
        # Summary panel
        summary_text = f"Sample Accuracy: {accuracy:.2f}%\n"
        summary_text += f"Correct Predictions: {np.sum(predicted_classes == y_true_samples)}/{sample_size}\n"
        summary_text += f"Models Used: {len(predictor.models)}"
        console.print(Panel(summary_text, title="[bold]Summary[/bold]", border_style="green"))
        
    except Exception as e:
        console.print(f"[red]Error loading or predicting data: {e}[/red]")

def load_test_data():
    """Load test data from the latest ensemble directory."""
    models_dir = "models"
    
    # First check for latest_ensemble_config.json to find the ensemble directory
    latest_config_path = os.path.join(models_dir, "latest_ensemble_config.json")
    if os.path.exists(latest_config_path):
        try:
            with open(latest_config_path, 'r') as f:
                config = json.load(f)
            
            test_set_path = config.get('test_set_path')
            if test_set_path and os.path.exists(test_set_path):
                console.print(f"[green]Loading test data from:[/green] {test_set_path}")
                with np.load(test_set_path) as data:
                    return data['X_test'], data['y_test']
        except Exception as e:
            console.print(f"[yellow]Failed to load from config: {e}[/yellow]")
    
    # Look for ensemble directories and their test sets
    ensemble_dirs = [d for d in os.listdir(models_dir) 
                    if os.path.isdir(os.path.join(models_dir, d)) and d.startswith('ensemble_')]
    
    if ensemble_dirs:
        # Get the most recent ensemble directory
        latest_ensemble_dir = max(ensemble_dirs)
        test_set_path = os.path.join(models_dir, latest_ensemble_dir, 'test_set.npz')
        
        if os.path.exists(test_set_path):
            console.print(f"[green]Loading test data from:[/green] {test_set_path}")
            with np.load(test_set_path) as data:
                return data['X_test'], data['y_test']
    
    # Fallback: look for individual test_set.npz files
    test_files = glob.glob(os.path.join(models_dir, "**", "test_set.npz"), recursive=True)
    
    if test_files:
        latest_test_file = max(test_files, key=os.path.getmtime)
        console.print(f"[yellow]Loading fallback test data from:[/yellow] {latest_test_file}")
        with np.load(latest_test_file) as data:
            return data['X_test'], data['y_test']
    
    # Last resort: create test data from processed data
    console.print("[yellow]No test set files found. Creating from processed data...[/yellow]")
    try:
        with np.load('processed_solar_data.npz') as data:
            X = data['X']
            y = data['y']
        
        # Use the same random state as training for consistency
        from sklearn.model_selection import train_test_split
        _, X_temp, _, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        
        return X_test, y_test
    except Exception as e:
        console.print(f"[red]Failed to load processed data: {e}[/red]")
        return None, None
    """Load test data from the latest model directory."""
    # Look for test_set.npz files in model directories
    test_files = glob.glob(os.path.join("models", "**", "test_set.npz"), recursive=True)
    
    if not test_files:
        print("No test set files found. Trying to load from processed data...")
        # Try to create test data from the main processed file
        try:
            with np.load('processed_solar_data.npz') as data:
                X = data['X']
                y = data['y']
            
            # Use the same random state as training for consistency
            from sklearn.model_selection import train_test_split
            _, X_temp, _, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
            
            return X_test, y_test
        except Exception as e:
            print(f"Failed to load processed data: {e}")
            return None, None
    
    # Use the most recent test file
    latest_test_file = max(test_files, key=os.path.getmtime)
    print(f"Loading test data from: {latest_test_file}")
    
    with np.load(latest_test_file) as data:
        return data['X_test'], data['y_test']

def main():
    try:
        console.clear()
        console.print(Panel.fit("[bold orange1]Ensemble Solar Flare Prediction[/bold orange1]", border_style="orange1"))
        
        # Check if we have command line arguments
        if len(sys.argv) > 1:
            # Command line mode
            parser = argparse.ArgumentParser(description='Ensemble Solar Flare Prediction')
            parser.add_argument('--config', type=str, help='Path to ensemble configuration file')
            parser.add_argument('--weights', type=str, help='Custom weights as JSON string (e.g., \'{"Bp": 0.3, "Br": 0.3, "Bt": 0.2, "continuum": 0.2}\')')
            parser.add_argument('--evaluate', action='store_true', help='Evaluate on test data')
            parser.add_argument('--predict', type=str, help='Path to data file for prediction')
            
            args = parser.parse_args()
            
            # Initialize ensemble predictor
            predictor = EnsemblePredictor(args.config)
            
            if not predictor.models:
                console.print("[red]No models loaded. Please train models first using train_multi_model.py[/red]")
                return
            
            # Set custom weights if provided
            if args.weights:
                try:
                    custom_weights = json.loads(args.weights)
                    predictor.set_custom_weights(custom_weights)
                except json.JSONDecodeError:
                    console.print("[red]Invalid weights format. Please provide a valid JSON string.[/red]")
                    return
            
            # Evaluation mode
            if args.evaluate:
                X_test, y_test = load_test_data()
                if X_test is not None and y_test is not None:
                    results = predictor.evaluate_ensemble(X_test, y_test)
                    predictor.print_evaluation_results(results)
                else:
                    console.print("[red]No test data available for evaluation.[/red]")
            
            # Prediction mode
            elif args.predict:
                try:
                    with np.load(args.predict) as data:
                        X_pred = data['X'] if 'X' in data else data['X_test']
                        y_true = data['y'] if 'y' in data else data.get('y_test', None)
                    
                    predictions, individual_preds = predictor.predict_batch(X_pred)
                    
                    console.print(f"\n[green]Predictions for {len(predictions)} samples:[/green]")
                    for i, (pred, individual) in enumerate(zip(predictions, individual_preds)):
                        console.print(f"Sample {i+1}: {pred:.4f}")
                        if y_true is not None:
                            console.print(f"  True label: {y_true[i]}")
                        console.print(f"  Individual predictions: {individual}")
                        console.print()
                
                except Exception as e:
                    console.print(f"[red]Failed to load prediction data: {e}[/red]")
            
            # Interactive mode info
            else:
                predictor = EnsemblePredictor()
                console.print(f"\n[green]Ensemble predictor loaded with {len(predictor.models)} models.[/green]")
                console.print("Use --evaluate to evaluate on test data")
                console.print("Use --predict <file> to make predictions on new data")
                console.print("Use --weights to set custom model weights")
        
        else:
            # Interactive mode
            # Create options menu
            options_table = Table(box=box.SIMPLE)
            options_table.add_column("Option", style="cyan", no_wrap=True)
            options_table.add_column("Description", style="white")
            options_table.add_row("1", "Evaluate ensemble on saved test data")
            options_table.add_row("2", "Test ensemble on 10 random samples from processed data")
            options_table.add_row("3", "Set custom model weights and test")
            
            console.print("\n")
            console.print(options_table)
            choice = console.input("\n[yellow]Enter your choice (1/2/3): [/yellow]")
            
            if choice == '1':
                # Evaluate on the saved test data
                predictor = EnsemblePredictor()
                if not predictor.models:
                    console.print("[red]No models loaded. Please train models first using train_multi_model.py[/red]")
                    return
                
                X_test, y_test = load_test_data()
                if X_test is not None and y_test is not None:
                    results = predictor.evaluate_ensemble(X_test, y_test)
                    predictor.print_evaluation_results(results)
                else:
                    console.print("[red]No test data available for evaluation.[/red]")
            
            elif choice == '2':
                # Test on random samples
                predict_on_random_samples()
            
            elif choice == '3':
                # Set custom weights and test
                predictor = EnsemblePredictor()
                if not predictor.models:
                    console.print("[red]No models loaded. Please train models first using train_multi_model.py[/red]")
                    return
                
                console.print(f"\n[cyan]Current models and weights:[/cyan]")
                for channel, weight in predictor.weights.items():
                    console.print(f"  {channel}: {weight:.3f}")
                
                console.print("\n[yellow]Enter custom weights (press Enter to use defaults):[/yellow]")
                custom_weights = {}
                for channel in predictor.models.keys():
                    weight_input = console.input(f"Weight for {channel} (current: {predictor.weights[channel]:.3f}): ")
                    if weight_input.strip():
                        try:
                            custom_weights[channel] = float(weight_input)
                        except ValueError:
                            console.print(f"[red]Invalid weight for {channel}, using default[/red]")
                
                if custom_weights:
                    predictor.set_custom_weights(custom_weights)
                
                # Now test on random samples
                predict_on_random_samples()
            
            else:
                console.print("[red]Invalid choice. Please run again and select 1, 2, or 3.[/red]")
    
    except KeyboardInterrupt:
        console.print("\n[red]Cancelled.[/red]")
        exit(0)

if __name__ == "__main__":
    main()
