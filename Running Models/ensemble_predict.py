import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os
import sys
import glob
from sklearn.metrics import confusion_matrix
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

class EnsemblePredictor:
    def __init__(self, ensemble_config_path=None):
        self.models = {}
        self.channel_names = []
        self.weights = {}
        
        config_path = ensemble_config_path or self.find_latest_ensemble_config()
        if config_path and os.path.exists(config_path):
            self.load_ensemble_config(config_path)
        else:
            print("No ensemble configuration found. Please train models first.")
    
    def find_latest_ensemble_config(self):
        models_dir = "../models"
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
        print(f"Loading ensemble configuration from: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.channel_names = config['channel_names']
        model_info = config['models']
        ensemble_dir = config.get('ensemble_dir', os.path.dirname(config_path))
        
        print(f"Loading {len(model_info)} models from ensemble directory...")
        
        for channel_name, info in model_info.items():
            model_path = os.path.join(ensemble_dir, f"solar_flare_model_{channel_name}.keras")
            
            if os.path.exists(model_path):
                try:
                    self.models[channel_name] = load_model(model_path)
                    test_accuracy = info.get('test_accuracy', 0.5)
                    self.weights[channel_name] = test_accuracy if test_accuracy and test_accuracy > 0 else 0.5
                    print(f"  ✓ Loaded {channel_name} model (accuracy: {test_accuracy*100:.2f}%)")
                except Exception as e:
                    print(f"  ✗ Failed to load {channel_name} model: {e}")
            else:
                print(f"  ✗ Model file not found: {model_path}")
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        print(f"Model weights: {self.weights}")
    
    def set_custom_weights(self, weights_dict):
        for channel_name, weight in weights_dict.items():
            if channel_name in self.models:
                self.weights[channel_name] = weight
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        print(f"Updated model weights: {self.weights}")
    
    def select_models(self, selected_models):
        """Remove models that are not in the selected_models list"""
        if not selected_models:
            return
        
        # Store original models for reference
        self._original_models = dict(self.models)
        
        # Remove models not in selection
        models_to_remove = [model for model in self.models.keys() if model not in selected_models]
        for model_name in models_to_remove:
            if model_name in self.models:
                del self.models[model_name]
                if model_name in self.weights:
                    del self.weights[model_name]
        
        # Normalize remaining weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        print(f"Selected models: {list(self.models.keys())}")
        print(f"Updated model weights: {self.weights}")
    
    def predict_single_sample(self, X_sample):
        if not self.models:
            raise ValueError("No models loaded. Please load ensemble configuration first.")
        
        individual_predictions = {}
        weighted_sum = 0
        total_weight = 0
        
        for channel_name, model in self.models.items():
            channel_idx = self.channel_names.index(channel_name)
            channel_data = X_sample[:, :, :, channel_idx:channel_idx+1]
            channel_data_batch = np.expand_dims(channel_data, axis=0)
            prediction = model.predict(channel_data_batch, verbose=0)[0][0]
            
            individual_predictions[channel_name] = prediction
            weight = self.weights.get(channel_name, 0)
            weighted_sum += prediction * weight
            total_weight += weight
        
        ensemble_prediction = weighted_sum / total_weight if total_weight > 0 else 0.5
        return ensemble_prediction, individual_predictions
    
    def predict_batch(self, X_batch):
        predictions = []
        all_individual_predictions = []
        
        for i in range(len(X_batch)):
            pred, individual = self.predict_single_sample(X_batch[i])
            predictions.append(pred)
            all_individual_predictions.append(individual)
        
        return np.array(predictions), all_individual_predictions
    
    def evaluate_ensemble(self, X_test, y_test, threshold=0.5):
        print("Evaluating ensemble model...")
        
        ensemble_predictions, individual_predictions = self.predict_batch(X_test)
        ensemble_classes = (ensemble_predictions > threshold).astype(int)
        
        cm = confusion_matrix(y_test, ensemble_classes)
        accuracy = np.mean(ensemble_classes == y_test)
        
        results = {
            'ensemble_accuracy': accuracy,
            'ensemble_predictions': ensemble_predictions,
            'ensemble_classes': ensemble_classes,
            'confusion_matrix': cm,
            'individual_predictions': individual_predictions
        }
        
        # Calculate TSS for binary classification
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            tss = tp/(tp+fn) - fp/(fp+tn) if (tp+fn) > 0 and (fp+tn) > 0 else 0
            results['tss'] = tss
        
        # Evaluate individual models
        individual_accuracies = {}
        individual_tss = {}
        for channel_name in self.models.keys():
            channel_preds = [pred[channel_name] for pred in individual_predictions]
            channel_classes = (np.array(channel_preds) > threshold).astype(int)
            individual_accuracies[channel_name] = np.mean(channel_classes == y_test)
            
            # Calculate TSS for each individual model
            individual_cm = confusion_matrix(y_test, channel_classes)
            if individual_cm.shape == (2, 2):
                tn, fp, fn, tp = individual_cm.ravel()
                tss = tp/(tp+fn) - fp/(fp+tn) if (tp+fn) > 0 and (fp+tn) > 0 else 0
                individual_tss[channel_name] = tss
            else:
                individual_tss[channel_name] = 0
        
        results['individual_accuracies'] = individual_accuracies
        results['individual_tss'] = individual_tss
        return results
    
    def print_evaluation_results(self, results):
        console.print(f"\n[bold blue]{'='*60}[/bold blue]")
        console.print("[bold blue]ENSEMBLE EVALUATION RESULTS[/bold blue]")
        console.print(f"[bold blue]{'='*60}[/bold blue]")
        
        # Main results table
        eval_table = Table(title="Ensemble Performance", box=box.ROUNDED)
        eval_table.add_column("Metric", style="cyan", no_wrap=True)
        eval_table.add_column("Value", style="magenta")
        eval_table.add_row("Ensemble Accuracy", f"{results['ensemble_accuracy']*100:.2f}%")
        if 'tss' in results:
            eval_table.add_row("True Skill Statistic (TSS)", f"{results['tss']:.4f}")
        console.print(eval_table)
        
        # Confusion matrix table
        cm = results['confusion_matrix']
        cm_table = Table(title="Confusion Matrix", box=box.ROUNDED)
        cm_table.add_column("", style="cyan")
        cm_table.add_column("Predicted No Flare", style="green")
        cm_table.add_column("Predicted Flare", style="red")
        cm_table.add_row("Actual No Flare", str(cm[0,0]), str(cm[0,1]))
        cm_table.add_row("Actual Flare", str(cm[1,0]), str(cm[1,1]))
        console.print(cm_table)
        
        # Individual model performance table
        individual_table = Table(title="Individual Model Performance", box=box.ROUNDED)
        individual_table.add_column("Model", style="cyan", no_wrap=True)
        individual_table.add_column("Accuracy", style="green")
        individual_table.add_column("TSS", style="blue")
        individual_table.add_column("Weight", style="yellow")
        
        for channel_name, accuracy in results['individual_accuracies'].items():
            weight = self.weights.get(channel_name, 0)
            tss = results.get('individual_tss', {}).get(channel_name, 0)
            individual_table.add_row(
                channel_name, 
                f"{accuracy*100:.2f}%", 
                f"{tss:.4f}",
                f"{weight:.3f}"
            )
        
        console.print(individual_table)

def predict_on_random_samples():
    try:
        with console.status("[bold green]Loading sample data..."):
            with np.load('processed_solar_data.npz') as data:
                X = data['X']
                y_true = data['y']
        
        console.print(f"[green]Loaded sample data with shape:[/green] {X.shape}")
        
        sample_size = min(10, X.shape[0])
        sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
        
        X_samples = X[sample_indices]
        y_true_samples = y_true[sample_indices]
        
        predictor = EnsemblePredictor()
        if not predictor.models:
            console.print("[red]No models loaded. Please train models first using train_multi_model.py[/red]")
            return
        
        with console.status("[bold green]Making ensemble predictions..."):
            predictions, individual_predictions = predictor.predict_batch(X_samples)
        
        predicted_classes = (predictions > 0.5).astype(int)
        accuracy = np.mean(predicted_classes == y_true_samples) * 100
        
        # Results table
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
        
        # Individual predictions for first 3 samples
        console.print("\n[bold]Individual Model Predictions (First 3 samples):[/bold]")
        for sample_idx in range(min(3, sample_size)):
            individual_table = Table(title=f"Sample {sample_idx + 1} - Individual Predictions", box=box.SIMPLE)
            individual_table.add_column("Model", style="cyan")
            individual_table.add_column("Prediction", style="magenta")
            individual_table.add_column("Weight", style="yellow")
            
            for channel_name, pred_value in individual_predictions[sample_idx].items():
                weight = predictor.weights.get(channel_name, 0)
                individual_table.add_row(channel_name, f"{pred_value:.4f}", f"{weight:.3f}")
            
            console.print(individual_table)
        
        # Summary
        summary_text = f"Sample Accuracy: {accuracy:.2f}%\n"
        summary_text += f"Correct Predictions: {np.sum(predicted_classes == y_true_samples)}/{sample_size}\n"
        summary_text += f"Models Used: {len(predictor.models)}"
        console.print(Panel(summary_text, title="[bold]Summary[/bold]", border_style="green"))
        
    except Exception as e:
        console.print(f"[red]Error loading or predicting data: {e}[/red]")

def load_test_data():
    models_dir = "models"
    
    # Check for latest_ensemble_config.json
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
    
    # Look for ensemble directories
    ensemble_dirs = [d for d in os.listdir(models_dir) 
                    if os.path.isdir(os.path.join(models_dir, d)) and d.startswith('ensemble_')]
    
    if ensemble_dirs:
        latest_ensemble_dir = max(ensemble_dirs)
        test_set_path = os.path.join(models_dir, latest_ensemble_dir, 'test_set.npz')
        
        if os.path.exists(test_set_path):
            console.print(f"[green]Loading test data from:[/green] {test_set_path}")
            with np.load(test_set_path) as data:
                return data['X_test'], data['y_test']
    
    # Fallback: look for test_set.npz files
    test_files = glob.glob(os.path.join(models_dir, "**", "test_set.npz"), recursive=True)
    if test_files:
        latest_test_file = max(test_files, key=os.path.getmtime)
        console.print(f"[yellow]Loading fallback test data from:[/yellow] {latest_test_file}")
        with np.load(latest_test_file) as data:
            return data['X_test'], data['y_test']
    
    # Last resort: create from processed data
    console.print("[yellow]No test set files found. Creating from processed data...[/yellow]")
    try:
        with np.load('processed_solar_data.npz') as data:
            X = data['X']
            y = data['y']
        
        from sklearn.model_selection import train_test_split
        _, X_temp, _, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        
        return X_test, y_test
    except Exception as e:
        console.print(f"[red]Failed to load processed data: {e}[/red]")
        return None, None

def main():
    try:
        console.clear()
        console.print(Panel.fit("[bold orange1]Ensemble Solar Flare Prediction[/bold orange1]", border_style="orange1"))

        # Detect available ensemble configuration files
        models_dir = "models"
        config_files = []

        latest_config_path = os.path.join(models_dir, "latest_ensemble_config.json")
        if os.path.exists(latest_config_path):
            config_files.append(latest_config_path)

        ensemble_dirs = [d for d in os.listdir(models_dir) 
                        if os.path.isdir(os.path.join(models_dir, d)) and d.startswith('ensemble_')]

        for ensemble_dir in ensemble_dirs:
            config_path = os.path.join(models_dir, ensemble_dir, "ensemble_config.json")
            if os.path.exists(config_path):
                config_files.append(config_path)

        if not config_files:
            console.print("[red]No ensemble configuration files found. Please train models first.[/red]")
            return

        console.print("\n[cyan]Available Ensemble Configurations:[/cyan]")
        for idx, config_file in enumerate(config_files, start=1):
            console.print(f"  {idx}. {config_file}")

        config_choice = console.input("\n[yellow]Enter the number corresponding to the configuration you'd like to use: [/yellow]").strip()
        try:
            config_index = int(config_choice) - 1
            if config_index < 0 or config_index >= len(config_files):
                raise ValueError
            config = config_files[config_index]
        except ValueError:
            console.print("[red]Invalid choice. Exiting.[/red]")
            return

        weights = console.input("[yellow]Custom weights as JSON string (press Enter to skip): [/yellow]").strip()
        models = console.input("[yellow]Comma-separated list of models to include (e.g., 'continuum,magnetogram') (press Enter to skip): [/yellow]").strip()
        evaluate = console.input("[yellow]Evaluate on test data? (yes/no): [/yellow]").strip().lower() == 'yes'
        predict = console.input("[yellow]Path to data file for prediction (press Enter to skip): [/yellow]").strip()

        predictor = EnsemblePredictor(config)
        if not predictor.models:
            console.print("[red]No models loaded. Please train models first using train_multi_model.py[/red]")
            return

        if models:
            # Remove quotes if present and split by comma
            models_clean = models.strip('\'"')  # Remove surrounding quotes
            selected_models = [model.strip().strip('\'"') for model in models_clean.split(',')]
            
            # Store available models before selection
            available_models = list(predictor.models.keys())
            
            predictor.select_models(selected_models)
            if not predictor.models:
                console.print("[red]No valid models selected. Please check model names.[/red]")
                console.print(f"[yellow]Available models: {available_models}[/yellow]")
                console.print(f"[yellow]You requested: {selected_models}[/yellow]")
                return

            # Confirm selected models
            console.print("\n[green]Selected models:[/green]")
            for model_name in predictor.models.keys():
                console.print(f"  - {model_name}")

        if weights:
            try:
                custom_weights = json.loads(weights)
                predictor.set_custom_weights(custom_weights)
            except json.JSONDecodeError:
                console.print("[red]Invalid weights format. Please provide a valid JSON string.[/red]")
                return

        if evaluate:
            # Ensure test_set.npz is loaded from the selected ensemble directory
            ensemble_dir = os.path.dirname(config)
            test_set_path = os.path.join(ensemble_dir, 'test_set.npz')
            if os.path.exists(test_set_path):
                console.print(f"[green]Loading test data from:[/green] {test_set_path}")
                with np.load(test_set_path) as data:
                    X_test, y_test = data['X_test'], data['y_test']
            else:
                console.print("[red]Test set not found in the selected ensemble directory.[/red]")
                return

            results = predictor.evaluate_ensemble(X_test, y_test)
            predictor.print_evaluation_results(results)

        elif predict:
            try:
                with np.load(predict) as data:
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

        else:
            console.print(f"\n[green]Ensemble predictor loaded with {len(predictor.models)} models.[/green]")
            console.print("Use 'Evaluate on test data' option to evaluate on test data")
            console.print("Use 'Path to data file for prediction' option to make predictions on new data")
            console.print("Use 'Custom weights' option to set custom model weights")

    except KeyboardInterrupt:
        console.print("\n[red]Cancelled.[/red]")
        exit(0)

if __name__ == "__main__":
    main()