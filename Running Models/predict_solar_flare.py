import os
import numpy as np
print("Starting...")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info and warning messages
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import confusion_matrix, classification_report
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box


# Initialize Rich console
console = Console()

<<<<<<< HEAD
def load_latest_model(models_dir="models"):
=======
def load_latest_model(models_dir="../models"):
>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11
    """Load the most recent model from the models directory."""
    if not os.path.exists(models_dir):
        console.print(f"[red]Models directory '{models_dir}' not found.[/red]")
        return None
    
    # List all model FILES (not directories) with .keras or .h5 endings
    model_files = [f for f in os.listdir(models_dir) 
                  if os.path.isfile(os.path.join(models_dir, f)) and 
                  f.startswith("solar_flare_model_") and 
                  (f.endswith(".keras") or f.endswith(".h5"))]
    
    if not model_files:
        console.print("[red]No saved models found.[/red]")
        return None
    
    # Find the most recent model file
    latest_model_file = max(model_files)
    model_path = os.path.join(models_dir, latest_model_file)
    
    console.print(f"[green]Loading model from:[/green] {model_path}")
    return load_model(model_path), os.path.splitext(latest_model_file)[0]

<<<<<<< HEAD
def load_specific_model(model_name, models_dir="models"):
=======
def load_specific_model(model_name, models_dir="../models"):
>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11
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
                console.print(f"[red]Model '{model_name}' not found in {models_dir}[/red]")
                return None
    else:
        # Model name already has an extension
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
            console.print(f"[red]Model '{model_name}' not found at {model_path}[/red]")
            return None
    
    console.print(f"[green]Loading model from:[/green] {model_path}")
    return load_model(model_path), model_name

def predict_solar_flare(model, input_data):
    """Make predictions using the loaded model."""
    predictions = model.predict(input_data)
    return predictions

<<<<<<< HEAD
def load_test_data(model_name, models_dir="models"):
=======
def load_test_data(model_name, models_dir="../models"):
>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11
    """Load the test data saved during model training."""
    # Strip file extension if present
    model_name = os.path.splitext(model_name)[0]
    
    artifacts_dir = os.path.join(models_dir, model_name)
    test_data_path = os.path.join(artifacts_dir, 'test_set.npz')
    
    if not os.path.exists(test_data_path):
        console.print(f"[red]Test data not found at {test_data_path}[/red]")
        return None, None
    
    with np.load(test_data_path) as data:
        X_test = data['X_test']
        y_test = data['y_test']
    
    console.print(f"[green]Loaded test data with shape:[/green] X={X_test.shape}, y={y_test.shape}")
    return X_test, y_test

<<<<<<< HEAD
def evaluate_model_on_test_data(model, model_name, models_dir="models"):
=======
def evaluate_model_on_test_data(model, model_name, models_dir="../models"):
>>>>>>> 2e83ace1dbaad6a0734e6a9bc820ded9df6f2c11
    """Evaluate model on the saved test data."""
    X_test, y_test = load_test_data(model_name, models_dir)
    
    if X_test is None or y_test is None:
        console.print("[red]Could not load test data.[/red]")
        return
    
    console.print("\n[bold blue]Evaluating model on saved test data...[/bold blue]")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Create evaluation results table
    eval_table = Table(title="Model Evaluation Results", box=box.ROUNDED)
    eval_table.add_column("Metric", style="cyan", no_wrap=True)
    eval_table.add_column("Value", style="magenta")
    eval_table.add_row("Test Set Accuracy", f"{accuracy * 100:.2f}%")
    eval_table.add_row("Test Set Loss", f"{loss:.4f}")
    console.print(eval_table)
    
    # Get predictions for the test set
    console.print("\n[yellow]Generating predictions...[/yellow]")
    predictions = model.predict(X_test)
    predicted_classes = (predictions > 0.5).astype(int)  # Convert probabilities to binary classes
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, predicted_classes)
    
    # Create confusion matrix table
    cm_table = Table(title="Confusion Matrix", box=box.ROUNDED)
    cm_table.add_column("", style="cyan")
    cm_table.add_column("Predicted No Flare", style="green")
    cm_table.add_column("Predicted Flare", style="red")
    cm_table.add_row("Actual No Flare", str(cm[0,0]), str(cm[0,1]))
    cm_table.add_row("Actual Flare", str(cm[1,0]), str(cm[1,1]))
    console.print(cm_table)
    
    # Calculate True Skill Statistic (TSS)
    if cm.size == 4:  # Binary classification with 2x2 matrix
        tn, fp, fn, tp = cm.ravel()
        tss = tp/(tp+fn) - fp/(fp+tn)
        console.print(f"\n[bold green]True Skill Statistic (TSS):[/bold green] {tss:.4f}")
    
    # Generate detailed classification metrics
    class_report = classification_report(y_test, predicted_classes, output_dict=True)
    
    # Create classification report table
    report_table = Table(title="Classification Report", box=box.ROUNDED)
    report_table.add_column("Class", style="cyan")
    report_table.add_column("Precision", style="green")
    report_table.add_column("Recall", style="yellow")
    report_table.add_column("F1-Score", style="magenta")
    report_table.add_column("Support", style="blue")
    
    for class_name, metrics in class_report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            class_label = "No Flare" if class_name == '0' else "Flare"
            report_table.add_row(
                class_label,
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1-score']:.3f}",
                str(int(metrics['support']))
            )
    
    # Add summary rows
    report_table.add_row("", "", "", "", "")  # Empty row
    report_table.add_row(
        "Macro Avg",
        f"{class_report['macro avg']['precision']:.3f}",
        f"{class_report['macro avg']['recall']:.3f}",
        f"{class_report['macro avg']['f1-score']:.3f}",
        str(int(class_report['macro avg']['support']))
    )
    report_table.add_row(
        "Weighted Avg",
        f"{class_report['weighted avg']['precision']:.3f}",
        f"{class_report['weighted avg']['recall']:.3f}",
        f"{class_report['weighted avg']['f1-score']:.3f}",
        str(int(class_report['weighted avg']['support']))
    )
    
    console.print(report_table)
    
    return predictions, y_test

if __name__ == "__main__":
    try:
        console.clear()
        console.print(Panel.fit("[bold orange1]Solar Flare Prediction[/bold orange1]", border_style="orange1"))
    
        # Load the model
        result = load_latest_model()
        
        if result is None:
            console.print("[red]No model available. Please train a model first.[/red]")
            exit()
        
        model, model_name = result
        
        # Create options menu
        options_table = Table(box=box.SIMPLE)
        options_table.add_column("Option", style="cyan", no_wrap=True)
        options_table.add_column("Description", style="white")
        options_table.add_row("1", "Test on saved test data")
        options_table.add_row("2", "Test on sample data from processed_solar_data.npz")
        
        console.print("\n")
        console.print(options_table)
        choice = console.input("\n[yellow]Enter your choice (1/2): [/yellow]")
        
        if choice == '1':
            # Evaluate on the saved test data
            evaluate_model_on_test_data(model, model_name)
        
        elif choice == '2':
            # Load some data to predict on
            try:
                with console.status("[bold green]Loading sample data..."):
                    with np.load('processed_solar_data.npz') as data:
                        X = data['X']
                        y_true = data['y']
                        
                console.print(f"[green]Loaded sample data with shape:[/green] {X.shape}")
                
                # Make predictions on a few samples
                sample_size = min(5, X.shape[0])
                sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
                
                X_samples = X[sample_indices]
                y_true_samples = y_true[sample_indices]
                
                with console.status("[bold green]Making predictions..."):
                    predictions = predict_solar_flare(model, X_samples)
                
                # Calculate accuracy
                predicted_classes = (predictions > 0.5).astype(int).flatten()
                accuracy = np.mean(predicted_classes == y_true_samples) * 100
                
                # Create results table
                results_table = Table(title=f"Sample Prediction Results (Accuracy: {accuracy:.2f}%)", box=box.ROUNDED)
                results_table.add_column("Sample", style="cyan", no_wrap=True)
                results_table.add_column("Prediction Score", style="magenta")
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
                        f"{predictions[i][0]:.4f}",
                        predicted_class,
                        actual_class,
                        f"[{correct_style}]{is_correct}[/{correct_style}]"
                    )
                
                console.print("\n")
                console.print(results_table)
                
                # Summary panel
                summary_text = f"Sample Accuracy: {accuracy:.2f}%\n"
                summary_text += f"Correct Predictions: {np.sum(predicted_classes == y_true_samples)}/{sample_size}"
                console.print(Panel(summary_text, title="[bold]Summary[/bold]", border_style="green"))
                    
            except Exception as e:
                console.print(f"[red]Error loading or predicting data: {e}[/red]")
                console.print("[yellow]You can modify this script to load your own data for predictions.[/yellow]")
        
        else:
            console.print("[red]Invalid choice. Please run again and select 1 or 2.[/red]")
    
    except KeyboardInterrupt:
        console.print("\n[red]Cancelled.[/red]")
        exit(0)