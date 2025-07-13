import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Input, Softmax, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import json
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TaskID
from rich import box
import matplotlib.pyplot as plt

console = Console()

class AttentionFusionTrainer:
    def __init__(self, ensemble_config_path=None, use_channels=None, ensemble_dir=None):
        self.base_models = {}
        self.channel_names = []
        self.fusion_model = None
        self.class_names = ['No Flare', 'Flare']  # Binary classification to match base models
        self.num_classes = 2
        self.use_channels = use_channels or ['Bp', 'Br', 'Bt', 'continuum']
        # Ask user for ensemble directory if not provided
        if ensemble_dir is None:
            ensemble_dir = self.prompt_for_ensemble_dir()
        # Load ensemble config from selected directory
        config_path = ensemble_config_path or os.path.join(ensemble_dir, "ensemble_config.json")
        if config_path and os.path.exists(config_path):
            self.load_base_models(config_path)
        else:
            console.print("[red]No ensemble configuration found. Please train base models first.[/red]")
            sys.exit(1)
        self.ensemble_dir = ensemble_dir

    def prompt_for_ensemble_dir(self):
        """Prompt user to select an ensemble directory from models folder"""
        models_dir = "models"
        if not os.path.exists(models_dir):
            console.print(f"[red]Models directory not found: {models_dir}[/red]")
            sys.exit(1)
        ensemble_dirs = [d for d in os.listdir(models_dir)
                        if os.path.isdir(os.path.join(models_dir, d)) and d.startswith('ensemble_')]
        if not ensemble_dirs:
            console.print(f"[red]No ensemble directories found in: {models_dir}[/red]")
            sys.exit(1)
        console.print("\n[bold cyan]Select an ensemble directory to use:[/bold cyan]")
        for idx, d in enumerate(ensemble_dirs):
            console.print(f"  [{idx}] {d}")
        while True:
            try:
                choice = int(console.input("Enter the number of the directory to use: "))
                if 0 <= choice < len(ensemble_dirs):
                    selected_dir = os.path.join(models_dir, ensemble_dirs[choice])
                    console.print(f"[green]Selected ensemble directory:[/green] {selected_dir}")
                    return selected_dir
            except Exception:
                pass
            console.print("[red]Invalid selection. Please try again.[/red]")
    
    def find_latest_ensemble_config(self):
        # Deprecated: now user selects ensemble dir interactively
        return None
    
    def load_base_models(self, config_path):
        """Load the pre-trained base models"""
        console.print(f"[green]Loading base models from:[/green] {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.channel_names = config['channel_names']
        model_info = config['models']
        ensemble_dir = config.get('ensemble_dir', os.path.dirname(config_path))
        
        console.print(f"Loading {len(model_info)} base models...")
        
        for channel_name, info in model_info.items():
            # Only load models that are in the use_channels list
            if channel_name not in self.use_channels:
                console.print(f"  - Skipping {channel_name} model (not in use_channels)")
                continue
                
            model_path = os.path.join(ensemble_dir, f"solar_flare_model_{channel_name}.keras")
            
            if os.path.exists(model_path):
                try:
                    # Load model and make it non-trainable
                    model = load_model(model_path)
                    model.trainable = False
                    self.base_models[channel_name] = model
                    console.print(f"  ✓ Loaded {channel_name} model")
                except Exception as e:
                    console.print(f"  ✗ Failed to load {channel_name} model: {e}")
            else:
                console.print(f"  ✗ Model file not found: {model_path}")
        
        if len(self.base_models) == 0:
            console.print("[red]No base models loaded successfully![/red]")
            sys.exit(1)
    
    def prepare_labels_for_binary(self, y):
        """Keep labels as binary (0=No Flare, 1=Flare) to match base models"""
        # Convert any multiclass labels back to binary
        # 0 = No Flare, 1+ = Flare (any type)
        binary_y = (y > 0).astype(int)
        
        unique_classes = np.unique(binary_y)
        console.print(f"[green]Using binary classification: {len(unique_classes)} classes {unique_classes}[/green]")
        return binary_y
    
    def extract_base_predictions(self, X):
        """Extract predictions from all base models"""
        predictions = {}
        
        with Progress() as progress:
            task = progress.add_task("Extracting base predictions...", total=len(self.base_models))
            
            for channel_name, model in self.base_models.items():
                channel_idx = self.channel_names.index(channel_name)
                # Fix: Extract channel correctly - the channel dimension is the last one
                channel_data = X[:, :, :, :, channel_idx:channel_idx+1]
                
                # Get predictions from base model (binary classification)
                channel_predictions = model.predict(channel_data, verbose=0)
                # Base models output single probability, flatten to 1D array
                predictions[channel_name] = channel_predictions.flatten()
                
                progress.advance(task)
        
        # Stack predictions - only use channels that we actually loaded models for
        prediction_matrix = np.column_stack([predictions[channel] for channel in self.use_channels if channel in predictions])
        return prediction_matrix
    
    def create_attention_fusion_model(self, input_dim):
        """Create attention-based fusion model"""
        console.print("[cyan]Creating attention fusion model...[/cyan]")
        
        # Input layer for base model predictions
        inputs = Input(shape=(input_dim,), name='base_predictions')
        
        # Attention mechanism
        # Transform predictions to a higher dimensional space
        attention_dense = Dense(64, activation='relu', name='attention_transform')(inputs)
        attention_dense = Dropout(0.3)(attention_dense)
        
        # Calculate attention weights
        attention_weights = Dense(input_dim, activation='softmax', name='attention_weights')(attention_dense)
        
        # Apply attention weights to input predictions
        weighted_predictions = tf.keras.layers.Multiply(name='weighted_predictions')([inputs, attention_weights])
        
        # Additional processing layers
        fusion_layer = Dense(32, activation='relu', name='fusion_layer')(weighted_predictions)
        fusion_layer = Dropout(0.4)(fusion_layer)
        
        # Final classification layer for binary classification (No Flare vs Flare)
        outputs = Dense(1, activation='sigmoid', name='final_classification')(fusion_layer)
        
        model = Model(inputs=inputs, outputs=outputs, name='attention_fusion_model')
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',  # Binary classification loss
            metrics=['accuracy']
        )
        
        return model
    
    def train_attention_fusion(self, X, y, test_size=0.2, validation_size=0.2, epochs=100):
        """Train the attention fusion model"""
        console.print("[bold blue]Training Attention Fusion Model[/bold blue]")
        
        # Convert labels to binary to match base models
        y_binary = self.prepare_labels_for_binary(y)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_binary, test_size=test_size, random_state=42, stratify=y_binary
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_size, random_state=42, stratify=y_temp
        )
        
        console.print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
        
        # Extract base model predictions
        console.print("[cyan]Extracting base model predictions...[/cyan]")
        console.print(f"[cyan]Using {len(self.base_models)} models: {list(self.base_models.keys())}[/cyan]")
        train_predictions = self.extract_base_predictions(X_train)
        val_predictions = self.extract_base_predictions(X_val)
        test_predictions = self.extract_base_predictions(X_test)
        
        console.print(f"[green]Prediction matrix shape: {train_predictions.shape}[/green]")
        
        # For binary classification, we don't need to_categorical
        # Just ensure labels are proper shape
        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32) 
        y_test = y_test.astype(np.float32)
        
        # Create fusion model
        self.fusion_model = self.create_attention_fusion_model(train_predictions.shape[1])
        
        # Print model summary
        console.print("\n[bold]Attention Fusion Model Architecture:[/bold]")
        self.fusion_model.summary()
        
        # Get output directory for callbacks
        config_path = self.find_latest_ensemble_config()
        if config_path:
            with open(config_path, 'r') as f:
                config = json.load(f)
            callback_output_dir = config.get('ensemble_dir', 'models')
        else:
            callback_output_dir = 'models'
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(callback_output_dir, 'attention_fusion_model_best.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        console.print("[cyan]Training attention fusion model...[/cyan]")
        history = self.fusion_model.fit(
            train_predictions, y_train,
            validation_data=(val_predictions, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        console.print("[cyan]Evaluating on test set...[/cyan]")
        test_loss, test_accuracy = self.fusion_model.evaluate(test_predictions, y_test, verbose=0)
        
        # Get individual model accuracies for comparison
        console.print("[cyan]Evaluating individual base models...[/cyan]")
        individual_accuracies = {}
        for channel_name, model in self.base_models.items():
            channel_idx = self.channel_names.index(channel_name)
            channel_test_data = X_test[:, :, :, :, channel_idx:channel_idx+1]
            channel_pred = model.predict(channel_test_data, verbose=0)
            
            # Base models are binary classifiers, so use threshold-based prediction
            channel_pred_classes = (channel_pred > 0.5).astype(int).flatten()
            
            # Since both models and test labels are now binary, direct comparison
            channel_accuracy = np.mean(channel_pred_classes == y_test)
            individual_accuracies[channel_name] = channel_accuracy
        
        # Get predictions for detailed evaluation
        y_pred_probs = self.fusion_model.predict(test_predictions, verbose=0)
        y_pred_classes = (y_pred_probs > 0.5).astype(int).flatten()  # Binary prediction
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        
        # Get attention weights for analysis
        attention_layer = self.fusion_model.get_layer('attention_weights')
        attention_weights_sample = attention_layer.output
        
        results = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'individual_accuracies': individual_accuracies,
            'confusion_matrix': cm,
            'y_true': y_test,
            'y_pred': y_pred_classes,
            'y_pred_probs': y_pred_probs,
            'history': history.history,
            'test_data': (test_predictions, y_test),
            'class_names': self.class_names
        }
        
        return results
    
    def save_attention_model(self, results, output_dir=None):
        """Save the trained attention fusion model and configuration"""
        # Use the selected ensemble directory by default
        if output_dir is None:
            output_dir = getattr(self, 'ensemble_dir', 'models')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Save the fusion model
        fusion_model_path = os.path.join(output_dir, 'attention_fusion_model.keras')
        self.fusion_model.save(fusion_model_path)
        # Save configuration
        config = {
            'model_type': 'attention_fusion',
            'channel_names': self.channel_names,
            'base_models': {channel: {'accuracy': 0.0} for channel in self.base_models.keys()},
            'fusion_model_path': fusion_model_path,
            'test_accuracy': results['test_accuracy'],
            'test_loss': results['test_loss'],
            'num_classes': self.num_classes,
            'class_names': self.class_names
        }
        config_path = os.path.join(output_dir, 'attention_fusion_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        console.print(f"[green]Attention fusion model saved to:[/green] {fusion_model_path}")
        console.print(f"[green]Configuration saved to:[/green] {config_path}")
        return config_path
    
    def print_results(self, results):
        """Print training results"""
        console.print(f"\n[bold blue]{'='*60}[/bold blue]")
        console.print("[bold blue]ATTENTION FUSION TRAINING RESULTS[/bold blue]")
        console.print(f"[bold blue]{'='*60}[/bold blue]")
        
        # Performance metrics
        metrics_table = Table(title="Performance Metrics", box=box.ROUNDED)
        metrics_table.add_column("Model", style="cyan")
        metrics_table.add_column("Test Accuracy", style="magenta")
        
        # Add individual model accuracies
        for channel_name, accuracy in results['individual_accuracies'].items():
            metrics_table.add_row(f"{channel_name} (base)", f"{accuracy*100:.2f}%")
        
        # Add fusion model accuracy
        metrics_table.add_row("Attention Fusion", f"{results['test_accuracy']*100:.2f}%")
        metrics_table.add_row("Test Loss", f"{results['test_loss']:.4f}")
        console.print(metrics_table)
        
        # Confusion Matrix
        cm = results['confusion_matrix']
        cm_table = Table(title="Confusion Matrix", box=box.ROUNDED)
        cm_table.add_column("", style="cyan")
        for class_name in self.class_names:
            cm_table.add_column(f"Pred {class_name}", style="green")
        
        for i, class_name in enumerate(self.class_names):
            row = [f"True {class_name}"] + [str(cm[i, j]) for j in range(len(self.class_names))]
            cm_table.add_row(*row)
        
        console.print(cm_table)
        
        # Classification Report
        report = classification_report(results['y_true'], results['y_pred'], 
                                     target_names=self.class_names, output_dict=True, zero_division=0)
        report_table = Table(title="Classification Report", box=box.ROUNDED)
        report_table.add_column("Class", style="cyan")
        report_table.add_column("Precision", style="green")
        report_table.add_column("Recall", style="yellow")
        report_table.add_column("F1-Score", style="magenta")

        # The keys in report dict are exactly the names in target_names
        for class_name in self.class_names:
            if class_name in report:
                class_report = report[class_name]
                report_table.add_row(
                    class_name,
                    f"{class_report['precision']:.3f}",
                    f"{class_report['recall']:.3f}",
                    f"{class_report['f1-score']:.3f}"
                )
        # Optionally add macro/micro/weighted avg
        for avg_type in ["macro avg", "weighted avg"]:
            if avg_type in report:
                avg_report = report[avg_type]
                report_table.add_row(
                    avg_type,
                    f"{avg_report['precision']:.3f}",
                    f"{avg_report['recall']:.3f}",
                    f"{avg_report['f1-score']:.3f}"
                )
        console.print(report_table)

def main():
    try:
        console.clear()
        console.print(Panel.fit("[bold orange1]Attention Fusion Model Training[/bold orange1]", border_style="orange1"))
        
        # Load processed data
        try:
            console.print("[cyan]Loading processed solar data...[/cyan]")
            with np.load('processed_HED_data.npz') as data:
                X = data['X']
                y = data['y']
            console.print(f"[green]Data loaded:[/green] X shape: {X.shape}, y shape: {y.shape}")
        except FileNotFoundError:
            console.print("[red]processed_HED_data.npz not found! Please run your data preprocessing script first.[/red]")
            return
        
        # Initialize trainer (user will be prompted to select ensemble directory)
        trainer = AttentionFusionTrainer()
        # Alternative: Use only specific channels
        # trainer = AttentionFusionTrainer(use_channels=['Bp', 'Br', 'Bt'])  # Example: use only 3 channels
        # trainer = AttentionFusionTrainer(use_channels=['Bp', 'continuum'])  # Example: use only 2 channels
        
        # Train the attention fusion model
        console.print("\n[bold]Starting attention fusion training...[/bold]")
        results = trainer.train_attention_fusion(X, y, epochs=100)
        
        # Print results
        trainer.print_results(results)
        
        # Save the model
        config_path = trainer.save_attention_model(results)
        
        console.print(f"\n[green]Training completed successfully![/green]")
        console.print(f"[green]Use the test script with config: {config_path}[/green]")
        
    except KeyboardInterrupt:
        console.print("\n[red]Training cancelled by user.[/red]")
    except Exception as e:
        console.print(f"[red]Error during training: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()