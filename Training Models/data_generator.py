import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import gc

class MemoryEfficientDataGenerator(Sequence):
    """
    Memory-efficient data generator that loads batches on-demand.
    Reduces memory usage by not keeping all data in memory simultaneously.
    """
    
    def __init__(self, X_images, X_scalars, y, batch_size=4, shuffle=True, 
                 cache_batches=False, max_cache_size=10):
        """
        Initialize the data generator.
        
        Args:
            X_images: Image data array
            X_scalars: Scalar features array  
            y: Labels array
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data between epochs
            cache_batches: Whether to cache recently used batches
            max_cache_size: Maximum number of batches to cache
        """
        self.X_images = X_images
        self.X_scalars = X_scalars
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cache_batches = cache_batches
        self.max_cache_size = max_cache_size
        
        self.n_samples = len(X_images)
        self.indices = np.arange(self.n_samples)
        
        # Cache for frequently accessed batches
        self.batch_cache = {}
        self.cache_access_count = {}
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Number of batches per epoch."""
        return int(np.ceil(self.n_samples / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data."""
        # Check cache first
        if self.cache_batches and index in self.batch_cache:
            self.cache_access_count[index] += 1
            return self.batch_cache[index]
        
        # Generate batch indices
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.n_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        # Load batch data
        batch_images = self.X_images[batch_indices].copy()
        batch_scalars = self.X_scalars[batch_indices].copy()
        batch_y = self.y[batch_indices].copy()
        
        # Ensure data types
        batch_images = batch_images.astype(np.float32)
        batch_scalars = batch_scalars.astype(np.float32)
        batch_y = batch_y.astype(np.float32)
        
        batch_data = ([batch_images, batch_scalars], batch_y)
        
        # Cache management
        if self.cache_batches:
            self._manage_cache(index, batch_data)
        
        return batch_data
    
    def _manage_cache(self, index, batch_data):
        """Manage batch cache to prevent memory overflow."""
        # Add to cache
        self.batch_cache[index] = batch_data
        self.cache_access_count[index] = 1
        
        # Remove least accessed batches if cache is full
        if len(self.batch_cache) > self.max_cache_size:
            # Find least accessed batch
            least_accessed = min(self.cache_access_count.items(), key=lambda x: x[1])
            least_accessed_index = least_accessed[0]
            
            # Remove from cache
            del self.batch_cache[least_accessed_index]
            del self.cache_access_count[least_accessed_index]
            
            # Force garbage collection
            gc.collect()
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch and clear cache."""
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        # Clear cache to free memory
        if self.cache_batches:
            self.batch_cache.clear()
            self.cache_access_count.clear()
            gc.collect()


class GradientAccumulationModel:
    """
    Wrapper for Keras model that implements gradient accumulation.
    Allows training with larger effective batch sizes without memory issues.
    """
    
    def __init__(self, model, accumulation_steps=4):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.accumulated_gradients = []
        self.step_count = 0
        
    def compile(self, optimizer, loss, metrics=None, **kwargs):
        self.optimizer = optimizer
        self.loss_fn = tf.keras.losses.get(loss)
        if metrics:
            self.compiled_metrics = [tf.keras.metrics.get(m) for m in metrics]
        else:
            self.compiled_metrics = []
        
        self._initialize_accumulated_gradients()
    
    def _initialize_accumulated_gradients(self):
        self.accumulated_gradients = [
            tf.Variable(tf.zeros_like(var), trainable=False) 
            for var in self.model.trainable_variables
        ]
    
    @tf.function
    def _accumulate_gradients(self, x, y, class_weight=None):
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            # --- FIX: Squeeze predictions from (batch, 1) to (batch,) to match labels ---
            predictions = tf.squeeze(predictions, axis=-1)
            
            per_sample_loss = self.loss_fn(y, predictions)
            
            if class_weight is not None:
                sample_weights = tf.gather(
                    list(class_weight.values()), 
                    tf.cast(y, dtype=tf.int32)
                )
                per_sample_loss *= tf.squeeze(tf.cast(sample_weights, tf.float32))

            loss = tf.nn.compute_average_loss(per_sample_loss) / self.accumulation_steps
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        for i, grad in enumerate(gradients):
            if grad is not None:
                self.accumulated_gradients[i].assign_add(grad)
        
        return loss, predictions
    
    @tf.function 
    def _apply_accumulated_gradients(self):
        self.optimizer.apply_gradients(
            zip(self.accumulated_gradients, self.model.trainable_variables)
        )
        for grad in self.accumulated_gradients:
            grad.assign(tf.zeros_like(grad))
    
    def train_step(self, data, class_weight=None):
        x, y = data
        loss, predictions = self._accumulate_gradients(x, y, class_weight)
        self.step_count += 1
        
        if self.step_count % self.accumulation_steps == 0:
            self._apply_accumulated_gradients()
        
        metric_results = {}
        for metric in self.compiled_metrics:
            metric.update_state(y, predictions)
            metric_results[metric.name] = metric.result()
        
        metric_results['loss'] = loss * self.accumulation_steps
        return metric_results
    
    def fit(self, train_generator, validation_data=None, epochs=1, 
            callbacks=None, verbose=1, class_weight=None, **kwargs):
        if callbacks is None:
            callbacks = []
        
        self.model.stop_training = False
        # Pass self (the wrapper) to the callback list, not self.model
        callback_list = tf.keras.callbacks.CallbackList(callbacks, add_history=True, model=self)
        
        callback_list.on_train_begin()
        
        for epoch in range(epochs):
            logs = {}
            callback_list.on_epoch_begin(epoch, logs)
            
            for metric in self.compiled_metrics:
                metric.reset_state()
            
            epoch_loss = 0
            num_batches = len(train_generator)
            progress_bar = tf.keras.utils.Progbar(target=num_batches, verbose=verbose)
            
            for batch_idx, batch_data in enumerate(train_generator):
                step_logs = self.train_step(batch_data, class_weight)
                epoch_loss += step_logs['loss']
                
                log_values = list(step_logs.items())
                progress_bar.update(batch_idx + 1, values=log_values)
            
            logs['loss'] = epoch_loss / num_batches
            for metric in self.compiled_metrics:
                logs[metric.name] = metric.result().numpy()
            
            if validation_data is not None:
                val_logs = self._evaluate_validation(validation_data)
                logs.update({f'val_{k}': v for k, v in val_logs.items()})

            progress_bar.update(num_batches, values=list(logs.items()), finalize=True)
            callback_list.on_epoch_end(epoch, logs)
            
            if self.model.stop_training:
                break
            
            if hasattr(train_generator, 'on_epoch_end'):
                train_generator.on_epoch_end()
        
        callback_list.on_train_end()

    def _evaluate_validation(self, validation_data):
        for metric in self.compiled_metrics:
            metric.reset_state()

        val_loss = 0
        num_batches = 0
        for x_val, y_val in validation_data:
            val_predictions = self.model(x_val, training=False)
            val_predictions = tf.squeeze(val_predictions, axis=-1)

            batch_loss = self.loss_fn(y_val, val_predictions)
            val_loss += tf.nn.compute_average_loss(batch_loss)
            num_batches += 1
            for metric in self.compiled_metrics:
                metric.update_state(y_val, val_predictions)
        
        val_loss /= num_batches if num_batches > 0 else 1.0
        val_logs = {'loss': val_loss.numpy()}
        val_logs.update({m.name: m.result().numpy() for m in self.compiled_metrics})
        return val_logs

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)
    
    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)
    
    def save(self, filepath, **kwargs):
        self.model.save(filepath, **kwargs)
    
    def save_weights(self, filepath, **kwargs):
        self.model.save_weights(filepath, **kwargs)