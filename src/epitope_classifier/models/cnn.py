"""
Convolutional Neural Network model for epitope classification.

This module implements a CNN model that works directly on one-hot encoded
amino acid sequences for epitope prediction.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from .base import BaseModel

logger = logging.getLogger(__name__)


class CNNModel(BaseModel):
    """
    Convolutional Neural Network for epitope classification.
    
    This model works directly on one-hot encoded amino acid sequences
    and learns sequence patterns for epitope prediction.
    """
    
    def __init__(self, input_shape: Tuple[int, int], config: Optional[Dict[str, Any]] = None):
        """
        Initialize the CNN model.
        
        Args:
            input_shape: Shape of input sequences (sequence_length, num_amino_acids)
            config: Configuration dictionary for the model
        """
        super().__init__(config)
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
        # Default configuration
        default_config = {
            'filters': [32, 64, 128],
            'kernel_size': 3,
            'dropout': 0.4,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'early_stopping_patience': 8,
            'use_batch_normalization': True,
            'l2_regularization': 0.001
        }
        
        # Update with provided config
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    def _build_model(self) -> Model:
        """
        Build the CNN architecture.
        
        Returns:
            Compiled Keras model
        """
        inputs = Input(shape=self.input_shape)
        x = inputs
        
        # Convolutional layers
        for i, filters in enumerate(self.config['filters']):
            x = Conv1D(
                filters=filters,
                kernel_size=self.config['kernel_size'],
                activation='relu',
                padding='same',
                kernel_regularizer=l2(self.config['l2_regularization']),
                name=f'conv1d_{i+1}'
            )(x)
            
            if self.config['use_batch_normalization']:
                x = BatchNormalization(name=f'batch_norm_{i+1}')(x)
            
            if i < len(self.config['filters']) - 1:  # No pooling after last conv layer
                x = MaxPooling1D(pool_size=2, padding='same', name=f'max_pool_{i+1}')(x)
        
        # Flatten and dense layers
        x = Flatten(name='flatten')(x)
        
        # First dense layer
        x = Dense(
            128, 
            activation='relu',
            kernel_regularizer=l2(self.config['l2_regularization']),
            name='dense_1'
        )(x)
        
        if self.config['use_batch_normalization']:
            x = BatchNormalization(name='batch_norm_dense_1')(x)
        
        x = Dropout(self.config['dropout'], name='dropout_1')(x)
        
        # Second dense layer
        x = Dense(
            64, 
            activation='relu',
            kernel_regularizer=l2(self.config['l2_regularization']),
            name='dense_2'
        )(x)
        
        if self.config['use_batch_normalization']:
            x = BatchNormalization(name='batch_norm_dense_2')(x)
        
        x = Dropout(self.config['dropout'] * 0.75, name='dropout_2')(x)
        
        # Output layer
        outputs = Dense(2, activation='softmax', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='epitope_cnn')
        
        # Compile model
        optimizer = Adam(learning_rate=self.config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        class_weights: Optional[Dict[int, float]] = None
    ) -> Dict[str, Any]:
        """
        Train the CNN model.
        
        Args:
            X_train: Training sequences (one-hot encoded)
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            class_weights: Optional class weights for imbalanced data
            
        Returns:
            Training history and metrics
        """
        logger.info("Building CNN model...")
        self.model = self._build_model()
        
        logger.info(f"Model architecture:\n{self.model.summary()}")
        
        # Setup callbacks
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=self.config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Model checkpoint (optional)
        if 'model_checkpoint_path' in self.config:
            checkpoint = ModelCheckpoint(
                self.config['model_checkpoint_path'],
                monitor='val_accuracy' if X_val is not None else 'accuracy',
                save_best_only=True,
                verbose=1
            )
            callbacks.append(checkpoint)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        logger.info("Starting training...")
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=validation_data,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        self.is_trained = True
        
        logger.info("Training completed!")
        
        return {
            'history': history.history,
            'final_train_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history.get('val_accuracy', [None])[-1],
            'epochs_trained': len(history.history['accuracy'])
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input sequences.
        
        Args:
            X: Input sequences (one-hot encoded)
            
        Returns:
            Predicted class labels
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before making predictions")
        
        probabilities = self.model.predict(X)
        return np.argmax(probabilities, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities for input sequences.
        
        Args:
            X: Input sequences (one-hot encoded)
            
        Returns:
            Prediction probabilities for each class
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before saving")
        
        # Save model
        model_path = Path(filepath)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        if filepath.endswith('.h5'):
            self.model.save(filepath)
        else:
            self.model.save(f"{filepath}.keras")
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = tf.keras.models.load_model(filepath)
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_maps(self, X: np.ndarray, layer_name: Optional[str] = None) -> np.ndarray:
        """
        Extract feature maps from a specific layer.
        
        Args:
            X: Input sequences
            layer_name: Name of layer to extract features from. If None, uses last conv layer.
            
        Returns:
            Feature maps from the specified layer
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before extracting features")
        
        if layer_name is None:
            # Find the last convolutional layer
            for layer in reversed(self.model.layers):
                if isinstance(layer, Conv1D):
                    layer_name = layer.name
                    break
        
        if layer_name is None:
            raise ValueError("No convolutional layers found in model")
        
        # Create a model that outputs the specified layer
        feature_model = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(layer_name).output
        )
        
        return feature_model.predict(X)
    
    def compute_saliency_map(self, X: np.ndarray, target_class: int = 1) -> np.ndarray:
        """
        Compute saliency maps for input sequences.
        
        Args:
            X: Input sequences (one-hot encoded)
            target_class: Target class for saliency computation
            
        Returns:
            Saliency maps showing important positions
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before computing saliency")
        
        saliency_maps = []
        
        for i in range(len(X)):
            x_input = tf.Variable(X[i:i+1], dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                tape.watch(x_input)
                predictions = self.model(x_input)
                target_output = predictions[0, target_class]
            
            # Compute gradients
            gradients = tape.gradient(target_output, x_input)
            
            # Calculate saliency as the max absolute gradient across amino acid dimension
            saliency = tf.reduce_max(tf.abs(gradients), axis=-1)
            saliency_maps.append(saliency.numpy().flatten())
        
        return np.array(saliency_maps)