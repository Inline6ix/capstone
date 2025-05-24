"""
Base model classes and interfaces.

This module defines the base model interface that all epitope classification
models should implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Abstract base class for epitope classification models.
    
    This class defines the standard interface that all models should implement
    for training, prediction, and evaluation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base model.
        
        Args:
            config: Configuration dictionary for the model
        """
        self.config = config or {}
        self.is_trained = False
        self.model = None
        self.feature_names = None
        self.class_names = ['Negative', 'Positive']
    
    @abstractmethod
    def train(
        self, 
        X_train: Union[np.ndarray, pd.DataFrame], 
        y_train: np.ndarray,
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with training metrics and history
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on the provided data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted class labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get prediction probabilities for the provided data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Prediction probabilities for each class
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        pass
    
    def evaluate(
        self, 
        X_test: Union[np.ndarray, pd.DataFrame], 
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")
        
        # Get predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
        }
        
        # Add ROC AUC if probabilities are available
        if y_pred_proba is not None and y_pred_proba.shape[1] == 2:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        return metrics
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores (if supported by the model).
        
        Returns:
            Feature importance scores or None if not supported
        """
        return None
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the model configuration.
        
        Returns:
            Model configuration dictionary
        """
        return self.config.copy()
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Update the model configuration.
        
        Args:
            config: New configuration dictionary
        """
        self.config.update(config)