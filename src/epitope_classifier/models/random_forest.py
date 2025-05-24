"""
Random Forest model for epitope classification.

This module implements a Random Forest model that uses engineered features
for epitope prediction.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .base import BaseModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """
    Random Forest model for epitope classification.
    
    This model uses engineered features (physicochemical properties,
    binding affinities, amino acid composition) for epitope prediction.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Random Forest model.
        
        Args:
            config: Configuration dictionary for the model
        """
        super().__init__(config)
        self.model = None
        self.scaler = None
        
        # Default configuration
        default_config = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'oob_score': True,
            'n_jobs': -1,
            'random_state': 42,
            'use_scaling': True
        }
        
        # Update with provided config
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    def train(
        self, 
        X_train: Union[np.ndarray, pd.DataFrame], 
        y_train: np.ndarray,
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[np.ndarray] = None,
        class_weights: Optional[Dict[int, float]] = None
    ) -> Dict[str, Any]:
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (not used in RF training)
            y_val: Validation labels (not used in RF training)
            class_weights: Optional class weights for imbalanced data
            
        Returns:
            Training metrics and model information
        """
        logger.info("Training Random Forest model...")
        
        # Convert to numpy arrays if needed
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        
        # Scale features if requested
        if self.config['use_scaling']:
            logger.info("Scaling features...")
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
        
        # Setup class weights
        class_weight = None
        if class_weights is not None:
            class_weight = class_weights
        elif 'class_weight' in self.config:
            class_weight = self.config['class_weight']
        else:
            class_weight = 'balanced'  # Automatically balance classes
        
        # Initialize model
        rf_config = {k: v for k, v in self.config.items() 
                    if k in ['n_estimators', 'max_depth', 'min_samples_split', 
                            'min_samples_leaf', 'max_features', 'bootstrap', 
                            'oob_score', 'n_jobs', 'random_state']}
        
        self.model = RandomForestClassifier(
            class_weight=class_weight,
            **rf_config
        )
        
        # Train the model
        logger.info(f"Training with {X_train.shape[0]} samples and {X_train.shape[1]} features...")
        self.model.fit(X_train, y_train)
        
        self.is_trained = True
        
        # Calculate training metrics
        train_accuracy = self.model.score(X_train, y_train)
        
        results = {
            'train_accuracy': train_accuracy,
            'n_features': X_train.shape[1],
            'n_samples': X_train.shape[0]
        }
        
        # Add OOB score if available
        if self.config['oob_score'] and hasattr(self.model, 'oob_score_'):
            results['oob_score'] = self.model.oob_score_
            logger.info(f"Out-of-bag score: {self.model.oob_score_:.4f}")
        
        # Validation metrics (if provided)
        if X_val is not None and y_val is not None:
            val_accuracy = self.evaluate(X_val, y_val)['accuracy']
            results['val_accuracy'] = val_accuracy
            logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        
        logger.info(f"Training completed! Training accuracy: {train_accuracy:.4f}")
        
        return results
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on input features.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale features if scaler was used during training
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get prediction probabilities for input features.
        
        Args:
            X: Input features
            
        Returns:
            Prediction probabilities for each class
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale features if scaler was used during training
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores from the trained Random Forest.
        
        Returns:
            Feature importance scores
        """
        if not self.is_trained or self.model is None:
            return None
        
        return self.model.feature_importances_
    
    def get_feature_importance_df(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance as a DataFrame with feature names.
        
        Returns:
            DataFrame with features and their importance scores
        """
        importance = self.get_feature_importance()
        if importance is None:
            return None
        
        feature_names = self.feature_names if self.feature_names else [f'feature_{i}' for i in range(len(importance))]
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before saving")
        
        model_path = Path(filepath)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model, scaler, and metadata
        save_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'feature_names': self.feature_names,
            'class_names': self.class_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.model = save_data['model']
        self.scaler = save_data.get('scaler')
        self.config = save_data.get('config', {})
        self.feature_names = save_data.get('feature_names')
        self.class_names = save_data.get('class_names', ['Negative', 'Positive'])
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_tree_paths(self, X: Union[np.ndarray, pd.DataFrame], tree_idx: int = 0) -> List[List[int]]:
        """
        Get the decision paths for samples through a specific tree.
        
        Args:
            X: Input features
            tree_idx: Index of the tree to analyze
            
        Returns:
            List of decision paths for each sample
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before analyzing tree paths")
        
        if tree_idx >= len(self.model.estimators_):
            raise ValueError(f"Tree index {tree_idx} out of range (model has {len(self.model.estimators_)} trees)")
        
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale features if scaler was used during training
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        tree = self.model.estimators_[tree_idx]
        paths = tree.decision_path(X)
        
        # Convert sparse matrix to list of paths
        paths_list = []
        for i in range(paths.shape[0]):
            path = paths[i].toarray().flatten().nonzero()[0].tolist()
            paths_list.append(path)
        
        return paths_list
    
    def explain_prediction(self, X: Union[np.ndarray, pd.DataFrame], sample_idx: int = 0) -> Dict[str, Any]:
        """
        Explain a single prediction using feature importance and tree paths.
        
        Args:
            X: Input features
            sample_idx: Index of the sample to explain
            
        Returns:
            Dictionary with explanation information
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before explaining predictions")
        
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if sample_idx >= len(X_array):
            raise ValueError(f"Sample index {sample_idx} out of range")
        
        sample = X_array[sample_idx:sample_idx+1]
        
        # Get prediction and probability
        prediction = self.predict(sample)[0]
        probabilities = self.predict_proba(sample)[0]
        
        # Get feature importance
        feature_importance = self.get_feature_importance()
        
        explanation = {
            'prediction': prediction,
            'probabilities': probabilities,
            'predicted_class': self.class_names[prediction],
            'confidence': np.max(probabilities)
        }
        
        # Add feature-level explanations
        if self.feature_names and feature_importance is not None:
            feature_values = X_array[sample_idx]
            feature_explanations = []
            
            for i, (name, value, importance) in enumerate(zip(self.feature_names, feature_values, feature_importance)):
                feature_explanations.append({
                    'feature': name,
                    'value': value,
                    'importance': importance,
                    'contribution': value * importance  # Simple approximation
                })
            
            # Sort by contribution magnitude
            feature_explanations.sort(key=lambda x: abs(x['contribution']), reverse=True)
            explanation['feature_explanations'] = feature_explanations[:10]  # Top 10
        
        return explanation