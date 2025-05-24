"""
Configuration management utilities.

This module provides functionality to load and manage configuration files
for the epitope classifier package.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class Config:
    """
    Configuration manager for the epitope classifier package.
    
    This class handles loading and accessing configuration values from
    YAML files, environment variables, and default values.
    """
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to YAML configuration file. If None, uses defaults.
        """
        self._config = {}
        self._load_defaults()
        
        if config_file:
            self.load_file(config_file)
        
        # Override with environment variables
        self._load_environment()
    
    def _load_defaults(self) -> None:
        """Load default configuration values."""
        self._config = {
            'data': {
                'train_split': 0.7,
                'val_split': 0.15,
                'test_split': 0.15,
                'negative_sample_ratio': 4.0,
                'sequence_length': 9,
                'random_state': 42
            },
            'features': {
                'use_physicochemical': True,
                'use_binding_affinity': True,
                'use_sequence_composition': True,
                'netmhcpan_path': '~/Documents/capstone/netMHCpan-4.1/run_netMHCpan.sh',
                'default_allele': 'HLA-A*02:01'
            },
            'model': {
                'type': 'cnn',
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'random_state': 42
                },
                'cnn': {
                    'filters': [32, 64, 128],
                    'kernel_size': 3,
                    'dropout': 0.4,
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 50,
                    'early_stopping_patience': 8,
                    'validation_split': 0.15
                }
            },
            'training': {
                'use_class_weights': True,
                'use_early_stopping': True,
                'save_best_model': True,
                'model_checkpoint_dir': 'models/',
                'log_dir': 'logs/'
            },
            'evaluation': {
                'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                'threshold_optimization': True,
                'cross_validation': False,
                'cv_folds': 5
            },
            'visualization': {
                'style': 'whitegrid',
                'figsize': [10, 6],
                'dpi': 300,
                'save_format': 'png'
            }
        }
    
    def _load_environment(self) -> None:
        """Load configuration from environment variables."""
        # Map environment variables to config paths
        env_mappings = {
            'EPITOPE_NETMHCPAN_PATH': 'features.netmhcpan_path',
            'EPITOPE_RANDOM_STATE': 'data.random_state',
            'EPITOPE_BATCH_SIZE': 'model.cnn.batch_size',
            'EPITOPE_LEARNING_RATE': 'model.cnn.learning_rate',
            'EPITOPE_MODEL_DIR': 'training.model_checkpoint_dir',
            'EPITOPE_LOG_DIR': 'training.log_dir'
        }
        
        for env_var, config_path in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                # Try to convert to appropriate type
                try:
                    # Try int first
                    if value.isdigit():
                        value = int(value)
                    # Try float
                    elif '.' in value and value.replace('.', '').isdigit():
                        value = float(value)
                    # Try boolean
                    elif value.lower() in ('true', 'false'):
                        value = value.lower() == 'true'
                except ValueError:
                    pass  # Keep as string
                
                self.set(config_path, value)
    
    def load_file(self, config_file: Union[str, Path]) -> None:
        """
        Load configuration from a YAML file.
        
        Args:
            config_file: Path to YAML configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
        
        # Deep merge with existing config
        self._deep_merge(self._config, file_config)
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary to update
            update: Dictionary with updates to apply
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'model.cnn.batch_size')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
            
        Examples:
            >>> config = Config()
            >>> config.get('model.cnn.batch_size')
            32
            >>> config.get('nonexistent.key', 'default_value')
            'default_value'
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'model.cnn.batch_size')
            value: Value to set
            
        Examples:
            >>> config = Config()
            >>> config.set('model.cnn.batch_size', 64)
            >>> config.get('model.cnn.batch_size')
            64
        """
        keys = key.split('.')
        current = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the final key
        current[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get the full configuration as a dictionary.
        
        Returns:
            Complete configuration dictionary
        """
        return self._config.copy()
    
    def save(self, output_file: Union[str, Path]) -> None:
        """
        Save the current configuration to a YAML file.
        
        Args:
            output_file: Path to output YAML file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style assignment."""
        self.set(key, value)


def load_config(config_file: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from file or use defaults.
    
    Args:
        config_file: Path to YAML configuration file
        
    Returns:
        Configured Config instance
    """
    return Config(config_file)


# Create a global default configuration instance
default_config = Config()