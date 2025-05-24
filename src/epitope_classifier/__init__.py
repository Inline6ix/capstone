"""
Epitope Classifier Package

A machine learning package for identifying T-cell epitopes in cancer antigens
for personalized immunotherapy applications.

This package provides:
- Data preprocessing and feature engineering for protein sequences
- Machine learning models (Random Forest, CNN) for epitope prediction
- Integration with external tools like netMHCpan for binding affinity prediction
- Utilities for visualization and analysis

Example:
    >>> from epitope_classifier.models import CNNModel
    >>> from epitope_classifier.features import physicochemical_features
    >>> 
    >>> # Load and train a model
    >>> model = CNNModel(input_shape=(9, 21))
    >>> predictions = model.predict(sequences)
"""

__version__ = "0.1.0"
__author__ = "Tariq Alagha"
__email__ = "your.email@example.com"

# Import main components for easy access (avoiding * imports for better control)
from epitope_classifier.models.base import BaseModel
from epitope_classifier.models.cnn import CNNModel
from epitope_classifier.models.random_forest import RandomForestModel

from epitope_classifier.features.binding_affinity import BindingAffinityPredictor
from epitope_classifier.features.physicochemical import calculate_physicochemical_features
from epitope_classifier.features.sequence_features import extract_sequence_features

from epitope_classifier.data.preprocessing import load_epitope_data, clean_epitope_data
from epitope_classifier.data.validation import validate_sequence_format

from epitope_classifier.utils.config import Config, load_config
from epitope_classifier.utils.logging import setup_logging
from epitope_classifier.utils.visualization import BindingAffinityVisualizer