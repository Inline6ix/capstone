"""
Machine learning models for epitope classification.

This module contains:
- CNNModel: Convolutional Neural Network for sequence classification
- RandomForestModel: Traditional ML approach using engineered features
- Base model classes and utilities
"""

from .base import BaseModel
from .cnn import CNNModel
from .random_forest import RandomForestModel