"""
Data validation utilities for epitope classification.

This module provides functions for validating data quality and consistency
in epitope datasets.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_sequence_format(sequences: pd.Series) -> pd.Series:
    """
    Validate that sequences contain only standard amino acids.
    
    Args:
        sequences: Series of amino acid sequences
        
    Returns:
        Boolean series indicating which sequences are valid
    """
    valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    
    def is_valid(seq):
        if pd.isna(seq) or not isinstance(seq, str):
            return False
        return all(aa.upper() in valid_amino_acids for aa in seq)
    
    return sequences.apply(is_valid)


def validate_allele_format(alleles: pd.Series) -> pd.Series:
    """
    Validate MHC allele format.
    
    Args:
        alleles: Series of MHC allele names
        
    Returns:
        Boolean series indicating which alleles have valid format
    """
    def is_valid_allele(allele):
        if pd.isna(allele) or not isinstance(allele, str):
            return False
        
        # Basic HLA format validation
        allele = allele.strip()
        if allele.startswith('HLA-'):
            return True
        if allele in ['HLA class 1', 'HLA class I', 'HLA class II']:
            return True
        
        return False
    
    return alleles.apply(is_valid_allele)


def check_data_quality(df: pd.DataFrame, sequence_col: str = 'peptide') -> Dict[str, any]:
    """
    Comprehensive data quality check for epitope datasets.
    
    Args:
        df: DataFrame to validate
        sequence_col: Column containing sequences
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'total_records': len(df),
        'missing_sequences': df[sequence_col].isna().sum(),
        'empty_sequences': (df[sequence_col] == '').sum(),
        'invalid_sequences': 0,
        'sequence_length_stats': {},
        'duplicate_sequences': 0,
        'warnings': [],
        'errors': []
    }
    
    # Check for invalid sequences
    if sequence_col in df.columns:
        valid_sequences = validate_sequence_format(df[sequence_col])
        results['invalid_sequences'] = (~valid_sequences).sum()
        
        if results['invalid_sequences'] > 0:
            results['warnings'].append(f"{results['invalid_sequences']} sequences contain invalid amino acids")
    
    # Sequence length statistics
    if sequence_col in df.columns:
        sequence_lengths = df[sequence_col].str.len()
        results['sequence_length_stats'] = {
            'min': sequence_lengths.min(),
            'max': sequence_lengths.max(),
            'mean': sequence_lengths.mean(),
            'median': sequence_lengths.median(),
            'std': sequence_lengths.std()
        }
        
        # Check for unusual lengths
        if sequence_lengths.min() < 6:
            results['warnings'].append(f"Found sequences shorter than 6 amino acids (min: {sequence_lengths.min()})")
        if sequence_lengths.max() > 20:
            results['warnings'].append(f"Found sequences longer than 20 amino acids (max: {sequence_lengths.max()})")
    
    # Check for duplicates
    if sequence_col in df.columns:
        duplicate_count = df[sequence_col].duplicated().sum()
        results['duplicate_sequences'] = duplicate_count
        if duplicate_count > 0:
            results['warnings'].append(f"{duplicate_count} duplicate sequences found")
    
    # Check for missing critical columns
    critical_columns = [sequence_col]
    for col in critical_columns:
        if col not in df.columns:
            results['errors'].append(f"Critical column '{col}' is missing")
    
    # Summary
    total_issues = results['missing_sequences'] + results['empty_sequences'] + results['invalid_sequences']
    results['quality_score'] = 1.0 - (total_issues / len(df)) if len(df) > 0 else 0.0
    
    return results


def validate_feature_matrix(X: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict[str, any]:
    """
    Validate feature matrix for training.
    
    Args:
        X: Feature matrix
        feature_names: Optional list of feature names
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'shape': X.shape,
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'missing_values': np.isnan(X).sum(),
        'infinite_values': np.isinf(X).sum(),
        'feature_stats': {},
        'warnings': [],
        'errors': []
    }
    
    # Check for missing values
    if results['missing_values'] > 0:
        results['warnings'].append(f"{results['missing_values']} missing values found in feature matrix")
    
    # Check for infinite values
    if results['infinite_values'] > 0:
        results['errors'].append(f"{results['infinite_values']} infinite values found in feature matrix")
    
    # Feature-wise statistics
    for i in range(X.shape[1]):
        feature_name = feature_names[i] if feature_names else f'feature_{i}'
        feature_data = X[:, i]
        
        stats = {
            'min': np.min(feature_data),
            'max': np.max(feature_data),
            'mean': np.mean(feature_data),
            'std': np.std(feature_data),
            'missing': np.isnan(feature_data).sum()
        }
        
        # Check for constant features
        if stats['std'] == 0:
            results['warnings'].append(f"Feature '{feature_name}' is constant (std=0)")
        
        # Check for extreme ranges
        if stats['max'] - stats['min'] > 1000:
            results['warnings'].append(f"Feature '{feature_name}' has very large range ({stats['min']:.2f} to {stats['max']:.2f})")
        
        results['feature_stats'][feature_name] = stats
    
    return results


def validate_labels(y: np.ndarray) -> Dict[str, any]:
    """
    Validate label array.
    
    Args:
        y: Label array
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'n_samples': len(y),
        'unique_values': np.unique(y).tolist(),
        'n_classes': len(np.unique(y)),
        'class_counts': {},
        'missing_values': np.isnan(y).sum() if y.dtype.kind == 'f' else 0,
        'warnings': [],
        'errors': []
    }
    
    # Count classes
    unique, counts = np.unique(y, return_counts=True)
    results['class_counts'] = {str(cls): int(count) for cls, count in zip(unique, counts)}
    
    # Check for missing labels
    if results['missing_values'] > 0:
        results['errors'].append(f"{results['missing_values']} missing labels found")
    
    # Check for class imbalance
    if len(counts) == 2:  # Binary classification
        minority_ratio = min(counts) / max(counts)
        if minority_ratio < 0.1:
            results['warnings'].append(f"Severe class imbalance detected (minority class ratio: {minority_ratio:.3f})")
        elif minority_ratio < 0.3:
            results['warnings'].append(f"Class imbalance detected (minority class ratio: {minority_ratio:.3f})")
    
    # Check for unexpected label values
    if results['n_classes'] == 2 and not set(unique).issubset({0, 1}):
        results['warnings'].append(f"Binary labels should be 0/1, found: {unique}")
    
    return results


def validate_train_test_split(
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    y_train: np.ndarray, 
    y_test: np.ndarray
) -> Dict[str, any]:
    """
    Validate train/test split consistency.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_features': X_train.shape[1],
        'test_features': X_test.shape[1],
        'warnings': [],
        'errors': []
    }
    
    # Check feature consistency
    if X_train.shape[1] != X_test.shape[1]:
        results['errors'].append("Training and test sets have different number of features")
    
    # Check label consistency
    if len(X_train) != len(y_train):
        results['errors'].append("Training features and labels have different lengths")
    
    if len(X_test) != len(y_test):
        results['errors'].append("Test features and labels have different lengths")
    
    # Check class distribution
    train_classes = set(np.unique(y_train))
    test_classes = set(np.unique(y_test))
    
    if train_classes != test_classes:
        results['warnings'].append(f"Different classes in train vs test: train={train_classes}, test={test_classes}")
    
    # Check for data leakage (basic check)
    if X_train.shape == X_test.shape and np.array_equal(X_train, X_test):
        results['errors'].append("Training and test sets are identical - possible data leakage")
    
    return results


def generate_data_quality_report(
    df: pd.DataFrame, 
    sequence_col: str = 'peptide',
    allele_col: Optional[str] = None
) -> str:
    """
    Generate a comprehensive data quality report.
    
    Args:
        df: DataFrame to analyze
        sequence_col: Column containing sequences
        allele_col: Optional column containing alleles
        
    Returns:
        Formatted report string
    """
    report_lines = ["DATA QUALITY REPORT", "=" * 50, ""]
    
    # Basic statistics
    report_lines.extend([
        f"Total records: {len(df)}",
        f"Columns: {list(df.columns)}",
        ""
    ])
    
    # Sequence validation
    seq_results = check_data_quality(df, sequence_col)
    report_lines.extend([
        "SEQUENCE VALIDATION:",
        f"  - Missing sequences: {seq_results['missing_sequences']}",
        f"  - Empty sequences: {seq_results['empty_sequences']}",
        f"  - Invalid sequences: {seq_results['invalid_sequences']}",
        f"  - Duplicate sequences: {seq_results['duplicate_sequences']}",
        f"  - Quality score: {seq_results['quality_score']:.3f}",
        ""
    ])
    
    # Length statistics
    if seq_results['sequence_length_stats']:
        stats = seq_results['sequence_length_stats']
        report_lines.extend([
            "SEQUENCE LENGTH STATISTICS:",
            f"  - Min length: {stats['min']:.0f}",
            f"  - Max length: {stats['max']:.0f}",
            f"  - Mean length: {stats['mean']:.1f}",
            f"  - Median length: {stats['median']:.0f}",
            f"  - Std deviation: {stats['std']:.1f}",
            ""
        ])
    
    # Allele validation (if applicable)
    if allele_col and allele_col in df.columns:
        valid_alleles = validate_allele_format(df[allele_col])
        unique_alleles = df[allele_col].nunique()
        
        report_lines.extend([
            "ALLELE VALIDATION:",
            f"  - Valid alleles: {valid_alleles.sum()} / {len(valid_alleles)}",
            f"  - Unique alleles: {unique_alleles}",
            ""
        ])
    
    # Warnings and errors
    if seq_results['warnings']:
        report_lines.extend(["WARNINGS:"] + [f"  - {w}" for w in seq_results['warnings']] + [""])
    
    if seq_results['errors']:
        report_lines.extend(["ERRORS:"] + [f"  - {e}" for e in seq_results['errors']] + [""])
    
    return "\n".join(report_lines)