"""
Data preprocessing utilities for epitope classification.

This module provides functions for loading, cleaning, and preprocessing
epitope datasets for machine learning.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


def load_epitope_data(
    epitope_file: Union[str, Path],
    assay_file: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Load epitope data from CSV files.
    
    Args:
        epitope_file: Path to epitope CSV file
        assay_file: Optional path to assay CSV file for additional metadata
        
    Returns:
        DataFrame with loaded epitope data
    """
    epitope_path = Path(epitope_file)
    if not epitope_path.exists():
        raise FileNotFoundError(f"Epitope file not found: {epitope_path}")
    
    logger.info(f"Loading epitope data from {epitope_path}")
    epitopes = pd.read_csv(epitope_path)
    
    # Clean column names
    epitopes.columns = epitopes.columns.str.lower()
    epitopes.columns = epitopes.columns.str.replace(' ', '_')
    epitopes.columns = epitopes.columns.str.replace('-', '_')
    
    logger.info(f"Loaded {len(epitopes)} epitope records")
    
    # Load assay data if provided
    if assay_file:
        assay_path = Path(assay_file)
        if assay_path.exists():
            logger.info(f"Loading assay data from {assay_path}")
            assays = pd.read_csv(assay_path)
            
            # Clean column names
            assays.columns = assays.columns.str.lower()
            assays.columns = assays.columns.str.replace(' ', '_')
            assays.columns = assays.columns.str.replace('-', '_')
            
            # Merge with epitope data if possible
            common_cols = set(epitopes.columns) & set(assays.columns)
            if common_cols:
                merge_col = list(common_cols)[0]  # Use first common column
                logger.info(f"Merging datasets on column: {merge_col}")
                epitopes = epitopes.merge(assays, on=merge_col, how='left')
                logger.info(f"Merged dataset has {len(epitopes)} records")
    
    return epitopes


def clean_epitope_data(
    df: pd.DataFrame,
    sequence_col: str = 'epitope_name',
    min_length: int = 8,
    max_length: int = 15
) -> pd.DataFrame:
    """
    Clean epitope data by removing invalid sequences and outliers.
    
    Args:
        df: DataFrame with epitope data
        sequence_col: Name of column containing epitope sequences
        min_length: Minimum valid sequence length
        max_length: Maximum valid sequence length
        
    Returns:
        Cleaned DataFrame
    """
    initial_count = len(df)
    df = df.copy()
    
    # Remove rows with missing sequences
    df = df.dropna(subset=[sequence_col])
    logger.info(f"Removed {initial_count - len(df)} rows with missing sequences")
    
    # Filter by sequence length
    df['sequence_length'] = df[sequence_col].str.len()
    df = df[(df['sequence_length'] >= min_length) & (df['sequence_length'] <= max_length)]
    logger.info(f"Filtered to sequences of length {min_length}-{max_length}: {len(df)} remaining")
    
    # Remove sequences with invalid amino acids
    valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    
    def is_valid_sequence(seq):
        return all(aa.upper() in valid_amino_acids for aa in str(seq))
    
    valid_mask = df[sequence_col].apply(is_valid_sequence)
    df = df[valid_mask]
    logger.info(f"Removed sequences with invalid amino acids: {len(df)} remaining")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=[sequence_col])
    logger.info(f"Removed duplicate sequences: {len(df)} unique sequences remaining")
    
    return df


def generate_negative_samples(
    epitope_df: pd.DataFrame,
    sequence_col: str = 'epitope_name',
    full_sequence_col: str = 'fullsequence',
    allele_col: str = 'mhcrestriction_name',
    ratio: float = 4.0,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate negative samples from full protein sequences.
    
    Args:
        epitope_df: DataFrame with epitope data
        sequence_col: Column with epitope sequences
        full_sequence_col: Column with full protein sequences
        allele_col: Column with MHC allele information
        ratio: Ratio of negatives to positives to generate
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with negative samples
    """
    np.random.seed(random_state)
    
    # Get epitope sequences for exclusion
    known_epitopes = set(epitope_df[sequence_col].str.upper())
    
    negatives = []
    target_count = int(len(epitope_df) * ratio)
    
    logger.info(f"Generating {target_count} negative samples...")
    
    for _, row in epitope_df.iterrows():
        if pd.isna(row[full_sequence_col]):
            continue
        
        epitope = str(row[sequence_col]).upper()
        full_seq = str(row[full_sequence_col]).upper()
        allele = row.get(allele_col, 'HLA-A*02:01')
        epitope_length = len(epitope)
        
        # Generate sliding window negatives
        for i in range(len(full_seq) - epitope_length + 1):
            if len(negatives) >= target_count:
                break
            
            window = full_seq[i:i + epitope_length]
            
            # Skip if this window is a known epitope
            if window in known_epitopes:
                continue
            
            # Skip if window contains invalid amino acids
            valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
            if not all(aa in valid_amino_acids for aa in window):
                continue
            
            negatives.append({
                'peptide': window,
                'allele': allele,
                'label': 0
            })
        
        if len(negatives) >= target_count:
            break
    
    # Convert to DataFrame and sample if we have too many
    negatives_df = pd.DataFrame(negatives)
    if len(negatives_df) > target_count:
        negatives_df = negatives_df.sample(n=target_count, random_state=random_state)
    
    logger.info(f"Generated {len(negatives_df)} negative samples")
    
    return negatives_df


def one_hot_encode_sequences(sequences: List[str], max_length: Optional[int] = None) -> np.ndarray:
    """
    One-hot encode amino acid sequences for CNN input.
    
    Args:
        sequences: List of amino acid sequences
        max_length: Maximum sequence length for padding. If None, uses longest sequence.
        
    Returns:
        One-hot encoded sequences as numpy array
    """
    # Standard amino acids
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}
    
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    # Initialize output array
    encoded = np.zeros((len(sequences), max_length, len(amino_acids)))
    
    for i, sequence in enumerate(sequences):
        sequence = sequence.upper()
        for j, aa in enumerate(sequence[:max_length]):  # Truncate if too long
            if aa in aa_to_index:
                encoded[i, j, aa_to_index[aa]] = 1.0
    
    return encoded


def prepare_training_data(
    positive_df: pd.DataFrame,
    negative_df: pd.DataFrame,
    sequence_col: str = 'peptide',
    test_size: float = 0.2,
    val_size: float = 0.15,
    random_state: int = 42,
    encoding_type: str = 'onehot'
) -> Dict[str, np.ndarray]:
    """
    Prepare training, validation, and test datasets.
    
    Args:
        positive_df: DataFrame with positive samples (epitopes)
        negative_df: DataFrame with negative samples
        sequence_col: Column name containing sequences
        test_size: Proportion of data for testing
        val_size: Proportion of remaining data for validation
        random_state: Random seed for reproducibility
        encoding_type: Type of encoding ('onehot' for CNN, 'features' for traditional ML)
        
    Returns:
        Dictionary with train/val/test splits
    """
    # Combine positive and negative samples
    positive_df = positive_df.copy()
    negative_df = negative_df.copy()
    
    positive_df['label'] = 1
    negative_df['label'] = 0
    
    combined_df = pd.concat([positive_df, negative_df], ignore_index=True)
    
    # Shuffle the data
    combined_df = combined_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    logger.info(f"Combined dataset: {len(combined_df)} samples ({sum(combined_df['label'])} positive, {len(combined_df) - sum(combined_df['label'])} negative)")
    
    # Split into train/temp, then temp into val/test
    train_df, temp_df = train_test_split(
        combined_df, 
        test_size=(test_size + val_size),
        random_state=random_state,
        stratify=combined_df['label']
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (test_size + val_size),
        random_state=random_state,
        stratify=temp_df['label']
    )
    
    logger.info(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Encode sequences
    if encoding_type == 'onehot':
        # One-hot encoding for CNN
        all_sequences = combined_df[sequence_col].tolist()
        max_length = max(len(seq) for seq in all_sequences)
        
        X_train = one_hot_encode_sequences(train_df[sequence_col].tolist(), max_length)
        X_val = one_hot_encode_sequences(val_df[sequence_col].tolist(), max_length)
        X_test = one_hot_encode_sequences(test_df[sequence_col].tolist(), max_length)
        
    else:
        # Feature-based encoding for traditional ML
        # This would require feature extraction - placeholder for now
        raise NotImplementedError("Feature-based encoding not yet implemented in this function")
    
    # Extract labels
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df
    }


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        y: Label array
        
    Returns:
        Dictionary mapping class indices to weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def filter_by_length(df: pd.DataFrame, sequence_col: str, target_length: int) -> pd.DataFrame:
    """
    Filter sequences to a specific length.
    
    Args:
        df: DataFrame with sequence data
        sequence_col: Column containing sequences
        target_length: Target sequence length
        
    Returns:
        Filtered DataFrame
    """
    initial_count = len(df)
    df_filtered = df[df[sequence_col].str.len() == target_length].copy()
    
    logger.info(f"Filtered to {target_length}-mer sequences: {len(df_filtered)} of {initial_count} remaining")
    
    return df_filtered