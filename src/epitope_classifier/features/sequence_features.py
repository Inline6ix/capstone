"""
Sequence-based feature extraction for protein sequences.

This module provides functions to extract various sequence-based features
from amino acid sequences for machine learning.
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_sequence_composition(sequence: str) -> Dict[str, int]:
    """
    Calculate amino acid composition of a sequence.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Dictionary with amino acid counts
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    composition = {aa: 0 for aa in amino_acids}
    
    for aa in sequence.upper():
        if aa in composition:
            composition[aa] += 1
    
    return composition


def calculate_sequence_frequencies(sequence: str) -> Dict[str, float]:
    """
    Calculate amino acid frequencies in a sequence.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Dictionary with amino acid frequencies
    """
    composition = calculate_sequence_composition(sequence)
    total = len(sequence)
    
    if total == 0:
        return composition
    
    return {aa: count / total for aa, count in composition.items()}


def calculate_dipeptide_frequencies(sequence: str) -> Dict[str, float]:
    """
    Calculate dipeptide frequencies in a sequence.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Dictionary with dipeptide frequencies
    """
    if len(sequence) < 2:
        return {}
    
    dipeptides = {}
    sequence = sequence.upper()
    
    for i in range(len(sequence) - 1):
        dipeptide = sequence[i:i+2]
        dipeptides[dipeptide] = dipeptides.get(dipeptide, 0) + 1
    
    # Convert to frequencies
    total = len(sequence) - 1
    return {dp: count / total for dp, count in dipeptides.items()}


def calculate_tripeptide_frequencies(sequence: str) -> Dict[str, float]:
    """
    Calculate tripeptide frequencies in a sequence.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Dictionary with tripeptide frequencies
    """
    if len(sequence) < 3:
        return {}
    
    tripeptides = {}
    sequence = sequence.upper()
    
    for i in range(len(sequence) - 2):
        tripeptide = sequence[i:i+3]
        tripeptides[tripeptide] = tripeptides.get(tripeptide, 0) + 1
    
    # Convert to frequencies
    total = len(sequence) - 2
    return {tp: count / total for tp, count in tripeptides.items()}


def calculate_positional_features(sequence: str) -> Dict[str, any]:
    """
    Calculate position-specific features for a sequence.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Dictionary with positional features
    """
    if not sequence:
        return {}
    
    sequence = sequence.upper()
    features = {
        'n_terminal': sequence[0] if sequence else '',
        'c_terminal': sequence[-1] if sequence else '',
        'length': len(sequence)
    }
    
    # Middle residue(s)
    mid_idx = len(sequence) // 2
    if len(sequence) % 2 == 1:
        features['middle_residue'] = sequence[mid_idx]
    else:
        features['middle_residues'] = sequence[mid_idx-1:mid_idx+1]
    
    # Position-specific amino acid content
    for i, aa in enumerate(sequence):
        features[f'pos_{i+1}'] = aa
    
    return features


def extract_sequence_features(
    sequences: Union[str, List[str]], 
    include_composition: bool = True,
    include_dipeptides: bool = False,
    include_tripeptides: bool = False,
    include_positional: bool = False
) -> pd.DataFrame:
    """
    Extract comprehensive sequence features for one or more sequences.
    
    Args:
        sequences: Single sequence or list of sequences
        include_composition: Whether to include amino acid composition
        include_dipeptides: Whether to include dipeptide frequencies
        include_tripeptides: Whether to include tripeptide frequencies
        include_positional: Whether to include positional features
        
    Returns:
        DataFrame with extracted features
    """
    if isinstance(sequences, str):
        sequences = [sequences]
    
    features_list = []
    
    for i, sequence in enumerate(sequences):
        if pd.isna(sequence) or not sequence:
            logger.warning(f"Skipping empty sequence at index {i}")
            continue
        
        features = {
            'sequence': sequence,
            'length': len(sequence)
        }
        
        # Amino acid composition
        if include_composition:
            composition = calculate_sequence_composition(sequence)
            features.update(composition)
        
        # Dipeptide frequencies
        if include_dipeptides:
            dipeptides = calculate_dipeptide_frequencies(sequence)
            features.update({f'dipeptide_{dp}': freq for dp, freq in dipeptides.items()})
        
        # Tripeptide frequencies
        if include_tripeptides:
            tripeptides = calculate_tripeptide_frequencies(sequence)
            features.update({f'tripeptide_{tp}': freq for tp, freq in tripeptides.items()})
        
        # Positional features
        if include_positional:
            positional = calculate_positional_features(sequence)
            features.update(positional)
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)


def one_hot_encode_sequence(sequence: str, max_length: Optional[int] = None) -> np.ndarray:
    """
    One-hot encode a single amino acid sequence.
    
    Args:
        sequence: Amino acid sequence
        max_length: Maximum length for padding/truncation
        
    Returns:
        One-hot encoded sequence
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}
    
    if max_length is None:
        max_length = len(sequence)
    
    # Initialize output array
    encoded = np.zeros((max_length, len(amino_acids)))
    
    sequence = sequence.upper()
    for i, aa in enumerate(sequence[:max_length]):
        if aa in aa_to_index:
            encoded[i, aa_to_index[aa]] = 1.0
    
    return encoded


def sequence_to_integers(sequence: str) -> List[int]:
    """
    Convert amino acid sequence to integer representation.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        List of integers representing amino acids
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_index = {aa: i+1 for i, aa in enumerate(amino_acids)}  # Start from 1, 0 for padding
    
    return [aa_to_index.get(aa.upper(), 0) for aa in sequence]


def calculate_sequence_similarity(seq1: str, seq2: str) -> float:
    """
    Calculate simple sequence similarity (fraction of identical positions).
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        
    Returns:
        Similarity score between 0 and 1
    """
    if len(seq1) != len(seq2):
        return 0.0
    
    if len(seq1) == 0:
        return 1.0
    
    matches = sum(1 for a, b in zip(seq1.upper(), seq2.upper()) if a == b)
    return matches / len(seq1)


def find_sequence_motifs(sequences: List[str], min_length: int = 3, min_frequency: int = 2) -> Dict[str, int]:
    """
    Find common motifs in a collection of sequences.
    
    Args:
        sequences: List of amino acid sequences
        min_length: Minimum motif length
        min_frequency: Minimum frequency for a motif to be reported
        
    Returns:
        Dictionary mapping motifs to their frequencies
    """
    motif_counts = {}
    
    for sequence in sequences:
        sequence = sequence.upper()
        
        # Extract all possible motifs of different lengths
        for length in range(min_length, len(sequence) + 1):
            for i in range(len(sequence) - length + 1):
                motif = sequence[i:i + length]
                motif_counts[motif] = motif_counts.get(motif, 0) + 1
    
    # Filter by minimum frequency
    return {motif: count for motif, count in motif_counts.items() if count >= min_frequency}


def calculate_sequence_entropy(sequence: str) -> float:
    """
    Calculate Shannon entropy of amino acid composition in a sequence.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Shannon entropy value
    """
    if not sequence:
        return 0.0
    
    # Count amino acid frequencies
    aa_counts = {}
    for aa in sequence.upper():
        aa_counts[aa] = aa_counts.get(aa, 0) + 1
    
    # Calculate probabilities
    total = len(sequence)
    probabilities = [count / total for count in aa_counts.values()]
    
    # Calculate entropy
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    
    return entropy


def calculate_sequence_complexity(sequence: str, window_size: int = 3) -> float:
    """
    Calculate sequence complexity based on local diversity.
    
    Args:
        sequence: Amino acid sequence
        window_size: Size of sliding window for complexity calculation
        
    Returns:
        Complexity score
    """
    if len(sequence) < window_size:
        return calculate_sequence_entropy(sequence)
    
    complexities = []
    
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i + window_size]
        complexity = calculate_sequence_entropy(window)
        complexities.append(complexity)
    
    return np.mean(complexities) if complexities else 0.0