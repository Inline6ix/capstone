"""
Physicochemical feature calculation for protein sequences.

This module provides functions to calculate various physicochemical properties
of protein sequences using BioPython.
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

logger = logging.getLogger(__name__)


def calculate_physicochemical_features(sequence: str) -> Optional[Dict[str, float]]:
    """
    Calculate physicochemical features for a protein sequence.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Dictionary with physicochemical features or None if calculation fails
    """
    try:
        # Create ProteinAnalysis object
        analyzer = ProteinAnalysis(sequence)
        
        # Calculate features
        features = {
            'molecular_weight': analyzer.molecular_weight(),
            'aromaticity': analyzer.aromaticity(),
            'isoelectric_point': analyzer.isoelectric_point(),
            'instability': analyzer.instability_index(),
            'charge_at_pH7': analyzer.charge_at_pH(7.0),
            'secondary_structure_helix': analyzer.secondary_structure_fraction()[0],
            'secondary_structure_turn': analyzer.secondary_structure_fraction()[1],
            'secondary_structure_sheet': analyzer.secondary_structure_fraction()[2]
        }
        
        return features
        
    except Exception as e:
        logger.warning(f"Failed to calculate physicochemical features for '{sequence}': {e}")
        return None


def calculate_hydrophobicity(sequence: str, scale: str = 'kyte_doolittle') -> Optional[float]:
    """
    Calculate average hydrophobicity of a protein sequence.
    
    Args:
        sequence: Amino acid sequence
        scale: Hydrophobicity scale to use ('kyte_doolittle', 'eisenberg', etc.)
        
    Returns:
        Average hydrophobicity score or None if calculation fails
    """
    # Kyte-Doolittle hydrophobicity scale
    kyte_doolittle = {
        'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5,
        'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8,
        'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5,
        'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5
    }
    
    # Eisenberg hydrophobicity scale
    eisenberg = {
        'I': 0.73, 'V': 0.54, 'L': 0.53, 'F': 0.51, 'C': 0.49,
        'M': 0.26, 'A': 0.25, 'T': -0.18, 'G': -0.32, 'S': -0.53,
        'W': -0.68, 'Y': -0.94, 'P': -0.76, 'H': -0.40, 'E': -0.62,
        'Q': -0.69, 'D': -0.72, 'N': -0.64, 'K': -1.10, 'R': -1.80
    }
    
    scales = {
        'kyte_doolittle': kyte_doolittle,
        'eisenberg': eisenberg
    }
    
    if scale not in scales:
        raise ValueError(f"Unknown hydrophobicity scale: {scale}")
    
    scale_dict = scales[scale]
    
    try:
        scores = [scale_dict.get(aa, 0) for aa in sequence.upper()]
        return sum(scores) / len(scores) if scores else 0.0
    except Exception as e:
        logger.warning(f"Failed to calculate hydrophobicity for '{sequence}': {e}")
        return None


def calculate_amino_acid_composition(sequence: str) -> Dict[str, int]:
    """
    Calculate amino acid composition of a protein sequence.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Dictionary with amino acid counts
    """
    # Standard amino acids
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    
    # Initialize counts
    composition = {aa: 0 for aa in amino_acids}
    
    # Count amino acids
    for aa in sequence.upper():
        if aa in composition:
            composition[aa] += 1
    
    return composition


def calculate_amino_acid_frequencies(sequence: str) -> Dict[str, float]:
    """
    Calculate amino acid frequencies of a protein sequence.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Dictionary with amino acid frequencies (0-1)
    """
    composition = calculate_amino_acid_composition(sequence)
    total = len(sequence)
    
    if total == 0:
        return composition
    
    return {aa: count / total for aa, count in composition.items()}


def calculate_dipeptide_composition(sequence: str) -> Dict[str, int]:
    """
    Calculate dipeptide composition of a protein sequence.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Dictionary with dipeptide counts
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    dipeptides = [aa1 + aa2 for aa1 in amino_acids for aa2 in amino_acids]
    
    # Initialize counts
    composition = {dipeptide: 0 for dipeptide in dipeptides}
    
    # Count dipeptides
    sequence = sequence.upper()
    for i in range(len(sequence) - 1):
        dipeptide = sequence[i:i+2]
        if dipeptide in composition:
            composition[dipeptide] += 1
    
    return composition


def extract_all_features(
    sequences: Union[str, List[str]], 
    include_composition: bool = True,
    include_dipeptides: bool = False
) -> pd.DataFrame:
    """
    Extract all physicochemical features for one or more sequences.
    
    Args:
        sequences: Single sequence or list of sequences
        include_composition: Whether to include amino acid composition
        include_dipeptides: Whether to include dipeptide composition
        
    Returns:
        DataFrame with features for each sequence
    """
    if isinstance(sequences, str):
        sequences = [sequences]
    
    features_list = []
    
    for i, sequence in enumerate(sequences):
        if pd.isna(sequence) or not sequence:
            logger.warning(f"Skipping empty sequence at index {i}")
            continue
        
        # Start with basic info
        features = {
            'sequence': sequence,
            'length': len(sequence)
        }
        
        # Add physicochemical features
        physico_features = calculate_physicochemical_features(sequence)
        if physico_features:
            features.update(physico_features)
        
        # Add hydrophobicity
        hydrophobicity = calculate_hydrophobicity(sequence)
        if hydrophobicity is not None:
            features['hydrophobicity'] = hydrophobicity
        
        # Add amino acid composition
        if include_composition:
            composition = calculate_amino_acid_composition(sequence)
            features.update({f'aa_{aa}': count for aa, count in composition.items()})
        
        # Add dipeptide composition
        if include_dipeptides:
            dipeptides = calculate_dipeptide_composition(sequence)
            features.update({f'dipeptide_{dp}': count for dp, count in dipeptides.items()})
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)


def validate_sequence(sequence: str) -> bool:
    """
    Validate that a sequence contains only standard amino acids.
    
    Args:
        sequence: Amino acid sequence to validate
        
    Returns:
        True if sequence is valid, False otherwise
    """
    if not sequence or pd.isna(sequence):
        return False
    
    valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    return all(aa.upper() in valid_amino_acids for aa in sequence)