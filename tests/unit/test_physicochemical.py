"""
Unit tests for physicochemical feature calculation.
"""

import pytest
import numpy as np
import pandas as pd

# Add src to path for testing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from epitope_classifier.features.physicochemical import (
    calculate_physicochemical_features,
    calculate_hydrophobicity,
    calculate_amino_acid_composition,
    validate_sequence
)


class TestPhysicochemicalFeatures:
    """Test physicochemical feature calculation functions."""
    
    def test_valid_sequence_features(self):
        """Test feature calculation for a valid sequence."""
        sequence = "AAGIGILTV"
        features = calculate_physicochemical_features(sequence)
        
        assert features is not None
        assert isinstance(features, dict)
        assert 'molecular_weight' in features
        assert 'aromaticity' in features
        assert 'isoelectric_point' in features
        assert features['molecular_weight'] > 0
    
    def test_invalid_sequence_features(self):
        """Test feature calculation for invalid sequences."""
        # Empty sequence
        assert calculate_physicochemical_features("") is None
        
        # Sequence with invalid characters
        assert calculate_physicochemical_features("AAGIGILTX") is None
    
    def test_hydrophobicity_calculation(self):
        """Test hydrophobicity calculation."""
        sequence = "AAGIGILTV"
        hydro = calculate_hydrophobicity(sequence)
        
        assert hydro is not None
        assert isinstance(hydro, float)
        
        # Test with different scale
        hydro_eisenberg = calculate_hydrophobicity(sequence, scale='eisenberg')
        assert hydro_eisenberg is not None
        assert hydro_eisenberg != hydro  # Should be different values
    
    def test_amino_acid_composition(self):
        """Test amino acid composition calculation."""
        sequence = "AAGIGILTV"
        composition = calculate_amino_acid_composition(sequence)
        
        assert isinstance(composition, dict)
        assert len(composition) == 20  # 20 standard amino acids
        assert sum(composition.values()) == len(sequence)
        assert composition['A'] == 2  # Two A's in the sequence
        assert composition['G'] == 2  # Two G's in the sequence
    
    def test_sequence_validation(self):
        """Test sequence validation function."""
        # Valid sequences
        assert validate_sequence("AAGIGILTV") is True
        assert validate_sequence("ACDEFGHIKLMNPQRSTVWY") is True
        
        # Invalid sequences
        assert validate_sequence("") is False
        assert validate_sequence("AAGIGILTX") is False
        assert validate_sequence("123456789") is False
        assert validate_sequence(None) is False
    
    @pytest.mark.parametrize("sequence,expected_length", [
        ("AAGIGILTV", 9),
        ("ACDEFGHIKLMNPQRSTVWY", 20),
        ("A", 1),
    ])
    def test_sequence_lengths(self, sequence, expected_length):
        """Test that feature calculation works for different sequence lengths."""
        features = calculate_physicochemical_features(sequence)
        assert features is not None
        
        composition = calculate_amino_acid_composition(sequence)
        assert sum(composition.values()) == expected_length


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_and_none_inputs(self):
        """Test handling of empty and None inputs."""
        test_cases = [None, "", "   ", pd.NA]
        
        for test_input in test_cases:
            assert calculate_physicochemical_features(test_input) is None
            assert calculate_hydrophobicity(test_input) is None
            assert validate_sequence(test_input) is False
    
    def test_case_insensitivity(self):
        """Test that calculations are case-insensitive."""
        seq_upper = "AAGIGILTV"
        seq_lower = "aagigiltv"
        seq_mixed = "AaGiGiLtV"
        
        features_upper = calculate_physicochemical_features(seq_upper)
        features_lower = calculate_physicochemical_features(seq_lower)
        features_mixed = calculate_physicochemical_features(seq_mixed)
        
        # All should be equal (or all None)
        if features_upper is not None:
            assert features_lower is not None
            assert features_mixed is not None
            # Compare molecular weights as a representative feature
            assert abs(features_upper['molecular_weight'] - features_lower['molecular_weight']) < 0.001
            assert abs(features_upper['molecular_weight'] - features_mixed['molecular_weight']) < 0.001


if __name__ == "__main__":
    pytest.main([__file__])