"""
Feature engineering and extraction modules.

This module provides:
- Physicochemical property calculation using BioPython
- MHC binding affinity prediction via netMHCpan integration
- Amino acid composition and sequence analysis
- Feature scaling and normalization utilities
"""

from .physicochemical import (
    calculate_physicochemical_features,
    calculate_hydrophobicity,
    calculate_amino_acid_composition,
    extract_all_features,
    validate_sequence
)

from .binding_affinity import (
    BindingAffinityPredictor,
    predict_binding_affinities,
    NetMHCpanError
)

from .sequence_features import (
    calculate_sequence_composition,
    calculate_sequence_frequencies,
    extract_sequence_features,
    one_hot_encode_sequence,
    calculate_sequence_similarity
)