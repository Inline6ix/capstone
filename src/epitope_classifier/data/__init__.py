"""
Data processing and preprocessing modules.

This module contains functionality for:
- Loading and validating epitope datasets
- Preprocessing protein sequences
- Generating negative samples
- Data quality checks and validation
"""

from .preprocessing import (
    load_epitope_data,
    clean_epitope_data,
    generate_negative_samples,
    one_hot_encode_sequences,
    prepare_training_data,
    compute_class_weights,
    filter_by_length
)

from .validation import (
    validate_sequence_format,
    validate_allele_format,
    check_data_quality,
    validate_feature_matrix,
    validate_labels,
    generate_data_quality_report
)