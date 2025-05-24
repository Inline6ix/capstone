"""
Input/Output utilities for data loading and saving.

This module provides functions for loading and saving various data formats
used in epitope classification.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_csv_data(
    filepath: Union[str, Path],
    clean_columns: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Load data from CSV file with optional column cleaning.
    
    Args:
        filepath: Path to CSV file
        clean_columns: Whether to clean column names
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        DataFrame with loaded data
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    logger.info(f"Loading CSV data from {filepath}")
    
    df = pd.read_csv(filepath, **kwargs)
    
    if clean_columns:
        # Clean column names
        df.columns = df.columns.str.lower()
        df.columns = df.columns.str.replace(' ', '_')
        df.columns = df.columns.str.replace('-', '_')
        df.columns = df.columns.str.replace('.', '_')
    
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    return df


def save_csv_data(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    create_dirs: bool = True,
    **kwargs
) -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        filepath: Output file path
        create_dirs: Whether to create parent directories
        **kwargs: Additional arguments for df.to_csv
    """
    filepath = Path(filepath)
    
    if create_dirs:
        filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving CSV data to {filepath}")
    
    # Set default arguments
    csv_kwargs = {'index': False}
    csv_kwargs.update(kwargs)
    
    df.to_csv(filepath, **csv_kwargs)
    
    logger.info(f"Saved {len(df)} rows to {filepath}")


def load_json_data(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary with loaded data
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    logger.info(f"Loading JSON data from {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data


def save_json_data(
    data: Dict[str, Any],
    filepath: Union[str, Path],
    create_dirs: bool = True,
    indent: int = 2
) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Output file path
        create_dirs: Whether to create parent directories
        indent: JSON indentation level
    """
    filepath = Path(filepath)
    
    if create_dirs:
        filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving JSON data to {filepath}")
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)  # default=str handles numpy types


def load_pickle_data(filepath: Union[str, Path]) -> Any:
    """
    Load data from pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Unpickled data
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    logger.info(f"Loading pickle data from {filepath}")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    return data


def save_pickle_data(
    data: Any,
    filepath: Union[str, Path],
    create_dirs: bool = True
) -> None:
    """
    Save data to pickle file.
    
    Args:
        data: Data to save
        filepath: Output file path
        create_dirs: Whether to create parent directories
    """
    filepath = Path(filepath)
    
    if create_dirs:
        filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving pickle data to {filepath}")
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_sequences_from_fasta(filepath: Union[str, Path]) -> Dict[str, str]:
    """
    Load sequences from FASTA file.
    
    Args:
        filepath: Path to FASTA file
        
    Returns:
        Dictionary mapping sequence IDs to sequences
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    logger.info(f"Loading FASTA sequences from {filepath}")
    
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                
                # Start new sequence
                current_id = line[1:]  # Remove '>'
                current_seq = []
            else:
                current_seq.append(line)
    
    # Save last sequence
    if current_id is not None:
        sequences[current_id] = ''.join(current_seq)
    
    logger.info(f"Loaded {len(sequences)} sequences")
    
    return sequences


def save_sequences_to_fasta(
    sequences: Dict[str, str],
    filepath: Union[str, Path],
    create_dirs: bool = True,
    line_length: int = 80
) -> None:
    """
    Save sequences to FASTA file.
    
    Args:
        sequences: Dictionary mapping sequence IDs to sequences
        filepath: Output file path
        create_dirs: Whether to create parent directories
        line_length: Maximum line length for sequences
    """
    filepath = Path(filepath)
    
    if create_dirs:
        filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving {len(sequences)} sequences to {filepath}")
    
    with open(filepath, 'w') as f:
        for seq_id, sequence in sequences.items():
            f.write(f'>{seq_id}\n')
            
            # Write sequence with line breaks
            for i in range(0, len(sequence), line_length):
                f.write(sequence[i:i + line_length] + '\n')


def save_numpy_array(
    array: np.ndarray,
    filepath: Union[str, Path],
    create_dirs: bool = True,
    compressed: bool = True
) -> None:
    """
    Save numpy array to file.
    
    Args:
        array: Numpy array to save
        filepath: Output file path
        create_dirs: Whether to create parent directories
        compressed: Whether to use compression
    """
    filepath = Path(filepath)
    
    if create_dirs:
        filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving numpy array {array.shape} to {filepath}")
    
    if compressed:
        np.savez_compressed(filepath, array=array)
    else:
        np.save(filepath, array)


def load_numpy_array(filepath: Union[str, Path]) -> np.ndarray:
    """
    Load numpy array from file.
    
    Args:
        filepath: Path to numpy file
        
    Returns:
        Loaded numpy array
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    logger.info(f"Loading numpy array from {filepath}")
    
    if filepath.suffix == '.npz':
        data = np.load(filepath)
        if 'array' in data:
            return data['array']
        else:
            # Return first array if 'array' key doesn't exist
            return data[list(data.keys())[0]]
    else:
        return np.load(filepath)


def create_output_directory(base_dir: Union[str, Path], experiment_name: str) -> Path:
    """
    Create output directory for an experiment.
    
    Args:
        base_dir: Base directory for outputs
        experiment_name: Name of the experiment
        
    Returns:
        Path to created output directory
    """
    from datetime import datetime
    
    base_dir = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / f"{experiment_name}_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created output directory: {output_dir}")
    
    return output_dir


def list_files_by_pattern(directory: Union[str, Path], pattern: str = "*") -> List[Path]:
    """
    List files in directory matching a pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern to match
        
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    files = list(directory.glob(pattern))
    logger.info(f"Found {len(files)} files matching pattern '{pattern}' in {directory}")
    
    return files


def get_file_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about a file.
    
    Args:
        filepath: Path to file
        
    Returns:
        Dictionary with file information
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    stat = filepath.stat()
    
    info = {
        'name': filepath.name,
        'size_bytes': stat.st_size,
        'size_mb': stat.st_size / (1024 * 1024),
        'modified_time': stat.st_mtime,
        'is_file': filepath.is_file(),
        'is_directory': filepath.is_dir(),
        'suffix': filepath.suffix,
        'absolute_path': str(filepath.absolute())
    }
    
    return info