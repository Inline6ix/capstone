"""
MHC binding affinity prediction using netMHCpan.

This module provides functionality to predict MHC Class I binding affinities
for peptide-allele pairs using the external netMHCpan tool.
"""

import logging
import os
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class NetMHCpanError(Exception):
    """Custom exception for netMHCpan-related errors."""
    pass


class BindingAffinityPredictor:
    """
    MHC binding affinity predictor using netMHCpan.
    
    This class provides a clean interface to the netMHCpan tool for predicting
    binding affinities between peptides and MHC alleles.
    
    Attributes:
        netmhcpan_path: Path to the netMHCpan executable
        default_allele: Default allele to use when none specified
    """
    
    def __init__(self, netmhcpan_path: Optional[Union[str, Path]] = None):
        """
        Initialize the binding affinity predictor.
        
        Args:
            netmhcpan_path: Path to netMHCpan executable. If None, uses default path.
        """
        if netmhcpan_path is None:
            netmhcpan_path = os.path.expanduser('~/Documents/capstone/netMHCpan-4.1/run_netMHCpan.sh')
        
        self.netmhcpan_path = Path(netmhcpan_path)
        self.default_allele = "HLA-A*02:01"
        
        # Validate netMHCpan installation
        self._validate_installation()
    
    def _validate_installation(self) -> None:
        """Validate that netMHCpan is properly installed and accessible."""
        if not self.netmhcpan_path.exists():
            raise NetMHCpanError(f"netMHCpan not found at {self.netmhcpan_path}")
        
        # Make sure the script is executable
        os.chmod(self.netmhcpan_path, 0o755)
        
        # Test execution
        try:
            result = subprocess.run(
                [str(self.netmhcpan_path), "-h"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                check=False,
                timeout=30
            )
            if result.returncode != 0:
                logger.warning(f"netMHCpan test run failed: {result.stderr.decode('utf-8')}")
        except subprocess.TimeoutExpired:
            logger.warning("netMHCpan test run timed out")
        except Exception as e:
            raise NetMHCpanError(f"Failed to test netMHCpan installation: {e}")
    
    @staticmethod
    def parse_netmhcpan_output(output: Union[str, bytes]) -> List[Dict[str, Union[str, float]]]:
        """
        Parse the output from netMHCpan and extract prediction results.
        
        Args:
            output: Raw output from netMHCpan command
            
        Returns:
            List of dictionaries containing prediction results
            
        Raises:
            NetMHCpanError: If output parsing fails
        """
        # Convert bytes to string if needed
        if isinstance(output, bytes):
            output = output.decode('utf-8')
        
        results = []
        lines = output.split('\n')
        header_found = False
        
        for line in lines:
            # Skip empty lines and comments
            if not line.strip() or line.startswith('#'):
                continue
            
            # Find the header line
            if 'Pos' in line and 'MHC' in line and 'Peptide' in line:
                header_found = True
                continue
            
            # Skip separator line
            if header_found and '-' * 20 in line:
                continue
            
            # Process data lines
            if header_found and not line.startswith('--'):
                parts = re.split(r'\s+', line.strip())
                if len(parts) >= 14:  # Ensure we have at least the essential fields
                    try:
                        result = {
                            'pos': parts[0],
                            'allele': parts[1],
                            'peptide': parts[2],
                            'Score_EL': float(parts[11]) if len(parts) > 11 else 0.0,
                            '%Rank_EL': float(parts[12]) if len(parts) > 12 else 0.0,
                            'Score_BA': float(parts[13]) if len(parts) > 13 else 0.0,
                            '%Rank_BA': float(parts[14]) if len(parts) > 14 else 0.0,
                            'ic50': float(parts[15]) if len(parts) > 15 else 0.0
                        }
                        results.append(result)
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error parsing line: {line} - {e}")
                        continue
        
        if not results:
            raise NetMHCpanError("No valid prediction results found in netMHCpan output")
        
        return results
    
    def predict_single(self, peptide: str, allele: str, verbose: bool = False) -> Optional[Dict[str, Union[str, float]]]:
        """
        Predict binding affinity for a single peptide-allele pair.
        
        Args:
            peptide: Amino acid sequence
            allele: MHC allele name
            verbose: Whether to print detailed output
            
        Returns:
            Dictionary with prediction results or None if prediction failed
            
        Raises:
            NetMHCpanError: If prediction fails
        """
        # Validate inputs
        if not peptide or not isinstance(peptide, str):
            raise ValueError("Peptide must be a non-empty string")
        
        if len(peptide) < 8 or len(peptide) > 15:
            logger.warning(f"Peptide length {len(peptide)} outside typical range (8-15)")
        
        # Create temporary file for peptide
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pep', delete=False) as f:
            f.write(peptide + '\n')
            pepfile = f.name
        
        try:
            # Format allele string (remove * if present for command line)
            allele_str = allele.replace('*', '')
            
            # Build command
            cmd = [
                str(self.netmhcpan_path),
                '-BA',
                '-f', pepfile,
                '-inptype', '1',
                '-a', allele_str
            ]
            
            if verbose:
                logger.info(f"Running command: {' '.join(cmd)}")
            
            # Execute netMHCpan
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=60
            )
            
            # Parse output
            predictions = self.parse_netmhcpan_output(result.stdout)
            
            if predictions:
                return predictions[0]  # Return first result
            else:
                logger.warning(f"No prediction result for {peptide} with {allele}")
                return None
                
        except subprocess.CalledProcessError as e:
            error_msg = f"netMHCpan execution failed: {e.stderr.decode('utf-8')}"
            logger.error(error_msg)
            raise NetMHCpanError(error_msg)
        except subprocess.TimeoutExpired:
            error_msg = f"netMHCpan execution timed out for {peptide}"
            logger.error(error_msg)
            raise NetMHCpanError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error predicting {peptide} with {allele}: {e}"
            logger.error(error_msg)
            raise NetMHCpanError(error_msg)
        finally:
            # Clean up temporary file
            try:
                os.unlink(pepfile)
            except OSError:
                pass
    
    def predict_batch(
        self, 
        peptides: List[str], 
        alleles: List[str], 
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Predict binding affinities for multiple peptide-allele pairs.
        
        Args:
            peptides: List of amino acid sequences
            alleles: List of MHC allele names (must match length of peptides)
            verbose: Whether to print progress information
            
        Returns:
            DataFrame with prediction results
            
        Raises:
            ValueError: If input lists have different lengths
            NetMHCpanError: If batch prediction fails
        """
        if len(peptides) != len(alleles):
            raise ValueError("Peptides and alleles lists must have the same length")
        
        results = []
        total = len(peptides)
        
        logger.info(f"Processing {total} peptide-allele pairs...")
        
        for i, (peptide, allele) in enumerate(zip(peptides, alleles)):
            try:
                # Handle missing or invalid alleles
                if pd.isna(allele) or allele.strip() == "HLA class 1":
                    allele = self.default_allele
                    if verbose:
                        logger.info(f"Using default allele {allele} for peptide {peptide}")
                
                # Skip invalid peptides
                if pd.isna(peptide) or not peptide or len(peptide) < 8:
                    logger.warning(f"Skipping invalid peptide: '{peptide}'")
                    continue
                
                if verbose:
                    logger.info(f"Processing {i+1}/{total}: {peptide} with {allele}")
                
                result = self.predict_single(peptide, allele, verbose=False)
                
                if result:
                    result.update({
                        'input_peptide': peptide,
                        'input_allele': allele
                    })
                    results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing peptide {peptide} with allele {allele}: {e}")
                continue
        
        if not results:
            raise NetMHCpanError("No successful predictions in batch")
        
        df = pd.DataFrame(results)
        logger.info(f"Successfully processed {len(df)} out of {total} peptide-allele pairs")
        
        return df


def predict_binding_affinities(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    peptide_col: str = 'peptide',
    allele_col: str = 'allele',
    netmhcpan_path: Optional[Union[str, Path]] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Convenience function to predict binding affinities from a CSV file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (optional)
        peptide_col: Name of peptide column in input file
        allele_col: Name of allele column in input file
        netmhcpan_path: Path to netMHCpan executable
        verbose: Whether to print progress information
        
    Returns:
        DataFrame with prediction results
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        KeyError: If required columns are missing
    """
    # Load input data
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows from {input_path}")
    
    # Validate required columns
    if peptide_col not in df.columns:
        raise KeyError(f"Peptide column '{peptide_col}' not found in input file")
    if allele_col not in df.columns:
        raise KeyError(f"Allele column '{allele_col}' not found in input file")
    
    # Initialize predictor
    predictor = BindingAffinityPredictor(netmhcpan_path)
    
    # Run predictions
    results_df = predictor.predict_batch(
        peptides=df[peptide_col].tolist(),
        alleles=df[allele_col].tolist(),
        verbose=verbose
    )
    
    # Save results if output file specified
    if output_file:
        output_path = Path(output_file)
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
    
    return results_df