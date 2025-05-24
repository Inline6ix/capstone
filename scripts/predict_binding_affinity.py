#!/usr/bin/env python3
"""
Command-line script for MHC binding affinity prediction.

This script provides a clean command-line interface to the binding affinity
prediction functionality using netMHCpan.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src directory to path so we can import our package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from epitope_classifier.features.binding_affinity import predict_binding_affinities
from epitope_classifier.utils.logging import setup_logging


def main():
    """Main entry point for binding affinity prediction script."""
    parser = argparse.ArgumentParser(
        description='Predict MHC binding affinity for peptides using netMHCpan',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python predict_binding_affinity.py -i data/epitopes.csv -o results/predictions.csv
  
  # Specify custom netMHCpan path
  python predict_binding_affinity.py -i data/epitopes.csv -o results/predictions.csv \\
    --netmhcpan ~/tools/netMHCpan-4.1/run_netMHCpan.sh
  
  # Custom column names
  python predict_binding_affinity.py -i data/epitopes.csv -o results/predictions.csv \\
    --peptide-col sequence --allele-col mhc_allele
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--input', '-i', 
        type=str, 
        required=True,
        help='Input CSV file with peptides and alleles'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output', '-o', 
        type=str,
        help='Output CSV file for results (default: auto-generated filename)'
    )
    
    parser.add_argument(
        '--netmhcpan', '-n', 
        type=str,
        help='Path to netMHCpan script (default: ~/Documents/capstone/netMHCpan-4.1/run_netMHCpan.sh)'
    )
    
    parser.add_argument(
        '--peptide-col', 
        type=str, 
        default='peptide',
        help='Column name for peptides in input file (default: peptide)'
    )
    
    parser.add_argument(
        '--allele-col', 
        type=str, 
        default='allele',
        help='Column name for alleles in input file (default: allele)'
    )
    
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true',
        help='Print verbose output'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting MHC binding affinity prediction...")
        logger.info(f"Input file: {args.input}")
        logger.info(f"Output file: {args.output or 'auto-generated'}")
        
        # Run prediction
        results_df = predict_binding_affinities(
            input_file=args.input,
            output_file=args.output,
            peptide_col=args.peptide_col,
            allele_col=args.allele_col,
            netmhcpan_path=args.netmhcpan,
            verbose=args.verbose
        )
        
        # Print summary
        logger.info("Prediction completed successfully!")
        logger.info(f"Total predictions: {len(results_df)}")
        
        if len(results_df) > 0:
            # Show sample results
            logger.info("Sample results:")
            logger.info(f"\n{results_df.head()}")
            
            # Show binding statistics
            strong_binders = (results_df['%Rank_BA'] <= 0.5).sum()
            weak_binders = ((results_df['%Rank_BA'] > 0.5) & (results_df['%Rank_BA'] <= 2.0)).sum()
            non_binders = (results_df['%Rank_BA'] > 2.0).sum()
            
            logger.info(f"\\nBinding statistics:")
            logger.info(f"  Strong binders (≤0.5%): {strong_binders} ({strong_binders/len(results_df)*100:.1f}%)")
            logger.info(f"  Weak binders (0.5-2.0%): {weak_binders} ({weak_binders/len(results_df)*100:.1f}%)")
            logger.info(f"  Non-binders (>2.0%): {non_binders} ({non_binders/len(results_df)*100:.1f}%)")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()