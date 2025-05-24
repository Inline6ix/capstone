#!/usr/bin/env python3
"""
Command-line script for visualizing MHC binding affinity predictions.

This script creates comprehensive visualizations of binding affinity prediction results.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src directory to path so we can import our package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from epitope_classifier.utils.visualization import visualize_predictions_from_file
from epitope_classifier.utils.logging import setup_logging


def main():
    """Main entry point for visualization script."""
    parser = argparse.ArgumentParser(
        description='Create visualizations for MHC binding affinity predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python visualize_predictions.py -i results/predictions.csv -o figures/
  
  # Custom prefix for output files
  python visualize_predictions.py -i results/predictions.csv -o figures/ \\
    --prefix experiment_1
  
  # Verbose output
  python visualize_predictions.py -i results/predictions.csv -o figures/ -v
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input CSV file with prediction results'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='figures',
        help='Output directory for plots (default: figures)'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='predictions',
        help='Prefix for output filenames (default: predictions)'
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
        logger.info("Starting visualization generation...")
        logger.info(f"Input file: {args.input}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Filename prefix: {args.prefix}")
        
        # Create visualizations
        plot_paths = visualize_predictions_from_file(
            input_file=args.input,
            output_dir=args.output_dir,
            prefix=args.prefix
        )
        
        # Report results
        logger.info("Visualization generation completed successfully!")
        logger.info(f"Created {len(plot_paths)} plots:")
        
        for plot_type, path in plot_paths.items():
            logger.info(f"  {plot_type}: {path}")
        
        logger.info(f"\\nAll plots saved to: {Path(args.output_dir).resolve()}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()