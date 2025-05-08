import pandas as pd
import os
import sys
import subprocess
import tempfile
import re
import argparse
from datetime import datetime

def parse_netmhcpan_output(output):
    """
    Parse the output from netMHCpan and extract the prediction results.
    
    Args:
        output (str or bytes): The output from netMHCpan
        
    Returns:
        list: A list of dictionaries containing the prediction results
    """
    # Convert bytes to string if needed
    if isinstance(output, bytes):
        output = output.decode('utf-8')
    
    # Manual parsing of the output
    results = []
    lines = output.split('\n')
    header_found = False
    
    for line in lines:
        # Skip empty lines and comments
        if not line.strip() or line.startswith('#'):
            continue
        
        # Skip until we find the header line
        if 'Pos' in line and 'MHC' in line and 'Peptide' in line:
            header_found = True
            continue
        
        # Skip the separator line
        if header_found and '-' * 20 in line:
            continue
        
        # Process data lines
        if header_found and not line.startswith('--'):
            # Split the line by whitespace
            parts = re.split(r'\s+', line.strip())
            if len(parts) >= 15:  # Make sure we have enough fields
                try:
                    pos = parts[0]
                    allele = parts[1]
                    peptide = parts[2]
                    score_el = float(parts[11])
                    rank_el = float(parts[12])
                    score_ba = float(parts[13])
                    rank_ba = float(parts[14])
                    ic50 = float(parts[15])
                    
                    results.append({
                        'pos': pos,
                        'allele': allele,
                        'peptide': peptide,
                        'Score_EL': score_el,
                        '%Rank_EL': rank_el,
                        'Score_BA': score_ba,
                        '%Rank_BA': rank_ba,
                        'ic50': ic50
                    })
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line: {line}")
                    print(f"Error details: {e}")
    
    return results

def predict_binding(peptide, allele, netmhcpan_path, verbose=False):
    """
    Predict binding affinity for a peptide-allele pair using netMHCpan.
    
    Args:
        peptide (str): The peptide sequence
        allele (str): The MHC allele
        netmhcpan_path (str): Path to the netMHCpan script
        verbose (bool): Whether to print verbose output
        
    Returns:
        dict: A dictionary containing the prediction results
    """
    # Create a temporary file for the peptide
    pepfile = tempfile.mktemp()+'.pep'
    with open(pepfile, 'w') as f:
        f.write(peptide+'\n')
    
    # Format the allele string (remove * if present)
    allele_str = allele.replace('*','')
    
    # Build the command
    cmd = f'{netmhcpan_path} -BA -f {pepfile} -inptype 1 -a {allele_str}'
    if verbose:
        print(f"Running command: {cmd}")
    
    try:
        # Run the command
        output = subprocess.check_output(cmd, shell=True, executable='/bin/bash')
        
        # Parse the output
        results = parse_netmhcpan_output(output)
        
        # Clean up temp file
        if os.path.exists(pepfile):
            os.remove(pepfile)
        
        if results:
            return results[0]  # Return the first result
        else:
            print(f"No prediction result for {peptide} with {allele}")
            return None
    except Exception as e:
        print(f"Error running prediction for {peptide} with {allele}: {e}")
        # Clean up temp file
        if os.path.exists(pepfile):
            os.remove(pepfile)
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict MHC binding affinity for peptides')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file with peptides and alleles')
    parser.add_argument('--output', '-o', help='Output CSV file for results')
    parser.add_argument('--netmhcpan', '-n', default=os.path.expanduser('~/Documents/capstone/netMHCpan-4.1/run_netMHCpan.sh'),
                        help='Path to netMHCpan script')
    parser.add_argument('--peptide-col', default='peptide', help='Column name for peptides in input file')
    parser.add_argument('--allele-col', default='allele', help='Column name for alleles in input file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print verbose output')
    args = parser.parse_args()
    
    # Check if netMHCpan exists
    if not os.path.exists(args.netmhcpan):
        print(f"Error: netMHCpan script not found at {args.netmhcpan}")
        sys.exit(1)
    
    # Make sure the script is executable
    os.chmod(args.netmhcpan, 0o755)
    
    # Set environment variables for netMHCpan
    os.environ['NETMHCPAN_PATH'] = args.netmhcpan
    os.environ['PATH'] = os.path.dirname(args.netmhcpan) + ":" + os.environ.get('PATH', '')
    
    # Verify netMHCpan is accessible
    try:
        result = subprocess.run([args.netmhcpan, "-h"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               check=False)
        if result.returncode != 0:
            print(f"Warning: netMHCpan test run failed with error: {result.stderr.decode('utf-8')}")
        else:
            print("netMHCpan is accessible")
    except Exception as e:
        print(f"Error testing netMHCpan: {e}")
        sys.exit(1)
    
    # Read input file
    try:
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} rows from {args.input}")
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)
    
    # Check if required columns exist
    if args.peptide_col not in df.columns:
        print(f"Error: Peptide column '{args.peptide_col}' not found in input file")
        sys.exit(1)
    if args.allele_col not in df.columns:
        print(f"Error: Allele column '{args.allele_col}' not found in input file")
        sys.exit(1)
    
    # Predict for each peptide-allele pair
    predictions = []
    total = len(df)
    print(f"Processing {total} epitopes...")
    
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            peptide = row[args.peptide_col]
            allele = row[args.allele_col]
            
            # Replace 'HLA class 1' with a default common allele
            if isinstance(allele, str) and allele.strip() == "HLA class 1":
                allele = "HLA-A*02:01"
                if args.verbose:
                    print(f"  Replacing 'HLA class 1' with {allele} for peptide {peptide}")

            # Skip if peptide is missing or too short
            if pd.isna(peptide) or peptide == '' or len(peptide) < 8:
                print(f"Skipping row {i+1}/{total} with missing or too short peptide: '{peptide}'")
                continue
                
            print(f"Processing {i+1}/{total}: peptide={peptide}, allele={allele}")
            
            result = predict_binding(peptide, allele, args.netmhcpan, args.verbose)
            
            if result:
                # Extract the score and rank
                score_ba = result.get('Score_BA')
                rank_ba = result.get('%Rank_BA')
                score_el = result.get('Score_EL')
                rank_el = result.get('%Rank_EL')
                ic50 = result.get('ic50')
                
                predictions.append({
                    'peptide': peptide,
                    'allele': allele,
                    'Score_BA': score_ba,
                    '%Rank_BA': rank_ba,
                    'Score_EL': score_el,
                    '%Rank_EL': rank_el,
                    'ic50': ic50
                })
                
                if args.verbose:
                    print(f"  Result: Score_BA={score_ba}, Rank_BA={rank_ba}, Score_EL={score_el}, Rank_EL={rank_el}, IC50={ic50}")
        except Exception as e:
            print(f"Error processing row {i+1}: {e}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(predictions)
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"mhc_predictions_{timestamp}.csv"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    results_df.to_csv(output_file, index=False)
    print(f"\nPrediction completed. Results saved to {output_file}")
    print(f"Total predictions: {len(results_df)}")
    
    # Display summary
    print("\nPrediction Summary:")
    print(results_df.head(10) if len(results_df) > 10 else results_df)
    if len(results_df) > 10:
        print(f"... and {len(results_df) - 10} more rows")

if __name__ == "__main__":
    main() 