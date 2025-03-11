# MHC Binding Affinity Prediction

This repository contains scripts for predicting MHC binding affinity using netMHCpan-4.1.

## Prerequisites

- Python 3.6+
- pandas
- matplotlib
- seaborn
- netMHCpan-4.1 installed and configured

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required Python packages:
   ```
   pip install pandas matplotlib seaborn
   ```

3. Make sure netMHCpan-4.1 is installed and the path to the `run_netMHCpan.sh` script is correctly set.

## Usage

### Prediction Script

```
python binding_affinity_prediction.py --input your_peptides.csv --output results.csv
```

#### Command Line Arguments

- `--input`, `-i`: Input CSV file with peptides and alleles (required)
- `--output`, `-o`: Output CSV file for results (optional, default: mhc_predictions_TIMESTAMP.csv)
- `--netmhcpan`, `-n`: Path to netMHCpan script (default: ~/Documents/capstone/netMHCpan-4.1/run_netMHCpan.sh)
- `--peptide-col`: Column name for peptides in input file (default: peptide)
- `--allele-col`: Column name for alleles in input file (default: allele)
- `--verbose`, `-v`: Print verbose output

### Visualization Script

```
python visualize_predictions.py --input results.csv --output-dir figures
```

#### Command Line Arguments

- `--input`, `-i`: Input CSV file with prediction results (required)
- `--output-dir`, `-o`: Output directory for figures (default: figures)

#### Generated Visualizations

The visualization script generates the following plots:

1. **Rank Distribution**: Histogram of binding affinity ranks
2. **Score vs Rank**: Scatter plot of binding affinity scores vs ranks
3. **IC50 Distribution**: Histogram of IC50 values
4. **IC50 vs Rank**: Scatter plot of IC50 values vs ranks
5. **Top Binders**: Bar chart of the top 10 binding peptides
6. **Binding Categories**: Pie chart showing the distribution of binding categories

### Input File Format

The input file should be a CSV file with at least two columns:
- A column for peptide sequences (default column name: "peptide")
- A column for MHC alleles (default column name: "allele")

Example:
```
peptide,allele
GILGFVFTL,HLA-A*02:01
NLVPMVATV,HLA-A*02:01
GLCTLVAML,HLA-A*02:01
```

### Output File Format

The output file is a CSV file with the following columns:
- peptide: The peptide sequence
- allele: The MHC allele
- Score_BA: Binding affinity score
- %Rank_BA: Binding affinity rank
- Score_EL: Eluted ligand score
- %Rank_EL: Eluted ligand rank
- ic50: IC50 value in nM

## Example

```
# Run prediction
python binding_affinity_prediction.py --input test_peptides.csv --output data/test_predictions.csv --verbose

# Visualize results
python visualize_predictions.py --input data/test_predictions.csv --output-dir figures
```

## Interpreting Results

- **Score_BA**: Binding affinity score (higher is better)
- **%Rank_BA**: Binding affinity rank (lower is better)
  - < 0.5: Strong binder
  - < 2.0: Weak binder
- **Score_EL**: Eluted ligand score (higher is better)
- **%Rank_EL**: Eluted ligand rank (lower is better)
  - < 0.5: Strong binder
  - < 2.0: Weak binder
- **ic50**: IC50 value in nM (lower is better)
  - < 50 nM: Strong binder
  - < 500 nM: Weak binder

## Troubleshooting

If you encounter issues with the script, try the following:

1. Make sure netMHCpan-4.1 is correctly installed and the path to `run_netMHCpan.sh` is correct.
2. Check that the input file has the correct format and column names.
3. Run with the `--verbose` flag to see more detailed output.
4. Check that the peptides are valid (8-11 amino acids for MHC-I).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
