# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning research project for **Cancer T-Cell Epitope Classification**. The goal is to develop computational models that can identify T-cell epitopes in cancer antigens for personalized immunotherapy. The project uses data from the Immune Epitope Database (IEDB) and implements both traditional ML (Random Forest) and deep learning (CNN) approaches.

## Key Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Alternative using UV (recommended)
uv pip install -r requirements.txt
```

### Data Processing
```bash
# Predict MHC binding affinities (requires netMHCpan installation)
python binding_affinity_prediction.py -i data/epitopes.csv -o data/predictions.csv

# Visualize prediction results
python visualize_predictions.py -i data/predictions.csv -o figures/
```

### Documentation
```bash
# Render analysis report
quarto render doc/epitope_classification.qmd

# Convert to PDF
quarto render doc/epitope_classification.qmd --to pdf
```

### Jupyter Analysis
```bash
# Launch main analysis notebook
jupyter notebook capstone.ipynb

# Or use specific cells for model training
```

## Architecture

### Data Pipeline
1. **Input**: Epitope sequences from IEDB with MHC allele information
2. **Feature Engineering**: BioPython-based physicochemical properties (hydrophobicity, molecular weight, aromaticity, isoelectric point, instability, charge)
3. **Negative Sampling**: Non-epitope peptides extracted from full protein sequences
4. **External Integration**: netMHCpan v4.1 for MHC binding affinity prediction

### Model Approaches
- **Random Forest**: Uses engineered features + binding affinity scores, achieves 91% accuracy but heavily dependent on netMHCpan predictions
- **CNN**: Direct sequence learning via one-hot encoding, achieves 82% accuracy with 65% epitope recall, learns patterns independently

### Key Dependencies
- **netMHCpan v4.1**: External binding affinity prediction tool (must be installed separately)
- **BioPython**: Protein sequence analysis and feature calculation
- **TensorFlow**: Deep learning framework for CNN implementation
- **Quarto**: Documentation rendering system

### Data Structure
- **Epitopes**: ~5,300 experimentally validated human cancer T-cell epitopes
- **Negatives**: Generated from full protein sequences, filtered to remove known epitopes
- **Features**: 7 physicochemical properties + binding affinity score
- **Focus**: 9-mer peptides (optimal for MHC Class I presentation)

## Important Notes

### netMHCpan Integration
- Requires separate installation of netMHCpan v4.1
- Default path: `~/Documents/capstone/netMHCpan-4.1/run_netMHCpan.sh`
- Used for critical binding affinity predictions that significantly impact model performance

### Model Performance Context
- MHC binding affinity is the dominant predictor (Random Forest drops from 75% to 18% epitope recall without it)
- CNN approach shows promise for discovering sequence patterns beyond binding affinity
- Class imbalance (4:1 negative:positive ratio) requires careful evaluation metrics

### Data Processing Considerations
- Negative sample generation uses sliding window approach on full protein sequences
- Risk of false negatives (untested epitopes labeled as negatives)
- MHC allele distribution heavily skewed toward HLA-A*02:01

### Development Workflow
1. Data preprocessing and feature engineering typically done in Jupyter notebooks
2. Model training and evaluation in both notebooks and standalone scripts
3. Results analysis and visualization through Quarto documents
4. External binding affinity prediction requires command-line netMHCpan calls