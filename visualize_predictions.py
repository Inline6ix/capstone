import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize MHC binding affinity predictions')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file with prediction results')
    parser.add_argument('--output-dir', '-o', default='figures', help='Output directory for figures')
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Read input file
    try:
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} rows from {args.input}")
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    # Check if required columns exist
    required_columns = ['peptide', 'allele', '%Rank_BA', 'Score_BA']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Set up the style
    sns.set(style="whitegrid")
    
    # 1. Binding Affinity Rank Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['%Rank_BA'], bins=20, kde=True)
    plt.axvline(x=0.5, color='r', linestyle='--', label='Strong Binder Threshold (0.5)')
    plt.axvline(x=2.0, color='orange', linestyle='--', label='Weak Binder Threshold (2.0)')
    plt.xlabel('Binding Affinity Rank (%Rank_BA)')
    plt.ylabel('Count')
    plt.title('Distribution of Binding Affinity Ranks')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'rank_distribution.png'))
    plt.close()
    
    # 2. Binding Affinity Score vs Rank
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Score_BA', y='%Rank_BA', data=df, hue='allele', s=100)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Strong Binder Threshold (0.5)')
    plt.axhline(y=2.0, color='orange', linestyle='--', label='Weak Binder Threshold (2.0)')
    plt.xlabel('Binding Affinity Score (Score_BA)')
    plt.ylabel('Binding Affinity Rank (%Rank_BA)')
    plt.title('Binding Affinity Score vs Rank')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'score_vs_rank.png'))
    plt.close()
    
    # 3. IC50 Distribution (if available)
    if 'ic50' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['ic50'], bins=20, kde=True)
        plt.axvline(x=50, color='r', linestyle='--', label='Strong Binder Threshold (50 nM)')
        plt.axvline(x=500, color='orange', linestyle='--', label='Weak Binder Threshold (500 nM)')
        plt.xlabel('IC50 (nM)')
        plt.ylabel('Count')
        plt.title('Distribution of IC50 Values')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'ic50_distribution.png'))
        plt.close()
        
        # 4. IC50 vs Rank
        plt.figure(figsize=(20, 15))
        sns.scatterplot(x='ic50', y='%Rank_BA', data=df, hue='allele', s=100)
        plt.axhline(y=0.5, color='r', linestyle='--', label='Strong Binder Threshold (0.5)')
        plt.axhline(y=2.0, color='orange', linestyle='--', label='Weak Binder Threshold (2.0)')
        plt.axvline(x=50, color='r', linestyle=':')
        plt.axvline(x=500, color='orange', linestyle=':')
        plt.xlabel('IC50 (nM)')
        plt.ylabel('Binding Affinity Rank (%Rank_BA)')
        plt.title('IC50 vs Binding Affinity Rank')
        plt.xscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'ic50_vs_rank.png'))
        plt.close()
    
    # 5. Top Binders Bar Chart
    # Sort by rank and take top 10
    top_binders = df.sort_values('%Rank_BA').head(10)
    plt.figure(figsize=(12, 6))
    bars = sns.barplot(x='peptide', y='%Rank_BA', data=top_binders)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Strong Binder Threshold (0.5)')
    plt.axhline(y=2.0, color='orange', linestyle='--', label='Weak Binder Threshold (2.0)')
    plt.xlabel('Peptide')
    plt.ylabel('Binding Affinity Rank (%Rank_BA)')
    plt.title('Top 10 Binding Peptides')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars.patches):
        bars.text(
            bar.get_x() + bar.get_width()/2.,
            bar.get_height() + 0.05,
            f"{bar.get_height():.3f}",
            ha='center',
            fontsize=9
        )
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'top_binders.png'))
    plt.close()
    
    # 6. Binding Category Pie Chart
    # Categorize peptides based on binding affinity rank
    df['binding_category'] = pd.cut(
        df['%Rank_BA'],
        bins=[0, 0.5, 2.0, float('inf')],
        labels=['Strong Binder', 'Weak Binder', 'Non-Binder']
    )
    
    # Count peptides in each category
    category_counts = df['binding_category'].value_counts()
    
    plt.figure(figsize=(8, 8))
    plt.pie(
        category_counts,
        labels=category_counts.index,
        autopct='%1.1f%%',
        colors=['#ff9999', '#ffc000', '#8cc3ff'],
        startangle=90
    )
    plt.axis('equal')
    plt.title('Distribution of Binding Categories')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'binding_categories.png'))
    plt.close()
    
    print(f"Visualizations saved to {args.output_dir}/")
    print(f"Summary of binding categories:")
    print(category_counts)

if __name__ == "__main__":
    main() 