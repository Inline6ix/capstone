"""
Visualization utilities for epitope classification results.

This module provides functions to create various plots and visualizations
for analyzing MHC binding affinity predictions and model performance.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Set default style
plt.style.use('default')
sns.set_palette("husl")


class BindingAffinityVisualizer:
    """
    Visualizer for MHC binding affinity prediction results.
    
    This class provides methods to create various plots for analyzing
    binding affinity predictions from netMHCpan or similar tools.
    """
    
    def __init__(self, style: str = "whitegrid", figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize the visualizer.
        
        Args:
            style: Seaborn style to use for plots
            figsize: Default figure size for plots
        """
        self.style = style
        self.figsize = figsize
        sns.set_style(style)
        
        # Define binding thresholds
        self.strong_binder_threshold = 0.5  # %Rank_BA
        self.weak_binder_threshold = 2.0    # %Rank_BA
        self.ic50_strong_threshold = 50     # nM
        self.ic50_weak_threshold = 500      # nM
    
    def plot_rank_distribution(
        self, 
        df: pd.DataFrame, 
        output_path: Optional[Union[str, Path]] = None,
        title: str = "Distribution of Binding Affinity Ranks"
    ) -> plt.Figure:
        """
        Plot distribution of binding affinity ranks.
        
        Args:
            df: DataFrame with '%Rank_BA' column
            output_path: Path to save the plot
            title: Plot title
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create histogram with KDE
        sns.histplot(data=df, x='%Rank_BA', bins=20, kde=True, ax=ax)
        
        # Add threshold lines
        ax.axvline(
            x=self.strong_binder_threshold, 
            color='r', 
            linestyle='--', 
            label=f'Strong Binder Threshold ({self.strong_binder_threshold})'
        )
        ax.axvline(
            x=self.weak_binder_threshold, 
            color='orange', 
            linestyle='--', 
            label=f'Weak Binder Threshold ({self.weak_binder_threshold})'
        )
        
        ax.set_xlabel('Binding Affinity Rank (%Rank_BA)')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved rank distribution plot to {output_path}")
        
        return fig
    
    def plot_score_vs_rank(
        self, 
        df: pd.DataFrame, 
        output_path: Optional[Union[str, Path]] = None,
        title: str = "Binding Affinity Score vs Rank"
    ) -> plt.Figure:
        """
        Plot binding affinity score vs rank with allele coloring.
        
        Args:
            df: DataFrame with 'Score_BA', '%Rank_BA', and 'allele' columns
            output_path: Path to save the plot
            title: Plot title
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create scatter plot
        sns.scatterplot(
            data=df, 
            x='Score_BA', 
            y='%Rank_BA', 
            hue='allele', 
            s=100, 
            alpha=0.7,
            ax=ax
        )
        
        # Add threshold lines
        ax.axhline(
            y=self.strong_binder_threshold, 
            color='r', 
            linestyle='--', 
            label=f'Strong Binder Threshold ({self.strong_binder_threshold})'
        )
        ax.axhline(
            y=self.weak_binder_threshold, 
            color='orange', 
            linestyle='--', 
            label=f'Weak Binder Threshold ({self.weak_binder_threshold})'
        )
        
        ax.set_xlabel('Binding Affinity Score (Score_BA)')
        ax.set_ylabel('Binding Affinity Rank (%Rank_BA)')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved score vs rank plot to {output_path}")
        
        return fig
    
    def plot_ic50_distribution(
        self, 
        df: pd.DataFrame, 
        output_path: Optional[Union[str, Path]] = None,
        title: str = "Distribution of IC50 Values"
    ) -> Optional[plt.Figure]:
        """
        Plot IC50 value distribution (if IC50 data is available).
        
        Args:
            df: DataFrame with 'ic50' column
            output_path: Path to save the plot
            title: Plot title
            
        Returns:
            Matplotlib figure object or None if IC50 data not available
        """
        if 'ic50' not in df.columns:
            logger.warning("IC50 column not found in DataFrame")
            return None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create histogram with KDE
        sns.histplot(data=df, x='ic50', bins=20, kde=True, ax=ax)
        
        # Add threshold lines
        ax.axvline(
            x=self.ic50_strong_threshold, 
            color='r', 
            linestyle='--', 
            label=f'Strong Binder Threshold ({self.ic50_strong_threshold} nM)'
        )
        ax.axvline(
            x=self.ic50_weak_threshold, 
            color='orange', 
            linestyle='--', 
            label=f'Weak Binder Threshold ({self.ic50_weak_threshold} nM)'
        )
        
        ax.set_xlabel('IC50 (nM)')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved IC50 distribution plot to {output_path}")
        
        return fig
    
    def plot_top_binders(
        self, 
        df: pd.DataFrame, 
        n_top: int = 10,
        output_path: Optional[Union[str, Path]] = None,
        title: str = "Top Binding Peptides"
    ) -> plt.Figure:
        """
        Plot top binding peptides as a bar chart.
        
        Args:
            df: DataFrame with 'peptide' and '%Rank_BA' columns
            n_top: Number of top binders to show
            output_path: Path to save the plot
            title: Plot title
            
        Returns:
            Matplotlib figure object
        """
        # Get top binders
        top_binders = df.nsmallest(n_top, '%Rank_BA')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bar plot
        bars = sns.barplot(
            data=top_binders, 
            x='peptide', 
            y='%Rank_BA',
            ax=ax
        )
        
        # Add threshold lines
        ax.axhline(
            y=self.strong_binder_threshold, 
            color='r', 
            linestyle='--', 
            label=f'Strong Binder Threshold ({self.strong_binder_threshold})'
        )
        ax.axhline(
            y=self.weak_binder_threshold, 
            color='orange', 
            linestyle='--', 
            label=f'Weak Binder Threshold ({self.weak_binder_threshold})'
        )
        
        # Add value labels on bars
        for i, bar in enumerate(bars.patches):
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 0.05,
                f"{bar.get_height():.3f}",
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        ax.set_xlabel('Peptide')
        ax.set_ylabel('Binding Affinity Rank (%Rank_BA)')
        ax.set_title(f'{title} (Top {n_top})')
        ax.legend()
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved top binders plot to {output_path}")
        
        return fig
    
    def plot_binding_categories(
        self, 
        df: pd.DataFrame, 
        output_path: Optional[Union[str, Path]] = None,
        title: str = "Distribution of Binding Categories"
    ) -> plt.Figure:
        """
        Plot pie chart of binding categories.
        
        Args:
            df: DataFrame with '%Rank_BA' column
            output_path: Path to save the plot
            title: Plot title
            
        Returns:
            Matplotlib figure object
        """
        # Categorize peptides
        df_copy = df.copy()
        df_copy['binding_category'] = pd.cut(
            df_copy['%Rank_BA'],
            bins=[0, self.strong_binder_threshold, self.weak_binder_threshold, float('inf')],
            labels=['Strong Binder', 'Weak Binder', 'Non-Binder']
        )
        
        category_counts = df_copy['binding_category'].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create pie chart
        colors = ['#ff9999', '#ffc000', '#8cc3ff']
        wedges, texts, autotexts = ax.pie(
            category_counts.values,
            labels=category_counts.index,
            autopct='%1.1f%%',
            colors=colors[:len(category_counts)],
            startangle=90
        )
        
        # Improve text appearance
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved binding categories plot to {output_path}")
        
        return fig
    
    def create_binding_affinity_report(
        self, 
        df: pd.DataFrame, 
        output_dir: Union[str, Path],
        prefix: str = "binding_affinity"
    ) -> Dict[str, Path]:
        """
        Create a complete set of binding affinity visualizations.
        
        Args:
            df: DataFrame with binding affinity prediction results
            output_dir: Directory to save all plots
            prefix: Prefix for output filenames
            
        Returns:
            Dictionary mapping plot types to output paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_paths = {}
        
        # Required columns check
        required_columns = ['%Rank_BA', 'Score_BA', 'peptide']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # 1. Rank distribution
        path = output_dir / f"{prefix}_rank_distribution.png"
        self.plot_rank_distribution(df, path)
        plot_paths['rank_distribution'] = path
        
        # 2. Score vs rank
        if 'allele' in df.columns:
            path = output_dir / f"{prefix}_score_vs_rank.png"
            self.plot_score_vs_rank(df, path)
            plot_paths['score_vs_rank'] = path
        
        # 3. IC50 distribution (if available)
        if 'ic50' in df.columns:
            path = output_dir / f"{prefix}_ic50_distribution.png"
            self.plot_ic50_distribution(df, path)
            if path:  # Only add if plot was created
                plot_paths['ic50_distribution'] = path
        
        # 4. Top binders
        path = output_dir / f"{prefix}_top_binders.png"
        self.plot_top_binders(df, output_path=path)
        plot_paths['top_binders'] = path
        
        # 5. Binding categories
        path = output_dir / f"{prefix}_binding_categories.png"
        self.plot_binding_categories(df, path)
        plot_paths['binding_categories'] = path
        
        logger.info(f"Created {len(plot_paths)} plots in {output_dir}")
        return plot_paths


def plot_model_performance(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_pred_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, plt.Figure]:
    """
    Create comprehensive model performance plots.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities (for ROC/PR curves)
        class_names: Names of the classes
        output_dir: Directory to save plots
        
    Returns:
        Dictionary of plot names to figure objects
    """
    from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
    
    figures = {}
    
    if class_names is None:
        class_names = ['Negative', 'Positive']
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    figures['confusion_matrix'] = fig
    
    if output_dir:
        fig.savefig(Path(output_dir) / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    
    # 2. ROC Curve (if probabilities provided)
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        figures['roc_curve'] = fig
        
        if output_dir:
            fig.savefig(Path(output_dir) / 'roc_curve.png', dpi=300, bbox_inches='tight')
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = auc(recall, precision)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
        ax.fill_between(recall, precision, alpha=0.2, color='blue')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        figures['precision_recall_curve'] = fig
        
        if output_dir:
            fig.savefig(Path(output_dir) / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
    
    logger.info(f"Created {len(figures)} performance plots")
    return figures


def visualize_predictions_from_file(
    input_file: Union[str, Path],
    output_dir: Union[str, Path] = "figures",
    prefix: str = "predictions"
) -> Dict[str, Path]:
    """
    Convenience function to create visualizations from a prediction file.
    
    Args:
        input_file: Path to CSV file with prediction results
        output_dir: Directory to save plots
        prefix: Prefix for output filenames
        
    Returns:
        Dictionary mapping plot types to output paths
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If required columns are missing
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load data
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows from {input_path}")
    
    # Create visualizations
    visualizer = BindingAffinityVisualizer()
    plot_paths = visualizer.create_binding_affinity_report(df, output_dir, prefix)
    
    # Print summary
    if '%Rank_BA' in df.columns:
        df_copy = df.copy()
        df_copy['binding_category'] = pd.cut(
            df_copy['%Rank_BA'],
            bins=[0, 0.5, 2.0, float('inf')],
            labels=['Strong Binder', 'Weak Binder', 'Non-Binder']
        )
        category_counts = df_copy['binding_category'].value_counts()
        
        logger.info("Summary of binding categories:")
        for category, count in category_counts.items():
            logger.info(f"  {category}: {count} ({count/len(df)*100:.1f}%)")
    
    return plot_paths