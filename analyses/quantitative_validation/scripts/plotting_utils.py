# scripts/plotting_utils.py

"""
Plotting utilities for validation figures
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


def ensure_output_dir(filepath):
    """Create parent directory if it doesn't exist"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)


def plot_dice_iou_comparison(df_metrics, output_path=None, title_prefix="", colors=None):
    """
    Create side-by-side Dice and IoU bar plots

    Parameters
    ----------
    df_metrics : pd.DataFrame
        Must have columns: FOV, Compartment, Dice, IoU
    output_path : str, optional
        Path to save figure (PDF)
    title_prefix : str, optional
        Prefix for plot titles (e.g., "CosMx: " or "Xenium: ")
    colors : dict, optional
        Custom colors for compartments, e.g., {'Tumor': 'red', 'Stroma': 'blue', 'Empty': 'gray'}
    """
    # Set color palette if provided
    if colors:
        palette = [colors.get(comp, 'gray') for comp in df_metrics['Compartment'].unique()]
    else:
        palette = None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Taller to fit legend

    # Dice
    sns.barplot(data=df_metrics, x='FOV', y='Dice', hue='Compartment', palette=palette, ax=axes[0])
    axes[0].set_title(f'{title_prefix}Dice Coefficient', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Dice Score', fontsize=11)
    axes[0].set_xlabel('')
    axes[0].set_ylim([0, 1.15])  # Extended to fit legend
    axes[0].axhline(y=0.7, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[0].legend(title='', loc='upper left', frameon=False, ncol=3)  # Horizontal legend at top
    axes[0].grid(axis='y', alpha=0.2, linestyle=':')

    # IoU
    sns.barplot(data=df_metrics, x='FOV', y='IoU', hue='Compartment', palette=palette, ax=axes[1])
    axes[1].set_title(f'{title_prefix}IoU (Jaccard Index)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('IoU Score', fontsize=11)
    axes[1].set_xlabel('')
    axes[1].set_ylim([0, 1.15])  # Extended to fit legend
    axes[1].axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[1].legend(title='', loc='upper left', frameon=False, ncol=3)  # Horizontal legend at top
    axes[1].grid(axis='y', alpha=0.2, linestyle=':')

    plt.tight_layout()
    if output_path:
        ensure_output_dir(output_path)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_precision_recall_per_fov(df_metrics, output_path=None, title_prefix="", colors=None):
    """
    Create precision vs recall scatter plot with FOV labels per compartment

    Parameters
    ----------
    df_metrics : pd.DataFrame
        Must have columns: FOV, Compartment, Precision, Recall
    output_path : str, optional
        Path to save figure (PDF)
    title_prefix : str, optional
        Prefix for plot title
    colors : dict, optional
        Custom colors for compartments
    """
    # Default colors if not provided
    if colors is None:
        colors = {'Tumor': '#e74c3c', 'Stroma': '#3498db', 'Empty': '#95a5a6'}

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for compartment in df_metrics['Compartment'].unique():
        comp_data = df_metrics[df_metrics['Compartment'] == compartment]
        ax.scatter(comp_data['Recall'], comp_data['Precision'],
                   s=120, alpha=0.8, label=compartment,
                   color=colors.get(compartment, 'gray'),
                   edgecolors='black', linewidth=1)

        # Add FOV labels
        for idx, row in comp_data.iterrows():
            ax.annotate(row['FOV'].replace('FOV', '').replace('ROI', ''),
                        (row['Recall'], row['Precision']),
                        fontsize=8, ha='center', va='center',
                        color='white', fontweight='bold')

    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title(f'{title_prefix}Precision vs Recall', fontsize=13, fontweight='bold')
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax.legend(title='', fontsize=10, frameon=False, loc='lower right')
    ax.grid(alpha=0.2, linestyle=':')

    plt.tight_layout()
    if output_path:
        ensure_output_dir(output_path)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_metrics_summary_barplot(df_metrics, output_path=None, title_prefix="", colors=None):
    """
    Create bar plot showing mean ± std for all metrics across compartments

    Parameters
    ----------
    df_metrics : pd.DataFrame
        Must have columns: Compartment, Dice, IoU, Precision, Recall, F1
    output_path : str, optional
        Path to save figure (PDF)
    title_prefix : str, optional
        Prefix for plot title
    colors : dict, optional
        Custom colors for compartments
    """
    # Default colors if not provided
    if colors is None:
        colors = {'Tumor': '#e74c3c', 'Stroma': '#3498db', 'Empty': '#95a5a6'}

    # Calculate mean and std per compartment
    metrics = ['Dice', 'IoU', 'Precision', 'Recall', 'F1']
    summary = df_metrics.groupby('Compartment')[metrics].agg(['mean', 'std']).reset_index()

    # Reshape for plotting
    plot_data = []
    for compartment in summary['Compartment']:
        for metric in metrics:
            mean_val = summary[summary['Compartment'] == compartment][metric]['mean'].values[0]
            std_val = summary[summary['Compartment'] == compartment][metric]['std'].values[0]
            plot_data.append({
                'Compartment': compartment,
                'Metric': metric,
                'Mean': mean_val,
                'Std': std_val
            })

    plot_df = pd.DataFrame(plot_data)

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))  # Taller

    x = np.arange(len(metrics))
    width = 0.25
    compartments = plot_df['Compartment'].unique()

    for i, compartment in enumerate(compartments):
        comp_data = plot_df[plot_df['Compartment'] == compartment]
        offset = (i - len(compartments) / 2 + 0.5) * width

        ax.bar(x + offset, comp_data['Mean'], width,
               yerr=comp_data['Std'],
               label=compartment,
               color=colors.get(compartment, 'gray'),
               capsize=5, alpha=0.8,
               error_kw={'linewidth': 1.5})

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_title(f'{title_prefix}Summary Metrics (Mean ± SD)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim([0, 1.15])  # Extended to fit legend
    ax.axhline(y=0.7, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend(title='', frameon=False, loc='upper left', ncol=3)  # Horizontal at top
    ax.grid(axis='y', alpha=0.2, linestyle=':')

    plt.tight_layout()
    if output_path:
        ensure_output_dir(output_path)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_metrics_heatmap(df_metrics, metric='Dice', output_path=None, title_prefix=""):
    """
    Create heatmap of metrics across FOVs and compartments

    Parameters
    ----------
    df_metrics : pd.DataFrame
        Must have columns: FOV, Compartment, and the specified metric
    metric : str
        Which metric to plot ('Dice', 'IoU', etc.)
    output_path : str, optional
        Path to save figure (PDF)
    title_prefix : str, optional
        Prefix for plot title (e.g., "CosMx: " or "Xenium: ")
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))

    heatmap_data = df_metrics.pivot_table(values=metric, index='Compartment', columns='FOV')

    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0.5, vmax=1.0, center=0.75,
                cbar_kws={'label': f'{metric} Coefficient'},
                linewidths=0.5, linecolor='white', ax=ax)
    ax.set_title(f'{title_prefix}{metric} Coefficient', fontsize=12, fontweight='bold')
    ax.set_xlabel('Field of View', fontsize=11, fontweight='bold')
    ax.set_ylabel('')

    plt.tight_layout()
    if output_path:
        ensure_output_dir(output_path)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_metrics_summary_table(df_metrics, output_csv=None):
    """
    Create summary statistics table (mean ± std)

    Parameters
    ----------
    df_metrics : pd.DataFrame
        Must have columns: Compartment, Dice, IoU, Precision, Recall
    output_csv : str, optional
        Path to save CSV

    Returns
    -------
    pd.DataFrame : Formatted summary table
    """
    summary_stats = df_metrics.groupby('Compartment')[['Dice', 'IoU', 'Precision', 'Recall']].agg(['mean', 'std'])

    summary_formatted = pd.DataFrame()
    for metric in ['Dice', 'IoU', 'Precision', 'Recall']:
        summary_formatted[metric] = summary_stats[metric].apply(
            lambda x: f"{x['mean']:.3f} ± {x['std']:.3f}", axis=1
        )

    if output_csv:
        ensure_output_dir(output_csv)
        summary_formatted.to_csv(output_csv)

    return summary_formatted