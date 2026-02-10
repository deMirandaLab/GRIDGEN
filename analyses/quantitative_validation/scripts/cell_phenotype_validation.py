"""
Cell-level validation functions for GRIDGENE mask validation.
Compares cell phenotypes (from Baysor segmentation) with GRIDGENE compartment predictions.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# COMPARTMENT MAPPING
# ============================================================================

COMPARTMENT_MAPPING = {
    'Epithelial cells': 'tumor',
    'Fibroblasts': 'stroma',
    'Lymphocytes': 'immune',
    'Myeloid cells': 'immune',
    'Plasma cells': 'immune',
    'Endothelial cells': 'stroma',
    'Other cells': 'other'
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_baysor_polygons(baysor_folder):
    """
    Load cell polygons from Baysor segmentation_polygons.json

    Args:
        baysor_folder: Path to folder containing segmentation_polygons.json

    Returns:
        dict: {cell_id: Polygon}
    """
    json_path = Path(baysor_folder) / "segmentation_polygons.json"

    if not json_path.exists():
        raise FileNotFoundError(f"Polygons not found: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    polygons = {}
    skipped = 0

    for geom in data['geometries']:
        cell_id = geom['cell']
        coords = geom['coordinates'][0]

        if len(coords) < 4:
            skipped += 1
            continue

        try:
            polygons[cell_id] = Polygon(coords)
        except Exception:
            skipped += 1

    if skipped > 0:
        print(f"  ({skipped} polygons skipped)", end=" ")

    return polygons


def map_phenotypes_to_polygons(polygons, phenotype_df, fov_name, cell_id_col='CellID', fov_col='orig.ident'):
    """
    Map phenotype labels to cell polygons

    Args:
        polygons: {cell_id: Polygon}
        phenotype_df: DataFrame with cell phenotype data
                     Must have 'compartment' column (from COMPARTMENT_MAPPING)
        fov_name: str, filter phenotype_df to this FOV
        cell_id_col: column name for cell IDs (default 'CellID')
        fov_col: column name for FOV identifier (default 'orig.ident')

    Returns:
        DataFrame: cell_id | polygon | major_cluster | compartment
    """
    fov_cells = phenotype_df[phenotype_df[fov_col] == fov_name].copy()

    cell_data = []
    for cell_id, polygon in polygons.items():
        cell_info = fov_cells[fov_cells[cell_id_col] == cell_id]

        if len(cell_info) == 0:
            continue

        cell_data.append({
            'cell_id': cell_id,
            'polygon': polygon,
            'major_cluster': cell_info['major_cluster'].iloc[0],
            'compartment': cell_info['compartment'].iloc[0]
        })

    return pd.DataFrame(cell_data)


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_cell_phenotype_distribution(cell_phenotypes_df, gridgene_masks):
    """
    Analyze overlap between cell phenotypes and GRIDGENE compartments.
    Creates masks from phenotyped cells and calculates overlap with GRIDGENE masks.

    Args:
        cell_phenotypes_df: DataFrame with 'polygon' and 'compartment' columns
        gridgene_masks: {'tumor': mask, 'stroma': mask}

    Returns:
        dict: Overlap statistics (pixel counts and percentages)
    """
    image_shape = gridgene_masks['tumor'].shape

    # Create masks for each cell type
    cell_masks = {
        'tumor': np.zeros(image_shape, dtype=bool),
        'stroma': np.zeros(image_shape, dtype=bool),
        'immune': np.zeros(image_shape, dtype=bool)
    }

    for _, row in cell_phenotypes_df.iterrows():
        compartment = row['compartment']
        polygon = row['polygon']

        if pd.isna(compartment) or compartment not in cell_masks:
            continue

        coords = np.array(polygon.exterior.coords, dtype=np.int32)
        temp_mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.fillPoly(temp_mask, [coords], 1)
        cell_masks[compartment] = cell_masks[compartment] | temp_mask.astype(bool)

    # Calculate overlaps
    gg_tumor = gridgene_masks['tumor'].astype(bool)
    gg_stroma = gridgene_masks['stroma'].astype(bool)

    analysis = {}
    for cell_type in ['tumor', 'stroma', 'immune']:
        # Map tumor→epithelial for clarity
        label = 'epithelial' if cell_type == 'tumor' else cell_type

        total = cell_masks[cell_type].sum()
        in_gg_tumor = (cell_masks[cell_type] & gg_tumor).sum()
        in_gg_stroma = (cell_masks[cell_type] & gg_stroma).sum()

        analysis[f'{label}_total'] = total
        analysis[f'{label}_in_gg_tumor'] = in_gg_tumor
        analysis[f'{label}_in_gg_stroma'] = in_gg_stroma

        if total > 0:
            analysis[f'{label}_in_tumor_pct'] = (in_gg_tumor / total) * 100
            analysis[f'{label}_in_stroma_pct'] = (in_gg_stroma / total) * 100
        else:
            analysis[f'{label}_in_tumor_pct'] = 0
            analysis[f'{label}_in_stroma_pct'] = 0

    return analysis


def create_distribution_dataframe(analysis_results):
    """
    Convert analysis results to long-format DataFrame for plotting

    Args:
        analysis_results: {fov_name: analysis_dict}

    Returns:
        DataFrame: FOV | Cell Type | In GG Tumor | In GG Stroma
    """
    records = []
    for fov_name, analysis in analysis_results.items():
        for cell_type in ['Epithelial', 'Stroma', 'Immune']:
            key = cell_type.lower()
            records.append({
                'FOV': fov_name,
                'Cell Type': cell_type,
                'In GG Tumor': analysis[f'{key}_in_tumor_pct'],
                'In GG Stroma': analysis[f'{key}_in_stroma_pct']
            })

    return pd.DataFrame(records)


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_cell_phenotypes(cell_phenotypes_df, image_shape, title="Cell Phenotypes"):
    """
    Draw polygons colored by compartment (simple visualization)

    Args:
        cell_phenotypes_df: DataFrame with 'polygon' and 'compartment' columns
        image_shape: (height, width)
        title: plot title
    """
    # Create RGB image (white background)
    image = np.ones((*image_shape, 3), dtype=np.uint8) * 255

    # Color mapping
    colors = {
        'tumor': (255, 0, 0),  # Red
        'stroma': (0, 0, 255),  # Blue
        'immune': (0, 255, 0),  # Green
        'other': (128, 128, 128)  # Gray
    }

    # Draw each polygon
    for _, row in cell_phenotypes_df.iterrows():
        compartment = row['compartment']
        polygon = row['polygon']

        if pd.isna(compartment):
            continue

        color = colors.get(compartment, (128, 128, 128))
        coords = np.array(polygon.exterior.coords, dtype=np.int32)

        # Fill polygon
        cv2.fillPoly(image, [coords], color)
        # Draw outline in black
        cv2.polylines(image, [coords], isClosed=True, color=(0, 0, 0), thickness=1)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.set_title(f"{title}\n(Red=Tumor, Blue=Stroma, Green=Immune)",
                 fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    # Print stats
    print(f"\nCell counts by compartment:")
    print(cell_phenotypes_df['compartment'].value_counts())
    print(f"Total cells: {len(cell_phenotypes_df)}")


def plot_distribution_boxplot(df_dist, save_path=None):
    """
    A) Boxplot showing distribution of cell types across GRIDGENE compartments

    Args:
        df_dist: DataFrame from create_distribution_dataframe()
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    colors = {'Epithelial': '#e74c3c', 'Stroma': '#3498db', 'Immune': '#2ecc71'}

    for idx, cell_type in enumerate(['Epithelial', 'Stroma', 'Immune']):
        data = df_dist[df_dist['Cell Type'] == cell_type]

        # Prepare data for boxplot
        boxplot_data = [data['In GG Tumor'], data['In GG Stroma']]

        bp = axes[idx].boxplot(boxplot_data,
                               labels=['GRIDGENE\nTumor', 'GRIDGENE\nStroma'],
                               patch_artist=True,
                               widths=0.6,
                               showfliers=False)

        # Color the boxes
        for patch in bp['boxes']:
            patch.set_facecolor(colors[cell_type])
            patch.set_alpha(0.6)

        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=1.5)

        # Overlay individual FOV points with jitter
        for i, location in enumerate(['In GG Tumor', 'In GG Stroma']):
            y_data = data[location].values
            x_data = np.random.normal(i + 1, 0.04, size=len(y_data))
            axes[idx].scatter(x_data, y_data, alpha=0.5, s=80,
                              edgecolors='black', linewidth=1, zorder=3,
                              color=colors[cell_type])

        axes[idx].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        axes[idx].set_ylim([0, 105])
        axes[idx].grid(axis='y', alpha=0.3, linestyle='--')

        # Titles
        if cell_type == 'Epithelial':
            title = f'{cell_type} Cells\n(Expected: Tumor)'
        elif cell_type == 'Stroma':
            title = f'{cell_type} Cells\n(Expected: Stroma)'
        else:
            title = f'{cell_type} Cells\n(Can infiltrate both)'

        axes[idx].set_title(title, fontsize=13, fontweight='bold', pad=10)

        # Add mean ± std
        mean_tumor = data['In GG Tumor'].mean()
        std_tumor = data['In GG Tumor'].std()
        mean_stroma = data['In GG Stroma'].mean()
        std_stroma = data['In GG Stroma'].std()

        axes[idx].text(1, 100, f'{mean_tumor:.1f}±{std_tumor:.1f}%',
                       fontsize=10, ha='center', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
        axes[idx].text(2, 100, f'{mean_stroma:.1f}±{std_stroma:.1f}%',
                       fontsize=10, ha='center', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))

    plt.suptitle('Cell Type Distribution Across GRIDGENE Compartments',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_distribution_barplot(df_dist, save_path=None):
    """
    B) Bar plot with mean ± SD (cleaner, publication standard)

    Args:
        df_dist: DataFrame from create_distribution_dataframe()
        save_path: Optional path to save figure
    """
    # Calculate summary statistics
    summary = df_dist.groupby('Cell Type')[['In GG Tumor', 'In GG Stroma']].agg(['mean', 'std']).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    cell_types = ['Epithelial', 'Stroma', 'Immune']
    colors = {'Epithelial': '#e74c3c', 'Stroma': '#3498db', 'Immune': '#2ecc71'}

    x = np.arange(len(cell_types))
    width = 0.35

    tumor_means = [summary[summary['Cell Type'] == ct]['In GG Tumor']['mean'].values[0] for ct in cell_types]
    tumor_stds = [summary[summary['Cell Type'] == ct]['In GG Tumor']['std'].values[0] for ct in cell_types]
    stroma_means = [summary[summary['Cell Type'] == ct]['In GG Stroma']['mean'].values[0] for ct in cell_types]
    stroma_stds = [summary[summary['Cell Type'] == ct]['In GG Stroma']['std'].values[0] for ct in cell_types]

    # Create bars
    bars1 = ax.bar(x - width / 2, tumor_means, width, yerr=tumor_stds,
                   label='In GRIDGENE Tumor', capsize=5,
                   color=[colors[ct] for ct in cell_types], alpha=0.7,
                   edgecolor='black', linewidth=1.5)

    bars2 = ax.bar(x + width / 2, stroma_means, width, yerr=stroma_stds,
                   label='In GRIDGENE Stroma', capsize=5,
                   color=[colors[ct] for ct in cell_types], alpha=0.4,
                   edgecolor='black', linewidth=1.5, hatch='///')

    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax.text(bar1.get_x() + bar1.get_width() / 2., height1 + tumor_stds[i] + 2,
                f'{height1:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax.text(bar2.get_x() + bar2.get_width() / 2., height2 + stroma_stds[i] + 2,
                f'{height2:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Cell Type', fontsize=12, fontweight='bold')
    ax.set_title('Cell Type Distribution in GRIDGENE Compartments (Mean ± SD)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(cell_types, fontsize=11)
    ax.set_ylim([0, 110])
    ax.legend(fontsize=11, loc='upper right', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_confusion_heatmap(df_dist, save_path=None):
    """
    C) Confusion matrix style heatmap

    Args:
        df_dist: DataFrame from create_distribution_dataframe()
        save_path: Optional path to save figure
    """
    # Prepare confusion matrix style data
    summary = df_dist.groupby('Cell Type')[['In GG Tumor', 'In GG Stroma']].mean()

    # Reorder for clarity
    summary = summary.reindex(['Epithelial', 'Stroma', 'Immune'])
    summary.columns = ['GRIDGENE Tumor', 'GRIDGENE Stroma']

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(summary, annot=True, fmt='.1f', cmap='RdYlGn', center=50,
                vmin=0, vmax=100, cbar_kws={'label': 'Percentage (%)'},
                linewidths=2, linecolor='black', ax=ax,
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})

    ax.set_xlabel('GRIDGENE Compartment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cell Phenotype', fontsize=12, fontweight='bold')
    ax.set_title(
        'Cell Phenotype vs GRIDGENE Compartment Assignment\n(Higher is better for Epithelial→Tumor, Stroma→Stroma)',
        fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_comprehensive_figure(df_dist, analysis_results, save_path=None):
    """
    D) Comprehensive figure with all metrics in publication-ready format

    Args:
        df_dist: DataFrame from create_distribution_dataframe()
        analysis_results: {fov_name: analysis_dict}
        save_path: Optional path to save figure
    """
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    colors = {'Epithelial': '#e74c3c', 'Stroma': '#3498db', 'Immune': '#2ecc71'}

    # ===== Top row: Boxplots for each cell type =====
    for idx, cell_type in enumerate(['Epithelial', 'Stroma', 'Immune']):
        ax = fig.add_subplot(gs[0, idx])
        data = df_dist[df_dist['Cell Type'] == cell_type]

        boxplot_data = [data['In GG Tumor'], data['In GG Stroma']]
        bp = ax.boxplot(boxplot_data,
                        labels=['GG Tumor', 'GG Stroma'],
                        patch_artist=True,
                        widths=0.6,
                        showfliers=False)

        for patch in bp['boxes']:
            patch.set_facecolor(colors[cell_type])
            patch.set_alpha(0.6)

        for element in ['whiskers', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=1.5)

        # Overlay points
        for i, location in enumerate(['In GG Tumor', 'In GG Stroma']):
            y_data = data[location].values
            x_data = np.random.normal(i + 1, 0.04, size=len(y_data))
            ax.scatter(x_data, y_data, alpha=0.5, s=60,
                       edgecolors='black', linewidth=1, zorder=3,
                       color=colors[cell_type])

        ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_title(f'{cell_type}', fontsize=12, fontweight='bold')

    # ===== Bottom left: Bar plot summary =====
    ax_bar = fig.add_subplot(gs[1, 0])

    summary = df_dist.groupby('Cell Type')[['In GG Tumor', 'In GG Stroma']].agg(['mean', 'std']).reset_index()
    cell_types = ['Epithelial', 'Stroma', 'Immune']
    x = np.arange(len(cell_types))
    width = 0.35

    tumor_means = [summary[summary['Cell Type'] == ct]['In GG Tumor']['mean'].values[0] for ct in cell_types]
    tumor_stds = [summary[summary['Cell Type'] == ct]['In GG Tumor']['std'].values[0] for ct in cell_types]
    stroma_means = [summary[summary['Cell Type'] == ct]['In GG Stroma']['mean'].values[0] for ct in cell_types]
    stroma_stds = [summary[summary['Cell Type'] == ct]['In GG Stroma']['std'].values[0] for ct in cell_types]

    ax_bar.bar(x - width / 2, tumor_means, width, yerr=tumor_stds,
               label='GG Tumor', capsize=4,
               color=[colors[ct] for ct in cell_types], alpha=0.7,
               edgecolor='black', linewidth=1.5)

    ax_bar.bar(x + width / 2, stroma_means, width, yerr=stroma_stds,
               label='GG Stroma', capsize=4,
               color=[colors[ct] for ct in cell_types], alpha=0.4,
               edgecolor='black', linewidth=1.5, hatch='///')

    ax_bar.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(cell_types, fontsize=10)
    ax_bar.set_ylim([0, 110])
    ax_bar.legend(fontsize=10)
    ax_bar.grid(axis='y', alpha=0.3, linestyle='--')
    ax_bar.set_title('Mean ± SD', fontsize=12, fontweight='bold')

    # ===== Bottom middle: Confusion heatmap =====
    ax_heat = fig.add_subplot(gs[1, 1])

    summary_heat = df_dist.groupby('Cell Type')[['In GG Tumor', 'In GG Stroma']].mean()
    summary_heat = summary_heat.reindex(['Epithelial', 'Stroma', 'Immune'])
    summary_heat.columns = ['GG Tumor', 'GG Stroma']

    sns.heatmap(summary_heat, annot=True, fmt='.1f', cmap='RdYlGn', center=50,
                vmin=0, vmax=100, cbar_kws={'label': '%'},
                linewidths=2, linecolor='black', ax=ax_heat,
                annot_kws={'fontsize': 11, 'fontweight': 'bold'})

    ax_heat.set_xlabel('GRIDGENE', fontsize=11, fontweight='bold')
    ax_heat.set_ylabel('Cell Type', fontsize=11, fontweight='bold')
    ax_heat.set_title('Confusion Matrix', fontsize=12, fontweight='bold')

    # ===== Bottom right: Summary statistics table =====
    ax_table = fig.add_subplot(gs[1, 2])
    ax_table.axis('off')

    table_data = []
    for cell_type in ['Epithelial', 'Stroma', 'Immune']:
        data = df_dist[df_dist['Cell Type'] == cell_type]
        tumor_mean = data['In GG Tumor'].mean()
        stroma_mean = data['In GG Stroma'].mean()
        n_fovs = len(data)

        # Calculate total pixels across all FOVs
        total_pixels = sum([analysis_results[fov][f'{cell_type.lower()}_total']
                            for fov in analysis_results.keys()
                            if f'{cell_type.lower()}_total' in analysis_results[fov]])

        table_data.append([cell_type, n_fovs, f'{total_pixels:,}',
                           f'{tumor_mean:.1f}%', f'{stroma_mean:.1f}%'])

    table = ax_table.table(cellText=table_data,
                           colLabels=['Cell Type', 'n FOVs', 'Total Pixels', 'In Tumor', 'In Stroma'],
                           cellLoc='center',
                           loc='center',
                           bbox=[0, 0.2, 1, 0.6])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color headers
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows by cell type
    for i, cell_type in enumerate(['Epithelial', 'Stroma', 'Immune'], start=1):
        table[(i, 0)].set_facecolor(colors[cell_type])
        table[(i, 0)].set_text_props(weight='bold', color='white')

    ax_table.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

    # Overall title
    fig.suptitle('GRIDGENE Cell-Level Validation: Comprehensive Analysis',
                 fontsize=16, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()