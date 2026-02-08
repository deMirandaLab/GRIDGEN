"""
Plot GRIDGEN Benchmark Results
Generates publication-ready figures from benchmark CSV
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Color schemes
PLATFORM_COLORS = {
    'CosMx': '#E64B35',
    'Xenium': '#4DBBD5'
}

METHOD_COLORS = {
    'Convolutional': '#3C5488',
    'KD-tree': '#F39B7F',
    'SOM': '#00A087'
}

METHOD_MARKERS = {
    'Convolutional': 'o',
    'KD-tree': 's',
    'SOM': '^'
}

# Output directory
OUTPUT_DIR = Path('benchmark_plots')
OUTPUT_DIR.mkdir(exist_ok=True)


# ==============================================================================
# LOAD DATA
# ==============================================================================

def load_benchmark_data(csv_file):
    """Load and preprocess benchmark results"""
    df = pd.read_csv(csv_file)

    # Convert n_cores to numeric where possible, keep "Full Slide" as string
    df['n_cores_numeric'] = pd.to_numeric(df['n_cores'], errors='coerce')

    # Separate regular and full-slide data
    df_cores = df[df['n_cores'] != 'Full Slide'].copy()
    df_full = df[df['n_cores'] == 'Full Slide'].copy()

    # Convert to numeric for cores data
    df_cores['n_cores_numeric'] = df_cores['n_cores_numeric'].astype(int)

    print(f"Loaded {len(df)} total results:")
    print(f"  - {len(df_cores)} core-based benchmarks")
    print(f"  - {len(df_full)} full-slide benchmarks")
    print(f"\nPlatforms: {df['platform'].unique()}")
    print(f"Methods: {df['method'].unique()}")
    print(f"Cores: {sorted(df_cores['n_cores_numeric'].unique())}")

    return df, df_cores, df_full


# ==============================================================================
# PLOT 1: RUNTIME VS NUMBER OF CORES
# ==============================================================================

def plot_runtime_vs_cores(df_cores, df_full, output_dir):
    """Plot runtime scaling with number of cores"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, platform in enumerate(['CosMx', 'Xenium']):
        ax = axes[idx]
        df_platform = df_cores[df_cores['platform'] == platform]
        df_full_platform = df_full[df_full['platform'] == platform]

        # Plot each method
        for method in df_platform['method'].unique():
            df_method = df_platform[df_platform['method'] == method]

            # Aggregate by cores (mean and std)
            grouped = df_method.groupby('n_cores_numeric')['total_time'].agg(['mean', 'std']).reset_index()

            ax.errorbar(
                grouped['n_cores_numeric'],
                grouped['mean'],
                yerr=grouped['std'],
                marker=METHOD_MARKERS[method],
                linewidth=2,
                markersize=8,
                capsize=5,
                label=method,
                color=METHOD_COLORS[method],
                alpha=0.8
            )

        # Add full-slide data as separate markers
        if not df_full_platform.empty:
            for method in df_full_platform['method'].unique():
                df_full_method = df_full_platform[df_full_platform['method'] == method]
                if not df_full_method.empty:
                    # Position full-slide at x=12 (beyond 10 cores)
                    ax.scatter(
                        [12] * len(df_full_method),
                        df_full_method['total_time'],
                        marker=METHOD_MARKERS[method],
                        s=200,
                        color=METHOD_COLORS[method],
                        edgecolors='black',
                        linewidths=2,
                        alpha=0.8,
                        zorder=10
                    )

        ax.set_xlabel('Number of Cores', fontweight='bold')
        ax.set_ylabel('Runtime (seconds)', fontweight='bold')
        ax.set_title(f'{platform} Platform', fontweight='bold')
        ax.legend(frameon=True, loc='upper left')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xticks([1, 5, 10, 12])
        ax.set_xticklabels(['1', '5', '10', 'Full\nSlide'])

    plt.tight_layout()
    plt.savefig(output_dir / 'runtime_vs_cores.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'runtime_vs_cores.pdf', bbox_inches='tight')
    print(f"✓ Saved: runtime_vs_cores.png/pdf")
    plt.close()


# ==============================================================================
# PLOT 2: MEMORY USAGE VS NUMBER OF CORES
# ==============================================================================

def plot_memory_vs_cores(df_cores, df_full, output_dir):
    """Plot memory usage scaling"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, platform in enumerate(['CosMx', 'Xenium']):
        ax = axes[idx]
        df_platform = df_cores[df_cores['platform'] == platform]
        df_full_platform = df_full[df_full['platform'] == platform]

        # Plot each method
        for method in df_platform['method'].unique():
            df_method = df_platform[df_platform['method'] == method]

            # Aggregate by cores
            grouped = df_method.groupby('n_cores_numeric')['peak_memory_gb'].agg(['mean', 'std']).reset_index()

            ax.errorbar(
                grouped['n_cores_numeric'],
                grouped['mean'],
                yerr=grouped['std'],
                marker=METHOD_MARKERS[method],
                linewidth=2,
                markersize=8,
                capsize=5,
                label=method,
                color=METHOD_COLORS[method],
                alpha=0.8
            )

        # Add full-slide data
        if not df_full_platform.empty:
            for method in df_full_platform['method'].unique():
                df_full_method = df_full_platform[df_full_platform['method'] == method]
                if not df_full_method.empty:
                    ax.scatter(
                        [12] * len(df_full_method),
                        df_full_method['peak_memory_gb'],
                        marker=METHOD_MARKERS[method],
                        s=200,
                        color=METHOD_COLORS[method],
                        edgecolors='black',
                        linewidths=2,
                        alpha=0.8,
                        zorder=10
                    )

        ax.set_xlabel('Number of Cores', fontweight='bold')
        ax.set_ylabel('Peak Memory (GB)', fontweight='bold')
        ax.set_title(f'{platform} Platform', fontweight='bold')
        ax.legend(frameon=True, loc='upper left')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xticks([1, 5, 10, 12])
        ax.set_xticklabels(['1', '5', '10', 'Full\nSlide'])

    plt.tight_layout()
    plt.savefig(output_dir / 'memory_vs_cores.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'memory_vs_cores.pdf', bbox_inches='tight')
    print(f"✓ Saved: memory_vs_cores.png/pdf")
    plt.close()


# ==============================================================================
# PLOT 3: METHOD COMPARISON AT 10 CORES
# ==============================================================================

def plot_method_comparison(df_cores, output_dir, n_cores=None):
    """Compare methods at specific scale"""
    # Use the highest n_cores available if not specified
    if n_cores is None:
        n_cores = df_cores['n_cores_numeric'].max()

    df_n = df_cores[df_cores['n_cores_numeric'] == n_cores]

    # If no data at requested n_cores, skip
    if df_n.empty:
        print(f"⚠ Skipping method comparison - no data at {n_cores} cores")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Runtime comparison
    ax = axes[0]
    data = []
    labels = []
    colors = []

    for platform in ['CosMx', 'Xenium']:
        for method in ['Convolutional', 'KD-tree', 'SOM']:
            df_subset = df_n[(df_n['platform'] == platform) & (df_n['method'] == method)]
            if not df_subset.empty:
                data.append(df_subset['total_time'].values)
                labels.append(f'{platform}\n{method}')
                colors.append(METHOD_COLORS[method])

    if data:  # Only plot if we have data
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Runtime (seconds)', fontweight='bold')
        ax.set_title(f'Method Comparison ({n_cores} Cores)', fontweight='bold')
        ax.grid(alpha=0.3, axis='y', linestyle='--')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Memory comparison
    ax = axes[1]
    data = []
    labels = []
    colors = []

    for platform in ['CosMx', 'Xenium']:
        for method in ['Convolutional', 'KD-tree', 'SOM']:
            df_subset = df_n[(df_n['platform'] == platform) & (df_n['method'] == method)]
            if not df_subset.empty:
                data.append(df_subset['peak_memory_gb'].values)
                labels.append(f'{platform}\n{method}')
                colors.append(METHOD_COLORS[method])

    if data:  # Only plot if we have data
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Peak Memory (GB)', fontweight='bold')
        ax.set_title(f'Memory Comparison ({n_cores} Cores)', fontweight='bold')
        ax.grid(alpha=0.3, axis='y', linestyle='--')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / f'method_comparison_{n_cores}cores.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'method_comparison_{n_cores}cores.pdf', bbox_inches='tight')
    print(f"✓ Saved: method_comparison_{n_cores}cores.png/pdf")
    plt.close()


# ==============================================================================
# PLOT 4: RUNTIME BREAKDOWN BY OPERATION
# ==============================================================================

def plot_runtime_breakdown(df_cores, output_dir):
    """Stacked bar chart showing time breakdown by operation"""

    # Use the highest n_cores available
    max_cores = df_cores['n_cores_numeric'].max()
    df_n = df_cores[df_cores['n_cores_numeric'] == max_cores].copy()

    if df_n.empty:
        print(f"⚠ Skipping runtime breakdown - no data")
        return

    # Calculate mean time per operation
    operations = ['load_time', 'tumor_contour_time', 'empty_contour_time', 'mask_time']
    op_labels = ['Loading', 'Tumor\nContours', 'Empty\nContours', 'Masks']
    op_colors = ['#8491B4', '#F39B7F', '#91D1C2', '#3C5488']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, platform in enumerate(['CosMx', 'Xenium']):
        ax = axes[idx]
        df_platform = df_n[df_n['platform'] == platform]

        methods = df_platform['method'].unique()
        x_pos = np.arange(len(methods))

        bottom = np.zeros(len(methods))

        for op, label, color in zip(operations, op_labels, op_colors):
            means = []
            for method in methods:
                df_method = df_platform[df_platform['method'] == method]
                # Handle NaN values (e.g., for SOM which doesn't have contour times)
                mean_val = df_method[op].fillna(0).mean()
                means.append(mean_val)

            ax.bar(x_pos, means, bottom=bottom, label=label, color=color, alpha=0.85, width=0.6)
            bottom += means

        # Add binning_time and som_time for SOM method
        if 'SOM' in methods:
            som_idx = list(methods).index('SOM')
            df_som = df_platform[df_platform['method'] == 'SOM']

            if 'binning_time' in df_som.columns:
                binning_mean = df_som['binning_time'].fillna(0).mean()
                som_mean = df_som['som_time'].fillna(0).mean()

                # Replace the bar for SOM
                ax.bar([som_idx], [binning_mean], label='Binning' if idx == 0 else '',
                       color='#E64B35', alpha=0.85, width=0.6)
                ax.bar([som_idx], [som_mean], bottom=[binning_mean],
                       label='SOM Clustering' if idx == 0 else '',
                       color='#00A087', alpha=0.85, width=0.6)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods)
        ax.set_ylabel('Time (seconds)', fontweight='bold')
        ax.set_title(f'{platform} - Runtime Breakdown (10 cores)', fontweight='bold')
        if idx == 0:
            ax.legend(loc='upper left', frameon=True)
        ax.grid(alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / 'runtime_breakdown.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'runtime_breakdown.pdf', bbox_inches='tight')
    print(f"✓ Saved: runtime_breakdown.png/pdf")
    plt.close()


# ==============================================================================
# PLOT 5: TRANSCRIPTS PROCESSED
# ==============================================================================

def plot_transcripts_processed(df_cores, df_full, output_dir):
    """Show data scale - transcripts processed"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Aggregate transcripts by platform and n_cores
    data = []
    for platform in ['CosMx', 'Xenium']:
        df_platform = df_cores[df_cores['platform'] == platform]
        for n_cores in sorted(df_platform['n_cores_numeric'].unique()):
            df_subset = df_platform[df_platform['n_cores_numeric'] == n_cores]
            total_transcripts = df_subset['n_transcripts'].sum()
            data.append({
                'platform': platform,
                'n_cores': n_cores,
                'transcripts_millions': total_transcripts / 1e6
            })

        # Add full-slide data
        df_full_platform = df_full[df_full['platform'] == platform]
        if not df_full_platform.empty:
            for _, row in df_full_platform.iterrows():
                data.append({
                    'platform': platform,
                    'n_cores': 12,  # Position for plotting
                    'transcripts_millions': row['n_transcripts'] / 1e6
                })

    df_plot = pd.DataFrame(data)

    # Plot
    for platform in ['CosMx', 'Xenium']:
        df_platform = df_plot[df_plot['platform'] == platform]
        ax.plot(
            df_platform['n_cores'],
            df_platform['transcripts_millions'],
            marker='o',
            linewidth=2.5,
            markersize=10,
            label=platform,
            color=PLATFORM_COLORS[platform]
        )

    ax.set_xlabel('Number of Cores', fontweight='bold')
    ax.set_ylabel('Total Transcripts (millions)', fontweight='bold')
    ax.set_title('Data Scale Processed', fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xticks([1, 5, 10, 12])
    ax.set_xticklabels(['1', '5', '10', 'Full\nSlide'])

    plt.tight_layout()
    plt.savefig(output_dir / 'transcripts_processed.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'transcripts_processed.pdf', bbox_inches='tight')
    print(f"✓ Saved: transcripts_processed.png/pdf")
    plt.close()


# ==============================================================================
# PLOT 6: COMBINED PANEL FIGURE
# ==============================================================================

def plot_combined_panel(df_cores, df_full, output_dir):
    """Create a combined multi-panel figure for publication"""

    # Get max cores for breakdown
    max_cores = df_cores['n_cores_numeric'].max()
    df_n = df_cores[df_cores['n_cores_numeric'] == max_cores].copy()

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Panel A: Runtime vs cores - CosMx
    ax1 = fig.add_subplot(gs[0, 0])
    df_cosmx = df_cores[df_cores['platform'] == 'CosMx']
    df_full_cosmx = df_full[df_full['platform'] == 'CosMx']

    for method in df_cosmx['method'].unique():
        df_method = df_cosmx[df_cosmx['method'] == method]
        grouped = df_method.groupby('n_cores_numeric')['total_time'].agg(['mean', 'std']).reset_index()
        ax1.errorbar(grouped['n_cores_numeric'], grouped['mean'], yerr=grouped['std'],
                     marker=METHOD_MARKERS[method], linewidth=2, markersize=8, capsize=5,
                     label=method, color=METHOD_COLORS[method], alpha=0.8)

    ax1.set_xlabel('Number of Cores', fontweight='bold')
    ax1.set_ylabel('Runtime (seconds)', fontweight='bold')
    ax1.set_title('A. CosMx Runtime Scaling', fontweight='bold', loc='left')
    ax1.legend(frameon=True, loc='upper left')
    ax1.grid(alpha=0.3, linestyle='--')

    # Panel B: Runtime vs cores - Xenium
    ax2 = fig.add_subplot(gs[0, 1])
    df_xenium = df_cores[df_cores['platform'] == 'Xenium']
    df_full_xenium = df_full[df_full['platform'] == 'Xenium']

    for method in df_xenium['method'].unique():
        df_method = df_xenium[df_xenium['method'] == method]
        grouped = df_method.groupby('n_cores_numeric')['total_time'].agg(['mean', 'std']).reset_index()
        ax2.errorbar(grouped['n_cores_numeric'], grouped['mean'], yerr=grouped['std'],
                     marker=METHOD_MARKERS[method], linewidth=2, markersize=8, capsize=5,
                     label=method, color=METHOD_COLORS[method], alpha=0.8)

    # Add full-slide markers
    for method in df_full_xenium['method'].unique():
        df_full_method = df_full_xenium[df_full_xenium['method'] == method]
        if not df_full_method.empty:
            ax2.scatter([12] * len(df_full_method), df_full_method['total_time'],
                        marker=METHOD_MARKERS[method], s=200, color=METHOD_COLORS[method],
                        edgecolors='black', linewidths=2, alpha=0.8, zorder=10)

    ax2.set_xlabel('Number of Cores', fontweight='bold')
    ax2.set_ylabel('Runtime (seconds)', fontweight='bold')
    ax2.set_title('B. Xenium Runtime Scaling', fontweight='bold', loc='left')
    ax2.legend(frameon=True, loc='upper left')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_xticks([1, 5, 10, 12])
    ax2.set_xticklabels(['1', '5', '10', 'Full\nSlide'])

    # Panel C: Memory comparison
    ax3 = fig.add_subplot(gs[1, 0])
    for platform in ['CosMx', 'Xenium']:
        df_platform = df_cores[df_cores['platform'] == platform]
        grouped = df_platform.groupby('n_cores_numeric')['peak_memory_gb'].mean().reset_index()
        ax3.plot(grouped['n_cores_numeric'], grouped['peak_memory_gb'],
                 marker='o', linewidth=2.5, markersize=10,
                 label=platform, color=PLATFORM_COLORS[platform])

    # Add full-slide
    for platform in ['CosMx', 'Xenium']:
        df_full_platform = df_full[df_full['platform'] == platform]
        if not df_full_platform.empty:
            ax3.scatter([12] * len(df_full_platform), df_full_platform['peak_memory_gb'],
                        s=200, color=PLATFORM_COLORS[platform],
                        edgecolors='black', linewidths=2, zorder=10)

    ax3.set_xlabel('Number of Cores', fontweight='bold')
    ax3.set_ylabel('Peak Memory (GB)', fontweight='bold')
    ax3.set_title('C. Memory Usage', fontweight='bold', loc='left')
    ax3.legend(frameon=True)
    ax3.grid(alpha=0.3, linestyle='--')
    ax3.set_xticks([1, 5, 10, 12])
    ax3.set_xticklabels(['1', '5', '10', 'Full\nSlide'])

    # Panel D: Transcripts processed
    ax4 = fig.add_subplot(gs[1, 1])
    data = []
    for platform in ['CosMx', 'Xenium']:
        df_platform = df_cores[df_cores['platform'] == platform]
        for n_cores in sorted(df_platform['n_cores_numeric'].unique()):
            df_subset = df_platform[df_platform['n_cores_numeric'] == n_cores]
            total_transcripts = df_subset['n_transcripts'].sum()
            data.append({'platform': platform, 'n_cores': n_cores,
                         'transcripts_millions': total_transcripts / 1e6})

        df_full_platform = df_full[df_full['platform'] == platform]
        if not df_full_platform.empty:
            for _, row in df_full_platform.iterrows():
                data.append({'platform': platform, 'n_cores': 12,
                             'transcripts_millions': row['n_transcripts'] / 1e6})

    df_plot = pd.DataFrame(data)
    for platform in ['CosMx', 'Xenium']:
        df_platform = df_plot[df_plot['platform'] == platform]
        ax4.plot(df_platform['n_cores'], df_platform['transcripts_millions'],
                 marker='o', linewidth=2.5, markersize=10,
                 label=platform, color=PLATFORM_COLORS[platform])

    ax4.set_xlabel('Number of Cores', fontweight='bold')
    ax4.set_ylabel('Total Transcripts (millions)', fontweight='bold')
    ax4.set_title('D. Data Scale', fontweight='bold', loc='left')
    ax4.legend(frameon=True)
    ax4.grid(alpha=0.3, linestyle='--')
    ax4.set_xticks([1, 5, 10, 12])
    ax4.set_xticklabels(['1', '5', '10', 'Full\nSlide'])

    # Panel E & F: Runtime breakdown
    df_10 = df_cores[df_cores['n_cores_numeric'] == 10].copy()
    operations = ['load_time', 'tumor_contour_time', 'empty_contour_time', 'mask_time']
    op_labels = ['Loading', 'Tumor', 'Empty', 'Masks']
    op_colors = ['#8491B4', '#F39B7F', '#91D1C2', '#3C5488']

    for idx, (platform, ax_pos) in enumerate([('CosMx', gs[2, 0]), ('Xenium', gs[2, 1])]):
        ax = fig.add_subplot(ax_pos)
        df_platform = df_n[df_n['platform'] == platform]

        methods = sorted(df_platform['method'].unique())
        x_pos = np.arange(len(methods))
        bottom = np.zeros(len(methods))

        for op, label, color in zip(operations, op_labels, op_colors):
            means = [df_platform[df_platform['method'] == m][op].fillna(0).mean()
                     for m in methods]
            ax.bar(x_pos, means, bottom=bottom, label=label if idx == 0 else '',
                   color=color, alpha=0.85, width=0.6)
            bottom += means

        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=0)
        ax.set_ylabel('Time (seconds)', fontweight='bold')
        ax.set_title(f'{"E" if idx == 0 else "F"}. {platform} Runtime Breakdown',
                     fontweight='bold', loc='left')
        if idx == 0:
            ax.legend(loc='upper left', frameon=True, ncol=2)
        ax.grid(alpha=0.3, axis='y', linestyle='--')

    plt.savefig(output_dir / 'combined_panel_figure.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'combined_panel_figure.pdf', bbox_inches='tight')
    print(f"✓ Saved: combined_panel_figure.png/pdf")
    plt.close()


# ==============================================================================
# SUMMARY TABLE
# ==============================================================================

def generate_summary_table(df, df_cores, df_full, output_dir):
    """Generate summary statistics table"""

    summary_data = []

    # Core-based summaries
    for platform in ['CosMx', 'Xenium']:
        for method in df_cores['method'].unique():
            for n_cores in sorted(df_cores['n_cores_numeric'].unique()):
                df_subset = df_cores[
                    (df_cores['platform'] == platform) &
                    (df_cores['method'] == method) &
                    (df_cores['n_cores_numeric'] == n_cores)
                    ]

                if not df_subset.empty:
                    summary_data.append({
                        'Platform': platform,
                        'Method': method,
                        'Scale': f'{n_cores} cores',
                        'N_samples': len(df_subset),
                        'Transcripts (M)': f"{df_subset['n_transcripts'].sum() / 1e6:.2f}",
                        'Runtime (s)': f"{df_subset['total_time'].mean():.2f} ± {df_subset['total_time'].std():.2f}",
                        'Memory (GB)': f"{df_subset['peak_memory_gb'].mean():.2f} ± {df_subset['peak_memory_gb'].std():.2f}"
                    })

    # Full-slide summaries
    for platform in ['CosMx', 'Xenium']:
        for method in df_full['method'].unique():
            df_subset = df_full[
                (df_full['platform'] == platform) &
                (df_full['method'] == method)
                ]

            if not df_subset.empty:
                summary_data.append({
                    'Platform': platform,
                    'Method': method,
                    'Scale': 'Full slide',
                    'N_samples': len(df_subset),
                    'Transcripts (M)': f"{df_subset['n_transcripts'].mean() / 1e6:.2f}",
                    'Runtime (s)': f"{df_subset['total_time'].mean():.2f}",
                    'Memory (GB)': f"{df_subset['peak_memory_gb'].mean():.2f}"
                })

    df_summary = pd.DataFrame(summary_data)

    # Save as CSV and text
    df_summary.to_csv(output_dir / 'benchmark_summary.csv', index=False)

    with open(output_dir / 'benchmark_summary.txt', 'w') as f:
        f.write("GRIDGEN BENCHMARK SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(df_summary.to_string(index=False))
        f.write("\n\n")

    print(f"✓ Saved: benchmark_summary.csv and benchmark_summary.txt")
    print("\nSummary:")
    print(df_summary.to_string(index=False))


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    # Find the most recent benchmark CSV
    csv_files = sorted(Path('.').glob('benchmark_results_*.csv'))

    if not csv_files:
        print("ERROR: No benchmark CSV files found!")
        print("Please run 'python run_benchmarks.py' first.")
        sys.exit(1)

    csv_file = csv_files[-1]
    print(f"\nLoading benchmark results from: {csv_file}")
    print("=" * 60)

    # Load data
    df, df_cores, df_full = load_benchmark_data(csv_file)

    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60 + "\n")

    # Generate all plots
    plot_runtime_vs_cores(df_cores, df_full, OUTPUT_DIR)
    plot_memory_vs_cores(df_cores, df_full, OUTPUT_DIR)
    plot_method_comparison(df_cores, OUTPUT_DIR)  # Auto-detect max cores
    plot_runtime_breakdown(df_cores, OUTPUT_DIR)
    plot_transcripts_processed(df_cores, df_full, OUTPUT_DIR)
    plot_combined_panel(df_cores, df_full, OUTPUT_DIR)

    # Generate summary table
    print("\n" + "=" * 60)
    print("GENERATING SUMMARY")
    print("=" * 60 + "\n")
    generate_summary_table(df, df_cores, df_full, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nAll outputs saved to: {OUTPUT_DIR.absolute()}")
    print("\nGenerated files:")
    print("  - runtime_vs_cores.png/pdf")
    print("  - memory_vs_cores.png/pdf")
    print("  - method_comparison_10cores.png/pdf")
    print("  - runtime_breakdown.png/pdf")
    print("  - transcripts_processed.png/pdf")
    print("  - combined_panel_figure.png/pdf")
    print("  - benchmark_summary.csv")
    print("  - benchmark_summary.txt")


if __name__ == "__main__":
    main()