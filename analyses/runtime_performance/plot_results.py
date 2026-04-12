"""
GRIDGENE Benchmark Plots - Clean publication style
Two panels: Runtime and Memory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ==============================================================================
# STYLE
# ==============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

METHOD_STYLES = {
    'Convolutional': {'color': 'black',  'marker': 'o', 'ls': '-'},
    'KD-tree':       {'color': '#555555', 'marker': 's', 'ls': '--'},
    'SOM':           {'color': '#999999', 'marker': '^', 'ls': ':'},
}

# Transcript counts per scale for x-axis labels
# Format: (n_cores_numeric, label)
COSMX_XTICKS = {
    1:  '1\n(1.8M)',
    5:  '5\n(5.4M)',
    10: '10\n(10.5M)',
}
XENIUM_XTICKS = {
    1:  '1\n(2.5M)',
    5:  '5\n(17.2M)',
    10: '10\n(31.0M)',
    12: 'Full slide¹\n(6.9M)',
}

OUTPUT_DIR = Path('benchmark_plots')
OUTPUT_DIR.mkdir(exist_ok=True)


# ==============================================================================
# LOAD
# ==============================================================================
csv_files = sorted(Path('.').glob('benchmark_results_*.csv'))
if not csv_files:
    raise FileNotFoundError("No benchmark CSV found.")
df = pd.read_csv(csv_files[-1])

df_cores = df[df['n_cores'] != 'Full Slide'].copy()
df_cores['n_cores_numeric'] = pd.to_numeric(df_cores['n_cores'], errors='coerce').astype(int)
df_full  = df[df['n_cores'] == 'Full Slide'].copy()


# ==============================================================================
# HELPER
# ==============================================================================
def plot_platform(ax, platform, metric, df_cores, df_full, xtick_map):
    df_p = df_cores[df_cores['platform'] == platform]
    df_f = df_full[df_full['platform'] == platform]

    for method, style in METHOD_STYLES.items():
        df_m = df_p[df_p['method'] == method]
        if df_m.empty:
            continue
        grp = df_m.groupby('n_cores_numeric')[metric].agg(['mean', 'std']).reset_index()
        ax.errorbar(
            grp['n_cores_numeric'], grp['mean'], yerr=grp['std'],
            marker=style['marker'], color=style['color'], ls=style['ls'],
            linewidth=1.5, markersize=6, capsize=4, label=method
        )
        # Full slide (Xenium only)
        if not df_f.empty:
            df_fm = df_f[df_f['method'] == method]
            if not df_fm.empty:
                ax.scatter(
                    [12], df_fm[metric].values,
                    marker=style['marker'], color=style['color'],
                    s=80, zorder=10, edgecolors='black', linewidths=0.8
                )

    xtick_vals = sorted(xtick_map.keys())
    ax.set_xticks(xtick_vals)
    ax.set_xticklabels([xtick_map[v] for v in xtick_vals])
    ax.set_xlim(0, max(xtick_vals) + 1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_title(platform, fontweight='bold')


# ==============================================================================
# FIGURE
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

# Row 0: Runtime
for col, (platform, xticks) in enumerate([('CosMx', COSMX_XTICKS), ('Xenium', XENIUM_XTICKS)]):
    ax = axes[0, col]
    plot_platform(ax, platform, 'total_time', df_cores, df_full, xticks)
    ax.set_ylabel('Runtime (seconds)' if col == 0 else '')
    if col == 0:
        ax.legend(frameon=False)

# Row 1: Memory
for col, (platform, xticks) in enumerate([('CosMx', COSMX_XTICKS), ('Xenium', XENIUM_XTICKS)]):
    ax = axes[1, col]
    plot_platform(ax, platform, 'peak_memory_gb', df_cores, df_full, xticks)
    ax.set_ylabel('Peak memory (GB)' if col == 0 else '')
    ax.set_xlabel('Number of samples\n(transcripts processed)')

# Panel labels
for ax, label in zip(axes.flat, ['A', 'B', 'C', 'D']):
    ax.text(-0.12, 1.05, label, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')

plt.savefig(OUTPUT_DIR / 'benchmark_figure.png', dpi=600, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'benchmark_figure.pdf', bbox_inches='tight')
print("Saved: benchmark_figure.png / .pdf")
plt.close()


# ==============================================================================
# SUMMARY TABLE
# ==============================================================================

summary_data = []

for platform in ['CosMx', 'Xenium']:
    for method in ['Convolutional', 'KD-tree', 'SOM']:
        for n_cores in sorted(df_cores['n_cores_numeric'].unique()):
            sub = df_cores[
                (df_cores['platform'] == platform) &
                (df_cores['method'] == method) &
                (df_cores['n_cores_numeric'] == n_cores)
            ]
            if sub.empty:
                continue
            summary_data.append({
                'Platform': platform,
                'Method': method,
                'Scale': f'{n_cores} cores',
                'Transcripts (M)': round(sub['n_transcripts'].sum() / 1e6, 2),
                'Runtime mean (s)': round(sub['total_time'].mean(), 2),
                'Runtime std (s)': round(sub['total_time'].std(), 2),
                'Memory mean (GB)': round(sub['peak_memory_gb'].mean(), 2),
                'Memory std (GB)': round(sub['peak_memory_gb'].std(), 2),
            })

        # Full slide
        sub_f = df_full[
            (df_full['platform'] == platform) &
            (df_full['method'] == method)
        ]
        if not sub_f.empty:
            summary_data.append({
                'Platform': platform,
                'Method': method,
                'Scale': 'Full slide (1/4 Xenium)',
                'Transcripts (M)': round(sub_f['n_transcripts'].mean() / 1e6, 2),
                'Runtime mean (s)': round(sub_f['total_time'].mean(), 2),
                'Runtime std (s)': float('nan'),
                'Memory mean (GB)': round(sub_f['peak_memory_gb'].mean(), 2),
                'Memory std (GB)': float('nan'),
            })

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv(OUTPUT_DIR / 'benchmark_summary.csv', index=False)

with open(OUTPUT_DIR / 'benchmark_summary.txt', 'w') as f:
    f.write("GRIDGENE BENCHMARK SUMMARY\n")
    f.write("System: Linux x86_64, 8 CPU cores, 62.6 GB RAM\n")
    f.write("Full slide = approximately 1/4 of a full Xenium slide\n")
    f.write("=" * 90 + "\n")
    f.write(df_summary.to_string(index=False))
    f.write("\n")

print("Saved: benchmark_summary.csv / .txt")