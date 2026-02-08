"""
GRIDGEN Benchmark Orchestrator
Runs benchmarks in isolated subprocesses for clean memory between runs
"""

import os
import sys
import subprocess
import json
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from natsort import natsorted

# ==============================================================================
# SETUP LOGGING
# ==============================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f'benchmark_{timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# SYSTEM SPECS
# ==============================================================================
import platform
import multiprocessing
import psutil

logger.info("=" * 60)
logger.info("SYSTEM SPECIFICATIONS")
logger.info("=" * 60)
logger.info(f"OS: {platform.system()} {platform.release()}")
logger.info(f"Processor: {platform.processor()}")
logger.info(f"CPU cores: {multiprocessing.cpu_count()}")
logger.info(f"RAM: {psutil.virtual_memory().total / (1024 ** 3):.1f} GB")
logger.info("=" * 60)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Paths
COSMX_PATH = '/home/martinha/PycharmProjects/phd/spatial_transcriptomics/cosmx_data/S0/S0/20230628_151317_S4/AnalysisResults/iz38iruwno'
XENIUM_PATH = '/home/martinha/PycharmProjects/phd/spatial_transcriptomics/xenium_data/HLA/GD_TMA1_S3/fov_filtered'
XENIUM_FULL_PATH = '/home/martinha/PycharmProjects/phd/spatial_transcriptomics/xenium_data/HLA/GD_TMA1_S3/original_data/transcripts.csv.gz'

XENIUM_EXCLUDE = [
    'TMA1_Selection14_filtered.csv',
    'TMA1_Selection15_filtered.csv',
    'TMA1_Selection18_filtered.csv',
    'TMA1_Selection24_filtered.csv',
    'TMA1_Selection27_filtered.csv',
    'TMA1_Selection32_filtered.csv',
    'TMA1_Selection33_filtered.csv'
]

# Gene sets
COSMX_TARGET_TUM = ['EPCAM', 'KRT19', 'KRT8', 'KRT18', 'KRT17', 'CEACAM6',
                    'SPINK1', 'CD24', 'S100A6', 'RPL37', 'S100P']
XENIUM_TARGET_TUM = ['EPCAM', 'SMIM22', 'CLDN3', 'KRT18', 'LGALS4', 'KRT8',
                     'ELF3', 'TSPAN8', 'STMN1', 'CD47', 'MYC', 'LGALS3']

# ============== CONVOLUTIONAL PARAMETERS ==============
COSMX_CONV_PARAMS = {
    'density_th_tum': 40,
    'min_area_th_tum': 1000,
    'kernel_size_tum': 80,
    'density_th_empty': 140,
    'min_area_th_empty': 2000,
    'kernel_size_empty': 80
}

XENIUM_CONV_PARAMS = {
    'kernel_size_tum': 10,
    'density_th_tum': 20,
    'min_area_th_tum': 700,
    'density_th_empty': 30,
    'min_area_th_empty': 400,
    'kernel_size_empty': 10
}

# ============== KD-TREE PARAMETERS ==============
COSMX_KDTREE_PARAMS = {
    'radius': 80,  # same as conv
    'density_th_empty': 140,
    'min_area_th_empty': 2000,
    'density_th_tum': 40,
    'min_area_th_tum': 1000
}

XENIUM_KDTREE_PARAMS = {
    'radius': 10,
    'density_th_empty': 30,
    'min_area_th_empty': 400,
    'density_th_tum': 20,
    'min_area_th_tum': 700
}

# ============== SOM PARAMETERS ==============
COSMX_SOM_PARAMS = {
    'bin_size': 80,
    'min_counts': 100
}

XENIUM_SOM_PARAMS = {
    'bin_size': 10,
    'min_counts': 20
}

# Scales to test
SCALES = [1, 5, 10]


# ==============================================================================
# FILE DISCOVERY
# ==============================================================================

def get_cosmx_files():
    """Get all CosMx transcript files"""
    files = []
    for folder in os.listdir(COSMX_PATH):
        folder_path = os.path.join(COSMX_PATH, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if '__target_call_coord.csv' in file:
                    files.append(os.path.join(folder_path, file))
    return natsorted(files)


def get_xenium_files():
    """Get all Xenium transcript files"""
    files = []
    for file in os.listdir(XENIUM_PATH):
        if file not in XENIUM_EXCLUDE and file.endswith('.csv'):
            files.append(os.path.join(XENIUM_PATH, file))
    return natsorted(files)


cosmx_files = get_cosmx_files()
xenium_files = get_xenium_files()

logger.info(f"\nFound {len(cosmx_files)} CosMx files")
logger.info(f"Found {len(xenium_files)} Xenium files")


# ==============================================================================
# RUN BENCHMARK IN SUBPROCESS
# ==============================================================================

def run_benchmark_subprocess(config):
    """Run a single benchmark in isolated subprocess"""
    try:
        result = subprocess.run(
            [sys.executable, 'run_single_benchmark.py'],
            input=json.dumps(config),
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        if result.returncode != 0:
            logger.error(f"Benchmark failed: {result.stderr}")
            return None

        metrics = json.loads(result.stdout)
        return metrics

    except subprocess.TimeoutExpired:
        logger.error("Benchmark timeout (>1 hour)")
        return None
    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        return None


# ==============================================================================
# MAIN BENCHMARK LOOP
# ==============================================================================

def run_all_benchmarks():
    """Run all benchmarks"""
    results = []

    logger.info("\n" + "=" * 60)
    logger.info("STARTING BENCHMARKS")
    logger.info("=" * 60)

    # ===== CosMx Convolutional =====
    logger.info("\n--- CosMx Convolutional ---")
    for n_cores in SCALES:
        logger.info(f"\nProcessing {n_cores} cores...")
        files_subset = cosmx_files[:n_cores]

        for file in tqdm(files_subset, desc=f"CosMx Conv {n_cores}"):
            config = {
                'benchmark_type': 'cosmx_conv',
                'file': file,
                'params': {**COSMX_CONV_PARAMS, 'target_tum': COSMX_TARGET_TUM}
            }
            metrics = run_benchmark_subprocess(config)
            if metrics:
                metrics['platform'] = 'CosMx'
                metrics['method'] = 'Convolutional'
                metrics['n_cores'] = n_cores
                metrics['file'] = os.path.basename(file)
                results.append(metrics)

    # ===== CosMx KD-tree =====
    logger.info("\n--- CosMx KD-tree ---")
    for n_cores in SCALES:
        logger.info(f"\nProcessing {n_cores} cores...")
        files_subset = cosmx_files[:n_cores]

        for file in tqdm(files_subset, desc=f"CosMx KDtree {n_cores}"):
            config = {
                'benchmark_type': 'cosmx_kdtree',
                'file': file,
                'params': {**COSMX_KDTREE_PARAMS, 'target_tum': COSMX_TARGET_TUM}
            }
            metrics = run_benchmark_subprocess(config)
            if metrics:
                metrics['platform'] = 'CosMx'
                metrics['method'] = 'KD-tree'
                metrics['n_cores'] = n_cores
                metrics['file'] = os.path.basename(file)
                results.append(metrics)

    # ===== CosMx SOM =====
    logger.info("\n--- CosMx SOM ---")
    for n_cores in SCALES:
        logger.info(f"\nProcessing {n_cores} cores...")
        files_subset = cosmx_files[:n_cores]

        config = {
            'benchmark_type': 'cosmx_som',
            'files': files_subset,
            'params': COSMX_SOM_PARAMS
        }
        metrics = run_benchmark_subprocess(config)
        if metrics:
            metrics['platform'] = 'CosMx'
            metrics['method'] = 'SOM'
            metrics['n_cores'] = n_cores
            metrics['file'] = f'{n_cores}_cores_combined'
            results.append(metrics)

    # ===== Xenium Convolutional =====
    logger.info("\n--- Xenium Convolutional ---")
    for n_cores in SCALES:
        logger.info(f"\nProcessing {n_cores} cores...")
        files_subset = xenium_files[:n_cores]

        for file in tqdm(files_subset, desc=f"Xenium Conv {n_cores}"):
            config = {
                'benchmark_type': 'xenium_conv',
                'file': file,
                'params': {**XENIUM_CONV_PARAMS, 'target_tum': XENIUM_TARGET_TUM}
            }
            metrics = run_benchmark_subprocess(config)
            if metrics:
                metrics['platform'] = 'Xenium'
                metrics['method'] = 'Convolutional'
                metrics['n_cores'] = n_cores
                metrics['file'] = os.path.basename(file)
                results.append(metrics)

    # ===== Xenium KD-tree =====
    logger.info("\n--- Xenium KD-tree ---")
    for n_cores in SCALES:
        logger.info(f"\nProcessing {n_cores} cores...")
        files_subset = xenium_files[:n_cores]

        for file in tqdm(files_subset, desc=f"Xenium KDtree {n_cores}"):
            config = {
                'benchmark_type': 'xenium_kdtree',
                'file': file,
                'params': {**XENIUM_KDTREE_PARAMS, 'target_tum': XENIUM_TARGET_TUM}
            }
            metrics = run_benchmark_subprocess(config)
            if metrics:
                metrics['platform'] = 'Xenium'
                metrics['method'] = 'KD-tree'
                metrics['n_cores'] = n_cores
                metrics['file'] = os.path.basename(file)
                results.append(metrics)

    # ===== Xenium SOM =====
    logger.info("\n--- Xenium SOM ---")
    for n_cores in SCALES:
        logger.info(f"\nProcessing {n_cores} cores...")
        files_subset = xenium_files[:n_cores]

        config = {
            'benchmark_type': 'xenium_som',
            'files': files_subset,
            'params': XENIUM_SOM_PARAMS
        }
        metrics = run_benchmark_subprocess(config)
        if metrics:
            metrics['platform'] = 'Xenium'
            metrics['method'] = 'SOM'
            metrics['n_cores'] = n_cores
            metrics['file'] = f'{n_cores}_cores_combined'
            results.append(metrics)

    # ===== Xenium Full Slide =====
    logger.info("\n" + "=" * 60)
    logger.info("XENIUM FULL SLIDE (1/3)")
    logger.info("=" * 60)

    # Convolutional
    logger.info("\nFull slide - Convolutional...")
    config = {
        'benchmark_type': 'xenium_fullslide_conv',
        'full_path': XENIUM_FULL_PATH,
        'params': {**XENIUM_CONV_PARAMS, 'target_tum': XENIUM_TARGET_TUM}
    }
    metrics = run_benchmark_subprocess(config)
    if metrics:
        metrics['platform'] = 'Xenium'
        metrics['method'] = 'Convolutional'
        metrics['n_cores'] = 'Full Slide'
        metrics['file'] = 'Third of Full Slide'
        results.append(metrics)

    # SOM
    logger.info("\nFull slide - SOM...")
    config = {
        'benchmark_type': 'xenium_fullslide_som',
        'full_path': XENIUM_FULL_PATH,
        'params': XENIUM_SOM_PARAMS
    }
    metrics = run_benchmark_subprocess(config)
    if metrics:
        metrics['platform'] = 'Xenium'
        metrics['method'] = 'SOM'
        metrics['n_cores'] = 'Full Slide'
        metrics['file'] = 'Third of Full Slide'
        results.append(metrics)

    return results


# ==============================================================================
# SAVE RESULTS
# ==============================================================================

def save_results(results):
    """Save benchmark results"""
    df_results = pd.DataFrame(results)
    output_file = f'benchmark_results_{timestamp}.csv'
    df_results.to_csv(output_file, index=False)
    logger.info(f"\n✓ Results saved to {output_file}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 60)

    summary = df_results.groupby(['platform', 'method', 'n_cores']).agg({
        'total_time': ['mean', 'std', 'min', 'max'],
        'peak_memory_gb': ['mean', 'std', 'max'],
        'n_transcripts': 'sum'
    }).round(3)

    logger.info("\n" + summary.to_string())

    return output_file


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import time

    start_time = time.time()

    logger.info("Starting GRIDGEN benchmark suite...")

    # Run all benchmarks
    results = run_all_benchmarks()

    # Save results
    if results:
        output_file = save_results(results)

        total_time = time.time() - start_time
        logger.info(f"\n" + "=" * 60)
        logger.info(f"BENCHMARK COMPLETE")
        logger.info(f"Total runtime: {total_time / 60:.2f} minutes")
        logger.info(f"Results: {output_file}")
        logger.info(f"Log: {log_file}")
        logger.info("=" * 60)
    else:
        logger.error("No results collected!")

    sys.exit(0)