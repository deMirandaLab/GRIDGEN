"""
GRIDGEN Performance Benchmark Script
Measures runtime and memory usage across different methods and scales
Outputs: benchmark_results.csv and benchmark.log
"""

import os
import sys
import time
import psutil
import pandas as pd
import numpy as np
from tqdm import tqdm
import tracemalloc
import logging
from datetime import datetime
from natsort import natsorted
import gc
from pathlib import Path

# Add GRIDGEN to path
gridgene_root = Path.cwd().parent.parent # analyses/quantification -> analyses -> GRIDGENE
sys.path.insert(0, str(gridgene_root))

print(f"Added to path: {gridgene_root}")
print(f"gridgene package location: {gridgene_root / 'gridgene'}")

# Import GRIDGEN modules
from gridgene import get_arrays as ga
from gridgene import contours
from gridgene import get_masks
from gridgene.mask_properties import MaskAnalysisPipeline, MaskDefinition
from gridgene.binsom import GetBins, GetContour

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
# SYSTEM SPECIFICATIONS
# ==============================================================================
import platform
import multiprocessing

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
    'radius': 80, # same as conv
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
# BENCHMARK FUNCTIONS - CONVOLUTIONAL
# ==============================================================================

def benchmark_cosmx_convolutional(file_csv, params):
    """CosMx convolutional pipeline"""
    metrics = {}
    tracemalloc.start()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 ** 3)

    try:
        # 1. LOAD DATA
        t0 = time.time()
        df_total = pd.read_csv(file_csv)
        df_total['X'] = (round(df_total['x'])).astype(int)
        df_total['Y'] = (round(df_total['y'])).astype(int)
        n_genes = len(df_total['target'].unique())
        height = max(df_total['X'] + 1)
        width = max(df_total['Y'] + 1)
        target_dict_total = {target: index for index, target in enumerate(df_total['target'].unique())}
        array_total = ga.transform_df_to_array(
            df=df_total, target_dict=target_dict_total,
            array_shape=(height, width, len(target_dict_total))
        ).astype(np.int8)
        metrics['load_time'] = time.time() - t0
        metrics['n_transcripts'] = len(df_total)
        metrics['n_genes'] = n_genes

        # 2. TUMOR CONTOURS
        t0 = time.time()
        df_subset_tum, array_subset_tum, _ = ga.get_subset_arrays(
            df_total, array_total, target_dict_total,
            target_list=COSMX_TARGET_TUM, target_col='target'
        )
        CTum = contours.ConvolutionContours(array_subset_tum, contour_name='tum')
        CTum.get_conv_sum(kernel_size=params['kernel_size_tum'], kernel_shape='square')
        CTum.contours_from_sum(
            density_threshold=params['density_th_tum'],
            min_area_threshold=params['min_area_th_tum'],
            directionality='higher'
        )
        metrics['tumor_contour_time'] = time.time() - t0

        # 3. EMPTY CONTOURS
        t0 = time.time()
        CEmpty = contours.ConvolutionContours(array_total, contour_name='empty')
        CEmpty.get_conv_sum(kernel_size=params['kernel_size_empty'], kernel_shape='square')
        CEmpty.contours_from_sum(
            density_threshold=params['density_th_empty'],
            min_area_threshold=params['min_area_th_empty'],
            directionality='lower'
        )
        metrics['empty_contour_time'] = time.time() - t0

        # 4. MASK GENERATION
        t0 = time.time()
        GM = get_masks.GetMasks(image_shape=(height, width))
        mask_empty = GM.create_mask(CEmpty.contours)
        mask_tum = GM.create_mask(CTum.contours)
        mask_tum = GM.fill_holes(mask_tum)
        mask_stroma = GM.subtract_masks(
            np.ones((height, width), dtype=np.uint8), mask_tum, mask_empty
        )
        mask_stroma = GM.filter_binary_mask_by_area(mask_stroma, min_area=700)
        metrics['mask_time'] = time.time() - t0

        # Memory tracking
        current_memory = process.memory_info().rss / (1024 ** 3)
        peak_memory = tracemalloc.get_traced_memory()[1] / (1024 ** 3)
        metrics['peak_memory_gb'] = max(peak_memory, current_memory - initial_memory)

        metrics['total_time'] = (metrics['load_time'] +
                                 metrics['tumor_contour_time'] +
                                 metrics['empty_contour_time'] +
                                 metrics['mask_time'])

    finally:
        tracemalloc.stop()
        gc.collect()

    return metrics


def benchmark_xenium_convolutional(file_csv, params):
    """Xenium convolutional pipeline"""
    metrics = {}
    tracemalloc.start()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 ** 3)

    try:
        # 1. LOAD DATA
        t0 = time.time()
        df_total = pd.read_csv(file_csv)
        df_total = df_total[['x_location', 'y_location', 'feature_name']]
        df_total = df_total.rename(columns={'feature_name': 'target'})
        df_total = df_total[~df_total['target'].str.contains('System|egative')]
        df_total['X'] = df_total['x_location'] - min(df_total['x_location'])
        df_total['Y'] = df_total['y_location'] - min(df_total['y_location'])
        n_genes = len(df_total['target'].unique())
        height = int(max(df_total['X'])) + 1
        width = int(max(df_total['Y'])) + 1
        target_dict_total = {target: index for index, target in enumerate(df_total['target'].unique())}
        array_total = ga.transform_df_to_array(
            df=df_total,
            target_dict=target_dict_total,
            array_shape=(height, width, len(target_dict_total))
        ).astype(np.int8)
        metrics['load_time'] = time.time() - t0
        metrics['n_transcripts'] = len(df_total)
        metrics['n_genes'] = n_genes

        # 2. TUMOR CONTOURS
        t0 = time.time()
        df_subset_tum, array_subset_tum, _ = ga.get_subset_arrays(
            df_total, array_total, target_dict_total,
            target_list=XENIUM_TARGET_TUM, target_col='target'
        )
        CTum = contours.ConvolutionContours(array_subset_tum, contour_name='tum')
        CTum.get_conv_sum(kernel_size=params['kernel_size_tum'], kernel_shape='square')
        CTum.contours_from_sum(
            density_threshold=params['density_th_tum'],
            min_area_threshold=params['min_area_th_tum'],
            directionality='higher'
        )
        metrics['tumor_contour_time'] = time.time() - t0

        # 3. EMPTY CONTOURS
        t0 = time.time()
        CEmpty = contours.ConvolutionContours(array_total, contour_name='empty')
        CEmpty.get_conv_sum(kernel_size=params['kernel_size_empty'], kernel_shape='square')
        CEmpty.contours_from_sum(
            density_threshold=params['density_th_empty'],
            min_area_threshold=params['min_area_th_empty'],
            directionality='lower'
        )
        metrics['empty_contour_time'] = time.time() - t0

        # 4. MASK GENERATION
        t0 = time.time()
        GM = get_masks.GetMasks(image_shape=(height, width))
        mask_empty = GM.create_mask(CEmpty.contours)
        mask_tum = GM.create_mask(CTum.contours)
        mask_tum = GM.fill_holes(mask_tum)
        mask_stroma = GM.subtract_masks(
            np.ones((height, width), dtype=np.uint8), mask_tum, mask_empty
        )
        mask_stroma = GM.filter_binary_mask_by_area(mask_stroma, min_area=700)
        metrics['mask_time'] = time.time() - t0

        # Memory tracking
        current_memory = process.memory_info().rss / (1024 ** 3)
        peak_memory = tracemalloc.get_traced_memory()[1] / (1024 ** 3)
        metrics['peak_memory_gb'] = max(peak_memory, current_memory - initial_memory)

        metrics['total_time'] = (metrics['load_time'] +
                                 metrics['tumor_contour_time'] +
                                 metrics['empty_contour_time'] +
                                 metrics['mask_time'])

    finally:
        tracemalloc.stop()
        gc.collect()

    return metrics


# ==============================================================================
# BENCHMARK FUNCTIONS - KD-TREE
# ==============================================================================

def benchmark_cosmx_kdtree(file_csv, params):
    """CosMx KD-tree pipeline"""
    metrics = {}
    tracemalloc.start()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 ** 3)

    try:
        # 1. LOAD DATA
        t0 = time.time()
        df_total = pd.read_csv(file_csv)
        df_total['X'] = (round(df_total['x'])).astype(int)
        df_total['Y'] = (round(df_total['y'])).astype(int)
        n_genes = len(df_total['target'].unique())
        height = max(df_total['X'] + 1)
        width = max(df_total['Y'] + 1)
        metrics['load_time'] = time.time() - t0
        metrics['n_transcripts'] = len(df_total)
        metrics['n_genes'] = n_genes

        # 2. EMPTY CONTOURS (KD-tree)
        t0 = time.time()
        CEmpty = contours.KDTreeContours(df_total[['target', 'X', 'Y']],
                                         contour_name='empty',
                                         height=height, width=width)
        CEmpty.get_kdt_dist(radius=params['radius'])
        array_total_nei = CEmpty.get_neighbour_array()
        CEmpty.contours_from_neighbors(
            density_threshold=params['density_th_empty'],
            min_area_threshold=params['min_area_th_empty'],
            directionality='lower'
        )
        metrics['empty_contour_time'] = time.time() - t0

        # 3. TUMOR CONTOURS (KD-tree)
        t0 = time.time()
        subset_condition = df_total['target'].isin(COSMX_TARGET_TUM)
        df_subset = df_total[subset_condition]
        Ctum = contours.KDTreeContours(df_subset[['target', 'X', 'Y']],
                                       contour_name='cancer',
                                       height=height, width=width)
        Ctum.get_kdt_dist(radius=params['radius'])
        array_tum_nei = Ctum.get_neighbour_array()
        array_tum_nei = Ctum.interpolate_array()
        Ctum.contours_from_neighbors(
            density_threshold=params['density_th_tum'],
            min_area_threshold=params['min_area_th_tum'],
            directionality='higher'
        )
        metrics['tumor_contour_time'] = time.time() - t0

        # 4. MASK GENERATION
        t0 = time.time()
        GM = get_masks.GetMasks(image_shape=(height, width))
        mask_empty = GM.create_mask(CEmpty.contours)
        mask_tum = GM.create_mask(Ctum.contours)
        mask_stroma = GM.subtract_masks(
            np.ones((height, width), dtype=np.uint8), mask_tum, mask_empty
        )
        mask_stroma = GM.filter_binary_mask_by_area(mask_stroma, min_area=700)
        metrics['mask_time'] = time.time() - t0

        # Memory tracking
        current_memory = process.memory_info().rss / (1024 ** 3)
        peak_memory = tracemalloc.get_traced_memory()[1] / (1024 ** 3)
        metrics['peak_memory_gb'] = max(peak_memory, current_memory - initial_memory)

        metrics['total_time'] = (metrics['load_time'] +
                                 metrics['tumor_contour_time'] +
                                 metrics['empty_contour_time'] +
                                 metrics['mask_time'])

    finally:
        tracemalloc.stop()
        gc.collect()

    return metrics


def benchmark_xenium_kdtree(file_csv, params):
    """Xenium KD-tree pipeline"""
    metrics = {}
    tracemalloc.start()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 ** 3)

    try:
        # 1. LOAD DATA
        t0 = time.time()
        df_total = pd.read_csv(file_csv)
        df_total = df_total[['x_location', 'y_location', 'feature_name']]
        df_total = df_total.rename(columns={'feature_name': 'target'})
        df_total = df_total[~df_total['target'].str.contains('System|egative')]
        df_total['X'] = df_total['x_location'] - min(df_total['x_location'])
        df_total['Y'] = df_total['y_location'] - min(df_total['y_location'])
        n_genes = len(df_total['target'].unique())
        height = int(max(df_total['X'])) + 1
        width = int(max(df_total['Y'])) + 1
        metrics['load_time'] = time.time() - t0
        metrics['n_transcripts'] = len(df_total)
        metrics['n_genes'] = n_genes

        # 2. EMPTY CONTOURS (KD-tree)
        t0 = time.time()
        CEmpty = contours.KDTreeContours(df_total[['target', 'X', 'Y']],
                                         contour_name='empty',
                                         height=height, width=width)
        CEmpty.get_kdt_dist(radius=params['radius'])
        array_total_nei = CEmpty.get_neighbour_array()
        CEmpty.contours_from_neighbors(
            density_threshold=params['density_th_empty'],
            min_area_threshold=params['min_area_th_empty'],
            directionality='lower'
        )
        metrics['empty_contour_time'] = time.time() - t0

        # 3. TUMOR CONTOURS (KD-tree)
        t0 = time.time()
        subset_condition = df_total['target'].isin(XENIUM_TARGET_TUM)
        df_subset = df_total[subset_condition]
        Ctum = contours.KDTreeContours(df_subset[['target', 'X', 'Y']],
                                       contour_name='cancer',
                                       height=height, width=width)
        Ctum.get_kdt_dist(radius=params['radius'])
        array_tum_nei = Ctum.get_neighbour_array()
        array_tum_nei = Ctum.interpolate_array()
        Ctum.contours_from_neighbors(
            density_threshold=params['density_th_tum'],
            min_area_threshold=params['min_area_th_tum'],
            directionality='higher'
        )
        metrics['tumor_contour_time'] = time.time() - t0

        # 4. MASK GENERATION
        t0 = time.time()
        GM = get_masks.GetMasks(image_shape=(height, width))
        mask_empty = GM.create_mask(CEmpty.contours)
        mask_tum = GM.create_mask(Ctum.contours)
        mask_stroma = GM.subtract_masks(
            np.ones((height, width), dtype=np.uint8), mask_tum, mask_empty
        )
        mask_stroma = GM.filter_binary_mask_by_area(mask_stroma, min_area=700)
        metrics['mask_time'] = time.time() - t0

        # Memory tracking
        current_memory = process.memory_info().rss / (1024 ** 3)
        peak_memory = tracemalloc.get_traced_memory()[1] / (1024 ** 3)
        metrics['peak_memory_gb'] = max(peak_memory, current_memory - initial_memory)

        metrics['total_time'] = (metrics['load_time'] +
                                 metrics['tumor_contour_time'] +
                                 metrics['empty_contour_time'] +
                                 metrics['mask_time'])

    finally:
        tracemalloc.stop()
        gc.collect()

    return metrics


# ==============================================================================
# BENCHMARK FUNCTIONS - SOM
# ==============================================================================

def benchmark_cosmx_som(files_list, params):
    """CosMx SOM pipeline - processes multiple files at once"""
    metrics = {}
    tracemalloc.start()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 ** 3)

    try:
        # 1. LOAD DATA
        t0 = time.time()
        df_list = []
        df_name_list = []
        total_transcripts = 0

        for file_csv in files_list:
            df_total = pd.read_csv(file_csv)
            df_total['X'] = (round(df_total['x'])).astype(int)
            df_total['Y'] = (round(df_total['y'])).astype(int)

            file_name = os.path.splitext(os.path.basename(file_csv))[0]
            df_list.append(df_total[['target', 'X', 'Y']])
            df_name_list.append(file_name)
            total_transcripts += len(df_total)

        unique_targets = df_list[0]['target'].unique()
        metrics['load_time'] = time.time() - t0
        metrics['n_transcripts'] = total_transcripts
        metrics['n_genes'] = len(unique_targets)

        # 2. BINNING
        t0 = time.time()
        GB = GetBins(params['bin_size'], unique_targets, logger=None)
        GB.get_bin_cohort(df_list, df_name_list, cohort_name='benchmark')
        GB.preprocess_bin(min_counts=params['min_counts'])
        adata = GB.adata
        metrics['binning_time'] = time.time() - t0

        # 3. SOM CLUSTERING
        t0 = time.time()
        GC = GetContour(adata, logger=None)
        GC.run_som(som_shape=(2, 1), n_iter=5000, sigma=0.5,
                   learning_rate=0.5, random_state=42)
        # som_images = GC.get_som_2d_image(bin_size=params['bin_size'])
        metrics['som_time'] = time.time() - t0

        # Note: No mask generation step - SOM outputs are classification images

        # Memory tracking
        current_memory = process.memory_info().rss / (1024 ** 3)
        peak_memory = tracemalloc.get_traced_memory()[1] / (1024 ** 3)
        metrics['peak_memory_gb'] = max(peak_memory, current_memory - initial_memory)

        metrics['total_time'] = (metrics['load_time'] +
                                 metrics['binning_time'] +
                                 metrics['som_time'])

        # Set empty/NA for fields that don't apply to SOM
        metrics['tumor_contour_time'] = None
        metrics['empty_contour_time'] = None
        metrics['mask_time'] = None

    finally:
        tracemalloc.stop()
        gc.collect()

    return metrics


def benchmark_xenium_som(files_list, params):
    """Xenium SOM pipeline - processes multiple files at once"""
    metrics = {}
    tracemalloc.start()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 ** 3)

    try:
        # 1. LOAD DATA
        t0 = time.time()
        df_list = []
        df_name_list = []
        total_transcripts = 0

        for file_csv in files_list:
            df_total = pd.read_csv(file_csv)
            df_total = df_total[['x_location', 'y_location', 'feature_name']]
            df_total = df_total.rename(columns={'feature_name': 'target'})
            df_total = df_total[~df_total['target'].str.contains('System|egative')]
            df_total['X'] = df_total['x_location'] - min(df_total['x_location'])
            df_total['Y'] = df_total['y_location'] - min(df_total['y_location'])

            file_name = os.path.splitext(os.path.basename(file_csv))[0]
            df_list.append(df_total)
            df_name_list.append(file_name)
            total_transcripts += len(df_total)

        unique_targets = df_list[0]['target'].unique()
        metrics['load_time'] = time.time() - t0
        metrics['n_transcripts'] = total_transcripts
        metrics['n_genes'] = len(unique_targets)

        # 2. BINNING
        t0 = time.time()
        GB = GetBins(params['bin_size'], unique_targets, logger=None)
        GB.get_bin_cohort(df_list, df_name_list, cohort_name='benchmark')
        GB.preprocess_bin(min_counts=params['min_counts'])
        adata = GB.adata
        metrics['binning_time'] = time.time() - t0

        # 3. SOM CLUSTERING
        t0 = time.time()
        GC = GetContour(adata, logger=None)
        GC.run_som(som_shape=(2, 1), n_iter=5000, sigma=0.5,
                   learning_rate=0.5, random_state=42)
        # som_images = GC.get_som_2d_image(bin_size=params['bin_size'])
        metrics['som_time'] = time.time() - t0

        # Note: No mask generation step - SOM outputs are classification images

        # Memory tracking
        current_memory = process.memory_info().rss / (1024 ** 3)
        peak_memory = tracemalloc.get_traced_memory()[1] / (1024 ** 3)
        metrics['peak_memory_gb'] = max(peak_memory, current_memory - initial_memory)

        metrics['total_time'] = (metrics['load_time'] +
                                 metrics['binning_time'] +
                                 metrics['som_time'])

        # Set empty/NA for fields that don't apply to SOM
        metrics['tumor_contour_time'] = None
        metrics['empty_contour_time'] = None
        metrics['mask_time'] = None

    finally:
        tracemalloc.stop()
        gc.collect()

    return metrics


# ==============================================================================
# FULL XENIUM SLIDE BENCHMARK
# ==============================================================================

def benchmark_xenium_half_slide_convolutional():
    """Benchmark third of full Xenium slide - Convolutional method"""
    logger.info("\n" + "=" * 60)
    logger.info("THIRD XENIUM SLIDE - CONVOLUTIONAL")
    logger.info("=" * 60)

    metrics = {}
    tracemalloc.start()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 ** 3)

    try:
        # 1. LOAD DATA - ONLY ONE THIRD OF Y COORDINATES
        t0 = time.time()
        df_total = pd.read_csv(XENIUM_FULL_PATH, compression='gzip')

        # Get one third of the slide
        y_max = df_total['y_location'].max()
        y_third = y_max / 3
        df_total = df_total[df_total['y_location'] <= y_third]

        logger.info(f"  Original transcripts: 89M, Using one-third: {len(df_total):,}")

        df_total = df_total[['x_location', 'y_location', 'feature_name']]
        df_total = df_total.rename(columns={'feature_name': 'target'})
        df_total = df_total[~df_total['target'].str.contains('System|egative')]
        df_total['X'] = df_total['x_location'] - min(df_total['x_location'])
        df_total['Y'] = df_total['y_location'] - min(df_total['y_location'])
        n_genes = len(df_total['target'].unique())
        height = int(max(df_total['X'])) + 1
        width = int(max(df_total['Y'])) + 1

        logger.info(f"  Genes: {n_genes}")
        logger.info(f"  Dimensions: {height} x {width}")
        logger.info(f"  Estimated memory needed: {height * width * n_genes / (1024 ** 3):.1f} GB")

        target_dict_total = {target: index for index, target in enumerate(df_total['target'].unique())}
        array_total = ga.transform_df_to_array(
            df=df_total,
            target_dict=target_dict_total,
            array_shape=(height, width, len(target_dict_total))
        ).astype(np.int8)
        metrics['load_time'] = time.time() - t0
        metrics['n_transcripts'] = len(df_total)
        metrics['n_genes'] = n_genes

        logger.info(f"  Load time: {metrics['load_time']:.2f}s")

        # 2. TUMOR CONTOURS
        t0 = time.time()
        df_subset_tum, array_subset_tum, _ = ga.get_subset_arrays(
            df_total, array_total, target_dict_total,
            target_list=XENIUM_TARGET_TUM, target_col='target'
        )
        CTum = contours.ConvolutionContours(array_subset_tum, contour_name='tum')
        CTum.get_conv_sum(kernel_size=XENIUM_CONV_PARAMS['kernel_size_tum'], kernel_shape='square')
        CTum.contours_from_sum(
            density_threshold=XENIUM_CONV_PARAMS['density_th_tum'],
            min_area_threshold=XENIUM_CONV_PARAMS['min_area_th_tum'],
            directionality='higher'
        )
        metrics['tumor_contour_time'] = time.time() - t0
        logger.info(f"  Tumor contours: {metrics['tumor_contour_time']:.2f}s")

        # 3. EMPTY CONTOURS
        t0 = time.time()
        CEmpty = contours.ConvolutionContours(array_total, contour_name='empty')
        CEmpty.get_conv_sum(kernel_size=XENIUM_CONV_PARAMS['kernel_size_empty'], kernel_shape='square')
        CEmpty.contours_from_sum(
            density_threshold=XENIUM_CONV_PARAMS['density_th_empty'],
            min_area_threshold=XENIUM_CONV_PARAMS['min_area_th_empty'],
            directionality='lower'
        )
        metrics['empty_contour_time'] = time.time() - t0
        logger.info(f"  Empty contours: {metrics['empty_contour_time']:.2f}s")

        # 4. MASK GENERATION
        t0 = time.time()
        GM = get_masks.GetMasks(image_shape=(height, width))
        mask_empty = GM.create_mask(CEmpty.contours)
        mask_tum = GM.create_mask(CTum.contours)
        mask_tum = GM.fill_holes(mask_tum)
        mask_stroma = GM.subtract_masks(
            np.ones((height, width), dtype=np.uint8), mask_tum, mask_empty
        )
        mask_stroma = GM.filter_binary_mask_by_area(mask_stroma, min_area=700)
        metrics['mask_time'] = time.time() - t0
        logger.info(f"  Mask generation: {metrics['mask_time']:.2f}s")

        # Memory tracking
        current_memory = process.memory_info().rss / (1024 ** 3)
        peak_memory = tracemalloc.get_traced_memory()[1] / (1024 ** 3)
        metrics['peak_memory_gb'] = max(peak_memory, current_memory - initial_memory)

        metrics['total_time'] = (metrics['load_time'] +
                                 metrics['tumor_contour_time'] +
                                 metrics['empty_contour_time'] +
                                 metrics['mask_time'])

        metrics['platform'] = 'Xenium'
        metrics['method'] = 'Convolutional'
        metrics['n_cores'] = 'Full Slide'
        metrics['file'] = 'Third of Full Slide'

        logger.info(f"\n  TOTAL TIME: {metrics['total_time']:.2f}s ({metrics['total_time'] / 60:.2f} min)")
        logger.info(f"  PEAK MEMORY: {metrics['peak_memory_gb']:.2f} GB")
        logger.info("\n✓ Third slide convolutional complete")

        return metrics

    except Exception as e:
        logger.error(f"\n✗ Error processing third slide (convolutional): {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        tracemalloc.stop()
        gc.collect()


def benchmark_xenium_half_slide_kdtree():
    """Benchmark third of full Xenium slide - KD-tree method"""
    logger.info("\n" + "=" * 60)
    logger.info("THIRD XENIUM SLIDE - KD-TREE")
    logger.info("=" * 60)

    metrics = {}
    tracemalloc.start()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 ** 3)

    try:
        # 1. LOAD DATA
        t0 = time.time()
        df_total = pd.read_csv(XENIUM_FULL_PATH, compression='gzip')

        y_max = df_total['y_location'].max()
        y_third = y_max / 3
        df_total = df_total[df_total['y_location'] <= y_third]

        logger.info(f"  Using one-third slide: {len(df_total):,} transcripts")

        df_total = df_total[['x_location', 'y_location', 'feature_name']]
        df_total = df_total.rename(columns={'feature_name': 'target'})
        df_total = df_total[~df_total['target'].str.contains('System|egative')]
        df_total['X'] = df_total['x_location'] - min(df_total['x_location'])
        df_total['Y'] = df_total['y_location'] - min(df_total['y_location'])
        n_genes = len(df_total['target'].unique())
        height = int(max(df_total['X'])) + 1
        width = int(max(df_total['Y'])) + 1

        metrics['load_time'] = time.time() - t0
        metrics['n_transcripts'] = len(df_total)
        metrics['n_genes'] = n_genes
        logger.info(f"  Load time: {metrics['load_time']:.2f}s")

        # 2. EMPTY CONTOURS (KD-tree)
        t0 = time.time()
        CEmpty = contours.KDTreeContours(df_total[['target', 'X', 'Y']],
                                         contour_name='empty',
                                         height=height, width=width)
        CEmpty.get_kdt_dist(radius=XENIUM_KDTREE_PARAMS['radius'])
        array_total_nei = CEmpty.get_neighbour_array()
        CEmpty.contours_from_neighbors(
            density_threshold=XENIUM_KDTREE_PARAMS['density_th_empty'],
            min_area_threshold=XENIUM_KDTREE_PARAMS['min_area_th_empty'],
            directionality='lower'
        )
        metrics['empty_contour_time'] = time.time() - t0
        logger.info(f"  Empty contours: {metrics['empty_contour_time']:.2f}s")

        # 3. TUMOR CONTOURS (KD-tree)
        t0 = time.time()
        subset_condition = df_total['target'].isin(XENIUM_TARGET_TUM)
        df_subset = df_total[subset_condition]
        Ctum = contours.KDTreeContours(df_subset[['target', 'X', 'Y']],
                                       contour_name='cancer',
                                       height=height, width=width)
        Ctum.get_kdt_dist(radius=XENIUM_KDTREE_PARAMS['radius'])
        array_tum_nei = Ctum.get_neighbour_array()
        array_tum_nei = Ctum.interpolate_array()
        Ctum.contours_from_neighbors(
            density_threshold=XENIUM_KDTREE_PARAMS['density_th_tum'],
            min_area_threshold=XENIUM_KDTREE_PARAMS['min_area_th_tum'],
            directionality='higher'
        )
        metrics['tumor_contour_time'] = time.time() - t0
        logger.info(f"  Tumor contours: {metrics['tumor_contour_time']:.2f}s")

        # 4. MASK GENERATION
        t0 = time.time()
        GM = get_masks.GetMasks(image_shape=(height, width))
        mask_empty = GM.create_mask(CEmpty.contours)
        mask_tum = GM.create_mask(Ctum.contours)
        mask_stroma = GM.subtract_masks(
            np.ones((height, width), dtype=np.uint8), mask_tum, mask_empty
        )
        mask_stroma = GM.filter_binary_mask_by_area(mask_stroma, min_area=700)
        metrics['mask_time'] = time.time() - t0
        logger.info(f"  Mask generation: {metrics['mask_time']:.2f}s")

        # Memory tracking
        current_memory = process.memory_info().rss / (1024 ** 3)
        peak_memory = tracemalloc.get_traced_memory()[1] / (1024 ** 3)
        metrics['peak_memory_gb'] = max(peak_memory, current_memory - initial_memory)

        metrics['total_time'] = (metrics['load_time'] +
                                 metrics['tumor_contour_time'] +
                                 metrics['empty_contour_time'] +
                                 metrics['mask_time'])

        metrics['platform'] = 'Xenium'
        metrics['method'] = 'KD-tree'
        metrics['n_cores'] = 'Full Slide'
        metrics['file'] = 'Third of Full Slide'

        logger.info(f"\n  TOTAL TIME: {metrics['total_time']:.2f}s ({metrics['total_time'] / 60:.2f} min)")
        logger.info(f"  PEAK MEMORY: {metrics['peak_memory_gb']:.2f} GB")
        logger.info("\n✓ Third slide KD-tree complete")

        return metrics

    except Exception as e:
        logger.error(f"\n✗ Error processing half slide (KD-tree): {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        tracemalloc.stop()
        gc.collect()


def benchmark_xenium_half_slide_som():
    """Benchmark third of full Xenium slide - SOM method"""
    logger.info("\n" + "=" * 60)
    logger.info("THIRD XENIUM SLIDE - SOM")
    logger.info("=" * 60)

    metrics = {}
    tracemalloc.start()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 ** 3)

    try:
        # 1. LOAD DATA
        t0 = time.time()
        df_total = pd.read_csv(XENIUM_FULL_PATH, compression='gzip')

        y_max = df_total['y_location'].max()
        y_third = y_max / 3
        df_total = df_total[df_total['y_location'] <= y_third]

        logger.info(f"  Using one-third slide: {len(df_total):,} transcripts")

        df_total = df_total[['x_location', 'y_location', 'feature_name']]
        df_total = df_total.rename(columns={'feature_name': 'target'})
        df_total = df_total[~df_total['target'].str.contains('System|egative')]
        df_total['X'] = df_total['x_location'] - min(df_total['x_location'])
        df_total['Y'] = df_total['y_location'] - min(df_total['y_location'])

        unique_targets = df_total['target'].unique()
        metrics['load_time'] = time.time() - t0
        metrics['n_transcripts'] = len(df_total)
        metrics['n_genes'] = len(unique_targets)
        logger.info(f"  Load time: {metrics['load_time']:.2f}s")

        # 2. BINNING
        t0 = time.time()
        GB = GetBins(XENIUM_SOM_PARAMS['bin_size'], unique_targets, logger=None)
        GB.get_bin_cohort([df_total], ['third_slide'], cohort_name='third_slide')
        GB.preprocess_bin(min_counts=XENIUM_SOM_PARAMS['min_counts'])
        adata = GB.adata
        metrics['binning_time'] = time.time() - t0
        logger.info(f"  Binning: {metrics['binning_time']:.2f}s")

        # 3. SOM CLUSTERING
        t0 = time.time()
        GC = GetContour(adata, logger=None)
        GC.run_som(som_shape=(2, 1), n_iter=5000, sigma=0.5,
                   learning_rate=0.5, random_state=42)
        # som_images = GC.get_som_2d_image(bin_size=XENIUM_SOM_PARAMS['bin_size'])
        metrics['som_time'] = time.time() - t0
        logger.info(f"  SOM clustering: {metrics['som_time']:.2f}s")

        # Note: No mask generation step - SOM outputs are classification images

        # Memory tracking
        current_memory = process.memory_info().rss / (1024 ** 3)
        peak_memory = tracemalloc.get_traced_memory()[1] / (1024 ** 3)
        metrics['peak_memory_gb'] = max(peak_memory, current_memory - initial_memory)

        metrics['total_time'] = (metrics['load_time'] +
                                 metrics['binning_time'] +
                                 metrics['som_time'])

        # Set empty/NA for fields that don't apply to SOM
        metrics['tumor_contour_time'] = None
        metrics['empty_contour_time'] = None
        metrics['mask_time'] = None

        metrics['platform'] = 'Xenium'
        metrics['method'] = 'SOM'
        metrics['n_cores'] = 'Full Slide'
        metrics['file'] = 'Third of Full Slide'

        logger.info(f"\n  TOTAL TIME: {metrics['total_time']:.2f}s ({metrics['total_time'] / 60:.2f} min)")
        logger.info(f"  PEAK MEMORY: {metrics['peak_memory_gb']:.2f} GB")
        logger.info("\n✓ Third slide SOM complete")

        return metrics

    except Exception as e:
        logger.error(f"\n✗ Error processing half slide (SOM): {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        tracemalloc.stop()
        gc.collect()


# ==============================================================================
# MAIN BENCHMARK LOOP
# ==============================================================================

def run_all_benchmarks():
    """Run all benchmarks and collect results"""
    results = []

    logger.info("\n" + "=" * 60)
    logger.info("STARTING BENCHMARKS")
    logger.info("=" * 60)

    # CosMx Convolutional
    logger.info("\n--- CosMx Platform (Convolutional) ---")
    for n_cores in SCALES:
        logger.info(f"\nProcessing {n_cores} cores...")
        files_subset = cosmx_files[:n_cores]

        for file in tqdm(files_subset, desc=f"CosMx Conv {n_cores} cores"):
            try:
                metrics = benchmark_cosmx_convolutional(file, COSMX_CONV_PARAMS)
                metrics['platform'] = 'CosMx'
                metrics['method'] = 'Convolutional'
                metrics['n_cores'] = n_cores
                metrics['file'] = os.path.basename(file)
                results.append(metrics)
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")

    # CosMx KD-tree
    logger.info("\n--- CosMx Platform (KD-tree) ---")
    for n_cores in SCALES:
        logger.info(f"\nProcessing {n_cores} cores...")
        files_subset = cosmx_files[:n_cores]

        for file in tqdm(files_subset, desc=f"CosMx KD-tree {n_cores} cores"):
            try:
                metrics = benchmark_cosmx_kdtree(file, COSMX_KDTREE_PARAMS)
                metrics['platform'] = 'CosMx'
                metrics['method'] = 'KD-tree'
                metrics['n_cores'] = n_cores
                metrics['file'] = os.path.basename(file)
                results.append(metrics)
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")

    # CosMx SOM
    logger.info("\n--- CosMx Platform (SOM) ---")
    for n_cores in SCALES:
        logger.info(f"\nProcessing {n_cores} cores...")
        files_subset = cosmx_files[:n_cores]

        try:
            metrics = benchmark_cosmx_som(files_subset, COSMX_SOM_PARAMS)
            metrics['platform'] = 'CosMx'
            metrics['method'] = 'SOM'
            metrics['n_cores'] = n_cores
            metrics['file'] = f'{n_cores}_cores_combined'
            results.append(metrics)
        except Exception as e:
            logger.error(f"Error processing SOM {n_cores} cores: {e}")

    # Xenium Convolutional
    logger.info("\n--- Xenium Platform (Convolutional) ---")
    for n_cores in SCALES:
        logger.info(f"\nProcessing {n_cores} cores...")
        files_subset = xenium_files[:n_cores]

        for file in tqdm(files_subset, desc=f"Xenium Conv {n_cores} cores"):
            try:
                metrics = benchmark_xenium_convolutional(file, XENIUM_CONV_PARAMS)
                metrics['platform'] = 'Xenium'
                metrics['method'] = 'Convolutional'
                metrics['n_cores'] = n_cores
                metrics['file'] = os.path.basename(file)
                results.append(metrics)
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")

    # Xenium KD-tree
    logger.info("\n--- Xenium Platform (KD-tree) ---")
    for n_cores in SCALES:
        logger.info(f"\nProcessing {n_cores} cores...")
        files_subset = xenium_files[:n_cores]

        for file in tqdm(files_subset, desc=f"Xenium KD-tree {n_cores} cores"):
            try:
                metrics = benchmark_xenium_kdtree(file, XENIUM_KDTREE_PARAMS)
                metrics['platform'] = 'Xenium'
                metrics['method'] = 'KD-tree'
                metrics['n_cores'] = n_cores
                metrics['file'] = os.path.basename(file)
                results.append(metrics)
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")

    # Xenium SOM
    logger.info("\n--- Xenium Platform (SOM) ---")
    for n_cores in SCALES:
        logger.info(f"\nProcessing {n_cores} cores...")
        files_subset = xenium_files[:n_cores]

        try:
            metrics = benchmark_xenium_som(files_subset, XENIUM_SOM_PARAMS)
            metrics['platform'] = 'Xenium'
            metrics['method'] = 'SOM'
            metrics['n_cores'] = n_cores
            metrics['file'] = f'{n_cores}_cores_combined'
            results.append(metrics)
        except Exception as e:
            logger.error(f"Error processing SOM {n_cores} cores: {e}")

    # Xenium Full Slide (1/3) - Conv and SOM only
    logger.info("\n" + "=" * 60)
    logger.info("XENIUM FULL SLIDE BENCHMARKS (1/3 OF SLIDE)")
    logger.info("=" * 60)

    full_slide_conv = benchmark_xenium_half_slide_convolutional()
    if full_slide_conv:
        results.append(full_slide_conv)

    full_slide_som = benchmark_xenium_half_slide_som()
    if full_slide_som:
        results.append(full_slide_som)

    return results


# ==============================================================================
# SAVE RESULTS
# ==============================================================================

def save_results(results):
    """Save benchmark results to CSV"""
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
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
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