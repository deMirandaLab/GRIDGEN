"""
Single Benchmark Worker
Runs ONE benchmark in isolation, prints JSON result, exits
This ensures complete memory cleanup between benchmarks
"""

import sys
import json
import time
import psutil
import pandas as pd
import numpy as np
import tracemalloc
from pathlib import Path

# Add GRIDGEN to path
gridgene_root = Path.cwd().parent.parent
sys.path.insert(0, str(gridgene_root))

from gridgene import get_arrays as ga
from gridgene import contours
from gridgene import get_masks
from gridgene.binsom import GetBins, GetContour


# ==============================================================================
# BENCHMARK FUNCTIONS
# ==============================================================================

def benchmark_cosmx_convolutional(file_csv, params):
    """CosMx convolutional pipeline"""
    metrics = {}
    tracemalloc.start()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 ** 3)

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
        target_list=params['target_tum'], target_col='target'
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
    tracemalloc.stop()

    metrics['total_time'] = (metrics['load_time'] +
                             metrics['tumor_contour_time'] +
                             metrics['empty_contour_time'] +
                             metrics['mask_time'])

    return metrics


def benchmark_xenium_convolutional(file_csv, params):
    """Xenium convolutional pipeline"""
    metrics = {}
    tracemalloc.start()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 ** 3)

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
        target_list=params['target_tum'], target_col='target'
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
    tracemalloc.stop()

    metrics['total_time'] = (metrics['load_time'] +
                             metrics['tumor_contour_time'] +
                             metrics['empty_contour_time'] +
                             metrics['mask_time'])

    return metrics


def benchmark_cosmx_kdtree(file_csv, params):
    """CosMx KD-tree pipeline"""
    metrics = {}
    tracemalloc.start()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 ** 3)

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

    # 2. EMPTY CONTOURS
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

    # 3. TUMOR CONTOURS
    t0 = time.time()
    subset_condition = df_total['target'].isin(params['target_tum'])
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
    tracemalloc.stop()

    metrics['total_time'] = (metrics['load_time'] +
                             metrics['tumor_contour_time'] +
                             metrics['empty_contour_time'] +
                             metrics['mask_time'])

    return metrics


def benchmark_xenium_kdtree(file_csv, params):
    """Xenium KD-tree pipeline"""
    metrics = {}
    tracemalloc.start()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 ** 3)

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

    # 2. EMPTY CONTOURS
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

    # 3. TUMOR CONTOURS
    t0 = time.time()
    subset_condition = df_total['target'].isin(params['target_tum'])
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
    tracemalloc.stop()

    metrics['total_time'] = (metrics['load_time'] +
                             metrics['tumor_contour_time'] +
                             metrics['empty_contour_time'] +
                             metrics['mask_time'])

    return metrics


def benchmark_cosmx_som(files_list, params):
    """CosMx SOM pipeline"""
    metrics = {}
    tracemalloc.start()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 ** 3)

    # 1. LOAD DATA
    t0 = time.time()
    df_list = []
    df_name_list = []
    total_transcripts = 0

    for file_csv in files_list:
        df_total = pd.read_csv(file_csv)
        df_total['X'] = (round(df_total['x'])).astype(int)
        df_total['Y'] = (round(df_total['y'])).astype(int)
        df_list.append(df_total[['target', 'X', 'Y']])

        file_name = Path(file_csv).stem
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

    # 3. SOM
    t0 = time.time()
    GC = GetContour(adata, logger=None)
    GC.run_som(som_shape=(2, 1), n_iter=5000, sigma=0.5,
               learning_rate=0.5, random_state=42)
    metrics['som_time'] = time.time() - t0

    # Memory tracking
    current_memory = process.memory_info().rss / (1024 ** 3)
    peak_memory = tracemalloc.get_traced_memory()[1] / (1024 ** 3)
    metrics['peak_memory_gb'] = max(peak_memory, current_memory - initial_memory)
    tracemalloc.stop()

    metrics['total_time'] = (metrics['load_time'] +
                             metrics['binning_time'] +
                             metrics['som_time'])
    metrics['tumor_contour_time'] = None
    metrics['empty_contour_time'] = None
    metrics['mask_time'] = None

    return metrics


def benchmark_xenium_som(files_list, params):
    """Xenium SOM pipeline"""
    metrics = {}
    tracemalloc.start()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 ** 3)

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
        df_list.append(df_total[['target', 'X', 'Y']])

        file_name = Path(file_csv).stem
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

    # 3. SOM
    t0 = time.time()
    GC = GetContour(adata, logger=None)
    GC.run_som(som_shape=(2, 1), n_iter=5000, sigma=0.5,
               learning_rate=0.5, random_state=42)
    metrics['som_time'] = time.time() - t0

    # Memory tracking
    current_memory = process.memory_info().rss / (1024 ** 3)
    peak_memory = tracemalloc.get_traced_memory()[1] / (1024 ** 3)
    metrics['peak_memory_gb'] = max(peak_memory, current_memory - initial_memory)
    tracemalloc.stop()

    metrics['total_time'] = (metrics['load_time'] +
                             metrics['binning_time'] +
                             metrics['som_time'])
    metrics['tumor_contour_time'] = None
    metrics['empty_contour_time'] = None
    metrics['mask_time'] = None

    return metrics


def benchmark_xenium_fullslide_conv(full_path, params):
    """Xenium full slide (1/3) - Convolutional"""
    metrics = {}
    tracemalloc.start()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 ** 3)

    # 1. LOAD DATA - ONE THIRD
    t0 = time.time()
    df_total = pd.read_csv(full_path, compression='gzip')
    y_max = df_total['y_location'].max()
    y_third = y_max / 3
    df_total = df_total[df_total['y_location'] <= y_third]

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
        target_list=params['target_tum'], target_col='target'
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
    tracemalloc.stop()

    metrics['total_time'] = (metrics['load_time'] +
                             metrics['tumor_contour_time'] +
                             metrics['empty_contour_time'] +
                             metrics['mask_time'])

    return metrics


def benchmark_xenium_fullslide_som(full_path, params):
    """Xenium full slide (1/3) - SOM"""
    metrics = {}
    tracemalloc.start()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 ** 3)

    # 1. LOAD DATA
    t0 = time.time()
    df_total = pd.read_csv(full_path, compression='gzip')
    y_max = df_total['y_location'].max()
    y_third = y_max / 3
    df_total = df_total[df_total['y_location'] <= y_third]

    df_total = df_total[['x_location', 'y_location', 'feature_name']]
    df_total = df_total.rename(columns={'feature_name': 'target'})
    df_total = df_total[~df_total['target'].str.contains('System|egative')]
    df_total['X'] = df_total['x_location'] - min(df_total['x_location'])
    df_total['Y'] = df_total['y_location'] - min(df_total['y_location'])

    unique_targets = df_total['target'].unique()
    metrics['load_time'] = time.time() - t0
    metrics['n_transcripts'] = len(df_total)
    metrics['n_genes'] = len(unique_targets)

    # 2. BINNING
    t0 = time.time()
    GB = GetBins(params['bin_size'], unique_targets, logger=None)
    GB.get_bin_cohort([df_total[['target', 'X', 'Y']]], ['third_slide'], cohort_name='third_slide')
    GB.preprocess_bin(min_counts=params['min_counts'])
    adata = GB.adata
    metrics['binning_time'] = time.time() - t0

    # 3. SOM
    t0 = time.time()
    GC = GetContour(adata, logger=None)
    GC.run_som(som_shape=(2, 1), n_iter=5000, sigma=0.5,
               learning_rate=0.5, random_state=42)
    metrics['som_time'] = time.time() - t0

    # Memory tracking
    current_memory = process.memory_info().rss / (1024 ** 3)
    peak_memory = tracemalloc.get_traced_memory()[1] / (1024 ** 3)
    metrics['peak_memory_gb'] = max(peak_memory, current_memory - initial_memory)
    tracemalloc.stop()

    metrics['total_time'] = (metrics['load_time'] +
                             metrics['binning_time'] +
                             metrics['som_time'])
    metrics['tumor_contour_time'] = None
    metrics['empty_contour_time'] = None
    metrics['mask_time'] = None

    return metrics


def benchmark_som(files_list, params, platform):
    """SOM pipeline for multiple files"""
    metrics = {}
    tracemalloc.start()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 ** 3)

    # 1. LOAD DATA
    t0 = time.time()
    df_list = []
    df_name_list = []
    total_transcripts = 0

    for file_csv in files_list:
        if platform == 'CosMx':
            df_total = pd.read_csv(file_csv)
            df_total['X'] = (round(df_total['x'])).astype(int)
            df_total['Y'] = (round(df_total['y'])).astype(int)
            df_list.append(df_total[['target', 'X', 'Y']])
        else:  # Xenium
            df_total = pd.read_csv(file_csv)
            df_total = df_total[['x_location', 'y_location', 'feature_name']]
            df_total = df_total.rename(columns={'feature_name': 'target'})
            df_total = df_total[~df_total['target'].str.contains('System|egative')]
            df_total['X'] = df_total['x_location'] - min(df_total['x_location'])
            df_total['Y'] = df_total['y_location'] - min(df_total['y_location'])
            df_list.append(df_total[['target', 'X', 'Y']])

        file_name = Path(file_csv).stem
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

    # 3. SOM
    t0 = time.time()
    GC = GetContour(adata, logger=None)
    GC.run_som(som_shape=(2, 1), n_iter=5000, sigma=0.5,
               learning_rate=0.5, random_state=42)
    metrics['som_time'] = time.time() - t0

    # Memory tracking
    current_memory = process.memory_info().rss / (1024 ** 3)
    peak_memory = tracemalloc.get_traced_memory()[1] / (1024 ** 3)
    metrics['peak_memory_gb'] = max(peak_memory, current_memory - initial_memory)
    tracemalloc.stop()

    metrics['total_time'] = (metrics['load_time'] +
                             metrics['binning_time'] +
                             metrics['som_time'])
    metrics['tumor_contour_time'] = None
    metrics['empty_contour_time'] = None
    metrics['mask_time'] = None

    return metrics


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import traceback as tb

    # Read config from stdin
    config = json.loads(sys.stdin.read())

    benchmark_type = config['benchmark_type']
    params = config['params']

    try:
        if benchmark_type == 'cosmx_conv':
            metrics = benchmark_cosmx_convolutional(config['file'], params)
        elif benchmark_type == 'cosmx_kdtree':
            metrics = benchmark_cosmx_kdtree(config['file'], params)
        elif benchmark_type == 'cosmx_som':
            metrics = benchmark_cosmx_som(config['files'], params)
        elif benchmark_type == 'xenium_conv':
            metrics = benchmark_xenium_convolutional(config['file'], params)
        elif benchmark_type == 'xenium_kdtree':
            metrics = benchmark_xenium_kdtree(config['file'], params)
        elif benchmark_type == 'xenium_som':
            metrics = benchmark_xenium_som(config['files'], params)
        elif benchmark_type == 'xenium_fullslide_conv':
            metrics = benchmark_xenium_fullslide_conv(config['full_path'], params)
        elif benchmark_type == 'xenium_fullslide_som':
            metrics = benchmark_xenium_fullslide_som(config['full_path'], params)
        else:
            raise ValueError(f"Unknown benchmark type: {benchmark_type}")

        # Print JSON result to stdout
        print(json.dumps(metrics))
        sys.exit(0)

    except Exception as e:
        error = {'error': str(e), 'traceback': tb.format_exc()}
        print(json.dumps(error), file=sys.stderr)
        sys.exit(1)