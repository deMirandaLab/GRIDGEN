"""
Metrics for quantitative validation of compartment masks.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def dice_coefficient(mask1, mask2):
    """
    Calculate Dice coefficient between two binary masks.

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Parameters:
    -----------
    mask1, mask2 : np.ndarray
        Binary masks (0 or 1)

    Returns:
    --------
    float : Dice coefficient (0-1, higher is better)
    """
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()

    if mask1.sum() + mask2.sum() == 0:
        return 1.0 if intersection == 0 else 0.0

    return 2.0 * intersection / (mask1.sum() + mask2.sum())


def iou_score(mask1, mask2):
    """
    Calculate Intersection over Union (IoU/Jaccard index).

    IoU = |A ∩ B| / |A ∪ B|

    Parameters:
    -----------
    mask1, mask2 : np.ndarray
        Binary masks

    Returns:
    --------
    float : IoU score (0-1, higher is better)
    """
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


def precision_recall_f1(predicted, reference):
    """
    Calculate precision, recall, and F1 score.

    Parameters:
    -----------
    predicted : np.ndarray
        Predicted binary mask
    reference : np.ndarray
        Reference/ground truth binary mask

    Returns:
    --------
    tuple : (precision, recall, f1)
    """
    pred = predicted.astype(bool)
    ref = reference.astype(bool)

    true_positive = np.logical_and(pred, ref).sum()
    false_positive = np.logical_and(pred, ~ref).sum()
    false_negative = np.logical_and(~pred, ref).sum()

    # Precision
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0

    # Recall
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0

    # F1
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def calculate_overlap_metrics(pred_tumor, pred_stroma, ref_tumor, ref_stroma):
    """
    Calculate all overlap metrics for tumor and stroma masks.

    Parameters:
    -----------
    pred_tumor : np.ndarray
        Predicted tumor mask from GRIDGENE
    pred_stroma : np.ndarray
        Predicted stroma mask from GRIDGENE
    ref_tumor : np.ndarray
        Reference tumor mask from IF/IMC
    ref_stroma : np.ndarray
        Reference stroma mask from IF/IMC

    Returns:
    --------
    dict : All metrics
    """

    metrics = {}

    # Tumor metrics
    metrics['dice_tumor'] = dice_coefficient(pred_tumor, ref_tumor)
    metrics['iou_tumor'] = iou_score(pred_tumor, ref_tumor)
    prec_t, rec_t, f1_t = precision_recall_f1(pred_tumor, ref_tumor)
    metrics['precision_tumor'] = prec_t
    metrics['recall_tumor'] = rec_t
    metrics['f1_tumor'] = f1_t

    # Stroma metrics
    metrics['dice_stroma'] = dice_coefficient(pred_stroma, ref_stroma)
    metrics['iou_stroma'] = iou_score(pred_stroma, ref_stroma)
    prec_s, rec_s, f1_s = precision_recall_f1(pred_stroma, ref_stroma)
    metrics['precision_stroma'] = prec_s
    metrics['recall_stroma'] = rec_s
    metrics['f1_stroma'] = f1_s

    # Overall pixel accuracy
    total_pixels = pred_tumor.size
    correct_pixels = (
            np.logical_and(pred_tumor, ref_tumor).sum() +
            np.logical_and(pred_stroma, ref_stroma).sum() +
            np.logical_and(~pred_tumor.astype(bool) & ~pred_stroma.astype(bool),
                           ~ref_tumor.astype(bool) & ~ref_stroma.astype(bool)).sum()
    )
    metrics['pixel_accuracy'] = correct_pixels / total_pixels

    return metrics


def calculate_enrichment(masks, marker_image, marker_names=None):
    """
    Calculate marker enrichment in each mask region.

    Parameters:
    -----------
    masks : dict
        Dictionary with mask names as keys, binary masks as values
        e.g., {'tumor': tumor_mask, 'stroma': stroma_mask}
    marker_image : np.ndarray
        Image with marker channels (can be multi-channel)
    marker_names : list of str, optional
        Names of markers/channels

    Returns:
    --------
    pd.DataFrame : Enrichment results
    """

    if marker_image.ndim == 2:
        marker_image = marker_image[:, :, np.newaxis]
        n_channels = 1
    else:
        n_channels = marker_image.shape[2]

    if marker_names is None:
        marker_names = [f'channel_{i}' for i in range(n_channels)]

    results = []

    for mask_name, mask in masks.items():
        mask_bool = mask.astype(bool)

        for ch_idx, ch_name in enumerate(marker_names):
            channel = marker_image[:, :, ch_idx]

            # Calculate mean intensity in mask region
            mean_intensity = channel[mask_bool].mean() if mask_bool.sum() > 0 else 0.0

            # Calculate mean intensity outside mask
            mean_outside = channel[~mask_bool].mean() if (~mask_bool).sum() > 0 else 0.0

            # Enrichment ratio
            enrichment = mean_intensity / mean_outside if mean_outside > 0 else 0.0

            results.append({
                'mask': mask_name,
                'marker': ch_name,
                'mean_intensity': mean_intensity,
                'mean_outside': mean_outside,
                'enrichment_ratio': enrichment
            })

    return pd.DataFrame(results)


# Add to scripts/metrics.py

def analyze_disagreement(pred_tumor, pred_stroma, ref_tumor, ref_stroma):
    """
    Analyze regions where predictions disagree with reference.

    Returns:
    --------
    dict : Disagreement statistics and masks
    """

    pred_tumor = pred_tumor.astype(bool)
    pred_stroma = pred_stroma.astype(bool)
    ref_tumor = ref_tumor.astype(bool)
    ref_stroma = ref_stroma.astype(bool)

    # Create confusion regions
    # True Positives
    tp_tumor = np.logical_and(pred_tumor, ref_tumor)
    tp_stroma = np.logical_and(pred_stroma, ref_stroma)

    # False Positives (predicted but not in reference)
    fp_tumor = np.logical_and(pred_tumor, ~ref_tumor)
    fp_stroma = np.logical_and(pred_stroma, ~ref_stroma)

    # False Negatives (in reference but not predicted)
    fn_tumor = np.logical_and(~pred_tumor, ref_tumor)
    fn_stroma = np.logical_and(~pred_stroma, ref_stroma)

    # Calculate areas
    total_pixels = pred_tumor.size

    disagreement = {
        # Tumor
        'tp_tumor_pixels': int(tp_tumor.sum()),
        'fp_tumor_pixels': int(fp_tumor.sum()),
        'fn_tumor_pixels': int(fn_tumor.sum()),
        'tp_tumor_pct': 100 * tp_tumor.sum() / total_pixels,
        'fp_tumor_pct': 100 * fp_tumor.sum() / total_pixels,
        'fn_tumor_pct': 100 * fn_tumor.sum() / total_pixels,

        # Stroma
        'tp_stroma_pixels': int(tp_stroma.sum()),
        'fp_stroma_pixels': int(fp_stroma.sum()),
        'fn_stroma_pixels': int(fn_stroma.sum()),
        'tp_stroma_pct': 100 * tp_stroma.sum() / total_pixels,
        'fp_stroma_pct': 100 * fp_stroma.sum() / total_pixels,
        'fn_stroma_pct': 100 * fn_stroma.sum() / total_pixels,

        # Masks for visualization
        'masks': {
            'tp_tumor': tp_tumor.astype(np.uint8),
            'fp_tumor': fp_tumor.astype(np.uint8),
            'fn_tumor': fn_tumor.astype(np.uint8),
            'tp_stroma': tp_stroma.astype(np.uint8),
            'fp_stroma': fp_stroma.astype(np.uint8),
            'fn_stroma': fn_stroma.astype(np.uint8),
        }
    }

    return disagreement