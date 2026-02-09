# scripts/validation_utils.py
"""
Validation utilities for comparing GRIDGENE masks with reference masks
"""
import numpy as np





def calculate_segmentation_metrics(reference_mask, predicted_mask, mask_name=""):
    """
    Calculate segmentation metrics: Dice, IoU, Precision, Recall, F1, Accuracy

    Parameters
    ----------
    reference_mask : np.ndarray
        Ground truth mask
    predicted_mask : np.ndarray
        Predicted mask (e.g., from GRIDGENE)
    mask_name : str
        Name for this mask (for output)

    Returns
    -------
    dict with metrics: dice, iou, precision, recall, f1, accuracy, TP, FP, FN, TN
    """

    # Flatten
    ref_flat = reference_mask.flatten().astype(bool)
    pred_flat = predicted_mask.flatten().astype(bool)

    # Confusion matrix
    TP = np.sum(ref_flat & pred_flat)
    FP = np.sum(~ref_flat & pred_flat)
    FN = np.sum(ref_flat & ~pred_flat)
    TN = np.sum(~ref_flat & ~pred_flat)

    # Edge case: both empty
    if TP + FP + FN == 0:
        return {
            'mask_name': mask_name,
            'dice': 1.0,
            'iou': 1.0,
            'precision': 1.0,
            'recall': 1.0,
            'f1': 1.0,
            'accuracy': 1.0,
            'TP': int(TP),
            'FP': int(FP),
            'FN': int(FN),
            'TN': int(TN),
            'note': 'Both masks empty'
        }

    # Calculate metrics
    dice = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    return {
        'mask_name': mask_name,
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'TP': int(TP),
        'FP': int(FP),
        'FN': int(FN),
        'TN': int(TN),
        'note': 'OK'
    }