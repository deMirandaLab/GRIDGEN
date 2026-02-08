"""
Preprocessing functions to create reference masks from imaging data.
"""

import numpy as np
import tifffile
from skimage import filters, morphology
from scipy import ndimage


def preprocess_if_image(if_composite_path,
                        red_percentile=99.0,
                        green_percentile=99.0,
                        min_area_tumor=500,
                        min_area_stroma=500,
                        fill_holes=True):
    """
    Create reference masks from IF composite image.

    Parameters:
    -----------
    if_composite_path : str or Path
        Path to the IF composite TIFF file (RGB or multi-channel)
    red_percentile : float
        Percentile threshold for red channel (tumor markers)
    green_percentile : float
        Percentile threshold for green channel (stroma markers)
    min_area_tumor : int
        Minimum area (pixels) for tumor regions
    min_area_stroma : int
        Minimum area (pixels) for stroma regions
    fill_holes : bool
        Whether to fill holes in masks

    Returns:
    --------
    tumor_mask : np.ndarray
        Binary mask for tumor regions
    stroma_mask : np.ndarray
        Binary mask for stroma regions
    metadata : dict
        Processing metadata (thresholds used, etc.)
    """

    # Load IF image
    image = tifffile.imread(if_composite_path)

    # Normalize if needed (assuming 0-1 range, convert to 0-255)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    # Extract channels (assuming RGB or similar multi-channel)
    if image.ndim == 3 and image.shape[2] >= 3:
        red_channel = image[:, :, 0]  # Tumor markers
        green_channel = image[:, :, 1]  # Stroma markers
    else:
        raise ValueError(f"Expected 3-channel image, got shape {image.shape}")

    # Calculate thresholds based on percentiles
    red_threshold = np.percentile(red_channel[red_channel > 0], red_percentile)
    green_threshold = np.percentile(green_channel[green_channel > 0], green_percentile)

    # Create binary masks
    tumor_mask = red_channel > red_threshold
    stroma_mask = green_channel > green_threshold

    # Optional: fill holes
    if fill_holes:
        tumor_mask = ndimage.binary_fill_holes(tumor_mask)
        stroma_mask = ndimage.binary_fill_holes(stroma_mask)

    # Remove small objects
    tumor_mask = morphology.remove_small_objects(tumor_mask, min_size=min_area_tumor)
    stroma_mask = morphology.remove_small_objects(stroma_mask, min_size=min_area_stroma)

    # Convert to uint8
    tumor_mask = tumor_mask.astype(np.uint8)
    stroma_mask = stroma_mask.astype(np.uint8)

    # Metadata
    metadata = {
        'red_threshold': float(red_threshold),
        'green_threshold': float(green_threshold),
        'red_percentile': red_percentile,
        'green_percentile': green_percentile,
        'tumor_pixels': int(tumor_mask.sum()),
        'stroma_pixels': int(stroma_mask.sum()),
    }

    return tumor_mask, stroma_mask, metadata


def register_masks_to_image(gridgene_masks, reference_image_shape,
                            pad_mode='constant', transpose=True):
    """
    Register GRIDGENE masks to match IF image dimensions.

    Parameters:
    -----------
    gridgene_masks : tuple of np.ndarray
        (tumor_mask, stroma_mask) from GRIDGENE
    reference_image_shape : tuple
        Shape of the reference IF image (height, width) or (height, width, channels)
    pad_mode : str
        Padding mode for np.pad
    transpose : bool
        Whether to transpose masks (sometimes needed for coordinate system alignment)

    Returns:
    --------
    registered_tumor : np.ndarray
        Tumor mask registered to image
    registered_stroma : np.ndarray
        Stroma mask registered to image
    """

    tumor_mask, stroma_mask = gridgene_masks

    # Get reference dimensions (ignore channel dimension if present)
    if len(reference_image_shape) == 3:
        ref_height, ref_width = reference_image_shape[:2]
    else:
        ref_height, ref_width = reference_image_shape

    # Calculate padding needed
    pad_y = (ref_height - tumor_mask.shape[0]) // 2
    pad_x = (ref_width - tumor_mask.shape[1]) // 2

    # Pad masks
    tumor_registered = np.pad(tumor_mask,
                              ((pad_y, pad_y), (pad_x, pad_x)),
                              mode=pad_mode, constant_values=0)
    stroma_registered = np.pad(stroma_mask,
                               ((pad_y, pad_y), (pad_x, pad_x)),
                               mode=pad_mode, constant_values=0)

    # Transpose if needed
    if transpose:
        tumor_registered = np.transpose(tumor_registered)
        stroma_registered = np.transpose(stroma_registered)

    # Ensure exact match (crop if padding was uneven)
    tumor_registered = tumor_registered[:ref_height, :ref_width]
    stroma_registered = stroma_registered[:ref_height, :ref_width]

    return tumor_registered, stroma_registered