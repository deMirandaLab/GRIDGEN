import logging
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon
from typing import Dict, List, Tuple, Union
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from gridgen.logger import get_logger
from typing import Optional, Tuple, Dict, Any, List
from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from scipy.ndimage import distance_transform_edt
import cv2
import numpy as np
from shapely.geometry import Polygon, box

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

class GetMasks:
    def __init__(self, logger: Optional[logging.Logger] = None, image_shape: Optional[Tuple[int, int]] = None):
        """
        Initialize the GetMasks class.

        :param logger: Logger instance for logging messages. If None, a default logger is configured.
        :param image_shape: Tuple representing the shape of the image (height, width).
        """

        self.image_shape = image_shape
        self.height = self.image_shape[0] if self.image_shape is not None else None
        self.width = self.image_shape[1] if self.image_shape is not None else None
        self.logger = logger or get_logger(f'{__name__}.{"GetMasks"}')
        self.logger.info("Initialized GetMasks")

    def filter_binary_mask_by_area(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        """
        Removes small connected components from a binary mask.

        :param mask: Binary mask (0 or 1).
        :param min_area: Minimum area threshold.
        :return: Filtered binary mask.
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)

        output_mask = np.zeros_like(mask, dtype=np.uint8)
        for i in range(1, num_labels):  # skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                output_mask[labels == i] = 1
        return output_mask

    def filter_labeled_mask_by_area(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        """
        Filters a labeled mask, keeping only components >= min_area. Preserves label IDs.

        :param mask: Input labeled mask (integers, not binary).
        :param min_area: Minimum area threshold.
        :param logger: Optional logger for debug info.
        :return: Filtered labeled mask.
        """
        mask = mask.astype(np.int32)
        unique_labels, counts = np.unique(mask, return_counts=True)
        labels_to_keep = unique_labels[(counts >= min_area) & (unique_labels != 0)]

        filtered_mask = np.zeros_like(mask, dtype=np.int32)
        for label in labels_to_keep:
            filtered_mask[mask == label] = label

        # if logger:
        self.logger.info(f'Filtered labeled mask by area >= {min_area}, kept {len(labels_to_keep)} components.')

        return filtered_mask

    def create_mask(self, contours: List[np.ndarray]) -> np.ndarray:
        """
        Creates a binary mask from given contours.

        :param contours: List of contours (numpy arrays).
        :return: Binary mask as a numpy array.
        """
        if self.height is None or self.width is None:
            raise ValueError("Image shape must be defined to create mask.")
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, color=1, thickness=cv2.FILLED)
        return mask

    def fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """
        Fills holes inside a binary mask using contours.

        :param mask: Binary mask.
        :return: Hole-filled binary mask.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(mask)
        cv2.drawContours(filled_mask, contours, -1, color=1, thickness=cv2.FILLED)
        return filled_mask


    def apply_morphology(self, mask: np.ndarray, operation: str = "open", kernel_size: int = 3) -> np.ndarray:
        """
        Applies morphological operations to refine binary masks.

        :param mask: Binary mask to be processed.
        :param operation: Type of morphological operation: "open", "close", "erode", or "dilate".
        :param kernel_size: Size of the structuring element.
        :return: Processed binary mask.
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        if operation == "open":
            result = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        elif operation == "close":
            result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        elif operation == "erode":
            result = cv2.erode(mask, kernel, iterations=1)
        elif operation == "dilate":
            result = cv2.dilate(mask, kernel, iterations=1)
        else:
            self.logger.warning(f"Unknown morphological operation '{operation}', returning original mask.")
            result = mask

        self.logger.info(f'Applied morphology operation "{operation}" with kernel size {kernel_size}.')
        return result

    def subtract_masks(self, base_mask: np.ndarray, *masks: np.ndarray) -> np.ndarray:
        """
        Subtracts multiple masks from a base mask.

        :param base_mask: The initial binary mask.
        :param masks: One or more masks to subtract.
        :return: Resulting mask after subtraction.
        """
        result_mask = base_mask.copy()
        for mask in masks:
            result_mask = cv2.subtract(result_mask, mask)
        self.logger.info(f'Subtracted masks from base mask.')
        return result_mask

    def save_masks_npy(self, mask: np.ndarray, save_path: str) -> None:
        """
        Save the mask as a .npy file.

        :param mask: Mask to be saved.
        :param save_path: Path where the mask will be saved.
        """
        np.save(save_path, mask)
        self.logger.info(f'Mask saved at {save_path}')

    def save_masks(self, mask: np.ndarray, path: str) -> None:
        """
        Save the mask as an image file.

        :param mask: Mask to be saved.
        :param path: Path where the mask will be saved.
        """
        cv2.imwrite(path, mask * 255)
        self.logger.info(f'Mask saved at {path}')

    def plot_masks(
        self,
        masks: List[np.ndarray],
        mask_names: List[str],
        background_color: Tuple[int, int, int] = (0, 0, 0),
        mask_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
        path: Optional[str] = None,
        show: bool = True,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (10, 10)
    ) -> None:
        """
        Plots the given masks with their corresponding names.

        :param masks: List of masks to plot.
        :param mask_names: List of names corresponding to the masks.
        :param background_color: Tuple to use for areas not assigned in any mask.
        :param mask_colors: Dictionary mapping mask names to colors.
        :param path: Directory path where the plots will be saved.
        :param show: Whether to display the plot.
        :param ax: Matplotlib axis object. If None, a new figure will be created.
        :param figsize: Tuple representing the figure size (width, height) in inches.
        """
        if len(masks) != len(mask_names):
            self.logger.error('The number of masks and mask names must be the same.')
            return

        # Create a background image filled with the background color
        background = np.full((self.height, self.width, 3), background_color)

        # Create a list to store the patches for the legend
        legend_patches = []

        # Choose a colormap based on the number of masks
        colormap = cm.get_cmap('tab10') if len(masks) <= 10 else cm.get_cmap('tab20')

        # Add each mask to the background image
        for i, (mask, mask_name) in enumerate(zip(masks, mask_names)):
            # Choose a color for the mask
            if mask_colors and mask_name in mask_colors:
                mask_color = np.array(mask_colors[mask_name])
            else:
                mask_color = (np.array(colormap(i % colormap.N)[:3]) * 255).astype(int)
            # Apply the mask color to the mask image
            background[mask!=0] = mask_color

            # Create a patch for the legend
            legend_patches.append(mpatches.Patch(color=mask_color / 255, label=mask_name))

        # Flip the mask horizontally and rotate 90 degrees clockwise
        background = np.fliplr(background)
        background = np.rot90(background, k=1)
        created_fig = False
        if ax is None:
            created_fig = True
            fig, ax = plt.subplots(figsize=figsize)

        # Plot the background image
        ax.imshow(background, origin='lower')
        ax.set_axis_off()

        # Add legend
        ax.legend(
            handles=legend_patches,
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            bbox_transform=ax.transAxes
        )

        # Save the image if path is provided
        if path is not None:
            save_path = os.path.join(
                path,
                f'masks_{"_".join(mask_names).replace(" ", "").lower()}.png'
            )
            plt.savefig(save_path, dpi=1000, bbox_inches='tight')
            self.logger.info(f'Plot saved at {save_path}')

        # Show the plot if required
        if show:
            plt.show()
            plt.close()

        # Close the figure if it was created within this function
        if created_fig:
            plt.close(fig)

# CancerStromaInterfaceanalysis
class ConstrainedMaskExpansion(GetMasks):
    """    Class for expanding a seed mask with constraints, generating binary, labeled, and referenced expansions.
    """
    def __init__(
        self,
        seed_mask: np.ndarray,
        constraint_mask: Optional[np.ndarray] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if seed_mask is None:
            raise ValueError("Seed mask cannot be None.")

        self.seed_mask_raw = seed_mask.astype(np.uint8)
        self.seed_mask = label(self.seed_mask_raw)  # connected components
        self.constraint_mask = (
            constraint_mask.astype(np.uint8)
            if constraint_mask is not None
            else np.ones_like(seed_mask, dtype=np.uint8)
        )

        image_shape = self.seed_mask.shape
        super().__init__(logger=logger, image_shape=image_shape)

        self.binary_expansions: Dict[str, np.ndarray] = {}
        self.labeled_expansions: Dict[str, np.ndarray] = {}
        self.referenced_expansions: Dict[str, np.ndarray] = {}

    def expand_mask(
        self,
        expansion_pixels: List[int],
        min_area: Optional[int] = None,
        restrict_to_limit: bool = True,
    ) -> None:
        """
        Expands the seed mask outward by specified pixel distances and stores binary, labeled,
        and label-propagated expansion masks.

        :param expansion_pixels: List of expansion distances (in pixels) from the seed mask.
        :param min_area: Optional minimum area for keeping connected components in each expansion ring.
        :param restrict_to_limit: If True, expansion is limited to the constraint mask.
        """
        sorted_dists = sorted(expansion_pixels)
        dist_map = distance_transform_edt(self.seed_mask == 0)

        previous_mask = np.zeros_like(self.seed_mask, dtype=bool)

        for dist in sorted_dists:
            if dist == sorted_dists[0]:
                ring = (dist_map <= dist) & (self.seed_mask == 0)
            else:
                prev_dist = sorted_dists[sorted_dists.index(dist) - 1]
                ring = (dist_map <= dist) & (dist_map > prev_dist) & (self.seed_mask == 0)

            if restrict_to_limit:
                ring &= self.constraint_mask.astype(bool)

            ring &= ~previous_mask

            if min_area:
                ring = self.filter_binary_mask_by_area(ring.astype(np.uint8), min_area).astype(bool)

            previous_mask |= ring

            # Store binary mask
            self.binary_expansions[f"expansion_{dist}"] = ring.astype(np.uint8)

            # Store labeled components using skimage
            self.labeled_expansions[f"expansion_{dist}"] = label(ring.astype(np.uint8))

            # Store label-referenced expansion using seed_mask
            referenced = self.propagate_labels(self.seed_mask, ring)
            self.referenced_expansions[f"expansion_{dist}"] = referenced

        self.binary_expansions["seed_mask"] = (self.seed_mask > 0).astype(np.uint8)
        self.labeled_expansions["seed_mask"] = self.seed_mask.copy()
        self.referenced_expansions["seed_mask"] = self.seed_mask.copy()

        constraint_remaining = (self.constraint_mask.astype(bool) & ~previous_mask).astype(np.uint8)
        self.binary_expansions["constraint_remaining"] = constraint_remaining
        self.labeled_expansions["constraint_remaining"] = np.zeros_like(self.seed_mask, dtype=np.int32)
        self.referenced_expansions["constraint_remaining"] = np.zeros_like(self.seed_mask, dtype=np.int32)


    def propagate_labels(self, seed_labeled: np.ndarray, expansion_mask: np.ndarray) -> np.ndarray:
        """
        Propagates labels from the seed labeled mask into the expansion region using morphological dilation.

        :param seed_labeled: Labeled seed mask (non-zero values represent different components).
        :param expansion_mask: Binary mask representing the region where labels should propagate.
        :return: Labeled mask with propagated labels in the expansion area.
        """
        output = np.zeros_like(seed_labeled, dtype=np.int32)
        output[seed_labeled > 0] = seed_labeled[seed_labeled > 0]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        expansion_mask = expansion_mask.astype(bool)
        iteration = 0

        while True:
            iteration += 1
            prev = output.copy()

            mask_to_fill = (output == 0) & expansion_mask

            # OpenCV only supports certain dtypes for dilation — use float32 safely
            dilated = cv2.dilate(output.astype(np.float32), kernel)
            dilated = dilated.astype(np.int32)

            output[mask_to_fill] = dilated[mask_to_fill]

            if np.array_equal(output, prev):
                break
            if iteration > 1000:
                if self.logger:
                    self.logger.warning("Label propagation exceeded 1000 iterations.")
                break

        return output


class SingleClassObjectAnalysis(GetMasks):
    """
    Analyze and expand a single binary object mask using distance-based ring expansion.

    This class computes concentric ring-based expansions of a binary mask,
    assigns unique labels to each expanded region, and tracks mask lineage
    through label propagation.

    Attributes:
        mask (np.ndarray): Binary mask of the object to be expanded.
        expansion_distances (List[int]): List of expansion radii in pixels.
        labelled_mask (np.ndarray): Resulting labeled mask with original and expanded areas.
        binary_masks (Dict[str, np.ndarray]): Dictionary of binary masks keyed by expansion distance.
        labelled_masks (Dict[str, np.ndarray]): Dictionary of labeled masks keyed by expansion distance.
        reference_masks (Dict[str, np.ndarray]): Masks encoding reference to original object.
    """

    def __init__(
        self,
        get_masks_instance: GetMasks,
        contours_object: List[np.ndarray],
        contour_name: str = ""
    ) -> None:
        """
        Initialize SingleClassObjectAnalysis with contour data and a GetMasks utility instance.

        :param get_masks_instance: Instance of GetMasks providing access to shape and filtering methods.
        :param contours_object: List of contours representing the object.
        :param contour_name: Optional name identifier for the object.
        """

        self.get_masks_instance = get_masks_instance
        self.height = get_masks_instance.height
        self.width = get_masks_instance.width
        self.logger = get_masks_instance.logger

        self.mask_object_SA: Optional[np.ndarray] = None
        self.binary_expansions: Dict[str, np.ndarray] = {}
        self.labeled_expansions: Dict[str, np.ndarray] = {}
        self.referenced_expansions: Dict[str, np.ndarray] = {}
        self.contours_object = contours_object
        self.contour_name = contour_name

    def get_mask_objects(
        self,
        exclude_masks: Optional[List[np.ndarray]] = None,
        filter_area: Optional[int] = None
    ) -> None:
        """
        Generate binary mask from object contours, with optional subtraction of other masks
        and area-based filtering.

        :param exclude_masks: List of masks to subtract from the generated object mask.
        :param filter_area: Minimum area threshold to retain components in the object mask.
        """
        mask_object = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.drawContours(mask_object, self.contours_object, -1, color=1, thickness=cv2.FILLED)

        if exclude_masks:
            for mask in exclude_masks:
                mask_object = cv2.subtract(mask_object, mask)

        if filter_area is not None:
            self.logger.info(f"Filtering object mask by area: {filter_area}")
            mask_object = self.get_masks_instance.filter_mask_by_area(mask_object, min_area=filter_area)

        self.mask_object_SA = mask_object
        self.logger.info("Mask for objects created.")

    def get_objects_expansion(
        self,
        expansions_pixels: Optional[List[int]] = None,
        filter_area: Optional[int] = None
    ) -> None:
        """
        Expand the object mask using distance-based rings and optionally filter each ring
        by minimum area. Generates binary, labeled, and propagated-label expansion masks.

        :param expansions_pixels: List of pixel distances for expansion.
        :param filter_area: Minimum area threshold to retain components in each expansion ring.

        """
        if self.mask_object_SA is None:
            self.logger.error("No object mask to expand.")
            return

        if expansions_pixels is None:
            expansions_pixels = []

        seed_mask = label(self.mask_object_SA)
        dist_map = distance_transform_edt(seed_mask == 0)
        previous_mask = np.zeros_like(seed_mask, dtype=bool)

        for i, dist in enumerate(sorted(expansions_pixels)):
            if i == 0:
                ring = (dist_map <= dist) & (seed_mask == 0)
            else:
                prev_dist = sorted(expansions_pixels)[i - 1]
                ring = (dist_map <= dist) & (dist_map > prev_dist) & (seed_mask == 0)

            ring &= ~previous_mask
            if filter_area:
                ring = self.get_masks_instance.filter_binary_mask_by_area(ring.astype(np.uint8), filter_area).astype(bool)

            previous_mask |= ring

            key = f"expansion_{dist}"
            self.binary_expansions[key] = ring.astype(np.uint8)
            self.labeled_expansions[key] = label(ring.astype(np.uint8))
            self.referenced_expansions[key] = self.propagate_labels(seed_mask, ring)

        # Store the base seed info
        self.binary_expansions["seed_mask"] = (seed_mask > 0).astype(np.uint8)
        self.labeled_expansions["seed_mask"] = seed_mask.copy()
        self.referenced_expansions["seed_mask"] = seed_mask.copy()

    def propagate_labels(self, seed_labeled: np.ndarray, expansion_mask: np.ndarray) -> np.ndarray:
        """
        Propagate labeled regions from a seed mask into the expansion area using iterative dilation.

        :param seed_labeled: Input labeled mask where each connected component has a unique integer label.
        :param expansion_mask: Binary mask indicating the region where labels should expand.
        :return: A labeled mask with labels propagated into the expansion region.
        """
        output = np.zeros_like(seed_labeled, dtype=np.int32)
        output[seed_labeled > 0] = seed_labeled[seed_labeled > 0]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        expansion_mask = expansion_mask.astype(bool)
        iteration = 0

        while True:
            iteration += 1
            prev = output.copy()

            mask_to_fill = (output == 0) & expansion_mask
            dilated = cv2.dilate(output.astype(np.float32), kernel)
            dilated = dilated.astype(np.int32)

            output[mask_to_fill] = dilated[mask_to_fill]

            if np.array_equal(output, prev):
                break
            if iteration > 1000:
                if self.logger:
                    self.logger.warning("Label propagation exceeded 1000 iterations.")
                break

        return output

# Propagate labels: If performance is a concern, the dilation-based propagation loop can be optimized with a queue-based BFS flood-fill instead.

class MultiClassObjectAnalysis(GetMasks):
    """
    Analyze and expand multiple object contours across different classes using Voronoi constraints.

    Constructs Voronoi diagrams to limit spatial expansion, assigns unique labels to each object,
    and tracks class-wise and parent-wise mask lineage for downstream analysis.

    Attributes:
        multiple_contours (Dict[str, List[np.ndarray]]): Input contours grouped by class.
        height (int): Image height.
        width (int): Image width.
        save_path (str): Optional path to save outputs.
        vor (scipy.spatial.Voronoi): Computed Voronoi diagram.
        all_centroids (np.ndarray): Coordinates of centroids of input objects.
        class_labels (List[str]): Class label for each object.
        binary_masks, labelled_masks, referenced_masks: Output mask collections.
    """

    def __init__(self, get_masks_instance, multiple_contours: dict, save_path: str = None):
        self.get_masks_instance = get_masks_instance

        self.height = self.get_masks_instance.height
        self.width = self.get_masks_instance.width
        self.logger = self.get_masks_instance.logger

        # Remove tumour/stroma mask references as per your note
        self.multiple_contours = multiple_contours
        self.masks = None
        self.vor = None
        self.list_of_polygons = None
        self.class_labels = None
        self.all_centroids = None
        self.voronoi_regions = None
        self.voronoi_vertices = None
        self.save_path = save_path

        for class_label, contours in self.multiple_contours.items():
            for i, contour in enumerate(contours):
                if contour.shape[0] < 4:
                    self.logger.warning(f"Skipping contour with less than 4 points for class '{class_label}'.")
                    continue
                self.multiple_contours[class_label][i] = contour[::-1]

    @staticmethod
    def voronoi_finite_polygons_2d(vor, radius=None):
        """
        Reconstruct finite Voronoi polygons in 2D by clipping infinite regions.

        Args:
            vor (Voronoi): The original Voronoi diagram from scipy.spatial.
            radius (float, optional): Distance to extend infinite edges.
                                      Defaults to 2x max image dimension.

        Returns:
            Tuple[List[List[int]], np.ndarray]:
                - List of polygon regions (as indices of vertices),
                - Array of Voronoi vertices.
        """
        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        if radius is None:
            radius = vor.points.ptp().max() * 2

        # Map of all ridges for a point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct finite polygons
        for p1, region_index in enumerate(vor.point_region):
            vertices = vor.regions[region_index]

            if all(v >= 0 for v in vertices):
                # Finite region
                new_regions.append(vertices)
                continue

            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v1 >= 0 and v2 >= 0:
                    continue

                t = vor.points[p2] - vor.points[p1]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal vector

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v1 if v1 >= 0 else v2] + direction * radius

                new_vertices.append(far_point.tolist())
                new_region.append(len(new_vertices) - 1)

            # Sort region counterclockwise
            vs = np.array([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = [new_region[i] for i in np.argsort(angles)]

            new_regions.append(new_region)

        return new_regions, np.asarray(new_vertices)

    def get_polygons_from_contours(self, contours):
        """
        Converts contour point arrays into Shapely Polygon objects.

        Args:
            contours (List[np.ndarray]): List of (N, 2) arrays representing contours.

        Returns:
            List[Polygon]: Corresponding Shapely polygon objects.
        """
        polygons = []
        for cnt in contours:
            if cnt.shape[0] < 4:
                continue  # Too few points to form a polygon

            coords = cnt.squeeze()

            if coords.shape[0] < 4:
                continue  # Still too few after squeezing

            # Ensure it's closed (first point == last point)
            if not np.array_equal(coords[0], coords[-1]):
                coords = np.vstack([coords, coords[0]])

            try:
                polygon = Polygon(coords)
                if not polygon.is_valid or polygon.area == 0:
                    continue  # Skip invalid or zero-area polygons
                polygons.append(polygon)
            except Exception:
                continue  # Defensive: skip any invalid contour
        return polygons

    def derive_voronoi_from_contours(self):
        """
        Constructs a Voronoi diagram from object centroids.

        Computes Voronoi regions and safely clips infinite edges. Also stores
        the corresponding regions and vertices for future use in expansion logic.
        """

        all_contours = [contour for contour_points in self.multiple_contours.values() for contour in contour_points if contour.shape[0] >= 4]
        if not all_contours:
            raise ValueError("No contours found to derive Voronoi diagram.")

        list_of_polygons = self.get_polygons_from_contours(all_contours)

        centroids = []
        class_labels = []
        for class_label, contours in self.multiple_contours.items():
            # for contour in contours:
            #     if len(contour) < 4:
            #         self.logger.warning(f"Skipping contour with less than 4 points for class '{class_label}'.")
            #         continue
            #     polygon = Polygon(contour)
            for polygon in list_of_polygons:
                centroids.append(polygon.centroid)
                class_labels.append(class_label)

        if len(centroids) < 4:
            # Not enough data to compute Voronoi
            self.logger.warning("Not enough valid centroids for Voronoi diagram. Skipping Voronoi computation.")
            self.list_of_polygons = list_of_polygons
            self.class_labels = class_labels
            self.all_centroids = np.array([(c.x, c.y) for c in centroids]) if centroids else None
            self.vor = None
            self.voronoi_regions = None
            self.voronoi_vertices = None
            return

        all_centroids = np.array([(c.x, c.y) for c in centroids])
        vor = Voronoi(all_centroids)

        # Use finite polygons clipped to a large radius (image max dimension * 2)
        regions, vertices = self.voronoi_finite_polygons_2d(vor, radius=max(self.height, self.width) * 2)

        self.list_of_polygons = list_of_polygons
        self.class_labels = class_labels
        self.all_centroids = all_centroids
        self.vor = vor
        self.voronoi_regions = regions
        self.voronoi_vertices = vertices

    def get_voronoi_mask(self, category_name):
        """
        Creates a binary mask of Voronoi regions for a given class category.

        Args:
            category_name (str): The class label to extract Voronoi regions for.

        Returns:
            np.ndarray: Binary mask of selected Voronoi regions.
        """

        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        for idx, (label, region) in enumerate(zip(self.class_labels, self.voronoi_regions)):
            if label != category_name:
                continue
            polygon = self.voronoi_vertices[region]
            # Clip coordinates inside image boundaries
            polygon[:, 0] = np.clip(polygon[:, 0], 0, self.width - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, self.height - 1)
            int_polygon = polygon.astype(np.int32)
            if len(int_polygon) >= 3:
                cv2.fillPoly(mask, [int_polygon], color=255)

        return mask

    def expand_mask(self, mask, expansion_distance):
        """
        Expands a binary mask using morphological dilation.

        Args:
            mask (np.ndarray): The binary mask to be expanded.
            expansion_distance (int): Distance (in pixels) to expand the mask.

        Returns:
            np.ndarray: The expanded mask minus the original mask.
        """

        kernel = np.ones((expansion_distance, expansion_distance), np.uint8)
        expanded_mask = cv2.dilate(mask, kernel, iterations=1)
        expanded_mask = cv2.subtract(expanded_mask, mask)
        return expanded_mask

    def generate_expanded_masks_limited_by_voronoi(self, expansion_distances):
        """
          Expands object masks for each class using Voronoi constraints.

          Each contour is expanded outward by specified pixel distances, but limited to
          remain within its associated Voronoi cell. All expansions are labeled and tracked.

          Args:
              expansion_distances (List[int]): List of pixel distances for expansion rings.

          Returns:
              Tuple[
                  Dict[str, np.ndarray],  # binary_masks
                  Dict[str, np.ndarray],  # labelled_masks
                  Dict[str, np.ndarray]   # referenced_masks
              ]
          """

        # Step 1: Generate masks for each contour, and label objects
        masks = {}
        labeled_masks = {}
        referenced_labeled_mask = np.zeros((self.height, self.width), dtype=np.int32)

        parent_id_counter = 1  # unique ID for each original object across all classes

        # Map from category -> list of (parent_id, mask)
        original_masks_info = {}

        # Create binary masks for each individual contour, label them, assign parent IDs
        for category_name, contours in self.multiple_contours.items():
            category_masks = []
            for contour in contours:
                mask = np.zeros((self.height, self.width), dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
                # Label connected components (should be 1 per mask but be safe)
                labeled = label(mask > 0)
                # Extract regionprops if needed, here we just assign parent_id directly
                labeled_mask = np.zeros_like(labeled, dtype=np.int32)
                # Assign the unique parent ID to all pixels in this object
                labeled_mask[labeled > 0] = parent_id_counter

                # Update global referenced mask
                referenced_labeled_mask[labeled_mask > 0] = parent_id_counter

                # Store original mask and label
                masks[f'{category_name}_{parent_id_counter}'] = mask
                labeled_masks[f'{category_name}_{parent_id_counter}'] = labeled_mask

                category_masks.append((parent_id_counter, mask))
                parent_id_counter += 1
            original_masks_info[category_name] = category_masks

        # Step 2: Generate expansions and label them, mapping back to parent IDs
        expanded_masks = {}
        expanded_labeled_masks = {}

        for category_name, masks_info in original_masks_info.items():
            voronoi_mask = self.get_voronoi_mask(category_name)
            for parent_id, base_mask in masks_info:
                previous_expansion_mask = np.zeros((self.height, self.width), dtype=np.uint8)
                for expansion_distance in expansion_distances:
                    current_expansion_mask = self.expand_mask(base_mask.copy(), expansion_distance)
                    current_expansion_mask = cv2.bitwise_and(current_expansion_mask,
                                                             cv2.bitwise_not(previous_expansion_mask))
                    current_expansion_mask = cv2.bitwise_and(current_expansion_mask, voronoi_mask)

                    # Label this expanded mask (connected components)
                    labeled_expansion = label(current_expansion_mask > 0)
                    labeled_mask = np.zeros_like(labeled_expansion, dtype=np.int32)

                    # For each component in expansion assign a unique label encoding:
                    # parent_id * 1000 + expansion_distance (assuming expansion_distance < 1000)
                    # This allows tracing expansions to parent
                    # label_value = parent_id * 1000 + expansion_distance
                    label_value = parent_id

                    labeled_mask[labeled_expansion > 0] = label_value

                    # Update global referenced mask — careful to avoid overwriting originals
                    referenced_labeled_mask[labeled_mask > 0] = label_value

                    key = f'{category_name}_expansion_{expansion_distance}_parent_{parent_id}'
                    expanded_masks[key] = current_expansion_mask
                    expanded_labeled_masks[key] = labeled_mask

                    previous_expansion_mask = cv2.bitwise_or(previous_expansion_mask, current_expansion_mask)

        # Combine all masks and labeled masks
        masks.update(expanded_masks)
        labeled_masks.update(expanded_labeled_masks)
        # Step 3: Aggregate masks by class and expansion name
        aggregate_binary = {}
        aggregate_labeled = {}
        aggregate_referenced = {}

        for key, mask in masks.items():
            parts = key.split('_')

            if 'expansion' in parts:
                category = parts[0]
                expansion_distance = parts[2]
                agg_key = f"{category}_expansion_{expansion_distance}"
            else:
                category = parts[0]
                agg_key = category

            if agg_key not in aggregate_binary:
                aggregate_binary[agg_key] = np.zeros_like(mask)
                aggregate_labeled[agg_key] = np.zeros_like(mask, dtype=np.int32)
                aggregate_referenced[agg_key] = np.zeros_like(mask, dtype=np.int32)

            aggregate_binary[agg_key] = cv2.bitwise_or(aggregate_binary[agg_key], mask)
            aggregate_labeled[agg_key] = np.maximum(aggregate_labeled[agg_key], labeled_masks[key])

            # Referenced mask is pulled from the global referenced_labeled_mask
            aggregate_referenced[agg_key] = np.maximum(
                aggregate_referenced[agg_key],
                np.where(mask > 0, referenced_labeled_mask, 0)
            )

        # Final output
        self.binary_masks = aggregate_binary
        self.labeled_masks = aggregate_labeled
        self.referenced_masks = aggregate_referenced
        return self.binary_masks, self.labeled_masks, self.referenced_masks

    def plot_masks_with_voronoi(self,
                                mask_colors,
                                background_color=(255, 255, 255),
                                show=True,
                                axes=None,
                                figsize=(8, 8)):
        """
        Plots the generated masks overlaid with Voronoi edges.

        Args:
            mask_colors (Dict[str, Tuple[int, int, int]]): Mapping from class name to RGB color.
            background_color (Tuple[int, int, int], optional): RGB color for background. Defaults to white.
            show (bool, optional): If True, displays the plot. Defaults to True.
            axes (matplotlib.axes.Axes, optional): Existing axes to plot on.
            figsize (Tuple[int, int], optional): Figure size for new plot.

        Returns:
            matplotlib.axes.Axes: The plot axes (if `axes` was not provided).
        """
        masks = self.binary_masks
        background = np.full((self.height, self.width, 3), background_color, dtype=np.uint8)
        fig, ax = plt.subplots(figsize=figsize) if axes is None else (None, axes)
        legend_patches = []
        seen_classes = set()

        for mask_name, mask in masks.items():
            # Identify base class: 'gd' or 'cd8' from names like 'gd_expansion_30_0'
            base_class = mask_name.split('_')[0]

            # Get color for this base class
            color = np.array(mask_colors.get(base_class, (128, 128, 128)))
            background[mask != 0] = color

            # Add legend entry only once per base class
            if base_class not in seen_classes:
                legend_patches.append(mpatches.Patch(color=color / 255, label=base_class))
                seen_classes.add(base_class)

        ax.imshow(background, origin='lower')

        # Draw Voronoi edges
        if self.vor:
            voronoi_plot_2d(self.vor, ax=ax, show_vertices=False, line_colors='black', line_alpha=0.6)

            # Plot centroids (smaller dots)
            if self.all_centroids is not None:
                centroids = np.array(self.all_centroids)
                ax.plot(centroids[:, 0], centroids[:, 1], '*', markersize=1, alpha=0.6)

        # Add clean legend (gd, cd8)
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', bbox_transform=ax.transAxes)

        if self.save_path:
            save_path = os.path.join(self.save_path, 'masks_with_voronoi_edges.png')
            plt.savefig(save_path, dpi=1000, bbox_inches='tight')
            self.logger.info(f'Plot saved at {save_path}')

        if show:
            plt.show()

        return ax if axes is not None else None
