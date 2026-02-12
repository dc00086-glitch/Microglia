"""Install appropriate packages if not already installed"""

import sys
import os
import re
import time
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QSlider, QSpinBox,
    QGroupBox, QMessageBox, QTextEdit, QLineEdit, QFormLayout, QTabWidget,
    QProgressBar, QListWidgetItem, QDialog, QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView,
    QCheckBox, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QBrush, QKeySequence
from PyQt5.QtWidgets import QShortcut
from PIL import Image
import tifffile
from skimage import restoration, color, measure
from scipy import ndimage, stats
from matplotlib.path import Path as mplPath
import cv2
import glob
import json
import math


def load_tiff_image(filepath):
    """Load TIFF image using PIL to handle all compression types"""
    img = Image.open(filepath)
    # Convert to numpy array
    img_array = np.array(img)
    return img_array


def ensure_grayscale(img):
    """Convert image to grayscale if needed"""
    if img is None:
        return None
    if img.ndim == 3:
        # Handle RGBA (4 channels) by dropping alpha channel
        if img.shape[2] == 4:
            img = img[:, :, :3]  # Keep only RGB channels
        # Convert RGB to grayscale
        img = (color.rgb2gray(img) * 255).astype(img.dtype)
    if img.ndim > 2:
        img = img.squeeze()
    return img


def extract_channel(img, channel_idx):
    """Extract a single channel from a color image as grayscale"""
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] > channel_idx:
        channel = img[:, :, channel_idx].astype(np.float32)
        # Normalize to 0-255
        c_min, c_max = channel.min(), channel.max()
        if c_max > c_min:
            channel = (channel - c_min) / (c_max - c_min) * 255
        return channel.astype(np.uint8)
    elif img.ndim == 2:
        return img
    return ensure_grayscale(img)


# ============================================================================
# AUTO-OUTLINE ALGORITHMS
# ============================================================================

MIN_OUTLINE_POINTS = 10

def _simplify_contour(contour, min_points=MIN_OUTLINE_POINTS):
    """Simplify a contour with approxPolyDP, ensuring at least min_points.
    Starts with epsilon=0.02*arcLength and halves it until enough points."""
    arc_len = cv2.arcLength(contour, True)
    epsilon = 0.02 * arc_len
    for _ in range(6):
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) >= min_points:
            return approx
        epsilon *= 0.5
    # Final attempt with very small epsilon
    return cv2.approxPolyDP(contour, 0.001 * arc_len, True)

def auto_outline_threshold(image, centroid, sensitivity=50, region_size=200):
    """
    Auto-outline using adaptive threshold + contour detection.
    Works well for fluorescent microscopy images.

    Args:
        image: Grayscale image (2D numpy array)
        centroid: (row, col) tuple of soma center
        sensitivity: 0-100, higher = larger outline
        region_size: Size of region around centroid to analyze

    Returns:
        List of (row, col) polygon points, or None if failed
    """
    try:
        # Ensure image is 2D
        if image is None:
            return None
        if image.ndim > 2:
            image = image[:, :, 0] if image.shape[2] > 0 else image.squeeze()

        h, w = image.shape[:2]
        cy, cx = int(centroid[0]), int(centroid[1])

        # Extract region around centroid
        half = region_size // 2
        y1, y2 = max(0, cy - half), min(h, cy + half)
        x1, x2 = max(0, cx - half), min(w, cx + half)
        region = image[y1:y2, x1:x2].copy()

        if region.size == 0:
            return None

        # Normalize region to 0-255
        region = region.astype(np.float64)
        rmin, rmax = region.min(), region.max()
        if rmax > rmin:
            region = (region - rmin) / (rmax - rmin) * 255
        region = region.astype(np.uint8)

        # Apply Gaussian blur
        region = cv2.GaussianBlur(region, (5, 5), 1.5)

        # Get local centroid coordinates
        local_cy, local_cx = cy - y1, cx - x1

        # Use Otsu's method as base, adjust with sensitivity
        otsu_thresh, _ = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Adjust threshold based on sensitivity (higher = lower threshold = more inclusive)
        adjustment = (sensitivity - 50) / 100 * otsu_thresh * 0.8
        thresh_val = max(5, min(250, int(otsu_thresh - adjustment)))

        # Try threshold
        _, binary = cv2.threshold(region, thresh_val, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # Try adaptive threshold as fallback
            binary = cv2.adaptiveThreshold(region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 21, -5)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find contour containing the centroid
        best_contour = None
        best_area = 0
        for contour in contours:
            if cv2.pointPolygonTest(contour, (local_cx, local_cy), False) >= 0:
                area = cv2.contourArea(contour)
                if area > best_area:
                    best_contour = contour
                    best_area = area

        # If centroid not inside any contour, find nearest large contour
        if best_contour is None:
            min_dist = float('inf')
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 50:  # Skip tiny contours
                    continue
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cont_cx = M['m10'] / M['m00']
                    cont_cy = M['m01'] / M['m00']
                    dist = ((cont_cx - local_cx)**2 + (cont_cy - local_cy)**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        best_contour = contour

        if best_contour is None and contours:
            best_contour = max(contours, key=cv2.contourArea)

        if best_contour is None:
            return None

        # Check if contour is reasonable size
        area = cv2.contourArea(best_contour)
        if area < 20:
            return None

        # Simplify contour ensuring minimum points
        approx = _simplify_contour(best_contour)

        # Convert back to image coordinates
        points = []
        for pt in approx:
            px, py = pt[0]
            img_x = px + x1
            img_y = py + y1
            points.append((img_y, img_x))  # (row, col) format

        return points if len(points) >= MIN_OUTLINE_POINTS else None

    except Exception as e:
        print(f"Auto-outline threshold error: {e}")
        return None


def auto_outline_region_growing(image, centroid, sensitivity=50, max_iterations=10000):
    """
    Auto-outline using region growing from centroid.
    Adapts to local intensity variations.

    Args:
        image: Grayscale image (2D numpy array)
        centroid: (row, col) tuple of soma center
        sensitivity: 0-100, higher = more tolerant intensity difference
        max_iterations: Maximum pixels to grow

    Returns:
        List of (row, col) polygon points, or None if failed
    """
    try:
        if image is None:
            return None
        if image.ndim > 2:
            image = image[:, :, 0] if image.shape[2] > 0 else image.squeeze()

        h, w = image.shape[:2]
        cy, cx = int(centroid[0]), int(centroid[1])

        if not (0 <= cy < h and 0 <= cx < w):
            return None

        # Normalize image for consistent comparison
        img_norm = image.astype(np.float64)
        imin, imax = img_norm.min(), img_norm.max()
        if imax > imin:
            img_norm = (img_norm - imin) / (imax - imin) * 255

        # Get seed intensity
        seed_val = float(img_norm[cy, cx])

        # Tolerance based on sensitivity (higher = more tolerant)
        tolerance = (sensitivity / 100) * 100 + 20  # Range: 20-120

        # Region growing
        visited = np.zeros((h, w), dtype=bool)
        region = np.zeros((h, w), dtype=bool)
        queue = [(cy, cx)]
        visited[cy, cx] = True
        region[cy, cx] = True

        iterations = 0
        while queue and iterations < max_iterations:
            y, x = queue.pop(0)
            iterations += 1

            # Check 8-connected neighbors
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                        visited[ny, nx] = True
                        pixel_val = float(img_norm[ny, nx])
                        if abs(pixel_val - seed_val) <= tolerance:
                            region[ny, nx] = True
                            queue.append((ny, nx))

        # Check if region is reasonable
        if np.sum(region) < 20:
            return None

        # Find contour of region
        region_uint8 = (region * 255).astype(np.uint8)
        contours, _ = cv2.findContours(region_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Get largest contour
        contour = max(contours, key=cv2.contourArea)

        # Simplify ensuring minimum points
        approx = _simplify_contour(contour)

        points = [(pt[0][1], pt[0][0]) for pt in approx]  # (row, col) format
        return points if len(points) >= MIN_OUTLINE_POINTS else None

    except Exception as e:
        print(f"Auto-outline region growing error: {e}")
        return None


def auto_outline_watershed(image, centroid, sensitivity=50, region_size=200):
    """
    Auto-outline using watershed segmentation.
    Good for separating touching somas.

    Args:
        image: Grayscale image (2D numpy array)
        centroid: (row, col) tuple of soma center
        sensitivity: 0-100, affects marker size and threshold
        region_size: Size of region around centroid

    Returns:
        List of (row, col) polygon points, or None if failed
    """
    try:
        if image is None:
            return None
        if image.ndim > 2:
            image = image[:, :, 0] if image.shape[2] > 0 else image.squeeze()

        h, w = image.shape[:2]
        cy, cx = int(centroid[0]), int(centroid[1])

        # Extract region
        half = region_size // 2
        y1, y2 = max(0, cy - half), min(h, cy + half)
        x1, x2 = max(0, cx - half), min(w, cx + half)
        region = image[y1:y2, x1:x2].copy()

        if region.size == 0:
            return None

        # Normalize
        region = region.astype(np.float64)
        rmin, rmax = region.min(), region.max()
        if rmax > rmin:
            region = (region - rmin) / (rmax - rmin) * 255
        region = region.astype(np.uint8)

        # Apply blur
        region = cv2.GaussianBlur(region, (5, 5), 0)

        # Create markers - soma center is foreground marker
        markers = np.zeros(region.shape, dtype=np.int32)
        local_cx, local_cy = cx - x1, cy - y1

        # Check bounds
        if not (0 <= local_cx < region.shape[1] and 0 <= local_cy < region.shape[0]):
            return None

        # Foreground marker (soma) - small circle at centroid
        marker_radius = max(3, int(5 + sensitivity / 20))
        cv2.circle(markers, (local_cx, local_cy), marker_radius, 1, -1)

        # Background marker - ring at edge
        edge_mask = np.zeros(region.shape, dtype=np.uint8)
        cv2.rectangle(edge_mask, (0, 0), (region.shape[1]-1, region.shape[0]-1), 255, 3)
        markers[edge_mask > 0] = 2

        # Convert to 3-channel for watershed
        region_color = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)

        # Apply watershed
        cv2.watershed(region_color, markers)

        # Extract soma region (marker == 1)
        soma_mask = (markers == 1).astype(np.uint8) * 255

        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        soma_mask = cv2.morphologyEx(soma_mask, cv2.MORPH_CLOSE, kernel)
        soma_mask = cv2.morphologyEx(soma_mask, cv2.MORPH_OPEN, kernel)

        # Find contour
        contours, _ = cv2.findContours(soma_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)

        # Check area is reasonable
        if cv2.contourArea(contour) < 20:
            return None

        approx = _simplify_contour(contour)

        # Convert back to image coordinates
        points = [(pt[0][1] + y1, pt[0][0] + x1) for pt in approx]
        return points if len(points) >= MIN_OUTLINE_POINTS else None

    except Exception as e:
        print(f"Auto-outline watershed error: {e}")
        return None


def auto_outline_active_contours(image, centroid, sensitivity=50, region_size=150):
    """
    Auto-outline using active contours (snakes).
    Produces smooth, precise outlines.

    Args:
        image: Grayscale image (2D numpy array)
        centroid: (row, col) tuple of soma center
        sensitivity: 0-100, affects initial circle size and snake parameters
        region_size: Size of region around centroid

    Returns:
        List of (row, col) polygon points, or None if failed
    """
    try:
        from skimage.segmentation import active_contour
        from skimage.filters import gaussian

        if image is None:
            return None
        if image.ndim > 2:
            image = image[:, :, 0] if image.shape[2] > 0 else image.squeeze()

        h, w = image.shape[:2]
        cy, cx = int(centroid[0]), int(centroid[1])

        # Extract region
        half = region_size // 2
        y1, y2 = max(0, cy - half), min(h, cy + half)
        x1, x2 = max(0, cx - half), min(w, cx + half)
        region = image[y1:y2, x1:x2].copy().astype(np.float64)

        if region.size == 0:
            return None

        # Normalize
        rmin, rmax = region.min(), region.max()
        if rmax > rmin:
            region = (region - rmin) / (rmax - rmin)

        # Smooth image
        region = gaussian(region, sigma=2)

        # Initial circle
        local_cx, local_cy = cx - x1, cy - y1

        # Check bounds
        if not (0 <= local_cx < region.shape[1] and 0 <= local_cy < region.shape[0]):
            return None

        # Initial radius based on sensitivity
        init_radius = 10 + sensitivity / 3  # Range: 10-43 pixels

        # Create initial snake points (circle)
        s = np.linspace(0, 2 * np.pi, 100)
        init_x = local_cx + init_radius * np.cos(s)
        init_y = local_cy + init_radius * np.sin(s)
        init = np.array([init_x, init_y]).T

        # Snake parameters based on sensitivity
        alpha = 0.01 + (100 - sensitivity) / 5000  # Smoothness
        beta = 0.1 + (100 - sensitivity) / 500     # Curvature

        # Run active contour
        snake = active_contour(
            region, init,
            alpha=alpha, beta=beta,
            gamma=0.01,
            max_num_iter=250
        )

        # Simplify snake to polygon, ensuring at least MIN_OUTLINE_POINTS
        target_pts = max(30, MIN_OUTLINE_POINTS)
        step = max(1, len(snake) // target_pts)
        simplified = snake[::step]

        # Convert back to image coordinates
        points = [(pt[1] + y1, pt[0] + x1) for pt in simplified]
        return points if len(points) >= MIN_OUTLINE_POINTS else None

    except Exception as e:
        print(f"Auto-outline active contours error: {e}")
        return None


def auto_outline_hybrid(image, centroid, sensitivity=50, region_size=200):
    """
    Hybrid approach: Try multiple methods and use the best result.
    Falls back through: Threshold -> Watershed -> Active Contours

    Args:
        image: Grayscale image (2D numpy array)
        centroid: (row, col) tuple of soma center
        sensitivity: 0-100
        region_size: Size of region around centroid

    Returns:
        List of (row, col) polygon points, or None if failed
    """
    try:
        if image is None:
            return None
        if image.ndim > 2:
            image = image[:, :, 0] if image.shape[2] > 0 else image.squeeze()

        # Try threshold first (most reliable)
        threshold_points = auto_outline_threshold(image, centroid, sensitivity, region_size)

        # Try watershed
        watershed_points = auto_outline_watershed(image, centroid, sensitivity, region_size)

        # Choose the better result based on some heuristics
        best_points = None
        best_score = 0

        for points, name in [(threshold_points, 'threshold'), (watershed_points, 'watershed')]:
            if points is None or len(points) < MIN_OUTLINE_POINTS:
                continue

            # Score based on: number of points (more = smoother), area, and compactness
            pts_array = np.array(points)

            # Calculate area using shoelace formula
            n = len(pts_array)
            area = 0.5 * abs(sum(pts_array[i, 0] * pts_array[(i + 1) % n, 1] -
                                 pts_array[(i + 1) % n, 0] * pts_array[i, 1]
                                 for i in range(n)))

            # Calculate perimeter
            perimeter = sum(np.sqrt((pts_array[(i + 1) % n, 0] - pts_array[i, 0]) ** 2 +
                                    (pts_array[(i + 1) % n, 1] - pts_array[i, 1]) ** 2)
                           for i in range(n))

            # Compactness (circularity) - closer to 1 is better for soma
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter ** 2)
            else:
                compactness = 0

            # Score: prefer reasonable area and good compactness
            if area > 50 and area < 50000:  # Reasonable soma size
                score = compactness * 100 + min(area / 100, 50)
                if score > best_score:
                    best_score = score
                    best_points = points

        if best_points is not None:
            return best_points

        # If nothing worked, try active contours as last resort
        active_points = auto_outline_active_contours(image, centroid, sensitivity, region_size)
        if active_points is not None and len(active_points) >= MIN_OUTLINE_POINTS:
            return active_points

        # Return whatever we have
        return threshold_points or watershed_points

    except Exception as e:
        print(f"Auto-outline hybrid error: {e}")
        # Try simple threshold as fallback
        return auto_outline_threshold(image, centroid, sensitivity, region_size)


class MorphologyCalculator:
    """
    Calculate morphological parameters for microglia.

    Calculates 10 basic metrics: roundness, eccentricity, soma area,
    mask area, perimeter, cell spread, polarity index, principal angle,
    major axis length, and minor axis length.
    """

    def __init__(self, image, pixel_size_um, use_imagej=False):
        self.image = image
        self.pixel_size = pixel_size_um
        # Note: use_imagej parameter kept for backwards compatibility but not used

    def calculate_all_parameters(self, cell_mask, soma_centroid, soma_area_um2=None):
        """Calculate ONLY simple parameters - Sholl, Fractal, Hull, Skeleton done in ImageJ"""
        params = {}
      
        params.update(self._calculate_simple_descriptors(cell_mask, soma_area_um2))

        return params

    def _calculate_simple_descriptors(self, mask, soma_area_um2=None):
        params = {}
        props = measure.regionprops(mask.astype(int))[0] if np.any(mask) else None
        if props:
            params['perimeter'] = props.perimeter * self.pixel_size
            params['mask_area'] = props.area * (self.pixel_size ** 2)

            major_axis = props.major_axis_length
            minor_axis = props.minor_axis_length
            
            # Range: 0 (perfect circle) to 1 (highly elongated)
            if major_axis > 0:
                axis_ratio = minor_axis / major_axis
                params['eccentricity'] = np.sqrt(1 - axis_ratio**2)
            else:
                params['eccentricity'] = 0
            
            # Roundness: minor/major ratio (unchanged)
            if major_axis > 0:
                params['roundness'] = minor_axis / major_axis
            else:
                params['roundness'] = 0

            centroid = np.array(props.centroid)
            coords = np.array(props.coords)

            top_point = coords[coords[:, 0].argmin()]
            bottom_point = coords[coords[:, 0].argmax()]
            left_point = coords[coords[:, 1].argmin()]
            right_point = coords[coords[:, 1].argmax()]

            extremities = np.array([top_point, bottom_point, left_point, right_point])
            distances = np.sqrt(np.sum((extremities - centroid) ** 2, axis=1))
            params['cell_spread'] = np.mean(distances) * self.pixel_size

            if soma_area_um2 is not None:
                params['soma_area'] = soma_area_um2
            else:
                params['soma_area'] = props.area * 0.1 * (self.pixel_size ** 2)

            # Directional polarity via PCA on mask coordinates
            params.update(self._calculate_polarity(coords, centroid))
        else:
            params = {k: 0 for k in ['perimeter', 'mask_area', 'eccentricity',
                                     'roundness', 'cell_spread', 'soma_area',
                                     'polarity_index', 'principal_angle',
                                     'major_axis_um', 'minor_axis_um']}
        return params

    def _calculate_polarity(self, coords, centroid):
        """Calculate directional polarity from PCA on mask pixel coordinates.

        Returns:
            dict with polarity_index (0=isotropic, 1=fully polarized),
            principal_angle (0-180 degrees), major_axis_um, minor_axis_um.
        """
        params = {}
        # Center coordinates on the centroid
        centered = coords - centroid

        if len(centered) < 3:
            params['polarity_index'] = 0
            params['principal_angle'] = 0
            params['major_axis_um'] = 0
            params['minor_axis_um'] = 0
            return params

        # Covariance matrix of (row, col) positions
        cov = np.cov(centered.T)

        # Eigenvalues = variance along principal axes
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # eigh returns ascending order; major axis is the last one
        major_val = eigenvalues[-1]
        minor_val = eigenvalues[0]
        major_vec = eigenvectors[:, -1]

        # Polarity index: 0 = circle, 1 = line
        if major_val > 0:
            params['polarity_index'] = round(1.0 - (minor_val / major_val), 4)
        else:
            params['polarity_index'] = 0

        # Principal direction angle (0-180 degrees, 0=horizontal)
        # major_vec is (row, col) so angle from horizontal = atan2(row, col)
        angle_rad = np.arctan2(major_vec[0], major_vec[1])
        angle_deg = np.degrees(angle_rad) % 180  # Map to 0-180 range
        params['principal_angle'] = round(angle_deg, 2)

        # Axis extents in microns (2 * sqrt(eigenvalue) gives the std dev extent)
        params['major_axis_um'] = round(2 * np.sqrt(major_val) * self.pixel_size, 4)
        params['minor_axis_um'] = round(2 * np.sqrt(max(minor_val, 0)) * self.pixel_size, 4)

        return params


class BatchProcessingThread(QThread):
    """Thread for batch processing images in the background"""
    progress = pyqtSignal(int)
    status_update = pyqtSignal(str)
    finished_image = pyqtSignal(str, str, object)
    error_occurred = pyqtSignal(str)

    def __init__(self, image_data_list, output_dir):
        super().__init__()
        self.image_data_list = image_data_list
        self.output_dir = output_dir

    def run(self):
        try:
            total = len(self.image_data_list)
            for i, (
                    img_path, img_name, radius, rb_enabled, denoise_enabled, denoise_size,
                    sharpen_enabled, sharpen_amount, process_channel) in enumerate(
                self.image_data_list):
                try:
                    self.status_update.emit(f"Processing: {img_name}")
                    img = load_tiff_image(img_path)
                    # Extract only the selected channel for processing
                    if img.ndim == 3:
                        img = extract_channel(img, process_channel)
                    img_dtype = img.dtype
                    result = img.copy()

                    # Apply optional rolling ball background subtraction
                    if rb_enabled:
                        background = restoration.rolling_ball(img, radius=radius)
                        result = img - background
                        result = np.clip(result, 0, np.iinfo(img_dtype).max)

                    # Apply optional denoising
                    if denoise_enabled:
                        result = ndimage.median_filter(result, size=denoise_size)

                    # Apply optional sharpening
                    if sharpen_enabled:
                        blurred = ndimage.gaussian_filter(result.astype(np.float32), sigma=2)
                        result_float = result.astype(np.float32)
                        sharpened = result_float + sharpen_amount * (result_float - blurred)
                        result = np.clip(sharpened, 0, np.iinfo(img_dtype).max).astype(img_dtype)

                    name = os.path.splitext(img_name)[0]
                    out_path = os.path.join(self.output_dir, f"{name}_processed.tif")
                    tifffile.imwrite(out_path, result.astype(img_dtype))
                    self.finished_image.emit(out_path, img_name, result)
                    self.progress.emit(int((i + 1) / total * 100))
                except Exception as e:
                    self.error_occurred.emit(f"Error: {img_name}: {e}")
        except Exception as e:
            self.error_occurred.emit(f"Fatal error: {e}")


class MorphologyCalculationThread(QThread):
    """Thread for calculating morphology parameters in the background"""
    progress = pyqtSignal(int, str)  # progress percentage, status message
    finished = pyqtSignal(list)  # list of results
    error_occurred = pyqtSignal(str)

    def __init__(self, approved_masks, pixel_size, use_imagej, images):
        super().__init__()
        self.approved_masks = approved_masks
        self.pixel_size = pixel_size
        self.use_imagej = use_imagej
        self.images = images

    def run(self):
        import sys
        import os

        # Suppress Java warnings during processing to speed up
        old_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')

        try:
            all_results = []
            total = len(self.approved_masks)

            for i, flat_data in enumerate(self.approved_masks):
                mask_data = flat_data['mask_data']
                processed_img = flat_data['processed_img']
                img_name = flat_data['image_name']

                self.progress.emit(
                    int((i + 1) / total * 100),
                    f"Processing {mask_data['soma_id']} ({i + 1}/{total})"
                )

                # Find soma centroid and area from original image data
                img_data = self.images[img_name]
                soma_centroid = img_data['somas'][mask_data['soma_idx']]
                soma_area_um2 = mask_data.get('soma_area_um2', None)

                # Time tracking for diagnostics
                import time
                start = time.time()

                calculator = MorphologyCalculator(processed_img, self.pixel_size, use_imagej=self.use_imagej)
                params = calculator.calculate_all_parameters(mask_data['mask'], soma_centroid, soma_area_um2)

                elapsed = time.time() - start
                # Temporarily restore stderr to print timing
                sys.stderr = old_stderr
                print(f"  ✓ {mask_data['soma_id']}: {elapsed:.1f}s")
                sys.stderr = open(os.devnull, 'w')

                params['image_name'] = os.path.splitext(img_name)[0]
                params['soma_id'] = mask_data['soma_id']
                params['soma_idx'] = mask_data['soma_idx']
                params['area_um2'] = mask_data['area_um2']

                all_results.append(params)

            self.finished.emit(all_results)

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            # Always restore stderr
            if sys.stderr != old_stderr:
                sys.stderr.close()
                sys.stderr = old_stderr


class BackgroundRemovalThread(QThread):
    progress = pyqtSignal(int)
    status_update = pyqtSignal(str)
    finished_image = pyqtSignal(str, str, object)
    error_occurred = pyqtSignal(str)

    def __init__(self, image_data_list, output_dir):
        super().__init__()
        self.image_data_list = image_data_list
        self.output_dir = output_dir

    def run(self):
        try:
            total = len(self.image_data_list)
            for i, (
                    img_path, img_name, radius, rb_enabled, denoise_enabled, denoise_size,
                    sharpen_enabled, sharpen_amount, process_channel) in enumerate(
                self.image_data_list):
                try:
                    self.status_update.emit(f"Processing: {img_name}")
                    img = load_tiff_image(img_path)
                    # Extract only the selected channel for processing
                    if img.ndim == 3:
                        img = extract_channel(img, process_channel)
                    img_dtype = img.dtype
                    result = img.copy()

                    # Apply optional rolling ball background subtraction
                    if rb_enabled:
                        background = restoration.rolling_ball(img, radius=radius)
                        result = img - background
                        result = np.clip(result, 0, np.iinfo(img_dtype).max)

                    # Apply optional denoising
                    if denoise_enabled:
                        result = ndimage.median_filter(result, size=denoise_size)

                    # Apply optional sharpening
                    if sharpen_enabled:
                        blurred = ndimage.gaussian_filter(result.astype(np.float32), sigma=2)
                        result_float = result.astype(np.float32)
                        sharpened = result_float + sharpen_amount * (result_float - blurred)
                        result = np.clip(sharpened, 0, np.iinfo(img_dtype).max).astype(img_dtype)

                    name = os.path.splitext(img_name)[0]
                    out_path = os.path.join(self.output_dir, f"{name}_processed.tif")
                    tifffile.imwrite(out_path, result.astype(img_dtype))
                    self.finished_image.emit(out_path, img_name, result)
                    self.progress.emit(int((i + 1) / total * 100))
                except Exception as e:
                    self.error_occurred.emit(f"Error: {img_name}: {e}")
        except Exception as e:
            self.error_occurred.emit(f"Fatal error: {e}")


class InteractiveImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.pix_source = None
        self.centroids = []
        self.mask_overlay = None
        self.polygon_pts = []
        self.soma_mode = False
        self.polygon_mode = False
        self.setMinimumSize(400, 400)
        # Don't use QLabel's auto-centering - we'll position the pixmap ourselves
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setStyleSheet("border: 2px solid palette(mid); background-color: palette(base);")
        # Allow label to expand
        from PyQt5.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setScaledContents(False)
        # Zoom and pan settings
        self.zoom_level = 1.0
        self.min_zoom = 0.5
        self.max_zoom = 10.0
        # View center in image coordinates (0-1 normalized)
        self.view_center_x = 0.5
        self.view_center_y = 0.5
        # Store the scaled pixmap for custom drawing
        self.scaled_pixmap = None
        # Point editing for polygon outlines
        self.point_edit_mode = False
        self.selected_point_idx = None
        self.dragging_point = False
        self.setMouseTracking(True)  # Enable mouse tracking for hover effects
        # Measurement tool
        self.measure_mode = False
        self.measure_pt1 = None  # (row, col) image coords
        self.measure_pt2 = None
        # Mask overlay opacity (0.0 - 1.0)
        self.overlay_opacity = 0.4
        # Centroid dragging
        self.dragging_centroid = False
        self.dragging_centroid_idx = None

    def set_image(self, qpix, centroids=None, mask_overlay=None, polygon_pts=None):
        self.pix_source = qpix
        self.centroids = centroids or []
        self.mask_overlay = mask_overlay
        self.polygon_pts = polygon_pts or []
        self._update_display()

    def _update_display(self):
        """Update the displayed image with current zoom"""
        if self.pix_source is None:
            return

        img_w = self.pix_source.width()
        img_h = self.pix_source.height()
        label_w = self.size().width()
        label_h = self.size().height()

        # Calculate base scale to fit
        base_scale = min(label_w / img_w, label_h / img_h)

        # Apply zoom
        final_w = int(img_w * base_scale * self.zoom_level)
        final_h = int(img_h * base_scale * self.zoom_level)

        # Store the scaled pixmap for custom drawing in paintEvent
        self.scaled_pixmap = self.pix_source.scaled(
            final_w,
            final_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        # Don't use QLabel's setPixmap - we draw it ourselves with pan offset
        self.repaint()

    def reset_zoom(self):
        """Reset zoom and pan to default"""
        self.zoom_level = 1.0
        self.view_center_x = 0.5
        self.view_center_y = 0.5
        self._update_display()

    def set_zoom(self, level):
        """Set zoom to a specific level"""
        self.zoom_level = max(self.min_zoom, min(self.max_zoom, level))
        self._update_display()

    def zoom_to_point(self, img_row, img_col, zoom_level=3.0):
        """Center the view on a specific image coordinate and zoom in."""
        if self.pix_source is None:
            return
        img_h = self.pix_source.height()
        img_w = self.pix_source.width()
        if img_w > 0 and img_h > 0:
            self.view_center_x = img_col / img_w
            self.view_center_y = img_row / img_h
            self.zoom_level = max(self.min_zoom, min(self.max_zoom, zoom_level))
            self._update_display()
            if self.parent_widget and hasattr(self.parent_widget, 'zoom_level_label'):
                self.parent_widget.zoom_level_label.setText(f"{self.zoom_level:.1f}x")

    def _get_pan_offset(self):
        """Calculate pan offset based on view center"""
        if not self.pix_source or not self.scaled_pixmap:
            return 0, 0

        label_w = self.size().width()
        label_h = self.size().height()

        pixmap_w = self.scaled_pixmap.width()
        pixmap_h = self.scaled_pixmap.height()

        # Center offset (where image would be if centered)
        center_offset_x = (label_w - pixmap_w) / 2
        center_offset_y = (label_h - pixmap_h) / 2

        # Pan offset based on view center
        # view_center is in normalized image coords (0-1)
        # We want that point to be at the label center
        pan_x = (0.5 - self.view_center_x) * pixmap_w
        pan_y = (0.5 - self.view_center_y) * pixmap_h

        return center_offset_x + pan_x, center_offset_y + pan_y

    def paintEvent(self, event):
        # Draw background first (from QLabel)
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().color(self.backgroundRole()))

        if not self.pix_source or not self.scaled_pixmap:
            painter.end()
            return

        # Draw the pixmap at the correct pan offset position
        offset_x, offset_y = self._get_pan_offset()
        painter.drawPixmap(int(offset_x), int(offset_y), self.scaled_pixmap)

        # Draw mask overlay
        if self.mask_overlay is not None:
            self._draw_mask_overlay(painter)

        # Draw soma markers (centroids)
        if self.centroids:
            pen = QPen(QColor(255, 0, 0), 3)
            painter.setPen(pen)
            for centroid in self.centroids:
                x, y = self._to_display_coords(centroid)
                # Only draw if visible
                if 0 <= x <= self.width() and 0 <= y <= self.height():
                    painter.drawEllipse(int(x - 6), int(y - 6), 12, 12)

        # Draw polygon points
        if self.polygon_mode and len(self.polygon_pts) > 0:
            self._draw_polygon(painter)

        # Draw measurement overlay
        if self.measure_mode and self.measure_pt1 is not None:
            pen = QPen(QColor(255, 255, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            x1, y1 = self._to_display_coords(self.measure_pt1)
            # Draw first point marker
            painter.setBrush(QColor(255, 255, 0, 150))
            painter.drawEllipse(int(x1 - 5), int(y1 - 5), 10, 10)
            if self.measure_pt2 is not None:
                x2, y2 = self._to_display_coords(self.measure_pt2)
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
                painter.drawEllipse(int(x2 - 5), int(y2 - 5), 10, 10)
                # Draw distance label at midpoint
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                if self.parent_widget and hasattr(self.parent_widget, '_get_measure_text'):
                    text = self.parent_widget._get_measure_text()
                    painter.setPen(QColor(0, 0, 0))
                    painter.setBrush(QColor(255, 255, 200, 220))
                    fm = painter.fontMetrics()
                    tw = fm.horizontalAdvance(text) + 8
                    painter.drawRect(int(mx - tw / 2), int(my - 18), tw, 20)
                    painter.setPen(QColor(0, 0, 0))
                    painter.drawText(int(mx - tw / 2 + 4), int(my - 2), text)

        # Draw zoom indicator
        if self.zoom_level != 1.0:
            bg = self.palette().color(self.backgroundRole())
            bg.setAlpha(200)
            fg = self.palette().color(self.foregroundRole())
            painter.setPen(QPen(fg))
            painter.setBrush(bg)
            painter.drawRect(5, 5, 70, 20)
            painter.setPen(fg)
            painter.drawText(10, 20, f"Zoom: {self.zoom_level:.1f}x")

        painter.end()

    def resizeEvent(self, event):
        """Re-scale the image when the label is resized"""
        super().resizeEvent(event)
        self._update_display()

    def center_on_soma(self):
        """Center the view on the current soma (first centroid)"""
        if not self.pix_source or not self.centroids:
            return
        # Get first centroid (current soma)
        soma = self.centroids[0]
        img_h, img_w = self.pix_source.height(), self.pix_source.width()
        # Convert to normalized coords
        self.view_center_x = soma[1] / img_w
        self.view_center_y = soma[0] / img_h
        self._update_display()

    def wheelEvent(self, event):
        """Mouse wheel disabled - use Z+click to zoom"""
        # Pass to parent (no zoom on scroll)
        event.ignore()

    def _find_nearest_point(self, click_pos, threshold=15):
        """Find the nearest polygon point within threshold pixels"""
        # Sync from parent's authoritative list to avoid stale data
        if self.parent_widget and hasattr(self.parent_widget, 'polygon_points'):
            self.polygon_pts = self.parent_widget.polygon_points
        if not self.polygon_pts:
            return None
        # Scale threshold with zoom so points are easier to grab when zoomed out
        effective_threshold = max(threshold, threshold / max(self.zoom_level, 0.5))
        min_dist = float('inf')
        nearest_idx = None
        for i, pt in enumerate(self.polygon_pts):
            display_x, display_y = self._to_display_coords(pt)
            dist = ((click_pos.x() - display_x) ** 2 + (click_pos.y() - display_y) ** 2) ** 0.5
            if dist < min_dist and dist < effective_threshold:
                min_dist = dist
                nearest_idx = i
        return nearest_idx

    def _find_nearest_centroid(self, click_pos, threshold=6):
        """Find the nearest centroid within threshold pixels for dragging"""
        if not self.centroids:
            return None
        effective_threshold = threshold / max(self.zoom_level, 0.5)
        min_dist = float('inf')
        nearest_idx = None
        for i, centroid in enumerate(self.centroids):
            display_x, display_y = self._to_display_coords(centroid)
            dist = ((click_pos.x() - display_x) ** 2 + (click_pos.y() - display_y) ** 2) ** 0.5
            if dist < min_dist and dist < effective_threshold:
                min_dist = dist
                nearest_idx = i
        return nearest_idx

    def mousePressEvent(self, event):
        # Check if Z key is held for zoom
        if self.parent_widget and hasattr(self.parent_widget, 'z_key_held') and self.parent_widget.z_key_held:
            # Z+click = zoom at clicked location
            if event.button() == Qt.LeftButton:
                self.zoom_at_point(event.pos(), zoom_in=True)
            elif event.button() == Qt.RightButton:
                self.zoom_at_point(event.pos(), zoom_in=False)
            return  # Don't place any points

        coords = self._to_image_coords(event.pos().x(), event.pos().y())
        if not coords:
            return

        # Measurement mode takes priority over other modes
        if self.measure_mode and event.button() == Qt.LeftButton:
            if self.measure_pt1 is None or self.measure_pt2 is not None:
                # Start new measurement
                self.measure_pt1 = coords
                self.measure_pt2 = None
            else:
                # Complete measurement
                self.measure_pt2 = coords
                if self.parent_widget and hasattr(self.parent_widget, '_show_measurement'):
                    self.parent_widget._show_measurement()
            self._update_display()
            return

        if self.soma_mode and self.parent_widget:
            # Check if clicking near an existing centroid to drag it
            if self.centroids and event.button() == Qt.LeftButton:
                nearest_idx = self._find_nearest_centroid(event.pos())
                if nearest_idx is not None:
                    self.dragging_centroid = True
                    self.dragging_centroid_idx = nearest_idx
                    return
            self.parent_widget.add_soma(coords)
        elif self.polygon_mode and self.parent_widget:
            # Sync polygon points from parent before checking for drag
            if hasattr(self.parent_widget, 'polygon_points'):
                self.polygon_pts = self.parent_widget.polygon_points
            # Check for point editing first (if there are existing points)
            if self.polygon_pts and event.button() == Qt.LeftButton:
                nearest_idx = self._find_nearest_point(event.pos())
                if nearest_idx is not None:
                    # Start dragging this point
                    self.selected_point_idx = nearest_idx
                    self.dragging_point = True
                    self._update_display()
                    return

            # Normal polygon mode behavior
            if event.button() == Qt.LeftButton:
                self.parent_widget.add_polygon_point(coords)
            elif event.button() == Qt.RightButton:
                self.parent_widget.finish_polygon()

    def mouseMoveEvent(self, event):
        """Handle mouse move for point/centroid dragging"""
        # Centroid dragging
        if self.dragging_centroid and self.dragging_centroid_idx is not None and self.parent_widget:
            coords = self._to_image_coords(event.pos().x(), event.pos().y())
            if coords and self.dragging_centroid_idx < len(self.centroids):
                self.centroids[self.dragging_centroid_idx] = coords
                # Update parent's somas list
                img_data = self.parent_widget.images.get(self.parent_widget.current_image_name)
                if img_data and self.dragging_centroid_idx < len(img_data['somas']):
                    img_data['somas'][self.dragging_centroid_idx] = coords
                self._update_display()
            return

        if self.dragging_point and self.selected_point_idx is not None and self.parent_widget:
            coords = self._to_image_coords(event.pos().x(), event.pos().y())
            if coords:
                # Update the point position in parent's polygon_points
                self.parent_widget.polygon_points[self.selected_point_idx] = coords
                # Update our local copy too
                self.polygon_pts = self.parent_widget.polygon_points.copy()
                self._update_display()

    def mouseReleaseEvent(self, event):
        """Handle mouse release to finish dragging"""
        # Centroid drag release - snap to brightest pixel within 5 µm
        if self.dragging_centroid and event.button() == Qt.LeftButton:
            self.dragging_centroid = False
            idx = self.dragging_centroid_idx
            self.dragging_centroid_idx = None
            if self.parent_widget:
                img_data = self.parent_widget.images.get(self.parent_widget.current_image_name)
                if img_data and idx is not None and idx < len(img_data['somas']):
                    snapped = self.parent_widget._snap_to_brightest(img_data['somas'][idx])
                    img_data['somas'][idx] = snapped
                    self.centroids[idx] = snapped
                    img_data['soma_ids'][idx] = f"soma_{snapped[0]}_{snapped[1]}"
                    self.parent_widget.log(f"  → Soma {idx + 1} repositioned (snapped to brightest)")
                self._update_display()
            return

        if self.dragging_point and event.button() == Qt.LeftButton:
            self.dragging_point = False
            # Keep the point selected for visibility but stop dragging
            if self.parent_widget and hasattr(self.parent_widget, 'polygon_points'):
                # Ensure parent's points are updated
                self.polygon_pts = self.parent_widget.polygon_points.copy()
                self._update_display()
                # Log the adjustment
                if hasattr(self.parent_widget, 'log'):
                    self.parent_widget.log(f"  → Point {self.selected_point_idx + 1} adjusted")

    def zoom_at_point(self, pos, zoom_in=True):
        """Zoom in or out centered on the clicked position"""
        if not self.pix_source:
            return

        # Convert click position to normalized image coordinates
        coords = self._to_image_coords(pos.x(), pos.y())
        if not coords:
            return

        img_h, img_w = self.pix_source.height(), self.pix_source.width()
        # coords is (row, col) = (y, x) in image coords
        self.view_center_x = coords[1] / img_w
        self.view_center_y = coords[0] / img_h

        # Apply zoom
        if zoom_in:
            new_zoom = self.zoom_level * 1.5
        else:
            new_zoom = self.zoom_level / 1.5

        self.zoom_level = max(self.min_zoom, min(self.max_zoom, new_zoom))
        self._update_display()

        # Update parent's zoom label if available
        if self.parent_widget and hasattr(self.parent_widget, 'zoom_level_label'):
            self.parent_widget.zoom_level_label.setText(f"{self.zoom_level:.1f}x")

    def _draw_mask_overlay(self, painter):
        if self.mask_overlay is None:
            return
        mask = self.mask_overlay
        if not self.pix_source or not self.scaled_pixmap:
            return
        mask_coords = np.argwhere(mask > 0)
        if len(mask_coords) == 0:
            return
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 255, 0))
        painter.setOpacity(self.overlay_opacity)
        img_h, img_w = self.pix_source.height(), self.pix_source.width()
        pixmap_w = self.scaled_pixmap.width()
        pixmap_h = self.scaled_pixmap.height()
        scale_x = pixmap_w / img_w
        scale_y = pixmap_h / img_h
        rect_w = max(1, int(scale_x))
        rect_h = max(1, int(scale_y))
        for coord in mask_coords:
            img_y, img_x = coord[0], coord[1]
            x_pos, y_pos = self._to_display_coords((img_y, img_x))
            painter.drawRect(int(x_pos), int(y_pos), rect_w, rect_h)
        painter.setOpacity(1.0)

    def _draw_polygon(self, painter):
        # Draw polygon lines
        pen = QPen(QColor(255, 165, 0), 3)
        painter.setPen(pen)
        for i in range(len(self.polygon_pts)):
            p1 = self._to_display_coords(self.polygon_pts[i])
            p2 = self._to_display_coords(self.polygon_pts[(i + 1) % len(self.polygon_pts)])
            painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))

        # Draw polygon points - highlight selected/dragging point
        for i, pt in enumerate(self.polygon_pts):
            x, y = self._to_display_coords(pt)
            if i == self.selected_point_idx:
                # Selected point - larger and green
                pen = QPen(QColor(0, 255, 0), 3)
                painter.setPen(pen)
                painter.setBrush(QColor(0, 255, 0))
                painter.drawEllipse(int(x - 6), int(y - 6), 12, 12)
            else:
                # Normal points - blue
                pen = QPen(QColor(0, 0, 255), 2)
                painter.setPen(pen)
                painter.setBrush(QColor(0, 0, 255))
                painter.drawEllipse(int(x - 4), int(y - 4), 8, 8)

    def _to_display_coords(self, img_coords):
        if not self.pix_source or not self.scaled_pixmap:
            return 0, 0
        img_h, img_w = self.pix_source.height(), self.pix_source.width()
        pixmap_w = self.scaled_pixmap.width()
        pixmap_h = self.scaled_pixmap.height()

        # Get pan offset from view center
        offset_x, offset_y = self._get_pan_offset()

        scale_x = pixmap_w / img_w
        scale_y = pixmap_h / img_h
        x = img_coords[1] * scale_x + offset_x
        y = img_coords[0] * scale_y + offset_y
        return x, y

    def _to_image_coords(self, display_x, display_y):
        if not self.pix_source or not self.scaled_pixmap:
            return None

        img_h, img_w = self.pix_source.height(), self.pix_source.width()
        pixmap_w = self.scaled_pixmap.width()
        pixmap_h = self.scaled_pixmap.height()

        # Get pan offset from view center
        offset_x, offset_y = self._get_pan_offset()

        scale_x = img_w / pixmap_w
        scale_y = img_h / pixmap_h
        img_x = round((display_x - offset_x) * scale_x)
        img_y = round((display_y - offset_y) * scale_y)
        return (img_y, img_x)

    def mouseDoubleClickEvent(self, event):
        if self.polygon_mode and self.parent_widget:
            self.parent_widget.finish_polygon()


class ChannelSelectDialog(QDialog):
    """Dialog to select which channels to display"""
    def __init__(self, parent=None, current_channels=None, channel_names=None, color_image=None):
        super().__init__(parent)
        self.setWindowTitle("Channel Display Settings")
        self.setModal(True)

        # Default channel settings - indexed by channel number
        self.channels = current_channels or {0: True, 1: True, 2: True}
        self.names = channel_names or {0: 'Channel 1', 1: 'Channel 2', 2: 'Channel 3'}

        # Detect number of channels
        self.num_channels = 3
        if color_image is not None and color_image.ndim == 3:
            self.num_channels = min(color_image.shape[2], 3)

        layout = QVBoxLayout(self)

        # Instructions
        instructions = QLabel("Select which channels to display:")
        layout.addWidget(instructions)

        # Channel checkboxes - simple numbered labels
        self.ch_checks = []
        for i in range(self.num_channels):
            user_name = self.names.get(i, '')
            label = f"Channel {i+1}"
            if user_name and user_name != f'Channel {i+1}':
                label += f": {user_name}"
            check = QCheckBox(label)
            check.setChecked(self.channels.get(i, True))
            layout.addWidget(check)
            self.ch_checks.append(check)

        # Channel naming section
        layout.addSpacing(10)
        name_label = QLabel("Customize channel names:")
        layout.addWidget(name_label)

        name_form = QFormLayout()
        self.name_inputs = []
        for i in range(self.num_channels):
            name_input = QLineEdit(self.names.get(i, ''))
            name_form.addRow(f"Channel {i+1}:", name_input)
            self.name_inputs.append(name_input)
        layout.addLayout(name_form)

        # Quick presets
        layout.addSpacing(10)
        preset_layout = QHBoxLayout()
        all_btn = QPushButton("All")
        all_btn.clicked.connect(self.select_all)
        preset_layout.addWidget(all_btn)

        for i in range(self.num_channels):
            btn = QPushButton(f"Ch{i+1} Only")
            btn.clicked.connect(lambda checked, idx=i: self.select_only(idx))
            preset_layout.addWidget(btn)

        layout.addLayout(preset_layout)

        # OK/Cancel buttons
        layout.addSpacing(10)
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def select_all(self):
        for check in self.ch_checks:
            check.setChecked(True)

    def select_only(self, ch_idx):
        for i, check in enumerate(self.ch_checks):
            check.setChecked(i == ch_idx)

    def get_settings(self):
        """Return the selected channel settings and names"""
        # Return by channel index
        channels = {i: check.isChecked() for i, check in enumerate(self.ch_checks)}
        return {
            'channels': channels,
            'names': {i: inp.text() for i, inp in enumerate(self.name_inputs)}
        }


class GrayscaleChannelDialog(QDialog):
    """Dialog to select channel for outlining AND channels for colocalization"""
    def __init__(self, parent=None, channel_names=None, color_image=None,
                 current_coloc_ch1=0, current_coloc_ch2=1):
        super().__init__(parent)
        self.setWindowTitle("Channel Selection")
        self.setModal(True)

        self.names = channel_names or {0: '', 1: '', 2: ''}

        # Detect number of channels
        self.num_channels = 3
        if color_image is not None and color_image.ndim == 3:
            self.num_channels = min(color_image.shape[2], 3)

        layout = QVBoxLayout(self)

        # === Section 1: Grayscale channel for outlining ===
        outline_label = QLabel("<b>1. Select channel for outlining (grayscale):</b>")
        layout.addWidget(outline_label)

        from PyQt5.QtWidgets import QRadioButton, QButtonGroup, QComboBox
        self.outline_group = QButtonGroup(self)

        self.outline_radios = []
        for i in range(self.num_channels):
            label = f"Channel {i+1}"
            radio = QRadioButton(label)
            if i == 0:
                radio.setChecked(True)
            self.outline_group.addButton(radio, i)
            layout.addWidget(radio)
            self.outline_radios.append(radio)

        layout.addSpacing(15)

        # === Section 2: Colocalization channels ===
        coloc_label = QLabel("<b>2. Select channels to colocalize:</b>")
        layout.addWidget(coloc_label)

        # Channel 1 dropdown
        ch1_layout = QHBoxLayout()
        ch1_label = QLabel("Colocalization Channel 1:")
        self.coloc_ch1_combo = QComboBox()
        for i in range(self.num_channels):
            self.coloc_ch1_combo.addItem(f"Channel {i+1}", i)
        self.coloc_ch1_combo.setCurrentIndex(current_coloc_ch1)
        ch1_layout.addWidget(ch1_label)
        ch1_layout.addWidget(self.coloc_ch1_combo)
        layout.addLayout(ch1_layout)

        # Channel 2 dropdown
        ch2_layout = QHBoxLayout()
        ch2_label = QLabel("Colocalization Channel 2:")
        self.coloc_ch2_combo = QComboBox()
        for i in range(self.num_channels):
            self.coloc_ch2_combo.addItem(f"Channel {i+1}", i)
        self.coloc_ch2_combo.setCurrentIndex(current_coloc_ch2 if current_coloc_ch2 < self.num_channels else 1)
        ch2_layout.addWidget(ch2_label)
        ch2_layout.addWidget(self.coloc_ch2_combo)
        layout.addLayout(ch2_layout)

        layout.addSpacing(15)

        # OK button
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        layout.addWidget(ok_btn)

    def get_selected_channel(self):
        """Get the selected grayscale channel for outlining"""
        return self.outline_group.checkedId()

    def get_coloc_channels(self):
        """Get the selected channels for colocalization"""
        ch1 = self.coloc_ch1_combo.currentData()
        ch2 = self.coloc_ch2_combo.currentData()
        return ch1, ch2


class MicrogliaAnalysisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.images = {}
        self.current_image_name = None
        self.batch_mode = False
        self.soma_picking_queue = []
        self.outlining_queue = []
        self.current_outline_idx = 0
        self.polygon_points = []
        self.output_dir = None
        self.masks_dir = None
        self.pixel_size = 0.3
        self.default_rolling_ball_radius = 50
        self.all_masks_flat = []
        self.mask_qa_idx = 0
        self.mask_qa_active = False
        self.last_qa_decisions = []
        self.soma_mode = False  # Initialize soma_mode to prevent crashes
        # Initialize display adjustment values
        self.brightness_value = 0
        self.contrast_value = 0
        # Per-channel brightness for colocalization mode
        self.channel_brightness = {'R': 0, 'G': 0, 'B': 0}
        # Mask generation settings (defaults)
        self.use_min_intensity = True
        self.min_intensity_percent = 30
        self.mask_min_area = 200
        self.mask_max_area = 800
        self.mask_step_size = 100
        self.use_imagej = False
        # Colocalization mode - show images in color
        self.colocalization_mode = False
        # Channel display settings: which channels to show (by index)
        self.display_channels = {0: True, 1: True, 2: True}
        # Which channel to use for grayscale during outlining
        self.grayscale_channel = 0
        # Which channels to use for colocalization analysis
        self.coloc_channel_1 = 0  # First channel for colocalization
        self.coloc_channel_2 = 1  # Second channel for colocalization
        # Channel names (can be customized by user)
        self.channel_names = {0: '', 1: '', 2: ''}
        # Color/grayscale display toggle
        self.show_color_view = False
        # Z key tracking for zoom functionality
        self.z_key_held = False
        # Measurement tool state
        self.measure_mode = False
        self.measure_point1 = None  # (row, col) image coords
        self.measure_point2 = None
        self.init_ui()

    def keyPressEvent(self, event):
        key = event.key()

        # Track Z key for zoom functionality
        if key == Qt.Key_Z:
            self.z_key_held = True
            return  # Don't process further, Z is for zoom

        # F1 shows help regardless of mode
        if key == Qt.Key_F1:
            self.show_shortcut_help()
            return

        # M key toggles measure tool (only when not in active input modes)
        if key == Qt.Key_M and not self.processed_label.soma_mode and not self.processed_label.polygon_mode and not self.mask_qa_active:
            self.toggle_measure_mode()
            return

        # U key resets zoom on current view
        if key == Qt.Key_U:
            self._reset_current_zoom()
            return

        # C key toggles color/grayscale display
        if key == Qt.Key_C:
            self.toggle_color_view()
            return

        # Handle polygon outlining mode shortcuts
        if self.processed_label.polygon_mode:
            if key == Qt.Key_Backspace:
                # Undo last point (use Backspace only, Z is for zoom)
                self.undo_last_polygon_point()
                return
            elif key == Qt.Key_Escape:
                # Restart current outline
                self.restart_polygon()
                return
            elif key == Qt.Key_Return or key == Qt.Key_Enter:
                # Alternative way to finish polygon
                self.finish_polygon()
                return

        # Handle soma picking mode shortcuts
        if self.processed_label.soma_mode:
            if key == Qt.Key_Backspace:
                # Undo last soma
                self.undo_last_soma()
                return
            elif key == Qt.Key_Return or key == Qt.Key_Enter:
                # Done picking somas for current image
                self.done_with_current()
                return
            elif key == Qt.Key_Escape:
                # Clear all somas on current image (undo all placed centroids)
                if self.current_image_name:
                    img_data = self.images[self.current_image_name]
                    if img_data['somas']:
                        count = len(img_data['somas'])
                        img_data['somas'].clear()
                        img_data['soma_ids'].clear()
                        self.log(f"↩ Cleared {count} soma(s) on {self.current_image_name}")
                        self._load_image_for_soma_picking()
                    else:
                        self.log("No somas to clear on this image")
                return

        # Handle mask QA mode shortcuts
        if not self.mask_qa_active:
            super().keyPressEvent(event)
            return
        try:
            if key == Qt.Key_A:
                self.approve_current_mask()
            elif key == Qt.Key_R:
                self.reject_current_mask()
            elif key == Qt.Key_Left:
                self.prev_mask()
            elif key == Qt.Key_Right:
                self.next_mask()
            elif key == Qt.Key_Space:
                self.approve_current_mask()
            else:
                super().keyPressEvent(event)
        except Exception as e:
            self.log(f"ERROR handling keypress: {e}")

    def keyReleaseEvent(self, event):
        """Track Z key release for zoom functionality"""
        if event.key() == Qt.Key_Z:
            self.z_key_held = False
        else:
            super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        """Reset zoom state when window loses focus to prevent getting stuck"""
        self.z_key_held = False
        super().focusOutEvent(event)

    def init_ui(self):
        self.setWindowTitle("Microglia Analysis - Multi-Image Batch Processing")
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel)
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, 1)

        # Try multiple approaches to ensure full screen
        # Get the screen geometry
        from PyQt5.QtWidgets import QDesktopWidget
        screen = QDesktopWidget().screenGeometry()
        self.setGeometry(screen)
        self.showMaximized()  # Also maximize to be sure

        # Global keyboard shortcuts (work regardless of focus)
        self.shortcut_color = QShortcut(QKeySequence('C'), self)
        self.shortcut_color.activated.connect(self.toggle_color_view)
        self.shortcut_zoom_reset = QShortcut(QKeySequence('U'), self)
        self.shortcut_zoom_reset.activated.connect(self._reset_current_zoom)
        self.shortcut_help = QShortcut(QKeySequence('F1'), self)
        self.shortcut_help.activated.connect(self.show_shortcut_help)
        self.shortcut_measure = QShortcut(QKeySequence('M'), self)
        self.shortcut_measure.activated.connect(self.toggle_measure_mode)

    def _create_left_panel(self):
        panel = QWidget()
        panel.setFixedWidth(450)
        layout = QVBoxLayout(panel)
        file_group = QGroupBox("1. File Selection")
        file_layout = QVBoxLayout()
        select_btn = QPushButton("Select Image Folder")
        select_btn.clicked.connect(self.select_folder)
        file_layout.addWidget(select_btn)
        output_btn = QPushButton("Select Output Folder")
        output_btn.clicked.connect(self.select_output)
        file_layout.addWidget(output_btn)
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_image_selected)
        self.file_list.itemChanged.connect(self.on_item_checkbox_changed)
        file_layout.addWidget(self.file_list)
        btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all_images)
        btn_layout.addWidget(select_all_btn)
        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.clicked.connect(self.clear_all_images)
        btn_layout.addWidget(clear_all_btn)
        label_btn = QPushButton("Image Labeling")
        label_btn.clicked.connect(self.open_image_labeling)
        btn_layout.addWidget(label_btn)
        file_layout.addLayout(btn_layout)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        param_group = QGroupBox("2. Parameters")
        param_layout = QVBoxLayout()
        form_layout = QFormLayout()
        self.pixel_size_input = QLineEdit(str(self.pixel_size))
        form_layout.addRow("Pixel size (μm/px):", self.pixel_size_input)
        param_layout.addLayout(form_layout)

        # Channel selection for processing
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(QLabel("Process channel:"))
        self.process_channel_combo = QComboBox()
        self.process_channel_combo.addItems(["Channel 1", "Channel 2", "Channel 3"])
        self.process_channel_combo.setCurrentIndex(0)
        self.process_channel_combo.currentIndexChanged.connect(self._on_process_channel_changed)
        channel_layout.addWidget(self.process_channel_combo)
        channel_layout.addStretch()
        param_layout.addLayout(channel_layout)


        self.use_imagej = False

        self.rb_check = QCheckBox("Apply Rolling Ball Background Subtraction")
        self.rb_check.setChecked(True)
        param_layout.addWidget(self.rb_check)

        rb_layout = QHBoxLayout()
        rb_layout.addWidget(QLabel("  Rolling ball radius:"))
        self.rb_slider = QSlider(Qt.Horizontal)
        self.rb_slider.setRange(5, 150)
        self.rb_slider.setValue(50)
        rb_layout.addWidget(self.rb_slider)
        self.rb_spinbox = QSpinBox()
        self.rb_spinbox.setRange(5, 150)
        self.rb_spinbox.setValue(50)
        self.rb_slider.valueChanged.connect(self.rb_spinbox.setValue)
        self.rb_spinbox.valueChanged.connect(self.rb_slider.setValue)
        rb_layout.addWidget(self.rb_spinbox)
        param_layout.addLayout(rb_layout)

        # Additional processing options
        extra_processing_group = QGroupBox("Additional Processing (Optional)")
        extra_processing_group.setMinimumHeight(150)
        extra_layout = QVBoxLayout()

        # Denoising
        self.denoise_check = QCheckBox("Apply Denoising (Median Filter)")
        self.denoise_check.setChecked(False)
        extra_layout.addWidget(self.denoise_check)

        denoise_layout = QHBoxLayout()
        denoise_layout.addWidget(QLabel("  Denoise size:"))
        self.denoise_spin = QSpinBox()
        self.denoise_spin.setRange(3, 7)
        self.denoise_spin.setValue(3)
        self.denoise_spin.setSingleStep(2)
        denoise_layout.addWidget(self.denoise_spin)
        denoise_layout.addWidget(QLabel("(3=gentle, 7=strong)"))
        denoise_layout.addStretch()
        extra_layout.addLayout(denoise_layout)

        # Sharpening
        self.sharpen_check = QCheckBox("Apply Sharpening (Unsharp Mask)")
        self.sharpen_check.setChecked(False)
        extra_layout.addWidget(self.sharpen_check)

        sharpen_layout = QHBoxLayout()
        sharpen_layout.addWidget(QLabel("  Sharpen amount:"))
        self.sharpen_slider = QSlider(Qt.Horizontal)
        self.sharpen_slider.setRange(10, 50)
        self.sharpen_slider.setValue(13)
        sharpen_layout.addWidget(self.sharpen_slider)
        self.sharpen_label = QLabel("1.3")
        self.sharpen_slider.valueChanged.connect(lambda v: self.sharpen_label.setText(f"{v / 10:.1f}"))
        sharpen_layout.addWidget(self.sharpen_label)
        extra_layout.addLayout(sharpen_layout)

        extra_processing_group.setLayout(extra_layout)
        param_layout.addWidget(extra_processing_group)

        self.preview_btn = QPushButton("Preview Current Image")
        self.preview_btn.clicked.connect(self.preview_current_image)
        param_layout.addWidget(self.preview_btn)
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        batch_group = QGroupBox("3. Batch Processing")
        batch_layout = QVBoxLayout()
        self.process_selected_btn = QPushButton("Process Selected Images")
        self.process_selected_btn.clicked.connect(self.process_selected_images)
        self.process_selected_btn.setEnabled(False)
        batch_layout.addWidget(self.process_selected_btn)
        self.batch_pick_somas_btn = QPushButton("Pick Somas (All Images)")
        self.batch_pick_somas_btn.clicked.connect(self.start_batch_soma_picking)
        self.batch_pick_somas_btn.setEnabled(False)
        batch_layout.addWidget(self.batch_pick_somas_btn)

        # Outline Somas - single button, controls hidden until outlining starts
        self.batch_outline_btn = QPushButton("Outline Somas")
        self.batch_outline_btn.clicked.connect(self.start_batch_outlining)
        self.batch_outline_btn.setEnabled(False)
        self.batch_outline_btn.setStyleSheet("font-weight: bold;")
        batch_layout.addWidget(self.batch_outline_btn)

        # Hidden auto-outline settings (store state but not shown in panel)
        self.auto_outline_method = QComboBox()
        self.auto_outline_method.addItems(["Threshold", "Region Grow", "Watershed", "Active Contour", "Hybrid"])
        self.auto_outline_method.setCurrentIndex(0)
        self.auto_outline_method.setVisible(False)
        self.auto_outline_sensitivity = QSlider(Qt.Horizontal)
        self.auto_outline_sensitivity.setRange(10, 90)
        self.auto_outline_sensitivity.setValue(20)
        self.auto_outline_sensitivity.setVisible(False)
        self.sensitivity_label = QLabel("20")
        self.auto_outline_sensitivity.valueChanged.connect(
            lambda v: self.sensitivity_label.setText(str(v))
        )

        # Outline controls container - shown only during active outlining
        self.outline_controls_widget = QWidget()
        outline_controls_layout = QVBoxLayout(self.outline_controls_widget)
        outline_controls_layout.setContentsMargins(0, 0, 0, 0)

        # Auto-outline settings row (visible during outlining)
        auto_settings = QHBoxLayout()
        auto_settings.addWidget(QLabel("Method:"))
        self.outline_method_display = QComboBox()
        self.outline_method_display.addItems(["Threshold", "Region Grow", "Watershed", "Active Contour", "Hybrid"])
        self.outline_method_display.setCurrentIndex(0)
        self.outline_method_display.setMaximumWidth(110)
        self.outline_method_display.currentIndexChanged.connect(
            lambda idx: self.auto_outline_method.setCurrentIndex(idx)
        )
        auto_settings.addWidget(self.outline_method_display)
        auto_settings.addWidget(QLabel("Sens:"))
        self.outline_sens_display = QSlider(Qt.Horizontal)
        self.outline_sens_display.setRange(10, 90)
        self.outline_sens_display.setValue(50)
        self.outline_sens_display.setMaximumWidth(60)
        self.outline_sens_display.valueChanged.connect(
            lambda v: self.auto_outline_sensitivity.setValue(v)
        )
        auto_settings.addWidget(self.outline_sens_display)
        self.outline_sens_label = QLabel("50")
        self.outline_sens_display.valueChanged.connect(
            lambda v: self.outline_sens_label.setText(str(v))
        )
        auto_settings.addWidget(self.outline_sens_label)
        outline_controls_layout.addLayout(auto_settings)

        # Action buttons row
        outline_btn_layout = QHBoxLayout()
        self.auto_outline_btn = QPushButton("Auto")
        self.auto_outline_btn.clicked.connect(self.auto_outline_current_soma)
        self.auto_outline_btn.setEnabled(False)
        self.auto_outline_btn.setStyleSheet("border: 2px solid #4CAF50; font-weight: bold;")
        self.auto_outline_btn.setToolTip("Auto-detect outline for current soma")
        outline_btn_layout.addWidget(self.auto_outline_btn)

        self.manual_draw_btn = QPushButton("Manual")
        self.manual_draw_btn.clicked.connect(self.start_manual_outline)
        self.manual_draw_btn.setEnabled(False)
        self.manual_draw_btn.setToolTip("Draw outline manually (click points)")
        outline_btn_layout.addWidget(self.manual_draw_btn)

        self.accept_outline_btn = QPushButton("Accept (Enter)")
        self.accept_outline_btn.clicked.connect(self.accept_current_outline)
        self.accept_outline_btn.setEnabled(False)
        self.accept_outline_btn.setStyleSheet("border: 2px solid #2196F3; font-weight: bold;")
        self.accept_outline_btn.setToolTip("Accept outline and move to next soma")
        outline_btn_layout.addWidget(self.accept_outline_btn)
        outline_controls_layout.addLayout(outline_btn_layout)

        # Redo button
        self.redo_outline_btn = QPushButton("↩ Redo Last")
        self.redo_outline_btn.clicked.connect(self.redo_last_outline)
        self.redo_outline_btn.setEnabled(False)
        self.redo_outline_btn.setStyleSheet("border: 2px solid #FF9800;")
        outline_controls_layout.addWidget(self.redo_outline_btn)

        # Outline progress bar
        self.outline_progress_bar = QProgressBar()
        self.outline_progress_bar.setVisible(False)
        self.outline_progress_bar.setMinimumHeight(20)
        self.outline_progress_bar.setFormat("%v / %m somas outlined")
        self.outline_progress_bar.setStyleSheet("""
            QProgressBar { border: 1px solid palette(mid); border-radius: 3px; text-align: center; }
            QProgressBar::chunk { background-color: #4CAF50; }
        """)
        outline_controls_layout.addWidget(self.outline_progress_bar)

        self.outline_controls_widget.setVisible(False)
        batch_layout.addWidget(self.outline_controls_widget)
        
        self.batch_generate_masks_btn = QPushButton("Generate All Masks")
        self.batch_generate_masks_btn.clicked.connect(self.batch_generate_masks)
        self.batch_generate_masks_btn.setEnabled(False)
        batch_layout.addWidget(self.batch_generate_masks_btn)

        # Add Clear All Masks button
        self.clear_masks_btn = QPushButton("🗑 Clear All Masks")
        self.clear_masks_btn.clicked.connect(self.clear_all_masks)
        self.clear_masks_btn.setEnabled(False)
        self.clear_masks_btn.setStyleSheet("border: 2px solid #F44336;")
        batch_layout.addWidget(self.clear_masks_btn)
        qa_row = QHBoxLayout()
        self.batch_qa_btn = QPushButton("QA All Masks")
        self.batch_qa_btn.clicked.connect(self.start_batch_qa)
        self.batch_qa_btn.setEnabled(False)
        qa_row.addWidget(self.batch_qa_btn)
        self.undo_qa_btn = QPushButton("Undo QA")
        self.undo_qa_btn.clicked.connect(self.undo_last_qa)
        self.undo_qa_btn.setEnabled(False)
        self.undo_qa_btn.setToolTip("Reset all mask approvals and restart QA")
        self.undo_qa_btn.setStyleSheet("border: 2px solid #FF9800;")
        qa_row.addWidget(self.undo_qa_btn)
        batch_layout.addLayout(qa_row)
        self.batch_calculate_btn = QPushButton("Calculate Simple Characteristics")
        self.batch_calculate_btn.clicked.connect(self.batch_calculate_morphology)
        self.batch_calculate_btn.setEnabled(False)
        batch_layout.addWidget(self.batch_calculate_btn)

        # ImageJ integration buttons
        imagej_layout = QHBoxLayout()
        self.launch_imagej_btn = QPushButton("Generate ImageJ Scripts")
        self.launch_imagej_btn.clicked.connect(self.generate_imagej_scripts)
        self.launch_imagej_btn.setEnabled(False)
        self.launch_imagej_btn.setToolTip("Generate Sholl & Skeleton analysis scripts for Fiji")
        imagej_layout.addWidget(self.launch_imagej_btn)
        self.import_imagej_btn = QPushButton("Import ImageJ Results")
        self.import_imagej_btn.clicked.connect(self.import_imagej_results)
        self.import_imagej_btn.setEnabled(False)
        self.import_imagej_btn.setToolTip("Import Sholl & Skeleton CSVs and merge with morphology results")
        imagej_layout.addWidget(self.import_imagej_btn)
        batch_layout.addLayout(imagej_layout)

        # Session save/restore buttons
        session_layout = QHBoxLayout()
        save_session_btn = QPushButton("Save Session")
        save_session_btn.clicked.connect(self.save_session)
        save_session_btn.setToolTip("Save current project state to resume later")
        session_layout.addWidget(save_session_btn)
        load_session_btn = QPushButton("Load Session")
        load_session_btn.clicked.connect(self.load_session)
        load_session_btn.setToolTip("Resume a previously saved session")
        session_layout.addWidget(load_session_btn)
        batch_layout.addLayout(session_layout)

        batch_group.setLayout(batch_layout)
        layout.addWidget(batch_group)
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # Legend button
        legend_btn = QPushButton("Legend")
        legend_btn.clicked.connect(self.show_legend)
        layout.addWidget(legend_btn)

        layout.addStretch()
        return panel

    def _create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)  # type: QVBoxLayout
        self.tabs = QTabWidget()
        self.original_label = InteractiveImageLabel(self)
        self.original_label.setText("Load images to begin")
        self.tabs.addTab(self.original_label, "Original")
        self.preview_label = InteractiveImageLabel(self)
        self.preview_label.setText("No preview yet")
        self.tabs.addTab(self.preview_label, "Preview")
        self.processed_label = InteractiveImageLabel(self)
        self.processed_label.setText("No processed images yet")
        self.tabs.addTab(self.processed_label, "Processed")
        self.mask_label = InteractiveImageLabel(self)
        self.mask_label.setText("No masks yet")
        self.tabs.addTab(self.mask_label, "Masks")

        # Give tabs most of the space (stretch factor)
        layout.addWidget(self.tabs, stretch=1)

        # Display adjustments buttons in a row
        display_btn_layout = QHBoxLayout()
        display_adjust_btn = QPushButton("Display Adjustments")
        display_adjust_btn.clicked.connect(self.open_display_adjustments)
        display_btn_layout.addWidget(display_adjust_btn)

        # Color/Grayscale toggle button
        self.color_toggle_btn = QPushButton("Show Color (C)")
        self.color_toggle_btn.clicked.connect(self.toggle_color_view)
        self.color_toggle_btn.setToolTip("Toggle between color and grayscale display")
        display_btn_layout.addWidget(self.color_toggle_btn)

        # Channel selection button (only visible when color view is on)
        self.channel_select_btn = QPushButton("Channel Display")
        self.channel_select_btn.clicked.connect(self.open_channel_selector)
        self.channel_select_btn.setToolTip("Select which color channels to display")
        self.channel_select_btn.setVisible(False)  # Hidden until color view is on
        display_btn_layout.addWidget(self.channel_select_btn)

        # Measure tool button
        self.measure_btn = QPushButton("Measure (M)")
        self.measure_btn.clicked.connect(self.toggle_measure_mode)
        self.measure_btn.setToolTip("Click two points to measure distance in microns")
        self.measure_btn.setCheckable(True)
        display_btn_layout.addWidget(self.measure_btn)

        # Help button
        help_btn = QPushButton("? (F1)")
        help_btn.setFixedWidth(55)
        help_btn.clicked.connect(self.show_shortcut_help)
        help_btn.setToolTip("Show keyboard shortcuts for current mode")
        display_btn_layout.addWidget(help_btn)

        layout.addLayout(display_btn_layout)

        # Mask overlay opacity slider
        opacity_layout = QHBoxLayout()
        opacity_label = QLabel("Mask Opacity:")
        opacity_label.setFixedWidth(85)
        opacity_layout.addWidget(opacity_label)
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(40)
        self.opacity_slider.setTickInterval(10)
        self.opacity_slider.setFixedHeight(20)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_layout.addWidget(self.opacity_slider)
        self.opacity_value_label = QLabel("40%")
        self.opacity_value_label.setFixedWidth(35)
        opacity_layout.addWidget(self.opacity_value_label)
        layout.addLayout(opacity_layout)

        # Zoom hint row
        zoom_layout = QHBoxLayout()
        zoom_hint = QLabel("Z + Left-click: zoom in, Z + Right-click: zoom out, U: reset, M: measure, F1: help")
        zoom_hint.setStyleSheet("color: palette(dark); font-size: 10px;")
        zoom_layout.addWidget(zoom_hint)
        zoom_layout.addStretch()

        # Redo masks button — bottom right, only visible during QA
        self.regen_masks_btn = QPushButton("Redo Masks (This Image)")
        self.regen_masks_btn.clicked.connect(self.regenerate_masks_current_image)
        self.regen_masks_btn.setVisible(False)
        self.regen_masks_btn.setStyleSheet("border: 2px solid #FF9800; font-weight: bold; padding: 4px 10px;")
        self.regen_masks_btn.setToolTip("Regenerate masks for this image with different settings")
        zoom_layout.addWidget(self.regen_masks_btn)

        self.zoom_level_label = QLabel("1.0x")
        self.zoom_level_label.setFixedWidth(50)
        self.zoom_level_label.setStyleSheet("color: palette(dark); font-size: 10px;")
        zoom_layout.addWidget(self.zoom_level_label)
        layout.addLayout(zoom_layout)

        # Progress bar with timer
        progress_container = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(25)
        progress_container.addWidget(self.progress_bar, stretch=3)

        self.timer_label = QLabel("")
        self.timer_label.setVisible(False)
        self.timer_label.setStyleSheet("font-family: monospace; font-size: 12pt; padding: 0 10px;")
        self.timer_label.setMinimumWidth(100)
        self.timer_label.setAlignment(Qt.AlignCenter)
        progress_container.addWidget(self.timer_label, stretch=0)

        layout.addLayout(progress_container)

        self.progress_status_label = QLabel("")
        self.progress_status_label.setVisible(False)
        self.progress_status_label.setStyleSheet("font-style: italic;")
        layout.addWidget(self.progress_status_label, stretch=0)

        # Timer for tracking processing time
        from PyQt5.QtCore import QTimer
        self.process_timer = QTimer()
        self.process_timer.timeout.connect(self.update_timer_display)
        self.process_start_time = None
        self.timer_running = False

        # Hidden navigation buttons (for compatibility - not displayed but needed by code)
        self.prev_btn = QPushButton("< Previous")
        self.prev_btn.clicked.connect(self.navigate_previous)
        self.prev_btn.setEnabled(False)
        self.prev_btn.setVisible(False)

        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self.navigate_next)
        self.next_btn.setEnabled(False)
        self.next_btn.setVisible(False)

        self.done_btn = QPushButton("Done with Current")
        self.done_btn.clicked.connect(self.done_with_current)
        self.done_btn.setEnabled(False)
        self.done_btn.setVisible(False)

        self.nav_status_label = QLabel("")
        self.nav_status_label.setVisible(False)

        self.approve_mask_btn = QPushButton("Approve (A)")
        self.approve_mask_btn.clicked.connect(self.approve_current_mask)
        self.approve_mask_btn.setEnabled(False)
        self.approve_mask_btn.setVisible(False)

        self.reject_mask_btn = QPushButton("Reject (R)")
        self.reject_mask_btn.clicked.connect(self.reject_current_mask)
        self.reject_mask_btn.setEnabled(False)
        self.reject_mask_btn.setVisible(False)

        return panel

    def update_timer_display(self):
        """Update the timer display during processing"""
        if self.process_start_time is not None:
            elapsed = time.time() - self.process_start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.timer_label.setText(f"{minutes:02d}:{seconds:02d}")

    def start_timer(self):
        """Start the processing timer"""
        self.process_start_time = time.time()
        self.timer_running = True
        self.timer_label.setVisible(True)
        self.timer_label.setText("00:00")
        self.process_timer.start(1000)  # Update every second

    def stop_timer(self):
        """Stop the processing timer"""
        self.process_timer.stop()
        self.timer_running = False
        # Keep the final time visible

    def calculate_colocalization(self, mask, img_name):
        """
        Calculate colocalization between two user-selected channels.

        Uses Otsu thresholding to separate signal from background.
        A channel with no real signal will have very few pixels above threshold.
        """
        if not self.colocalization_mode:
            return {}

        img_data = self.images.get(img_name)
        if img_data is None or 'color_image' not in img_data:
            return {'coloc_status': 'no_color_data'}

        color_img = img_data['color_image']
        if color_img.ndim != 3:
            return {'coloc_status': 'not_multichannel'}

        # Validate channel indices
        n_channels = color_img.shape[2]
        if self.coloc_channel_1 >= n_channels or self.coloc_channel_2 >= n_channels:
            return {'coloc_status': 'invalid_channels'}

        # Apply mask - ONLY analyze pixels within the cell mask
        mask_bool = mask > 0
        if not np.any(mask_bool):
            return {'coloc_status': 'empty_mask'}

        n_mask_pixels = np.sum(mask_bool)

        # Get user-selected channels within mask only
        ch1_full = color_img[:, :, self.coloc_channel_1].astype(np.float64)
        ch2_full = color_img[:, :, self.coloc_channel_2].astype(np.float64)

        ch1_masked = ch1_full[mask_bool]
        ch2_masked = ch2_full[mask_bool]

        results = {
            'coloc_status': 'ok',
            'coloc_ch1': self.coloc_channel_1 + 1,
            'coloc_ch2': self.coloc_channel_2 + 1,
            'n_mask_pixels': int(n_mask_pixels)
        }

        # === OTSU THRESHOLDING ===
        # Use Otsu's method to find optimal threshold separating background from signal
        from skimage.filters import threshold_otsu

        # For each channel, find Otsu threshold and check if there's real signal
        def get_signal_pixels(channel_data):
            """
            Use Otsu thresholding to find signal pixels.
            Returns: threshold, signal_mask, has_real_signal
            """
            ch_min = np.min(channel_data)
            ch_max = np.max(channel_data)
            ch_range = ch_max - ch_min

            # If the range is tiny, there's no real signal
            if ch_range < 5:
                return ch_max, np.zeros(len(channel_data), dtype=bool), False

            try:
                # Otsu threshold
                thresh = threshold_otsu(channel_data)
            except:
                # If Otsu fails, use median
                thresh = np.median(channel_data)

            signal_mask = channel_data > thresh
            n_signal = np.sum(signal_mask)

            # Check if this is real signal or just noise
            # Real signal: threshold should be well below max, and significant pixels above it
            # No signal: threshold near max, very few pixels above
            signal_fraction = n_signal / len(channel_data)
            threshold_position = (thresh - ch_min) / ch_range if ch_range > 0 else 1

            # If threshold is > 80% of the way to max, and < 10% of pixels are "signal",
            # this channel likely has no real signal
            has_real_signal = not (threshold_position > 0.8 and signal_fraction < 0.1)

            # Also check: if the "signal" pixels aren't much brighter than threshold, it's noise
            if n_signal > 0:
                signal_values = channel_data[signal_mask]
                mean_signal = np.mean(signal_values)
                # Signal should be at least 20% brighter than threshold
                if mean_signal < thresh * 1.2:
                    has_real_signal = False

            return thresh, signal_mask, has_real_signal

        ch1_thresh, ch1_signal_mask, ch1_has_signal = get_signal_pixels(ch1_masked)
        ch2_thresh, ch2_signal_mask, ch2_has_signal = get_signal_pixels(ch2_masked)

        n_ch1_signal = np.sum(ch1_signal_mask)
        n_ch2_signal = np.sum(ch2_signal_mask)

        # Store diagnostic info
        results['ch1_threshold'] = round(float(ch1_thresh), 2)
        results['ch1_has_signal'] = ch1_has_signal
        results['ch1_min'] = round(float(np.min(ch1_masked)), 2)
        results['ch1_max'] = round(float(np.max(ch1_masked)), 2)
        results['ch2_threshold'] = round(float(ch2_thresh), 2)
        results['ch2_has_signal'] = ch2_has_signal
        results['ch2_min'] = round(float(np.min(ch2_masked)), 2)
        results['ch2_max'] = round(float(np.max(ch2_masked)), 2)
        results['n_ch1_signal'] = int(n_ch1_signal)
        results['n_ch2_signal'] = int(n_ch2_signal)

        # === COLOCALIZATION ===
        # Only count colocalization if BOTH channels have real signal
        if not ch1_has_signal or not ch2_has_signal:
            results['n_coloc_pixels'] = 0
            results['ch1_coloc_percent'] = 0.0
            results['ch2_coloc_percent'] = 0.0
            results['pearson_r'] = 0.0
            results['coloc_status'] = 'no_signal_in_channel'
            return results

        # A pixel is colocalized ONLY if BOTH channels have signal at that location
        colocalized_mask = ch1_signal_mask & ch2_signal_mask
        n_coloc = np.sum(colocalized_mask)
        results['n_coloc_pixels'] = int(n_coloc)

        # Percent colocalized
        results['ch1_coloc_percent'] = round((n_coloc / n_ch1_signal) * 100, 2) if n_ch1_signal > 0 else 0.0
        results['ch2_coloc_percent'] = round((n_coloc / n_ch2_signal) * 100, 2) if n_ch2_signal > 0 else 0.0

        # === PEARSON'S R ===
        # Only calculate if there are enough colocalized pixels
        if n_coloc >= 10:
            ch1_coloc_values = ch1_masked[colocalized_mask]
            ch2_coloc_values = ch2_masked[colocalized_mask]

            if np.std(ch1_coloc_values) > 0 and np.std(ch2_coloc_values) > 0:
                pearson_r, _ = stats.pearsonr(ch1_coloc_values, ch2_coloc_values)
                results['pearson_r'] = round(pearson_r, 4)
            else:
                results['pearson_r'] = 0.0
        else:
            results['pearson_r'] = 0.0

        return results

        return results

    def log(self, message):
        self.log_text.append(str(message))

    # ========================================================================
    # KEYBOARD SHORTCUT HELP
    # ========================================================================

    def show_shortcut_help(self):
        """Show context-sensitive keyboard shortcuts overlay"""
        # Determine current mode
        mode = "General"
        if self.mask_qa_active:
            mode = "Mask QA"
        elif hasattr(self, 'processed_label') and self.processed_label.polygon_mode:
            mode = "Soma Outlining"
        elif hasattr(self, 'processed_label') and self.processed_label.soma_mode:
            mode = "Soma Picking"

        html = "<h3>Keyboard Shortcuts</h3>"
        html += f"<p><b>Current mode: {mode}</b></p>"
        html += "<table cellpadding='4' style='border-collapse: collapse;'>"

        # Always-available shortcuts
        html += "<tr><td colspan='2' style='border-bottom: 1px solid #ccc;'><b>Always Available</b></td></tr>"
        always = [
            ("F1", "Show this help"),
            ("C", "Toggle color / grayscale"),
            ("U", "Reset zoom"),
            ("Z + Left-click", "Zoom in"),
            ("Z + Right-click", "Zoom out"),
            ("M", "Toggle measure tool"),
        ]
        for key, desc in always:
            html += f"<tr><td><code>{key}</code></td><td>{desc}</td></tr>"

        if mode == "Soma Picking":
            html += "<tr><td colspan='2' style='border-bottom: 1px solid #ccc;'><b>Soma Picking</b></td></tr>"
            shortcuts = [
                ("Left-click", "Place soma / drag existing centroid"),
                ("Backspace", "Remove last soma"),
                ("Enter", "Done with current image"),
                ("Escape", "Cancel soma picking"),
            ]
            for key, desc in shortcuts:
                html += f"<tr><td><code>{key}</code></td><td>{desc}</td></tr>"

        elif mode == "Soma Outlining":
            html += "<tr><td colspan='2' style='border-bottom: 1px solid #ccc;'><b>Soma Outlining</b></td></tr>"
            shortcuts = [
                ("Left-click", "Place outline point / drag existing point"),
                ("Right-click", "Finish outline"),
                ("Double-click", "Finish outline"),
                ("Backspace", "Remove last point"),
                ("Escape", "Restart current outline"),
                ("Enter", "Accept outline"),
            ]
            for key, desc in shortcuts:
                html += f"<tr><td><code>{key}</code></td><td>{desc}</td></tr>"

        elif mode == "Mask QA":
            html += "<tr><td colspan='2' style='border-bottom: 1px solid #ccc;'><b>Mask QA Review</b></td></tr>"
            shortcuts = [
                ("A / Space", "Approve current mask"),
                ("R", "Reject current mask"),
                ("Left arrow", "Previous mask"),
                ("Right arrow", "Next mask"),
            ]
            for key, desc in shortcuts:
                html += f"<tr><td><code>{key}</code></td><td>{desc}</td></tr>"

        html += "</table>"

        dialog = QMessageBox(self)
        dialog.setWindowTitle("Keyboard Shortcuts")
        dialog.setIcon(QMessageBox.Information)
        dialog.setText(html)
        dialog.setTextFormat(Qt.RichText)
        dialog.exec_()

    # ========================================================================
    # MEASUREMENT TOOL
    # ========================================================================

    def _on_opacity_changed(self, value):
        """Update mask overlay opacity on all labels"""
        opacity = value / 100.0
        self.opacity_value_label.setText(f"{value}%")
        for label in [self.original_label, self.preview_label, self.processed_label, self.mask_label]:
            label.overlay_opacity = opacity
            label._update_display()

    def toggle_measure_mode(self):
        """Toggle the measurement tool on/off"""
        self.measure_mode = not self.measure_mode
        self.measure_btn.setChecked(self.measure_mode)

        # Set mode on the active display label
        for label in [self.original_label, self.preview_label, self.processed_label, self.mask_label]:
            label.measure_mode = self.measure_mode
            if not self.measure_mode:
                label.measure_pt1 = None
                label.measure_pt2 = None
                label._update_display()

        if self.measure_mode:
            self.log("Measure tool ON - click two points to measure distance")
        else:
            self.log("Measure tool OFF")

    def _get_measure_text(self):
        """Get formatted measurement text for display overlay"""
        label = self._get_active_label()
        if not label or label.measure_pt1 is None or label.measure_pt2 is None:
            return ""
        try:
            pixel_size = float(self.pixel_size_input.text())
        except (ValueError, AttributeError):
            pixel_size = 0.316

        pt1 = label.measure_pt1
        pt2 = label.measure_pt2
        dx = (pt2[1] - pt1[1])
        dy = (pt2[0] - pt1[0])
        dist_px = math.sqrt(dx * dx + dy * dy)
        dist_um = dist_px * pixel_size
        return f"{dist_um:.2f} um ({dist_px:.0f} px)"

    def _show_measurement(self):
        """Log the measurement result"""
        text = self._get_measure_text()
        if text:
            self.log(f"Measurement: {text}")

    def _get_active_label(self):
        """Return the InteractiveImageLabel in the currently visible tab"""
        idx = self.tabs.currentIndex()
        labels = [self.original_label, self.preview_label, self.processed_label, self.mask_label]
        if 0 <= idx < len(labels):
            return labels[idx]
        return self.processed_label

    # ========================================================================
    # SESSION SAVE / RESTORE
    # ========================================================================

    def _build_session_dict(self):
        """Build a serializable session dictionary from current state."""
        session = {
            'version': 2,
            'output_dir': self.output_dir,
            'masks_dir': self.masks_dir,
            'colocalization_mode': self.colocalization_mode,
            'pixel_size': self.pixel_size_input.text(),
            'rolling_ball_radius': self.default_rolling_ball_radius,
            'use_min_intensity': self.use_min_intensity,
            'min_intensity_percent': self.min_intensity_percent,
            'mask_min_area': self.mask_min_area,
            'mask_max_area': self.mask_max_area,
            'mask_step_size': self.mask_step_size,
            'coloc_channel_1': self.coloc_channel_1,
            'coloc_channel_2': self.coloc_channel_2,
            'grayscale_channel': self.grayscale_channel,
            'images': {}
        }

        for img_name, img_data in self.images.items():
            # Build path to processed TIFF if it exists on disk
            processed_path = None
            if self.output_dir:
                candidate = os.path.join(
                    self.output_dir,
                    os.path.splitext(img_name)[0] + "_processed.tif"
                )
                if os.path.exists(candidate):
                    processed_path = candidate

            img_session = {
                'raw_path': img_data['raw_path'],
                'processed_path': processed_path,
                'status': img_data['status'],
                'selected': img_data['selected'],
                'animal_id': img_data.get('animal_id', ''),
                'treatment': img_data.get('treatment', ''),
                'rolling_ball_radius': img_data.get('rolling_ball_radius', 50),
                'somas': [(float(s[0]), float(s[1])) for s in img_data.get('somas', [])],
                'soma_ids': img_data.get('soma_ids', []),
                'soma_outlines': [],
            }
            # Save soma outlines with full metadata
            for outline in img_data.get('soma_outlines', []):
                outline_data = {
                    'soma_idx': outline.get('soma_idx', 0),
                    'soma_id': outline.get('soma_id', ''),
                    'centroid': [float(outline['centroid'][0]), float(outline['centroid'][1])] if 'centroid' in outline else None,
                    'soma_area_um2': outline.get('soma_area_um2', 0),
                    'polygon_points': [(float(pt[0]), float(pt[1])) for pt in outline.get('polygon_points', [])],
                }
                img_session['soma_outlines'].append(outline_data)

            # Record which masks exist on disk for this image
            if self.masks_dir:
                prefix = os.path.splitext(img_name)[0]
                mask_files = [
                    f for f in os.listdir(self.masks_dir)
                    if f.startswith(prefix) and f.endswith('_mask.tif')
                ] if os.path.isdir(self.masks_dir) else []
                img_session['mask_files'] = mask_files
            else:
                img_session['mask_files'] = []

            session['images'][img_name] = img_session

        return session

    def save_session(self):
        """Save the entire project state to a JSON file"""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "", "Session Files (*.mmps_session);;All Files (*)",
            options=QFileDialog.DontUseNativeDialog
        )
        if not path:
            return

        if not path.endswith('.mmps_session'):
            path += '.mmps_session'

        try:
            session = self._build_session_dict()
            with open(path, 'w') as f:
                json.dump(session, f, indent=2)

            self.log(f"Session saved to: {path}")
            QMessageBox.information(self, "Session Saved", f"Session saved to:\n{path}")

        except Exception as e:
            self.log(f"ERROR saving session: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save session:\n{e}")

    def _auto_save(self):
        """Silently auto-save the session to the output directory."""
        if not self.output_dir or not self.images:
            return
        try:
            path = os.path.join(self.output_dir, "autosave.mmps_session")
            session = self._build_session_dict()
            with open(path, 'w') as f:
                json.dump(session, f, indent=2)
        except Exception:
            pass  # Silent - never interrupt the user's workflow

    def load_session(self):
        """Restore a previously saved session"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "", "Session Files (*.mmps_session);;All Files (*)",
            options=QFileDialog.DontUseNativeDialog
        )
        if not path:
            return

        try:
            with open(path, 'r') as f:
                session = json.load(f)

            if session.get('version', 1) < 2:
                QMessageBox.warning(self, "Warning", "Incompatible session file version.")
                return

            # Verify image files still exist (check both raw and processed)
            missing = []
            for img_name, img_session in session['images'].items():
                raw_exists = os.path.exists(img_session['raw_path'])
                proc_exists = img_session.get('processed_path') and os.path.exists(img_session['processed_path'])
                if not raw_exists and not proc_exists:
                    missing.append(img_name)

            if missing:
                reply = QMessageBox.question(
                    self, "Missing Files",
                    f"{len(missing)} image(s) not found at original paths:\n\n" +
                    "\n".join(missing[:5]) +
                    ("\n..." if len(missing) > 5 else "") +
                    "\n\nContinue loading available images?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return

            # Restore settings
            self.output_dir = session.get('output_dir')
            self.masks_dir = session.get('masks_dir')
            self.colocalization_mode = session.get('colocalization_mode', False)
            self.use_min_intensity = session.get('use_min_intensity', True)
            self.min_intensity_percent = session.get('min_intensity_percent', 30)
            self.mask_min_area = session.get('mask_min_area', 200)
            self.mask_max_area = session.get('mask_max_area', 800)
            self.mask_step_size = session.get('mask_step_size', 100)
            self.coloc_channel_1 = session.get('coloc_channel_1', 0)
            self.coloc_channel_2 = session.get('coloc_channel_2', 1)
            self.grayscale_channel = session.get('grayscale_channel', 0)

            pixel_size = session.get('pixel_size', '0.316')
            self.pixel_size_input.setText(str(pixel_size))
            self.default_rolling_ball_radius = session.get('rolling_ball_radius', 50)

            if self.colocalization_mode:
                self.show_color_view = True
                self.color_toggle_btn.setText("Show Grayscale (C)")
                self.channel_select_btn.setVisible(True)

            # Ensure output dirs exist
            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
            if self.masks_dir:
                os.makedirs(self.masks_dir, exist_ok=True)
                self.somas_dir = os.path.join(self.output_dir, "somas") if self.output_dir else None
                if self.somas_dir:
                    os.makedirs(self.somas_dir, exist_ok=True)

            # Restore images
            from PyQt5.QtGui import QColor, QBrush
            self.images = {}
            self.file_list.clear()

            for img_name, img_session in session['images'].items():
                if img_name in missing:
                    continue

                # Try to reload processed image from disk
                processed_data = None
                processed_path = img_session.get('processed_path')
                if processed_path and os.path.exists(processed_path):
                    try:
                        processed_data = tifffile.imread(processed_path)
                    except Exception:
                        processed_data = None

                # Reconstruct soma outlines with full metadata
                restored_outlines = []
                for outline_data in img_session.get('soma_outlines', []):
                    if isinstance(outline_data, dict) and 'soma_idx' in outline_data:
                        # New format: full outline dict
                        polygon_pts = [tuple(pt) for pt in outline_data.get('polygon_points', [])]
                        centroid = tuple(outline_data['centroid']) if outline_data.get('centroid') else None
                        # Reconstruct the outline mask from polygon points if we have the processed image
                        outline_mask = None
                        if polygon_pts and len(polygon_pts) >= 3 and processed_data is not None:
                            outline_mask = self._polygon_to_mask(polygon_pts, processed_data.shape)
                        restored_outlines.append({
                            'soma_idx': outline_data['soma_idx'],
                            'soma_id': outline_data['soma_id'],
                            'centroid': centroid,
                            'outline': outline_mask,
                            'polygon_points': polygon_pts,
                            'soma_area_um2': outline_data.get('soma_area_um2', 0),
                        })
                    else:
                        # Legacy format: flat list of points
                        polygon_pts = [tuple(pt) for pt in outline_data]
                        restored_outlines.append({
                            'soma_idx': len(restored_outlines),
                            'soma_id': '',
                            'centroid': None,
                            'outline': None,
                            'polygon_points': polygon_pts,
                            'soma_area_um2': 0,
                        })

                self.images[img_name] = {
                    'raw_path': img_session['raw_path'],
                    'processed': processed_data,
                    'rolling_ball_radius': img_session.get('rolling_ball_radius', 50),
                    'somas': [tuple(s) for s in img_session.get('somas', [])],
                    'soma_ids': img_session.get('soma_ids', []),
                    'soma_outlines': restored_outlines,
                    'masks': [],
                    'status': img_session.get('status', 'loaded'),
                    'selected': img_session.get('selected', False),
                    'animal_id': img_session.get('animal_id', ''),
                    'treatment': img_session.get('treatment', ''),
                }

                # If processed image wasn't found, downgrade status
                # But don't downgrade qa_complete — masks on disk are still valid
                if processed_data is None and img_session.get('status') not in ('loaded', 'qa_complete', 'analyzed'):
                    self.images[img_name]['status'] = 'loaded'

                # Load mask TIFFs from disk for qa_complete/analyzed images
                orig_status = img_session.get('status', 'loaded')
                if orig_status in ('qa_complete', 'analyzed') and self.masks_dir and os.path.isdir(self.masks_dir):
                    img_basename = os.path.splitext(img_name)[0]
                    mask_pattern = re.compile(
                        re.escape(img_basename) + r'_(soma_\d+_\d+)_area(\d+)_mask\.tif$'
                    )
                    # Build a lookup of soma outlines for soma_area_um2
                    outline_lookup = {}
                    for ol in restored_outlines:
                        outline_lookup[ol.get('soma_id', '')] = ol.get('soma_area_um2', 0)

                    for mf in sorted(os.listdir(self.masks_dir)):
                        m = mask_pattern.match(mf)
                        if not m:
                            continue
                        soma_id = m.group(1)
                        area_um2 = int(m.group(2))
                        mask_path = os.path.join(self.masks_dir, mf)
                        try:
                            mask_arr = tifffile.imread(mask_path)
                            # Find soma_idx from soma_ids list
                            soma_ids_list = self.images[img_name]['soma_ids']
                            soma_idx = soma_ids_list.index(soma_id) if soma_id in soma_ids_list else 0
                            self.images[img_name]['masks'].append({
                                'image_name': img_name,
                                'soma_idx': soma_idx,
                                'soma_id': soma_id,
                                'area_um2': area_um2,
                                'mask': mask_arr,
                                'approved': True,
                                'soma_area_um2': outline_lookup.get(soma_id, 0),
                            })
                        except Exception as e:
                            print(f"Warning: Could not load mask {mf}: {e}")
                    n_loaded_masks = len(self.images[img_name]['masks'])
                    if n_loaded_masks > 0:
                        print(f"Loaded {n_loaded_masks} masks from disk for {img_name}")
                    else:
                        print(f"Warning: No masks found on disk for {img_name} in {self.masks_dir}")

                status = self.images[img_name]['status']
                status_colors = {
                    'loaded': ('#808080', '⚪'),
                    'processed': ('#009600', '🟢'),
                    'somas_picked': ('#0064C8', '🔵'),
                    'outlined': ('#C89600', '🟡'),
                    'masks_generated': ('#FF8C00', '🟠'),
                    'qa_complete': ('#800080', '🟣'),
                    'analyzed': ('#00B400', '✅'),
                }
                color_hex, icon = status_colors.get(status, ('#808080', '⚪'))
                check_icon = "☑" if img_session.get('selected') else "☐"
                item = QListWidgetItem(f"{check_icon} {icon} {img_name} [{status}]")
                item.setData(Qt.UserRole, img_name)
                item.setCheckState(Qt.Checked if img_session.get('selected') else Qt.Unchecked)
                item.setForeground(QBrush(QColor(color_hex)))
                self.file_list.addItem(item)

            # Load and display first image
            if self.images:
                first_name = sorted(self.images.keys())[0]
                self.current_image_name = first_name
                self._display_current_image()
                self.file_list.setCurrentRow(0)
                self.process_selected_btn.setEnabled(True)

            # Rebuild all_masks_flat from loaded masks
            self.all_masks_flat = []
            for iname, idata in self.images.items():
                if not idata['selected']:
                    continue
                for mask_data in idata['masks']:
                    self.all_masks_flat.append({
                        'image_name': iname,
                        'mask_data': mask_data,
                        'processed_img': idata['processed'],
                    })

            # Enable buttons based on restored state
            self._update_buttons_after_session_load()

            n_loaded = len(self.images)
            n_with_somas = sum(1 for d in self.images.values() if d['somas'])
            n_with_outlines = sum(1 for d in self.images.values() if d['soma_outlines'])
            n_with_processed = sum(1 for d in self.images.values() if d['processed'] is not None)

            # Count mask files found on disk
            n_mask_files = 0
            if self.masks_dir and os.path.isdir(self.masks_dir):
                n_mask_files = len([f for f in os.listdir(self.masks_dir) if f.endswith('_mask.tif')])

            self.log("=" * 50)
            self.log(f"Session loaded: {n_loaded} images")
            if n_with_processed:
                self.log(f"  {n_with_processed} processed images restored from disk")
            if n_with_somas:
                self.log(f"  {n_with_somas} images with somas picked")
            if n_with_outlines:
                self.log(f"  {n_with_outlines} images with soma outlines")
            if n_mask_files:
                self.log(f"  {n_mask_files} exported masks found in {self.masks_dir}")
            self.log("=" * 50)

            note = ""
            if n_with_processed < n_loaded:
                n_need = n_loaded - n_with_processed
                note += f"\n{n_need} image(s) need re-processing (Run 'Process Selected Images')."
            if n_mask_files > 0:
                note += f"\n{n_mask_files} mask TIFFs found on disk (ready for ImageJ)."

            QMessageBox.information(self, "Session Loaded",
                f"Session restored:\n\n"
                f"Images: {n_loaded}\n"
                f"Processed: {n_with_processed}\n"
                f"With somas: {n_with_somas}\n"
                f"With outlines: {n_with_outlines}\n"
                f"Mask files on disk: {n_mask_files}"
                f"{note}"
            )

        except Exception as e:
            self.log(f"ERROR loading session: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load session:\n{e}")

    def _update_buttons_after_session_load(self):
        """Enable appropriate buttons based on restored session state"""
        has_selected = any(d['selected'] for d in self.images.values())
        has_processed = any(d['processed'] is not None for d in self.images.values())
        has_somas = any(d['somas'] for d in self.images.values())
        has_outlines = any(d['soma_outlines'] for d in self.images.values())

        self.process_selected_btn.setEnabled(has_selected)
        if has_processed:
            self.batch_pick_somas_btn.setEnabled(True)
        if has_somas:
            self.batch_outline_btn.setEnabled(True)
        if has_outlines:
            self.batch_generate_masks_btn.setEnabled(True)

        # Enable QA and calculate buttons based on image statuses
        has_masks = any(d['status'] in ('masks_generated', 'qa_complete', 'analyzed') for d in self.images.values())
        has_qa_complete = any(d['status'] in ('qa_complete', 'analyzed') for d in self.images.values())
        if has_masks:
            self.batch_qa_btn.setEnabled(True)
        if has_qa_complete:
            self.batch_calculate_btn.setEnabled(True)

        if self.output_dir:
            self.launch_imagej_btn.setEnabled(True)
            self.import_imagej_btn.setEnabled(True)

        # Check for partially-outlined session and offer to resume
        if has_somas and has_outlines:
            total_somas = sum(len(d['somas']) for d in self.images.values()
                              if d['selected'] and d['status'] in ('somas_picked', 'outlined'))
            total_outlines = sum(len(d['soma_outlines']) for d in self.images.values()
                                 if d['selected'] and d['status'] in ('somas_picked', 'outlined'))
            if 0 < total_outlines < total_somas:
                reply = QMessageBox.question(
                    self, 'Resume Outlining',
                    f"Found {total_outlines}/{total_somas} soma outlines completed.\n\n"
                    f"Resume outlining the remaining {total_somas - total_outlines} somas?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    self.start_batch_outlining()

    # ========================================================================
    # IMAGEJ SCRIPT GENERATION
    # ========================================================================

    def generate_imagej_scripts(self):
        """Generate Fiji/ImageJ scripts for Sholl and Skeleton analysis"""
        if not self.masks_dir or not self.output_dir:
            QMessageBox.warning(self, "Warning",
                "Please set an output folder and export masks first.")
            return

        try:
            pixel_size = float(self.pixel_size_input.text())
        except ValueError:
            pixel_size = 0.316

        masks_path = self.masks_dir.replace("\\", "/")
        output_path = self.output_dir.replace("\\", "/")
        somas_path = os.path.join(self.output_dir, "somas").replace("\\", "/")

        # Determine scale factor from pixel size
        scale_factor = 2 if pixel_size > 0.2 else 1

        # Generate Sholl analysis script
        sholl_script = f'''// Sholl Analysis - Auto-generated by MMPS
// Run this in Fiji with the SNT plugin installed
// Masks dir: {masks_path}
// Output dir: {output_path}

#@File(label="Masks Directory", style="directory", value="{masks_path}") masksDir
#@File(label="Somas Directory", style="directory", value="{somas_path}") somasDir
#@File(label="Output Directory", style="directory", value="{output_path}") outputDir
#@Float(label="Pixel Size (um/pixel)", value={pixel_size}) pixelSize

// This script requires Sholl_Attempt3.py to be run in Fiji's script editor.
// 1. Open Fiji
// 2. File > Open... > select Sholl_Attempt3.py
// 3. Click Run
// 4. Set the directories when prompted

print("=== Sholl Analysis ===");
print("Masks: " + masksDir);
print("Somas: " + somasDir);
print("Output: " + outputDir);
print("Pixel size: " + pixelSize + " um/px");
'''

        # Generate Skeleton analysis script
        skeleton_script = f'''// Skeleton Analysis - Auto-generated by MMPS
// Run this in Fiji
// Masks dir: {masks_path}
// Output dir: {output_path}

#@File(label="Masks Directory", style="directory", value="{masks_path}") masksDir
#@File(label="Output Directory", style="directory", value="{output_path}") outputDir
#@Float(label="Pixel Size (um/pixel)", value={pixel_size}) pixelSize
#@Integer(label="Upscale Factor (2 for 20x, 1 for 40x)", value={scale_factor}) scaleFactor

// This script requires SkeletonAnalysisImageJ.py to be run in Fiji's script editor.
// 1. Open Fiji
// 2. File > Open... > select SkeletonAnalysisImageJ.py
// 3. Click Run
// 4. Set the directories when prompted

print("=== Skeleton Analysis ===");
print("Masks: " + masksDir);
print("Output: " + outputDir);
print("Pixel size: " + pixelSize + " um/px");
print("Scale factor: " + scaleFactor + "x");
'''

        # Generate a combined batch runner script
        script_dir = self.output_dir

        # Write instructions file
        instructions = f"""MMPS ImageJ Analysis Instructions
===================================

Your masks are in: {masks_path}
Your soma files are in: {somas_path}
Output goes to: {output_path}
Pixel size: {pixel_size} um/pixel
Upscale factor: {scale_factor}x

Step 1: Sholl Analysis
-----------------------
1. Open Fiji/ImageJ
2. File > Open > select "Sholl_Attempt3.py" from your Microglia folder
3. Click "Run" in the script editor
4. When prompted, set:
   - Masks Directory: {masks_path}
   - Somas Directory: {somas_path}
   - Output Directory: {output_path}
   - Pixel Size: {pixel_size}

Step 2: Skeleton Analysis
--------------------------
1. In Fiji, File > Open > select "SkeletonAnalysisImageJ.py"
2. Click "Run" in the script editor
3. When prompted, set:
   - Masks Directory: {masks_path}
   - Output Directory: {output_path}
   - Pixel Size: {pixel_size}
   - Upscale Factor: {scale_factor}

Step 3: Import Results Back
----------------------------
1. Return to MMPS
2. Click "Import ImageJ Results"
3. Select the output folder
4. The Sholl and Skeleton results will be merged with your morphology CSV
"""

        instructions_path = os.path.join(script_dir, "ImageJ_Analysis_Instructions.txt")
        with open(instructions_path, 'w') as f:
            f.write(instructions)

        # Write parameter file that Sholl/Skeleton scripts can read
        params = {
            'masks_dir': masks_path,
            'somas_dir': somas_path,
            'output_dir': output_path,
            'pixel_size': pixel_size,
            'scale_factor': scale_factor,
        }
        params_path = os.path.join(script_dir, "imagej_params.json")
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2)

        self.log("=" * 50)
        self.log("ImageJ analysis files generated:")
        self.log(f"  Instructions: {instructions_path}")
        self.log(f"  Parameters: {params_path}")
        self.log("")
        self.log("To run analysis:")
        self.log("  1. Open Fiji/ImageJ")
        self.log(f"  2. Open Sholl_Attempt3.py and run with masks dir: {masks_path}")
        self.log(f"  3. Open SkeletonAnalysisImageJ.py and run with masks dir: {masks_path}")
        self.log("  4. Return here and click 'Import ImageJ Results'")
        self.log("=" * 50)

        QMessageBox.information(self, "ImageJ Scripts Generated",
            f"Analysis files saved to:\n{script_dir}\n\n"
            f"See ImageJ_Analysis_Instructions.txt for step-by-step guide.\n\n"
            f"Directories pre-configured:\n"
            f"  Masks: {masks_path}\n"
            f"  Somas: {somas_path}\n"
            f"  Output: {output_path}\n"
            f"  Pixel size: {pixel_size} um/px"
        )

    # ========================================================================
    # IMPORT IMAGEJ RESULTS
    # ========================================================================

    def import_imagej_results(self):
        """Import Sholl and Skeleton CSVs from ImageJ and merge with morphology results"""
        if not self.output_dir:
            QMessageBox.warning(self, "Warning", "Please set an output folder first.")
            return

        import csv

        # Look for ImageJ output files
        sholl_path = os.path.join(self.output_dir, "Sholl_Combined_Results.csv")
        skeleton_path = os.path.join(self.output_dir, "Skeleton_Analysis_Results.csv")
        morphology_path = os.path.join(self.output_dir, "combined_morphology_results.csv")

        # Allow user to locate files if not found in expected location
        found_sholl = os.path.exists(sholl_path)
        found_skeleton = os.path.exists(skeleton_path)
        found_morphology = os.path.exists(morphology_path)

        if not found_sholl and not found_skeleton:
            reply = QMessageBox.question(
                self, "Files Not Found",
                "No Sholl or Skeleton results found in the output folder.\n\n"
                "Would you like to browse for them?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

            # Browse for Sholl results
            sholl_path, _ = QFileDialog.getOpenFileName(
                self, "Select Sholl Results CSV (or Cancel to skip)", self.output_dir,
                "CSV Files (*.csv);;All Files (*)",
                options=QFileDialog.DontUseNativeDialog
            )
            found_sholl = bool(sholl_path) and os.path.exists(sholl_path)

            # Browse for Skeleton results
            skeleton_path, _ = QFileDialog.getOpenFileName(
                self, "Select Skeleton Results CSV (or Cancel to skip)", self.output_dir,
                "CSV Files (*.csv);;All Files (*)",
                options=QFileDialog.DontUseNativeDialog
            )
            found_skeleton = bool(skeleton_path) and os.path.exists(skeleton_path)

        if not found_sholl and not found_skeleton:
            QMessageBox.information(self, "No Files", "No ImageJ results to import.")
            return

        self.log("=" * 50)
        self.log("Importing ImageJ results...")

        # Read Sholl results
        sholl_data = {}
        if found_sholl:
            try:
                with open(sholl_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Match by cell_name which corresponds to mask filename pattern
                        cell_name = row.get('Cell', row.get('cell_name', ''))
                        if cell_name:
                            sholl_data[cell_name] = {f"sholl_{k}": v for k, v in row.items() if k != 'Cell' and k != 'cell_name'}
                self.log(f"  Sholl: loaded {len(sholl_data)} cells from {os.path.basename(sholl_path)}")
            except Exception as e:
                self.log(f"  ERROR reading Sholl results: {e}")

        # Read Skeleton results
        skeleton_data = {}
        if found_skeleton:
            try:
                with open(skeleton_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        cell_name = row.get('cell_name', '')
                        if cell_name:
                            skeleton_data[cell_name] = {f"skel_{k}": v for k, v in row.items() if k != 'cell_name'}
                self.log(f"  Skeleton: loaded {len(skeleton_data)} cells from {os.path.basename(skeleton_path)}")
            except Exception as e:
                self.log(f"  ERROR reading Skeleton results: {e}")

        # Read existing morphology results
        morphology_rows = []
        morph_fieldnames = []
        if found_morphology:
            try:
                with open(morphology_path, 'r') as f:
                    reader = csv.DictReader(f)
                    morph_fieldnames = list(reader.fieldnames)
                    morphology_rows = list(reader)
                self.log(f"  Morphology: loaded {len(morphology_rows)} cells")
            except Exception as e:
                self.log(f"  ERROR reading morphology results: {e}")

        if not morphology_rows:
            self.log("  No morphology results to merge with. Saving ImageJ results separately.")
            # Save a combined ImageJ-only file
            all_ij_data = {}
            for cell_name, data in sholl_data.items():
                all_ij_data.setdefault(cell_name, {'cell_name': cell_name}).update(data)
            for cell_name, data in skeleton_data.items():
                all_ij_data.setdefault(cell_name, {'cell_name': cell_name}).update(data)

            if all_ij_data:
                combined_ij_path = os.path.join(self.output_dir, "imagej_combined_results.csv")
                all_keys = set()
                for d in all_ij_data.values():
                    all_keys.update(d.keys())
                ordered = ['cell_name'] + sorted(k for k in all_keys if k != 'cell_name')
                with open(combined_ij_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=ordered)
                    writer.writeheader()
                    writer.writerows(all_ij_data.values())
                self.log(f"  Saved ImageJ results to: {combined_ij_path}")
        else:
            # Merge with morphology results
            matched_sholl = 0
            matched_skel = 0

            new_sholl_keys = set()
            new_skel_keys = set()
            for d in sholl_data.values():
                new_sholl_keys.update(d.keys())
            for d in skeleton_data.values():
                new_skel_keys.update(d.keys())

            for row in morphology_rows:
                img_name = row.get('image_name', '')
                soma_id = row.get('soma_id', '')
                # Build possible cell name patterns to match against
                cell_key = f"{img_name}_{soma_id}" if img_name and soma_id else ''

                # Try to match Sholl data
                for key in [cell_key] + [k for k in sholl_data.keys() if cell_key and cell_key in k]:
                    if key in sholl_data:
                        row.update(sholl_data[key])
                        matched_sholl += 1
                        break

                # Try to match Skeleton data
                for key in [cell_key] + [k for k in skeleton_data.keys() if cell_key and cell_key in k]:
                    if key in skeleton_data:
                        row.update(skeleton_data[key])
                        matched_skel += 1
                        break

            # Write merged results
            all_keys = morph_fieldnames + sorted(new_sholl_keys) + sorted(new_skel_keys)
            # Remove duplicates while preserving order
            seen = set()
            ordered_keys = []
            for k in all_keys:
                if k not in seen:
                    seen.add(k)
                    ordered_keys.append(k)

            merged_path = os.path.join(self.output_dir, "combined_all_results.csv")
            with open(merged_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=ordered_keys, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(morphology_rows)

            self.log(f"  Matched Sholl data: {matched_sholl}/{len(morphology_rows)} cells")
            self.log(f"  Matched Skeleton data: {matched_skel}/{len(morphology_rows)} cells")
            self.log(f"  Merged results saved to: {merged_path}")

        self.log("=" * 50)

        summary = "ImageJ Results Import Complete\n\n"
        if found_sholl:
            summary += f"Sholl: {len(sholl_data)} cells\n"
        if found_skeleton:
            summary += f"Skeleton: {len(skeleton_data)} cells\n"
        if morphology_rows:
            summary += f"\nMerged with {len(morphology_rows)} morphology results\n"
            summary += f"Output: combined_all_results.csv"
        else:
            summary += "\nSaved as: imagej_combined_results.csv"

        QMessageBox.information(self, "Import Complete", summary)

    def show_legend(self):
        """Display a popup dialog with the workflow status legend"""
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Workflow Status Legend")
        dialog.setIcon(QMessageBox.Information)

        legend_text = """
<h3>Workflow Status Colors:</h3>
<p style='line-height: 1.8;'>
<span style='color: #808080;'><b>⚪ Image(s) Loaded</b></span> - Images have been loaded into the application<br>
<span style='color: #009600;'><b>🟢 Image(s) Processed</b></span> - Background removal completed<br>
<span style='color: #0064C8;'><b>🔵 Somas Selected</b></span> - Soma centers have been marked<br>
<span style='color: #C89600;'><b>🟡 Somas Outlined</b></span> - Soma boundaries have been outlined<br>
<span style='color: #FF8C00;'><b>🟠 Masks Generated</b></span> - Cell masks have been generated<br>
<span style='color: #800080;'><b>🟣 Masks QA'ed</b></span> - Quality assurance review completed<br>
<span style='color: #00B400;'><b>✅ Mask Characteristics Processed</b></span> - Morphological analysis complete
</p>
        """

        dialog.setText(legend_text)
        dialog.setTextFormat(Qt.RichText)
        dialog.exec_()

    def open_display_adjustments(self):
        """Open display adjustments dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Display Adjustments (Visual Only)")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout(dialog)

        info_label = QLabel(
            "<b>Adjust display for better visibility</b><br>"
            "<i>These adjustments do NOT affect image processing or analysis</i>"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Per-channel brightness in colocalization mode
        channel_sliders = {}
        if self.colocalization_mode:
            channel_group = QGroupBox("Channel Brightness")
            channel_layout = QVBoxLayout()

            # Green channel
            green_layout = QHBoxLayout()
            green_label = QLabel("Green:")
            green_label.setStyleSheet("color: green; font-weight: bold;")
            green_label.setFixedWidth(50)
            green_slider = QSlider(Qt.Horizontal)
            green_slider.setRange(-100, 100)
            green_slider.setValue(self.channel_brightness.get('G', 0))
            green_value = QLabel(str(self.channel_brightness.get('G', 0)))
            green_value.setFixedWidth(40)
            green_layout.addWidget(green_label)
            green_layout.addWidget(green_slider)
            green_layout.addWidget(green_value)
            channel_layout.addLayout(green_layout)
            channel_sliders['G'] = (green_slider, green_value)

            # Red channel
            red_layout = QHBoxLayout()
            red_label = QLabel("Red:")
            red_label.setStyleSheet("color: red; font-weight: bold;")
            red_label.setFixedWidth(50)
            red_slider = QSlider(Qt.Horizontal)
            red_slider.setRange(-100, 100)
            red_slider.setValue(self.channel_brightness.get('R', 0))
            red_value = QLabel(str(self.channel_brightness.get('R', 0)))
            red_value.setFixedWidth(40)
            red_layout.addWidget(red_label)
            red_layout.addWidget(red_slider)
            red_layout.addWidget(red_value)
            channel_layout.addLayout(red_layout)
            channel_sliders['R'] = (red_slider, red_value)

            # Blue channel
            blue_layout = QHBoxLayout()
            blue_label = QLabel("Blue:")
            blue_label.setStyleSheet("color: blue; font-weight: bold;")
            blue_label.setFixedWidth(50)
            blue_slider = QSlider(Qt.Horizontal)
            blue_slider.setRange(-100, 100)
            blue_slider.setValue(self.channel_brightness.get('B', 0))
            blue_value = QLabel(str(self.channel_brightness.get('B', 0)))
            blue_value.setFixedWidth(40)
            blue_layout.addWidget(blue_label)
            blue_layout.addWidget(blue_slider)
            blue_layout.addWidget(blue_value)
            channel_layout.addLayout(blue_layout)
            channel_sliders['B'] = (blue_slider, blue_value)

            channel_group.setLayout(channel_layout)
            layout.addWidget(channel_group)

            # Connect channel sliders
            def make_channel_updater(ch, value_label):
                def updater(value):
                    value_label.setText(str(value))
                    self.channel_brightness[ch] = value
                    self.update_display()
                return updater

            green_slider.valueChanged.connect(make_channel_updater('G', green_value))
            red_slider.valueChanged.connect(make_channel_updater('R', red_value))
            blue_slider.valueChanged.connect(make_channel_updater('B', blue_value))

        # Global Brightness slider
        brightness_group = QGroupBox("Global Brightness")
        brightness_layout = QVBoxLayout()
        brightness_slider = QSlider(Qt.Horizontal)
        brightness_slider.setRange(-100, 100)
        brightness_slider.setValue(self.brightness_value)
        brightness_slider.setTickPosition(QSlider.TicksBelow)
        brightness_slider.setTickInterval(25)
        brightness_value_label = QLabel(str(self.brightness_value))
        brightness_value_label.setAlignment(Qt.AlignCenter)
        brightness_layout.addWidget(brightness_slider)
        brightness_layout.addWidget(brightness_value_label)
        brightness_group.setLayout(brightness_layout)
        layout.addWidget(brightness_group)

        # Contrast slider
        contrast_group = QGroupBox("Contrast")
        contrast_layout = QVBoxLayout()
        contrast_slider = QSlider(Qt.Horizontal)
        contrast_slider.setRange(-100, 100)
        contrast_slider.setValue(self.contrast_value)
        contrast_slider.setTickPosition(QSlider.TicksBelow)
        contrast_slider.setTickInterval(25)
        contrast_value_label = QLabel(str(self.contrast_value))
        contrast_value_label.setAlignment(Qt.AlignCenter)
        contrast_layout.addWidget(contrast_slider)
        contrast_layout.addWidget(contrast_value_label)
        contrast_group.setLayout(contrast_layout)
        layout.addWidget(contrast_group)

        # Connect sliders to update labels and live preview
        def update_brightness(value):
            brightness_value_label.setText(str(value))
            self.brightness_value = value
            self.update_display()

        def update_contrast(value):
            contrast_value_label.setText(str(value))
            self.contrast_value = value
            self.update_display()

        brightness_slider.valueChanged.connect(update_brightness)
        contrast_slider.valueChanged.connect(update_contrast)

        # Buttons
        button_layout = QHBoxLayout()

        reset_btn = QPushButton("Reset All")

        def reset_values():
            brightness_slider.setValue(0)
            contrast_slider.setValue(0)
            if self.colocalization_mode:
                for ch, (slider, _) in channel_sliders.items():
                    slider.setValue(0)

        reset_btn.clicked.connect(reset_values)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)

        button_layout.addWidget(reset_btn)
        button_layout.addStretch()
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)

        dialog.exec_()

    def open_channel_selector(self):
        """Open channel selection dialog to choose which channels to display"""
        if not self.show_color_view:
            QMessageBox.information(
                self, "Channel Selection",
                "Channel selection is only available in color view mode.\nPress C to toggle color view."
            )
            return

        # Use the current image to detect channel count
        sample_color_img = None
        if self.current_image_name and self.current_image_name in self.images:
            img_data = self.images[self.current_image_name]
            if 'color_image' in img_data:
                sample_color_img = img_data['color_image']
        if sample_color_img is None:
            for img_data in self.images.values():
                if 'color_image' in img_data:
                    sample_color_img = img_data['color_image']
                    break

        dialog = ChannelSelectDialog(
            self,
            current_channels=self.display_channels,
            channel_names=self.channel_names,
            color_image=sample_color_img
        )

        if dialog.exec_() == QDialog.Accepted:
            settings = dialog.get_settings()
            self.display_channels = settings['channels']
            self.channel_names = settings['names']

            # Log the change
            active_channels = [f"Ch{ch+1}" for ch, active in self.display_channels.items() if active]
            self.log(f"Channel display: {', '.join(active_channels)}")

            # Refresh the current display
            self._refresh_color_display()

    def _refresh_color_display(self):
        """Refresh the display with current channel settings"""
        if not self.current_image_name or self.current_image_name not in self.images:
            return

        img_data = self.images[self.current_image_name]

        # Refresh if color view is on and we have color data
        if self.show_color_view and 'color_image' in img_data:
            # Use processed channel in color composite if available
            proc_color = self._build_processed_color_image(img_data)
            if proc_color is not None:
                adjusted = self._apply_display_adjustments_color(proc_color)
            else:
                adjusted = self._apply_display_adjustments_color(img_data['color_image'])
            pixmap = self._array_to_pixmap_color(adjusted)
            # Preserve markers
            if self.processed_label.soma_mode:
                self.processed_label.set_image(pixmap, centroids=img_data['somas'])
            elif self.processed_label.polygon_mode:
                queue_idx = getattr(self, 'current_outline_idx', 0)
                if queue_idx < len(self.outlining_queue):
                    img_name, soma_idx = self.outlining_queue[queue_idx]
                    if img_name == self.current_image_name:
                        soma = img_data['somas'][soma_idx]
                        self.processed_label.set_image(pixmap, centroids=[soma],
                                                       polygon_pts=self.polygon_points)
            else:
                self.processed_label.set_image(pixmap, centroids=img_data['somas'])

            # Also refresh original tab if viewing (original stays unprocessed)
            raw_img = load_tiff_image(img_data['raw_path'])
            if raw_img.ndim == 3:
                adjusted_orig = self._apply_display_adjustments_color(raw_img)
                orig_pixmap = self._array_to_pixmap_color(adjusted_orig)
                self.original_label.set_image(orig_pixmap)

    def reset_display_adjustments(self):
        """Reset brightness and contrast to default"""
        self.brightness_value = 0
        self.contrast_value = 0
        self.update_display()

    def toggle_color_view(self):
        """Toggle between color and grayscale display"""
        self.show_color_view = not self.show_color_view
        if self.show_color_view:
            self.color_toggle_btn.setText("Show Grayscale (C)")
            self.channel_select_btn.setVisible(True)
        else:
            self.color_toggle_btn.setText("Show Color (C)")
            self.channel_select_btn.setVisible(False)
        self.update_display()

    def _on_process_channel_changed(self, index):
        """Update the grayscale channel when user changes the dropdown"""
        self.grayscale_channel = index
        self.log(f"Processing channel set to: Channel {index + 1}")
        # Refresh display if not in color mode
        if not self.show_color_view:
            self.update_display()

    def _reset_current_zoom(self):
        """Reset zoom on all image labels"""
        for label in [self.original_label, self.preview_label, self.processed_label, self.mask_label]:
            if label:
                label.reset_zoom()
        self.zoom_level_label.setText("1.0x")

    def update_display(self):
        """Update the display with current brightness/contrast settings"""
        try:
            # Refresh the current image display
            if self.current_image_name and self.current_image_name in self.images:
                current_tab = self.tabs.currentIndex()
                img_data = self.images[self.current_image_name]

                # Determine which image to display based on current tab
                use_color = self.show_color_view
                if current_tab == 0:  # Original
                    if 'raw_path' in img_data:
                        raw_img = load_tiff_image(img_data['raw_path'])
                        if use_color and raw_img.ndim == 3:
                            adjusted = self._apply_display_adjustments_color(raw_img)
                            pixmap = self._array_to_pixmap_color(adjusted)
                        else:
                            # Use selected channel for grayscale
                            if raw_img.ndim == 3:
                                gray_img = extract_channel(raw_img, self.grayscale_channel)
                            else:
                                gray_img = raw_img
                            adjusted = self._apply_display_adjustments(gray_img)
                            pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
                        self.original_label.set_image(pixmap)
                elif current_tab == 1:  # Preview
                    if 'preview' in img_data and img_data['preview'] is not None:
                        adjusted = self._apply_display_adjustments(img_data['preview'])
                        pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
                        self.preview_label.set_image(pixmap)
                elif current_tab == 2:  # Processed
                    # Show color image when color view is on
                    if use_color and 'color_image' in img_data:
                        # Use processed channel in color composite if available
                        proc_color = self._build_processed_color_image(img_data)
                        if proc_color is not None:
                            adjusted = self._apply_display_adjustments_color(proc_color)
                        else:
                            adjusted = self._apply_display_adjustments_color(img_data['color_image'])
                        pixmap = self._array_to_pixmap_color(adjusted)
                        # Preserve polygon if in outlining mode
                        if self.processed_label.polygon_mode:
                            queue_idx = getattr(self, 'current_outline_idx', 0)
                            if queue_idx < len(self.outlining_queue):
                                img_name, soma_idx = self.outlining_queue[queue_idx]
                                soma = img_data['somas'][soma_idx] if img_name == self.current_image_name else (img_data['somas'][0] if img_data['somas'] else None)
                                if soma is not None:
                                    self.processed_label.set_image(pixmap, centroids=[soma],
                                                                   polygon_pts=self.polygon_points)
                                else:
                                    self.processed_label.set_image(pixmap, polygon_pts=self.polygon_points)
                            else:
                                self.processed_label.set_image(pixmap, centroids=img_data['somas'])
                        else:
                            self.processed_label.set_image(pixmap, centroids=img_data['somas'])
                    elif img_data['processed'] is not None:
                        # Always use the processed image for grayscale display
                        gray_img = img_data['processed']
                        adjusted = self._apply_display_adjustments(gray_img)
                        pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
                        # Preserve soma markers if in soma picking mode
                        if self.soma_mode:
                            self.processed_label.set_image(pixmap, centroids=img_data['somas'])
                        # Preserve polygon if in outlining mode
                        elif self.processed_label.polygon_mode:
                            queue_idx = getattr(self, 'current_outline_idx', 0)
                            if queue_idx < len(self.outlining_queue):
                                img_name, soma_idx = self.outlining_queue[queue_idx]
                                soma = img_data['somas'][soma_idx] if img_name == self.current_image_name else (img_data['somas'][0] if img_data['somas'] else None)
                                if soma is not None:
                                    self.processed_label.set_image(pixmap, centroids=[soma],
                                                                   polygon_pts=self.polygon_points)
                                else:
                                    self.processed_label.set_image(pixmap, polygon_pts=self.polygon_points)
                            else:
                                self.processed_label.set_image(pixmap, centroids=img_data['somas'])
                        else:
                            self.processed_label.set_image(pixmap, centroids=img_data['somas'])
                elif current_tab == 3:  # Masks
                    # Re-render the current mask with updated color settings
                    if self.mask_qa_active:
                        self._show_current_mask()
        except Exception as e:
            self.log(f"ERROR updating display: {str(e)}")
            import traceback
            traceback.print_exc()

    def _apply_display_adjustments(self, img):
        """Apply brightness and contrast adjustments for display only (does not modify original data)"""
        if img is None:
            return None

        # Work with a copy so we don't modify the original
        adjusted = img.astype(np.float32).copy()

        # Normalize to 0-255 range first
        img_min = adjusted.min()
        img_max = adjusted.max()
        if img_max > img_min:
            adjusted = (adjusted - img_min) / (img_max - img_min) * 255.0

        # Apply contrast first (multiply around midpoint)
        contrast = self.contrast_value
        if contrast != 0:
            # Convert contrast from -100,100 to a multiplier
            # -100 = 0.1x (much less contrast), 0 = 1x (no change), 100 = 3x (much more contrast)
            if contrast > 0:
                factor = 1.0 + (contrast / 100.0) * 2.0  # 0 to +2 (max 3x)
            else:
                factor = 1.0 + (contrast / 100.0) * 0.9  # -0.9 to 0 (min 0.1x)

            midpoint = 127.5
            adjusted = (adjusted - midpoint) * factor + midpoint

        # Apply brightness (simple addition)
        brightness = self.brightness_value
        if brightness != 0:
            # Scale brightness to have more visible effect
            adjusted = adjusted + (brightness * 1.5)

        # Clip to valid range
        adjusted = np.clip(adjusted, 0, 255)

        return adjusted.astype(np.uint8)

    def _apply_display_adjustments_color(self, img):
        """Apply per-channel brightness adjustments for color images"""
        if img is None:
            return None

        # Work with a copy
        adjusted = img.astype(np.float32).copy()

        # Handle different array shapes - convert to RGB format
        if adjusted.ndim == 2:
            adjusted = np.stack([adjusted, adjusted, adjusted], axis=-1)
        elif adjusted.ndim == 3:
            if adjusted.shape[2] == 4:
                adjusted = adjusted[:, :, :3]
            elif adjusted.shape[2] != 3 and adjusted.shape[2] >= 2:
                # Multi-channel - map to RGB (Green, Red, Blue)
                h, w, c = adjusted.shape
                rgb = np.zeros((h, w, 3), dtype=np.float32)
                rgb[:, :, 1] = adjusted[:, :, 0]  # Green
                rgb[:, :, 0] = adjusted[:, :, 1]  # Red
                if c >= 3:
                    rgb[:, :, 2] = adjusted[:, :, 2]  # Blue
                adjusted = rgb

        # Normalize each channel to 0-255
        for i in range(3):
            channel = adjusted[:, :, i]
            c_min, c_max = channel.min(), channel.max()
            if c_max > c_min:
                adjusted[:, :, i] = (channel - c_min) / (c_max - c_min) * 255.0

        # Apply per-channel brightness
        # R channel (index 0)
        brightness_r = self.channel_brightness.get('R', 0)
        if brightness_r != 0:
            adjusted[:, :, 0] = adjusted[:, :, 0] + (brightness_r * 1.5)

        # G channel (index 1)
        brightness_g = self.channel_brightness.get('G', 0)
        if brightness_g != 0:
            adjusted[:, :, 1] = adjusted[:, :, 1] + (brightness_g * 1.5)

        # B channel (index 2)
        brightness_b = self.channel_brightness.get('B', 0)
        if brightness_b != 0:
            adjusted[:, :, 2] = adjusted[:, :, 2] + (brightness_b * 1.5)

        # Apply global brightness/contrast on top
        if self.brightness_value != 0:
            adjusted = adjusted + (self.brightness_value * 1.5)

        if self.contrast_value != 0:
            if self.contrast_value > 0:
                factor = 1.0 + (self.contrast_value / 100.0) * 2.0
            else:
                factor = 1.0 + (self.contrast_value / 100.0) * 0.9
            midpoint = 127.5
            adjusted = (adjusted - midpoint) * factor + midpoint

        # Clip and return
        adjusted = np.clip(adjusted, 0, 255)
        return adjusted.astype(np.uint8)

    def _build_processed_color_image(self, img_data):
        """Build a color composite with processed channel replacing the original channel"""
        if 'color_image' not in img_data or img_data['processed'] is None:
            return None

        color_img = img_data['color_image']
        processed = img_data['processed']

        # Start with a copy of the original color image
        if color_img.ndim != 3:
            return None

        # Build RGB composite
        h, w = color_img.shape[:2]
        c = min(color_img.shape[2], 3)
        composite = np.zeros((h, w, 3), dtype=np.float32)

        # Map channels: index 0=Green, 1=Red, 2=Blue in display
        for i in range(c):
            if i == self.grayscale_channel:
                # Use the processed (cleaned) version for this channel
                # Normalize processed to match original range for display
                proc_norm = processed.astype(np.float32)
                composite[:, :, i] = proc_norm
            else:
                # Use original channel
                composite[:, :, i] = color_img[:, :, i].astype(np.float32)

        return composite

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Image Folder",
            options=QFileDialog.DontUseNativeDialog
        )
        if not folder:
            return

        # Ask about colocalization mode
        reply = QMessageBox.question(
            self, 'Colocalization Analysis',
            'Do you want to perform colocalization analysis?\n\n'
            'If YES: Images will be displayed in color showing all channels.\n'
            'Colocalization metrics will be calculated for selected cells.\n\n'
            'If NO: Images will be converted to grayscale for standard analysis.',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        self.colocalization_mode = (reply == QMessageBox.Yes)

        if self.colocalization_mode:
            self.log("=" * 50)
            self.log("COLOCALIZATION MODE ENABLED")
            self.log("Images will be displayed in color")
            self.log("Use 'Channel Display' button to select which channels to show")
            self.log("Press C to toggle between color and grayscale")
            self.log("Grayscale conversion will occur when outlining begins")
            self.log("=" * 50)
            # Auto-enable color view in colocalization mode
            self.show_color_view = True
            self.color_toggle_btn.setText("Show Grayscale (C)")
            self.channel_select_btn.setVisible(True)
        else:
            self.show_color_view = False
            self.color_toggle_btn.setText("Show Color (C)")
            self.channel_select_btn.setVisible(False)

        # Include both lowercase and uppercase extensions for macOS compatibility
        exts = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg',
                '*.TIF', '*.TIFF', '*.PNG', '*.JPG', '*.JPEG']
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(folder, ext)))
        # Remove duplicates in case filesystem is case-insensitive
        files = list(set(files))
        self.images = {}
        self.file_list.clear()

        from PyQt5.QtGui import QColor, QBrush

        for f in sorted(files):
            img_name = os.path.basename(f)
            self.images[img_name] = {
                'raw_path': f,
                'processed': None,
                'rolling_ball_radius': self.default_rolling_ball_radius,
                'somas': [],
                'soma_ids': [],
                'soma_outlines': [],
                'masks': [],
                'status': 'loaded',
                'selected': False,
                'animal_id': '',
                'treatment': ''
            }
            item = QListWidgetItem(f"☐ ⚪ {img_name} [loaded]")
            item.setData(Qt.UserRole, img_name)
            item.setCheckState(Qt.Unchecked)
            item.setForeground(QBrush(QColor(128, 128, 128)))  # Gray for loaded
            self.file_list.addItem(item)

        if self.images:
            self.process_selected_btn.setEnabled(True)
            self.log(f"Loaded {len(self.images)} images")
            # self.update_workflow_status()

            # Automatically load and display the first image
            first_image_name = sorted(self.images.keys())[0]
            self.current_image_name = first_image_name
            self._display_current_image()

            # Select the first item in the list
            self.file_list.setCurrentRow(0)

            self.log(f"Displaying: {first_image_name}")

    def select_output(self):
        folder = QFileDialog.getExistingDirectory(
            self, 
            "Select Output Folder",
            options=QFileDialog.DontUseNativeDialog
        )
        if folder:
            self.output_dir = folder
            self.masks_dir = os.path.join(folder, "masks")
            self.somas_dir = os.path.join(folder, "somas")
            os.makedirs(self.masks_dir, exist_ok=True)
            os.makedirs(self.somas_dir, exist_ok=True)
            self.log(f"Output folder: {folder}")
            self.log(f"Masks will be saved to: {self.masks_dir}")
            self.log(f"Somas will be saved to: {self.somas_dir}")
            self.launch_imagej_btn.setEnabled(True)
            self.import_imagej_btn.setEnabled(True)

    def select_all_images(self):
        """Select all images and check all checkboxes"""
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            item.setCheckState(Qt.Checked)  # Check the box
            img_name = item.data(Qt.UserRole)
            self.images[img_name]['selected'] = True
        self.log(f"Selected all {self.file_list.count()} images")
        # self.update_workflow_status()

    def clear_all_images(self):
        """Deselect all images and uncheck all checkboxes"""
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            item.setCheckState(Qt.Unchecked)  # Uncheck the box
            img_name = item.data(Qt.UserRole)
            self.images[img_name]['selected'] = False
        self.log(f"Cleared selection for all images")
        # self.update_workflow_status()
        # self.update_workflow_status()

    def on_item_checkbox_changed(self, item):
        """Handle checkbox state changes for images in the list"""
        img_name = item.data(Qt.UserRole)
        if img_name and img_name in self.images:
            is_checked = item.checkState() == Qt.Checked
            self.images[img_name]['selected'] = is_checked

    def on_image_selected(self, item):
        img_name = item.data(Qt.UserRole)
        is_checked = item.checkState() == Qt.Checked
        self.images[img_name]['selected'] = is_checked
        self.current_image_name = img_name
        self._display_current_image()
        # self.update_workflow_status()

    def _display_current_image(self):
        if not self.current_image_name or self.current_image_name not in self.images:
            return
        try:
            img_data = self.images[self.current_image_name]
            raw_img = load_tiff_image(img_data['raw_path'])

            # Always store color image for toggle functionality
            if raw_img.ndim == 3:
                img_data['color_image'] = raw_img.copy()
                img_data['num_channels'] = raw_img.shape[2]

            # Display in color or grayscale based on toggle, with adjustments
            if self.show_color_view and raw_img.ndim == 3:
                adjusted = self._apply_display_adjustments_color(raw_img)
                pixmap = self._array_to_pixmap_color(adjusted)
                self.original_label.set_image(pixmap)
            else:
                # Use only the selected channel for grayscale display
                if raw_img.ndim == 3:
                    raw_gray = extract_channel(raw_img, self.grayscale_channel)
                else:
                    raw_gray = raw_img
                adjusted_raw = self._apply_display_adjustments(raw_gray)
                pixmap = self._array_to_pixmap(adjusted_raw, skip_rescale=True)
                self.original_label.set_image(pixmap)

            if img_data['processed'] is not None:
                # Processed images - show color or grayscale based on toggle
                if self.show_color_view and 'color_image' in img_data:
                    # Build composite with processed channel replacing original
                    proc_color = self._build_processed_color_image(img_data)
                    if proc_color is not None:
                        adjusted_proc = self._apply_display_adjustments_color(proc_color)
                    else:
                        adjusted_proc = self._apply_display_adjustments_color(img_data['color_image'])
                    pixmap_proc = self._array_to_pixmap_color(adjusted_proc)
                else:
                    # Always use the processed image for grayscale display
                    gray_proc = img_data['processed']
                    adjusted_proc = self._apply_display_adjustments(gray_proc)
                    pixmap_proc = self._array_to_pixmap(adjusted_proc, skip_rescale=True)
                self.processed_label.set_image(pixmap_proc, centroids=img_data['somas'])
            else:
                # Before processing, show color image in processed tab too if color view is on
                if self.show_color_view and raw_img.ndim == 3:
                    adjusted_proc = self._apply_display_adjustments_color(raw_img)
                    pixmap_proc = self._array_to_pixmap_color(adjusted_proc)
                    self.processed_label.set_image(pixmap_proc, centroids=img_data['somas'])
                else:
                    self.processed_label.set_image(self._create_blank_pixmap())
                    self.processed_label.setText("Not processed yet")
        except Exception as e:
            self.log(f"ERROR displaying image: {str(e)}")
            import traceback
            traceback.print_exc()

    def _create_blank_pixmap(self):
        blank = np.ones((500, 500), dtype=np.uint8) * 128
        return self._array_to_pixmap(blank)

    def _array_to_pixmap(self, arr, skip_rescale=False):
        arr_disp = arr.astype(float)

        # Only rescale if not already adjusted
        if not skip_rescale:
            arr_disp -= arr_disp.min()
            if arr_disp.max() > 0:
                arr_disp = arr_disp / arr_disp.max() * 255

        arr8 = arr_disp.clip(0, 255).astype(np.uint8)
        arr8 = np.ascontiguousarray(arr8)
        h, w = arr8.shape
        bytes_per_line = w
        img = QImage(arr8.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        img = img.copy()
        return QPixmap.fromImage(img)

    def _array_to_pixmap_color(self, arr, num_source_channels=None):
        """Convert a color (RGB) numpy array to QPixmap, respecting channel selection.

        num_source_channels: how many channels the original image actually has.
            If provided, channels beyond this count are treated as empty.
        """
        if arr is None:
            return self._create_blank_pixmap()

        # Track which RGB slots have real data
        real_channels = {0: True, 1: True, 2: True}

        # Handle different array shapes
        if arr.ndim == 2:
            # Grayscale - convert to RGB
            arr = np.stack([arr, arr, arr], axis=-1)
            real_channels = {0: True, 1: False, 2: False}
        elif arr.ndim == 3:
            n_ch = arr.shape[2]
            if n_ch == 4:
                # RGBA - take RGB only
                arr = arr[:, :, :3]
            elif n_ch < 3:
                # Fewer than 3 channels — map to RGB, mark missing as empty
                h, w, c = arr.shape
                rgb = np.zeros((h, w, 3), dtype=arr.dtype)
                if c >= 1:
                    rgb[:, :, 1] = arr[:, :, 0]  # Ch1 -> Green
                if c >= 2:
                    rgb[:, :, 0] = arr[:, :, 1]  # Ch2 -> Red
                # Mark channels that don't exist
                if c < 3:
                    real_channels[2] = False  # No blue
                if c < 2:
                    real_channels[0] = False  # No red
                if c < 1:
                    real_channels[1] = False  # No green
                arr = rgb

        # Determine real source channel count — from parameter, or from current image
        if num_source_channels is None and self.current_image_name and self.current_image_name in self.images:
            num_source_channels = self.images[self.current_image_name].get('num_channels')

        # Apply source channel limit (handles pre-expanded 3-channel arrays from composites)
        if num_source_channels is not None and num_source_channels < 3:
            # Ch1->Green(1), Ch2->Red(0), Ch3->Blue(2) mapping
            if num_source_channels < 3:
                real_channels[2] = False  # No blue
            if num_source_channels < 2:
                real_channels[0] = False  # No red
            if num_source_channels < 1:
                real_channels[1] = False  # No green

        # Apply channel selection mask — zero out disabled or missing channels
        arr_display = arr.copy()
        for i in range(min(3, arr_display.shape[2])):
            if not real_channels.get(i, False) or not self.display_channels.get(i, True):
                arr_display[:, :, i] = 0

        # Normalize to 0-255 — only for real, enabled channels
        arr_float = arr_display.astype(np.float32)
        for i in range(arr_display.shape[2]):
            if real_channels.get(i, False) and self.display_channels.get(i, True):
                channel = arr_float[:, :, i]
                c_min, c_max = channel.min(), channel.max()
                if c_max > c_min:
                    arr_float[:, :, i] = (channel - c_min) / (c_max - c_min) * 255

        arr8 = arr_float.clip(0, 255).astype(np.uint8)
        arr8 = np.ascontiguousarray(arr8)
        h, w, c = arr8.shape
        bytes_per_line = 3 * w
        img = QImage(arr8.data, w, h, bytes_per_line, QImage.Format_RGB888)
        img = img.copy()
        return QPixmap.fromImage(img)

    def preview_current_image(self):
        if not self.current_image_name:
            QMessageBox.warning(self, "Warning", "Select an image first")
            return
        img_data = self.images[self.current_image_name]
        raw_img = load_tiff_image(img_data['raw_path'])
        # Extract only the selected channel for processing
        if raw_img.ndim == 3:
            channel_img = extract_channel(raw_img, self.grayscale_channel)
        else:
            channel_img = raw_img
        result = channel_img.copy()

        # Apply optional rolling ball background subtraction
        rb_enabled = self.rb_check.isChecked()
        if rb_enabled:
            radius = self.rb_slider.value()
            background = restoration.rolling_ball(channel_img, radius=radius)
            result = channel_img - background
            result = np.clip(result, 0, channel_img.max())

        # Apply optional denoising
        if self.denoise_check.isChecked():
            denoise_size = self.denoise_spin.value()
            result = ndimage.median_filter(result, size=denoise_size)

        # Apply optional sharpening
        if self.sharpen_check.isChecked():
            sharpen_amount = self.sharpen_slider.value() / 10.0
            blurred = ndimage.gaussian_filter(result.astype(np.float32), sigma=2)
            result_float = result.astype(np.float32)
            sharpened = result_float + sharpen_amount * (result_float - blurred)
            result = np.clip(sharpened, 0, channel_img.max()).astype(result.dtype)

        # Store the preview (without adjustments)
        img_data['preview'] = result
        # Apply display adjustments for viewing
        adjusted = self._apply_display_adjustments(result)
        pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
        self.preview_label.set_image(pixmap)
        self.tabs.setCurrentIndex(1)
        steps = [f"Ch{self.grayscale_channel + 1}"]
        if rb_enabled:
            steps.append(f"RB={self.rb_slider.value()}")
        if self.denoise_check.isChecked():
            steps.append(f"Denoise={self.denoise_spin.value()}")
        if self.sharpen_check.isChecked():
            steps.append(f"Sharpen={self.sharpen_slider.value() / 10:.1f}")
        if len(steps) == 1:
            steps.append("no processing")
        self.log(f"Preview {self.current_image_name}: {', '.join(steps)}")

    def process_selected_images(self):
        if not self.output_dir:
            QMessageBox.warning(self, "Warning", "Select output folder first")
            return
        selected_images = [(name, data) for name, data in self.images.items() if data['selected']]
        if not selected_images:
            QMessageBox.warning(self, "Warning", "No images selected")
            return
        radius = self.rb_slider.value()
        rb_enabled = self.rb_check.isChecked()
        denoise_enabled = self.denoise_check.isChecked()
        denoise_size = self.denoise_spin.value()
        sharpen_enabled = self.sharpen_check.isChecked()
        sharpen_amount = self.sharpen_slider.value() / 10.0

        # Log what processing steps will be applied
        process_channel = self.grayscale_channel
        steps = [f"Channel {process_channel + 1}"]
        if rb_enabled:
            steps.append(f"Rolling Ball (r={radius})")
        if denoise_enabled:
            steps.append(f"Denoise ({denoise_size})")
        if sharpen_enabled:
            steps.append(f"Sharpen ({sharpen_amount:.1f})")
        if len(steps) == 1:
            self.log(f"Processing Channel {process_channel + 1} only - no additional processing")
        else:
            self.log(f"Processing with: {', '.join(steps)}")

        process_list = []
        for img_name, img_data in selected_images:
            process_list.append((img_data['raw_path'], img_name, radius, rb_enabled,
                                 denoise_enabled, denoise_size, sharpen_enabled, sharpen_amount,
                                 process_channel))
        self.thread = BackgroundRemovalThread(process_list, self.output_dir)
        self.thread.status_update.connect(self.log)
        self.thread.progress.connect(self._update_progress)
        self.thread.finished_image.connect(self._handle_processed_image)
        self.thread.finished.connect(self._background_removal_finished)
        self.thread.error_occurred.connect(lambda msg: self.log(f"ERROR: {msg}"))
        self.progress_bar.setVisible(True)
        self.progress_status_label.setVisible(True)
        if len(steps) > 1:
            self.progress_status_label.setText(f"Processing Channel {process_channel + 1}...")
        else:
            self.progress_status_label.setText(f"Extracting Channel {process_channel + 1}...")
        self.process_selected_btn.setEnabled(False)
        self.thread.start()

    def _update_progress(self, value):
        self.progress_bar.setValue(value)

    def _handle_processed_image(self, output_path, img_name, processed_data):
        if img_name in self.images:
            self.images[img_name]['processed'] = processed_data
            self.images[img_name]['status'] = 'processed'

            # Always store color image for toggle functionality
            if 'color_image' not in self.images[img_name]:
                try:
                    raw_img = load_tiff_image(self.images[img_name]['raw_path'])
                    if raw_img.ndim == 3:
                        self.images[img_name]['color_image'] = raw_img.copy()
                except:
                    pass

            self._update_file_list_item(img_name)
            if img_name == self.current_image_name:
                if self.show_color_view and 'color_image' in self.images[img_name]:
                    # Use processed channel in color composite
                    proc_color = self._build_processed_color_image(self.images[img_name])
                    if proc_color is not None:
                        adjusted = self._apply_display_adjustments_color(proc_color)
                    else:
                        adjusted = self._apply_display_adjustments_color(self.images[img_name]['color_image'])
                    pixmap = self._array_to_pixmap_color(adjusted)
                else:
                    # Use selected channel from color image if available
                    if 'color_image' in self.images[img_name]:
                        gray_img = extract_channel(self.images[img_name]['color_image'], self.grayscale_channel)
                    else:
                        gray_img = processed_data
                    adjusted = self._apply_display_adjustments(gray_img)
                    pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
                self.processed_label.set_image(pixmap)
                self.tabs.setCurrentIndex(2)

    def _update_file_list_item(self, img_name):
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(Qt.UserRole) == img_name:
                check_mark = "☑" if self.images[img_name]['selected'] else "☐"
                status = self.images[img_name]['status']

                # Add visual indicators for different statuses
                status_icons = {
                    'loaded': '⚪',
                    'processed': '🟢',
                    'somas_picked': '🔵',
                    'outlined': '🟡',
                    'masks_generated': '🟠',
                    'qa_complete': '🟣',
                    'analyzed': '✅'
                }

                status_icon = status_icons.get(status, '⚪')
                item.setText(f"{check_mark} {status_icon} {img_name} [{status}]")

                # Color code the item based on status
                from PyQt5.QtGui import QColor, QBrush
                if status == 'loaded':
                    item.setForeground(QBrush(QColor(128, 128, 128)))  # Gray
                elif status == 'processed':
                    item.setForeground(QBrush(QColor(0, 150, 0)))  # Green
                elif status == 'somas_picked':
                    item.setForeground(QBrush(QColor(0, 100, 200)))  # Blue
                elif status == 'outlined':
                    item.setForeground(QBrush(QColor(200, 150, 0)))  # Yellow/Gold
                elif status == 'masks_generated':
                    item.setForeground(QBrush(QColor(255, 140, 0)))  # Orange
                elif status == 'qa_complete':
                    item.setForeground(QBrush(QColor(128, 0, 128)))  # Purple
                elif status == 'analyzed':
                    item.setForeground(QBrush(QColor(0, 180, 0)))  # Bright Green
                    item.setBackground(QBrush(QColor(230, 255, 230)))  # Light green background
                break

    def _background_removal_finished(self):

        self.progress_bar.setVisible(False)
        self.progress_status_label.setVisible(False)
        self.process_selected_btn.setEnabled(True)
        self.batch_pick_somas_btn.setEnabled(True)
        self.log("=" * 50)
        self.log("✓ Background removal complete!")
        self.log("✓ Ready for batch soma picking")
        self.log("=" * 50)
        self._auto_save()
        QMessageBox.information(
            self, "Complete",
            "All selected images processed!\n\nYou can now pick somas."
        )

    def update_workflow_status(self):
        selected = [name for name, data in self.images.items() if data['selected']]
        if not selected:
            self.workflow_status_label.setText("No images selected")
            return
        status_counts = {}
        for img_name in selected:
            status = self.images[img_name]['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        status_text = f"Selected: {len(selected)} images\n"
        for status, count in sorted(status_counts.items()):
            status_text += f"{status}: {count}\n"
        self.workflow_status_label.setText(status_text)

    def get_current_processed_image(self):
        if not self.current_image_name or self.current_image_name not in self.images:
            return None
        return self.images[self.current_image_name]['processed']

    def start_batch_soma_picking(self):
        self.soma_picking_queue = [name for name, data in self.images.items()
                                   if data['selected'] and data['status'] == 'processed']
        if not self.soma_picking_queue:
            QMessageBox.warning(self, "Warning", "No processed images to pick somas from")
            return
        self.batch_mode = True
        self.current_image_name = self.soma_picking_queue[0]
        self.processed_label.soma_mode = True
        self.original_label.soma_mode = False
        self.preview_label.soma_mode = False
        self.mask_label.soma_mode = False
        self._load_image_for_soma_picking()
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.done_btn.setEnabled(True)
        self.log("=" * 50)
        self.log("🎯 BATCH SOMA PICKING MODE")
        self.log(f"Click somas on: {self.current_image_name}")
        self.log("Click 'Done with Current' when finished with this image")
        self.log("=" * 50)

    def _load_image_for_soma_picking(self):
        if not self.current_image_name:
            return
        img_data = self.images[self.current_image_name]

        # Show color or grayscale based on toggle, with display adjustments
        if self.show_color_view and 'color_image' in img_data:
            # Use processed channel in color composite if available
            proc_color = self._build_processed_color_image(img_data)
            if proc_color is not None:
                adjusted = self._apply_display_adjustments_color(proc_color)
            else:
                adjusted = self._apply_display_adjustments_color(img_data['color_image'])
            pixmap = self._array_to_pixmap_color(adjusted)
        else:
            # Use processed image if available, otherwise extract from color
            if img_data['processed'] is not None:
                gray_img = img_data['processed']
            elif 'color_image' in img_data:
                gray_img = extract_channel(img_data['color_image'], self.grayscale_channel)
            else:
                gray_img = img_data['processed']
            adjusted = self._apply_display_adjustments(gray_img)
            pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)

        self.processed_label.set_image(pixmap, centroids=img_data['somas'])
        self.tabs.setCurrentIndex(2)
        current_idx = self.soma_picking_queue.index(
            self.current_image_name) if self.current_image_name in self.soma_picking_queue else -1
        self.nav_status_label.setText(
            f"Image {current_idx + 1}/{len(self.soma_picking_queue)}: {self.current_image_name} | "
            f"Somas: {len(img_data['somas'])}"
        )

    def _snap_to_brightest(self, coords):
        """Snap coords to the brightest pixel within 5 µm radius."""
        if not self.current_image_name:
            return coords
        img_data = self.images[self.current_image_name]
        # Get the grayscale image to search
        if img_data['processed'] is not None:
            gray = img_data['processed']
        elif 'color_image' in img_data:
            gray = extract_channel(img_data['color_image'], self.grayscale_channel)
        else:
            return coords
        try:
            pixel_size = float(self.pixel_size_input.text())
        except (ValueError, AttributeError):
            pixel_size = 0.3
        radius_px = max(1, int(round(2.0 / pixel_size)))
        row, col = int(coords[0]), int(coords[1])
        h, w = gray.shape[:2]
        r_min = max(0, row - radius_px)
        r_max = min(h, row + radius_px + 1)
        c_min = max(0, col - radius_px)
        c_max = min(w, col + radius_px + 1)
        best_val = -1
        best_r, best_c = row, col
        for r in range(r_min, r_max):
            for c in range(c_min, c_max):
                if (r - row) ** 2 + (c - col) ** 2 <= radius_px ** 2:
                    val = float(gray[r, c])
                    if val > best_val:
                        best_val = val
                        best_r, best_c = r, c
        return (best_r, best_c)

    def add_soma(self, coords):
        if not self.current_image_name:
            return
        img_data = self.images[self.current_image_name]
        coords = self._snap_to_brightest(coords)
        img_data['somas'].append(coords)
        soma_id = f"soma_{coords[0]}_{coords[1]}"
        img_data['soma_ids'].append(soma_id)

        # Show color or grayscale based on toggle, with display adjustments
        if self.show_color_view and 'color_image' in img_data:
            # Use processed channel in color composite if available
            proc_color = self._build_processed_color_image(img_data)
            if proc_color is not None:
                adjusted = self._apply_display_adjustments_color(proc_color)
            else:
                adjusted = self._apply_display_adjustments_color(img_data['color_image'])
            pixmap = self._array_to_pixmap_color(adjusted)
        else:
            # Use processed image if available, otherwise extract from color
            if img_data['processed'] is not None:
                gray_img = img_data['processed']
            elif 'color_image' in img_data:
                gray_img = extract_channel(img_data['color_image'], self.grayscale_channel)
            else:
                gray_img = img_data['processed']
            adjusted = self._apply_display_adjustments(gray_img)
            pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)

        self.processed_label.set_image(pixmap, centroids=img_data['somas'])
        self.log(f"Soma {len(img_data['somas'])} added to {self.current_image_name} | ID: {soma_id}")
        self._load_image_for_soma_picking()

    def undo_last_soma(self):
        """Remove the last picked soma location"""
        if not self.current_image_name:
            return
        img_data = self.images[self.current_image_name]
        if len(img_data['somas']) == 0:
            self.log("No somas to undo")
            return

        # Remove the last soma and its ID
        removed_soma = img_data['somas'].pop()
        removed_id = img_data['soma_ids'].pop() if img_data['soma_ids'] else None

        # Update the display with adjustments
        if self.show_color_view and 'color_image' in img_data:
            # Use processed channel in color composite if available
            proc_color = self._build_processed_color_image(img_data)
            if proc_color is not None:
                adjusted = self._apply_display_adjustments_color(proc_color)
            else:
                adjusted = self._apply_display_adjustments_color(img_data['color_image'])
            pixmap = self._array_to_pixmap_color(adjusted)
        else:
            # Use processed image if available, otherwise extract from color
            if img_data['processed'] is not None:
                gray_img = img_data['processed']
            elif 'color_image' in img_data:
                gray_img = extract_channel(img_data['color_image'], self.grayscale_channel)
            else:
                gray_img = img_data['processed']
            adjusted = self._apply_display_adjustments(gray_img)
            pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)

        self.processed_label.set_image(pixmap, centroids=img_data['somas'])
        self.log(f"Soma removed from {self.current_image_name} | Remaining: {len(img_data['somas'])}")
        self._load_image_for_soma_picking()

    def done_with_current(self):
        if not self.current_image_name:
            return
        img_data = self.images[self.current_image_name]
        if len(img_data['somas']) > 0:
            img_data['status'] = 'somas_picked'
            self._update_file_list_item(self.current_image_name)
        self.navigate_next()

    def navigate_next(self):
        if not self.soma_picking_queue:
            return
        current_idx = self.soma_picking_queue.index(
            self.current_image_name) if self.current_image_name in self.soma_picking_queue else -1
        if current_idx < len(self.soma_picking_queue) - 1:
            self.current_image_name = self.soma_picking_queue[current_idx + 1]
            self._load_image_for_soma_picking()
        else:
            self._finish_soma_picking()

    def navigate_previous(self):
        if not self.soma_picking_queue:
            return
        current_idx = self.soma_picking_queue.index(
            self.current_image_name) if self.current_image_name in self.soma_picking_queue else 0
        if current_idx > 0:
            self.current_image_name = self.soma_picking_queue[current_idx - 1]
            self._load_image_for_soma_picking()

    def _finish_soma_picking(self):
        self.batch_mode = False
        self.processed_label.soma_mode = False
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.done_btn.setEnabled(False)
        total_somas = sum(len(data['somas']) for data in self.images.values() if data['selected'])
        self.batch_outline_btn.setEnabled(True)
        # self.update_workflow_status()
        self.log("=" * 50)
        self.log(f"✓ Soma picking complete! Total somas: {total_somas}")
        self.log("✓ Ready for outlining")
        self.log("=" * 50)
        self._auto_save()
        QMessageBox.information(
            self, "Complete",
            f"Soma picking complete!\n\nTotal somas marked: {total_somas}\n\nReady to outline."
        )

    def start_batch_outlining(self):
        self.outlining_queue = []
        for img_name, img_data in self.images.items():
            if not img_data['selected']:
                continue
            if img_data['status'] not in ('somas_picked', 'outlined'):
                continue
            for soma_idx in range(len(img_data['somas'])):
                self.outlining_queue.append((img_name, soma_idx))
        if not self.outlining_queue:
            QMessageBox.warning(self, "Warning", "No somas to outline")
            return

        # Find first unoutlined soma
        first_unoutlined = self._find_next_unoutlined_idx()
        if first_unoutlined is None:
            QMessageBox.information(self, "Complete", "All somas are already outlined!")
            return

        already_done = first_unoutlined
        remaining = len(self.outlining_queue) - already_done

        # In colocalization mode, ask which channel to use for grayscale outlining
        if self.colocalization_mode:
            sample_color_img = None
            for img_name, img_data in self.images.items():
                if 'color_image' in img_data:
                    sample_color_img = img_data['color_image']
                    break

            dialog = GrayscaleChannelDialog(
                self,
                channel_names=self.channel_names,
                color_image=sample_color_img,
                current_coloc_ch1=self.coloc_channel_1,
                current_coloc_ch2=self.coloc_channel_2
            )
            if dialog.exec_() == QDialog.Accepted:
                self.grayscale_channel = dialog.get_selected_channel()
                self.coloc_channel_1, self.coloc_channel_2 = dialog.get_coloc_channels()
                channel_name = self.channel_names.get(self.grayscale_channel, f'Channel {self.grayscale_channel + 1}')
                self.log(f"Using Channel {self.grayscale_channel + 1} for grayscale outlining")
                self.log(f"Colocalization: Channel {self.coloc_channel_1 + 1} vs Channel {self.coloc_channel_2 + 1}")
            else:
                return

        # Ask user: Manual or Auto?
        dialog = QDialog(self)
        dialog.setWindowTitle("Outline Method")
        dialog.setModal(True)
        layout = QVBoxLayout()

        if already_done > 0:
            label = QLabel(
                f"<b>{remaining} somas remaining</b> ({already_done}/{len(self.outlining_queue)} already done)"
                f"<br><br>Choose outline method for remaining somas:"
            )
        else:
            label = QLabel(f"<b>{len(self.outlining_queue)} somas to outline</b><br><br>Choose outline method:")
        layout.addWidget(label)

        manual_btn = QPushButton("Manual - Draw each outline by hand")
        manual_btn.clicked.connect(lambda: dialog.done(1))
        layout.addWidget(manual_btn)

        auto_btn = QPushButton("Auto - Auto-detect all, then review/fix")
        auto_btn.clicked.connect(lambda: dialog.done(2))
        auto_btn.setStyleSheet("border: 2px solid #4CAF50; font-weight: bold;")
        layout.addWidget(auto_btn)

        # Auto settings
        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("Method:"))
        method_combo = QComboBox()
        method_combo.addItems(["Threshold", "Region Grow", "Watershed", "Active Contour", "Hybrid"])
        method_combo.setCurrentIndex(self.auto_outline_method.currentIndex())
        settings_layout.addWidget(method_combo)
        settings_layout.addWidget(QLabel("Sens:"))
        sens_spin = QSpinBox()
        sens_spin.setRange(10, 90)
        sens_spin.setValue(self.auto_outline_sensitivity.value())
        settings_layout.addWidget(sens_spin)
        layout.addLayout(settings_layout)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(lambda: dialog.done(0))
        layout.addWidget(cancel_btn)

        dialog.setLayout(layout)
        result = dialog.exec_()

        if result == 0:
            return  # Cancelled

        # Update auto settings from dialog
        self.auto_outline_method.setCurrentIndex(method_combo.currentIndex())
        self.auto_outline_sensitivity.setValue(sens_spin.value())

        # Initialize outlining state
        self.batch_mode = True
        self.polygon_points = []
        self.processed_label.polygon_mode = True
        self.processed_label.soma_mode = False
        self.processed_label.point_edit_mode = False
        self.processed_label.selected_point_idx = None
        self.original_label.polygon_mode = False
        self.preview_label.polygon_mode = False
        self.mask_label.polygon_mode = False
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.done_btn.setEnabled(False)

        # Store for review mode
        self.auto_outlined_points = {}  # {queue_idx: points}
        self.failed_auto_outlines = []  # List of queue_idx that failed
        self.review_mode = False
        self.current_review_idx = 0
        self.current_outline_idx = 0

        # Show outline controls and progress bar
        self.outline_controls_widget.setVisible(True)
        self.outline_method_display.setCurrentIndex(self.auto_outline_method.currentIndex())
        self.outline_sens_display.setValue(self.auto_outline_sensitivity.value())
        self._update_outline_progress()

        if result == 1:
            # Manual mode
            self._start_manual_outlining()
        else:
            # Auto mode - outline all, then review
            self._run_auto_outline_all()

    def _start_manual_outlining(self):
        """Start manual outlining mode - draw each soma one by one"""
        start_idx = self._find_next_unoutlined_idx()
        if start_idx is None:
            self._finish_outlining()
            return
        self._load_soma_for_outlining(start_idx)

        self.auto_outline_btn.setEnabled(True)
        self.manual_draw_btn.setEnabled(True)
        self.accept_outline_btn.setEnabled(False)

        self.log("=" * 50)
        self.log(f"📐 MANUAL OUTLINING MODE")
        self.log(f"Total somas: {len(self.outlining_queue)}")
        self.log("")
        self.log("Click to add points, right-click to complete")
        self.log("Press Enter or [Accept] to save and move to next")
        self.log("=" * 50)

    def _run_auto_outline_all(self):
        """Run auto-outline on all somas, then start review mode"""
        method = self._get_auto_outline_method()
        sensitivity = self.auto_outline_sensitivity.value()

        self.log("=" * 50)
        self.log(f"📐 AUTO-OUTLINING ALL SOMAS...")
        self.log(f"Method: {self.auto_outline_method.currentText()}")
        self.log(f"Sensitivity: {sensitivity}")
        self.log("=" * 50)

        success_count = 0
        fail_count = 0
        self.auto_outlined_points = {}
        self.failed_auto_outlines = []

        for i, (img_name, soma_idx) in enumerate(self.outlining_queue):
            # Skip already-outlined somas
            if self._soma_has_outline(img_name, soma_idx):
                continue

            img_data = self.images[img_name]
            soma = img_data['somas'][soma_idx]

            # Get image for outlining
            outline_img = self._get_image_for_outlining(img_data)
            if outline_img is None:
                fail_count += 1
                self.failed_auto_outlines.append(i)
                continue

            try:
                points = method(outline_img, soma, sensitivity)
            except Exception as e:
                self.log(f"Error on soma {i+1}: {e}")
                points = None

            if points is None or len(points) < 3:
                fail_count += 1
                self.failed_auto_outlines.append(i)
            else:
                self.auto_outlined_points[i] = list(points)
                success_count += 1

        self.log("")
        self.log(f"✓ Auto-outlined: {success_count}/{len(self.outlining_queue)}")
        if fail_count > 0:
            self.log(f"⚠ Failed: {fail_count} (will need manual)")

        # Check if too many failed
        if fail_count > len(self.outlining_queue) * 0.5:
            reply = QMessageBox.question(
                self, "Many Failures",
                f"{fail_count} out of {len(self.outlining_queue)} somas failed to auto-outline.\n\n"
                f"Do you want to:\n"
                f"• Yes - Continue and manually draw the failed ones\n"
                f"• No - Restart with different settings",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                self.batch_mode = False
                self.processed_label.polygon_mode = False
                self.start_batch_outlining()  # Restart
                return

        # Start review mode
        self._start_review_mode()

    def _start_review_mode(self):
        """Start reviewing auto-outlined somas one by one"""
        self.review_mode = True
        self.current_review_idx = 0

        self.log("")
        self.log("=" * 50)
        self.log("📋 REVIEW MODE - Check each outline")
        self.log("• Drag points to adjust")
        self.log("• Press Enter or [Accept] to approve")
        self.log("• Click [Manual] to redraw from scratch")
        self.log("=" * 50)

        self._load_review_soma(0)

    def _load_review_soma(self, review_idx):
        """Load a soma for review"""
        if review_idx >= len(self.outlining_queue):
            self._finish_review_mode()
            return

        self.current_review_idx = review_idx
        self.current_outline_idx = review_idx
        img_name, soma_idx = self.outlining_queue[review_idx]
        img_data = self.images[img_name]
        soma = img_data['somas'][soma_idx]
        soma_id = img_data['soma_ids'][soma_idx]

        # Check if this soma has auto-outlined points or needs manual
        if review_idx in self.auto_outlined_points:
            self.polygon_points = self.auto_outlined_points[review_idx].copy()
            status = "Auto - Review/Edit"
        else:
            self.polygon_points = []
            status = "MANUAL NEEDED"

        pixmap = self._get_outlining_pixmap(img_data)
        self.processed_label.set_image(pixmap, centroids=[soma], polygon_pts=self.polygon_points)
        self.processed_label.zoom_to_point(soma[0], soma[1], zoom_level=3.0)
        self.tabs.setCurrentIndex(2)

        self.nav_status_label.setText(
            f"Review {review_idx + 1}/{len(self.outlining_queue)} | "
            f"{img_name} | {soma_id} | {status}"
        )

        # Enable/disable buttons
        self.auto_outline_btn.setEnabled(True)
        self.manual_draw_btn.setEnabled(True)
        self.accept_outline_btn.setEnabled(len(self.polygon_points) >= 3)

        if review_idx in self.auto_outlined_points:
            self.log(f"Reviewing {soma_id} - {len(self.polygon_points)} points")
        else:
            self.log(f"⚠ {soma_id} needs manual outline")

    def _finish_review_mode(self):
        """Finish review mode and complete outlining"""
        self.review_mode = False
        self._finish_outlining()

    def _get_outlining_pixmap(self, img_data):
        """Get the appropriate pixmap for outlining with display adjustments"""
        # Support color view toggle during outlining
        if self.show_color_view and 'color_image' in img_data:
            # Use processed channel in color composite if available
            proc_color = self._build_processed_color_image(img_data)
            if proc_color is not None:
                adjusted = self._apply_display_adjustments_color(proc_color)
            else:
                adjusted = self._apply_display_adjustments_color(img_data['color_image'])
            return self._array_to_pixmap_color(adjusted)

        # Grayscale mode
        if self.colocalization_mode and 'color_image' in img_data:
            # Extract the selected channel from color image for grayscale display
            color_img = img_data['color_image']
            if color_img.ndim == 3 and color_img.shape[2] > self.grayscale_channel:
                # Get the selected channel
                channel_img = color_img[:, :, self.grayscale_channel].astype(np.float32)
                # Normalize
                c_min, c_max = channel_img.min(), channel_img.max()
                if c_max > c_min:
                    channel_img = (channel_img - c_min) / (c_max - c_min) * 255
                # Apply display adjustments
                adjusted = self._apply_display_adjustments(channel_img.astype(np.uint8))
                return self._array_to_pixmap(adjusted, skip_rescale=True)
        # Default: use processed image with adjustments
        adjusted = self._apply_display_adjustments(img_data['processed'])
        return self._array_to_pixmap(adjusted, skip_rescale=True)

    def _load_soma_for_outlining(self, queue_idx):
        if queue_idx >= len(self.outlining_queue):
            self._finish_outlining()
            return
        self.current_outline_idx = queue_idx
        img_name, soma_idx = self.outlining_queue[queue_idx]
        self.current_image_name = img_name
        img_data = self.images[img_name]
        soma = img_data['somas'][soma_idx]
        soma_id = img_data['soma_ids'][soma_idx]
        pixmap = self._get_outlining_pixmap(img_data)
        self.processed_label.set_image(pixmap, centroids=[soma], polygon_pts=self.polygon_points)
        self.processed_label.zoom_to_point(soma[0], soma[1], zoom_level=3.0)
        self.tabs.setCurrentIndex(2)
        self.nav_status_label.setText(
            f"Soma {queue_idx + 1}/{len(self.outlining_queue)} | "
            f"Image: {img_name} | ID: {soma_id}"
        )
        self.log(f"Outlining {soma_id} ({queue_idx + 1}/{len(self.outlining_queue)})")

    def add_polygon_point(self, coords):
        # Reset zoom state when adding points (prevents getting stuck)
        self.z_key_held = False

        self.polygon_points.append(coords)
        queue_idx = getattr(self, 'current_outline_idx', 0)
        if queue_idx < len(self.outlining_queue):
            img_name, soma_idx = self.outlining_queue[queue_idx]
            img_data = self.images[img_name]
            soma = img_data['somas'][soma_idx]
            soma_id = img_data['soma_ids'][soma_idx]
            pixmap = self._get_outlining_pixmap(img_data)
            self.processed_label.set_image(pixmap, centroids=[soma], polygon_pts=self.polygon_points)
            # Update status to show point count
            self.nav_status_label.setText(
                f"Soma {queue_idx + 1}/{len(self.outlining_queue)} | "
                f"Image: {img_name} | ID: {soma_id} | Points: {len(self.polygon_points)}"
            )

        # Enable accept button when we have enough points
        if len(self.polygon_points) >= 3:
            self.accept_outline_btn.setEnabled(True)

    def undo_last_polygon_point(self):
        """Remove the last point added to the polygon"""
        if len(self.polygon_points) > 0:
            self.polygon_points.pop()
            self.log(f"↩️ Undid last point ({len(self.polygon_points)} points remaining)")
            # Refresh the display
            queue_idx = getattr(self, 'current_outline_idx', 0)
            if queue_idx < len(self.outlining_queue):
                img_name, soma_idx = self.outlining_queue[queue_idx]
                img_data = self.images[img_name]
                soma = img_data['somas'][soma_idx]
                soma_id = img_data['soma_ids'][soma_idx]
                pixmap = self._get_outlining_pixmap(img_data)
                self.processed_label.set_image(pixmap, centroids=[soma], polygon_pts=self.polygon_points)
                self.nav_status_label.setText(
                    f"Soma {queue_idx + 1}/{len(self.outlining_queue)} | "
                    f"Image: {img_name} | ID: {soma_id} | Points: {len(self.polygon_points)}"
                )
            # Update accept button state
            self.accept_outline_btn.setEnabled(len(self.polygon_points) >= 3)
        else:
            self.log("⚠️ No points to undo")

    def restart_polygon(self):
        """Clear all points and restart the current outline"""
        if len(self.polygon_points) > 0:
            self.polygon_points = []
            self.log("🔄 Restarted outline (all points cleared)")
            # Refresh the display
            queue_idx = getattr(self, 'current_outline_idx', 0)
            if queue_idx < len(self.outlining_queue):
                img_name, soma_idx = self.outlining_queue[queue_idx]
                img_data = self.images[img_name]
                soma = img_data['somas'][soma_idx]
                soma_id = img_data['soma_ids'][soma_idx]
                pixmap = self._get_outlining_pixmap(img_data)
                self.processed_label.set_image(pixmap, centroids=[soma], polygon_pts=self.polygon_points)
                self.nav_status_label.setText(
                    f"Soma {queue_idx + 1}/{len(self.outlining_queue)} | "
                    f"Image: {img_name} | ID: {soma_id} | Points: {len(self.polygon_points)}"
                )
            # Disable accept button since no points
            self.accept_outline_btn.setEnabled(False)
        else:
            self.log("⚠️ No points to clear")

    def finish_polygon(self):
        # Reset zoom state to prevent getting stuck
        self.z_key_held = False
        self.processed_label.selected_point_idx = None
        self.processed_label.dragging_point = False

        if len(self.polygon_points) < 3:
            QMessageBox.warning(self, "Warning", "Need at least 3 points")
            return

        queue_idx = getattr(self, 'current_outline_idx', 0)

        if queue_idx >= len(self.outlining_queue):
            return

        img_name, soma_idx = self.outlining_queue[queue_idx]
        img_data = self.images[img_name]
        mask = self._polygon_to_mask(self.polygon_points, img_data['processed'].shape)
        soma_id = img_data['soma_ids'][soma_idx]

        # Calculate soma area from the outline
        pixel_size = float(self.pixel_size_input.text())
        soma_area_um2 = np.sum(mask) * (pixel_size ** 2)

        img_data['soma_outlines'].append({
            'soma_idx': soma_idx,
            'soma_id': soma_id,
            'centroid': img_data['somas'][soma_idx],
            'outline': mask,
            'polygon_points': self.polygon_points.copy(),
            'soma_area_um2': soma_area_um2
        })

        # Export soma outline to file
        self._export_soma_outline(img_name, soma_id, mask, pixel_size, soma_area_um2)

        self.log(f"✓ {soma_id} approved (soma area: {soma_area_um2:.1f} µm²)")
        self.polygon_points = []

        # Update outline progress
        self._update_outline_progress()

        # Auto-save after each outline
        self._auto_save()

        # Enable Redo button after first outline is complete
        self.redo_outline_btn.setEnabled(True)

        # Reset accept button for next soma
        self.accept_outline_btn.setEnabled(False)

        # Move to next unoutlined soma
        next_idx = self._find_next_unoutlined_idx(start_from=queue_idx + 1)
        if next_idx is None:
            self._finish_outlining()
        elif hasattr(self, 'review_mode') and self.review_mode:
            self._load_review_soma(next_idx)
        else:
            self._load_soma_for_outlining(next_idx)

    def redo_last_outline(self):
        """Delete the last completed outline and go back to redo it"""
        # Find the last outline by walking the outlining_queue in reverse
        # and checking which ones still have outlines saved
        last_outline_img = None
        last_outline_data = None
        last_queue_idx = None

        # Count outlines in queue order to find the last completed one
        outline_count = 0
        for qi, (img_name, soma_idx) in enumerate(self.outlining_queue):
            img_data = self.images[img_name]
            soma_id = img_data['soma_ids'][soma_idx]
            # Check if this queue entry has a matching outline
            for ol in img_data['soma_outlines']:
                if ol['soma_idx'] == soma_idx and ol['soma_id'] == soma_id:
                    last_outline_img = img_name
                    last_outline_data = ol
                    last_queue_idx = qi
                    break

        if not last_outline_data:
            QMessageBox.warning(self, "Warning", "No outlines to redo")
            return

        # Remove the outline
        self.images[last_outline_img]['soma_outlines'].remove(last_outline_data)

        # Delete the exported soma file if it exists
        if self.masks_dir:
            soma_id = last_outline_data['soma_id']
            soma_file = os.path.join(self.masks_dir, f"{os.path.splitext(last_outline_img)[0]}_{soma_id}_soma.tif")
            if os.path.exists(soma_file):
                os.remove(soma_file)

        self.log(f"↩ Undid outline for {last_outline_data['soma_id']} - ready to redo")

        # Update outline progress
        self._update_outline_progress()

        # Go back to that soma for re-outlining
        self.polygon_points = []

        # If no more outlines left in this session, disable the redo button
        has_outlines_in_queue = any(
            self._soma_has_outline(in_, si) for in_, si in self.outlining_queue
        )
        if not has_outlines_in_queue:
            self.redo_outline_btn.setEnabled(False)

        self._load_soma_for_outlining(last_queue_idx)

    def _get_auto_outline_method(self):
        """Get the selected auto-outline method function"""
        method_idx = self.auto_outline_method.currentIndex()
        methods = [
            auto_outline_threshold,
            auto_outline_region_growing,
            auto_outline_watershed,
            auto_outline_active_contours,
            auto_outline_hybrid
        ]
        return methods[method_idx]

    def _get_image_for_outlining(self, img_data):
        """Get the appropriate grayscale image for auto-outlining"""
        if img_data['processed'] is not None:
            return img_data['processed']
        elif 'color_image' in img_data:
            return extract_channel(img_data['color_image'], self.grayscale_channel)
        return None

    def auto_outline_current_soma(self):
        """Auto-outline the current soma using selected method"""
        if not self.outlining_queue:
            QMessageBox.warning(self, "Warning", "No somas in outlining queue")
            return

        queue_idx = getattr(self, 'current_outline_idx', 0)
        if queue_idx >= len(self.outlining_queue):
            QMessageBox.warning(self, "Warning", "All somas already outlined")
            return

        img_name, soma_idx = self.outlining_queue[queue_idx]
        img_data = self.images[img_name]
        soma = img_data['somas'][soma_idx]
        soma_id = img_data['soma_ids'][soma_idx]

        # Get the image for outlining
        outline_img = self._get_image_for_outlining(img_data)
        if outline_img is None:
            QMessageBox.warning(self, "Warning", "No processed image available")
            return

        # Get method and sensitivity
        method = self._get_auto_outline_method()
        sensitivity = self.auto_outline_sensitivity.value()

        self.log(f"Auto-outlining {soma_id} using {self.auto_outline_method.currentText()}...")

        # Run auto-outline
        try:
            points = method(outline_img, soma, sensitivity)
        except Exception as e:
            self.log(f"Error: {str(e)}")
            points = None

        if points is None or len(points) < 3:
            self.log(f"⚠ Auto-outline failed for {soma_id} - please outline manually")
            QMessageBox.warning(
                self, "Auto-Outline Failed",
                f"Could not auto-outline {soma_id}.\n\n"
                "Try adjusting sensitivity or use manual outlining."
            )
            return

        # Set the polygon points and display
        self.polygon_points = list(points)
        pixmap = self._get_outlining_pixmap(img_data)
        self.processed_label.set_image(pixmap, centroids=[soma], polygon_pts=self.polygon_points)

        # Enable point editing mode
        self.processed_label.point_edit_mode = True
        self.processed_label.selected_point_idx = None

        self.nav_status_label.setText(
            f"Soma {queue_idx + 1}/{len(self.outlining_queue)} | "
            f"Image: {img_name} | ID: {soma_id} | Points: {len(self.polygon_points)} (Auto)"
        )
        self.log(f"✓ Auto-outlined {soma_id} with {len(self.polygon_points)} points")
        self.log("  → Drag points to adjust, then press Enter or [Accept]")

        # Enable accept button for review
        self.accept_outline_btn.setEnabled(True)
        self.manual_draw_btn.setEnabled(True)  # Can still switch to manual

    def auto_outline_all_somas(self):
        """Auto-outline all remaining somas in the queue"""
        if not self.outlining_queue:
            QMessageBox.warning(self, "Warning", "No somas in outlining queue")
            return

        queue_idx = getattr(self, 'current_outline_idx', 0)
        remaining = len(self.outlining_queue) - queue_idx

        if remaining <= 0:
            QMessageBox.warning(self, "Warning", "All somas already outlined")
            return

        reply = QMessageBox.question(
            self, 'Auto-Outline All',
            f"Auto-outline {remaining} remaining soma(s)?\n\n"
            f"Method: {self.auto_outline_method.currentText()}\n"
            f"Sensitivity: {self.auto_outline_sensitivity.value()}\n\n"
            "You can review and adjust each outline afterward.",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.No:
            return

        method = self._get_auto_outline_method()
        sensitivity = self.auto_outline_sensitivity.value()

        success_count = 0
        fail_count = 0
        failed_somas = []

        self.log(f"Auto-outlining {remaining} somas...")

        for i in range(queue_idx, len(self.outlining_queue)):
            img_name, soma_idx = self.outlining_queue[i]
            # Skip already-outlined somas
            if self._soma_has_outline(img_name, soma_idx):
                continue
            img_data = self.images[img_name]
            soma = img_data['somas'][soma_idx]
            soma_id = img_data['soma_ids'][soma_idx]

            outline_img = self._get_image_for_outlining(img_data)
            if outline_img is None:
                fail_count += 1
                failed_somas.append(soma_id)
                continue

            try:
                points = method(outline_img, soma, sensitivity)
            except Exception:
                points = None

            if points is None or len(points) < 3:
                fail_count += 1
                failed_somas.append(soma_id)
                continue

            # Create mask from points
            mask = self._polygon_to_mask(points, outline_img.shape)
            pixel_size = float(self.pixel_size_input.text())
            soma_area_um2 = np.sum(mask) * (pixel_size ** 2)

            # Save outline
            img_data['soma_outlines'].append({
                'soma_idx': soma_idx,
                'soma_id': soma_id,
                'centroid': soma,
                'outline': mask,
                'polygon_points': list(points),
                'soma_area_um2': soma_area_um2,
                'auto_outlined': True  # Mark as auto-outlined for review
            })

            # Export soma outline
            self._export_soma_outline(img_name, soma_id, mask, pixel_size, soma_area_um2)
            success_count += 1

        self.log(f"✓ Auto-outlined {success_count} somas")
        if fail_count > 0:
            self.log(f"⚠ Failed to auto-outline {fail_count} somas: {', '.join(failed_somas)}")

        # Update outline progress
        self._update_outline_progress()

        # Check if all done — find first queue entry without an outline
        next_idx = self._find_next_unoutlined_idx()
        if next_idx is None:
            self._finish_outlining()
        else:
            # Load the first failed soma for manual outlining
            self._load_soma_for_outlining(next_idx)
            self.auto_outline_btn.setEnabled(True)
            self.manual_draw_btn.setEnabled(True)

        QMessageBox.information(
            self, "Auto-Outline Complete",
            f"Successfully outlined: {success_count}\n"
            f"Failed (need manual): {fail_count}\n\n"
            f"{'All somas outlined!' if fail_count == 0 else 'Please manually outline the remaining somas.'}"
        )

    def start_manual_outline(self):
        """Switch to manual outline mode - clear any auto points and let user draw"""
        # Clear any existing points
        self.polygon_points = []
        self.processed_label.selected_point_idx = None
        self.processed_label.dragging_point = False

        queue_idx = getattr(self, 'current_outline_idx', 0)
        if queue_idx < len(self.outlining_queue):
            img_name, soma_idx = self.outlining_queue[queue_idx]
            img_data = self.images[img_name]
            soma = img_data['somas'][soma_idx]
            soma_id = img_data['soma_ids'][soma_idx]
            pixmap = self._get_outlining_pixmap(img_data)
            self.processed_label.set_image(pixmap, centroids=[soma], polygon_pts=[])
            self.nav_status_label.setText(
                f"Soma {queue_idx + 1}/{len(self.outlining_queue)} | "
                f"Image: {img_name} | ID: {soma_id} | Manual Mode"
            )

        self.log("✏ Manual mode - click to add points, right-click to complete")
        self.accept_outline_btn.setEnabled(False)  # Re-enable after points are drawn

    def accept_current_outline(self):
        """Accept the current outline and move to next soma (same as finish_polygon)"""
        if len(self.polygon_points) < 3:
            QMessageBox.warning(self, "Warning", "Need at least 3 points to accept outline")
            return
        self.finish_polygon()

    def manual_override_outline(self):
        """Legacy function - redirects to start_manual_outline"""
        self.start_manual_outline()

    def clear_all_masks(self):
        """Delete all generated masks and allow regeneration"""
        # Count total masks
        total_masks = sum(len(data['masks']) for data in self.images.values())
        
        if total_masks == 0:
            QMessageBox.information(self, "Info", "No masks to clear")
            return
        
        reply = QMessageBox.question(
            self, 'Clear All Masks',
            f"Delete all {total_masks} generated masks?\n\n"
            f"This will allow you to regenerate masks with different settings.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        # Clear masks from all images
        masks_cleared = 0
        for img_name, img_data in self.images.items():
            if img_data['masks']:
                masks_cleared += len(img_data['masks'])
                img_data['masks'] = []
                # Update status back to 'outlined'
                if img_data['status'] in ['masks_generated', 'qa_complete', 'morphology_calculated']:
                    img_data['status'] = 'outlined'
                    self._update_file_list_item(img_name)
        
        # Delete mask files from disk if they exist
        if self.masks_dir and os.path.exists(self.masks_dir):
            mask_files = glob.glob(os.path.join(self.masks_dir, "*_mask.tif"))
            for mask_file in mask_files:
                try:
                    os.remove(mask_file)
                except Exception as e:
                    self.log(f"Warning: Could not delete {os.path.basename(mask_file)}: {e}")
        
        # Disable buttons that depend on masks
        self.batch_qa_btn.setEnabled(False)
        self.batch_calculate_btn.setEnabled(False)
        self.clear_masks_btn.setEnabled(False)
        
        # Re-enable mask generation
        self.batch_generate_masks_btn.setEnabled(True)
        
        self.log("=" * 50)
        self.log(f"🗑 Cleared {masks_cleared} masks")
        self.log("✓ Ready to regenerate masks with new settings")
        self.log("=" * 50)

    def _polygon_to_mask(self, polygon, shape):
        if len(polygon) < 3:
            return np.zeros(shape, dtype=np.uint8)
        poly_array = np.array([[p[1], p[0]] for p in polygon])
        h, w = shape[:2]
        yy, xx = np.mgrid[:h, :w]
        points = np.c_[xx.ravel(), yy.ravel()]
        path = mplPath(poly_array)
        mask = path.contains_points(points).reshape(h, w)
        return mask.astype(np.uint8)

    def _soma_has_outline(self, img_name, soma_idx):
        """Check if a soma already has a completed outline."""
        img_data = self.images[img_name]
        soma_id = img_data['soma_ids'][soma_idx]
        return any(
            ol['soma_idx'] == soma_idx and ol['soma_id'] == soma_id
            for ol in img_data['soma_outlines']
        )

    def _find_next_unoutlined_idx(self, start_from=0):
        """Find the first queue entry that doesn't have an outline yet."""
        for qi in range(start_from, len(self.outlining_queue)):
            img_name, soma_idx = self.outlining_queue[qi]
            if not self._soma_has_outline(img_name, soma_idx):
                return qi
        return None  # All done

    def _update_outline_progress(self):
        """Update the outline progress bar with current completion count."""
        if not hasattr(self, 'outlining_queue') or not self.outlining_queue:
            return
        completed = sum(1 for in_, si in self.outlining_queue if self._soma_has_outline(in_, si))
        total = len(self.outlining_queue)
        self.outline_progress_bar.setMaximum(total)
        self.outline_progress_bar.setValue(completed)
        self.outline_progress_bar.setVisible(True)

    def _finish_outlining(self):
        self.outline_progress_bar.setVisible(False)
        self.outline_controls_widget.setVisible(False)
        self.batch_mode = False
        self.processed_label.polygon_mode = False
        self.processed_label.point_edit_mode = False
        self.processed_label.selected_point_idx = None
        self.redo_outline_btn.setEnabled(False)  # Disable redo button when outlining complete

        # Disable outline buttons
        self.auto_outline_btn.setEnabled(False)
        self.manual_draw_btn.setEnabled(False)
        self.accept_outline_btn.setEnabled(False)

        for img_name, img_data in self.images.items():
            if img_data['selected'] and len(img_data['soma_outlines']) == len(img_data['somas']):
                img_data['status'] = 'outlined'
                self._update_file_list_item(img_name)
        self.batch_generate_masks_btn.setEnabled(True)
        # self.update_workflow_status()
        self.log("=" * 50)
        self.log("✓ All somas outlined!")
        self.log("✓ Ready to generate masks")
        self.log("=" * 50)
        self._auto_save()
        QMessageBox.information(self, "Complete", "All somas outlined!\n\nReady to generate masks.")

    def batch_generate_masks(self):
        # Show mask generation settings dialog first
        dialog = QDialog(self)
        dialog.setWindowTitle("Mask Generation Settings")
        dialog.setModal(True)

        layout = QVBoxLayout()

        # Title
        title = QLabel("Configure Mask Generation Settings")
        title_font = title.font()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        layout.addWidget(title)

        # --- Mask Size Settings ---
        size_group = QLabel("<b>Mask Sizes (µm²)</b>")
        layout.addWidget(size_group)

        size_grid = QHBoxLayout()

        size_grid.addWidget(QLabel("Min:"))
        min_area_spin = QSpinBox()
        min_area_spin.setRange(100, 2000)
        min_area_spin.setSingleStep(50)
        min_area_spin.setValue(self.mask_min_area)
        min_area_spin.setSuffix(" µm²")
        size_grid.addWidget(min_area_spin)

        size_grid.addWidget(QLabel("Max:"))
        max_area_spin = QSpinBox()
        max_area_spin.setRange(200, 5000)
        max_area_spin.setSingleStep(50)
        max_area_spin.setValue(self.mask_max_area)
        max_area_spin.setSuffix(" µm²")
        size_grid.addWidget(max_area_spin)

        size_grid.addWidget(QLabel("Step:"))
        step_spin = QSpinBox()
        step_spin.setRange(10, 500)
        step_spin.setSingleStep(10)
        step_spin.setValue(self.mask_step_size)
        step_spin.setSuffix(" µm²")
        size_grid.addWidget(step_spin)

        layout.addLayout(size_grid)

        # Preview of mask sizes that will be generated
        preview_label = QLabel("")
        preview_label.setStyleSheet("color: palette(dark); font-size: 10px;")
        preview_label.setWordWrap(True)
        layout.addWidget(preview_label)

        def update_size_preview():
            mn = min_area_spin.value()
            mx = max_area_spin.value()
            st = step_spin.value()
            if mn > mx:
                preview_label.setText("⚠ Min must be ≤ Max")
                return
            sizes = list(range(mn, mx + 1, st))
            if sizes[-1] != mx:
                sizes.append(mx)
            preview_label.setText(f"Masks: {', '.join(str(s) for s in sizes)} µm²  ({len(sizes)} masks per cell)")

        min_area_spin.valueChanged.connect(lambda: update_size_preview())
        max_area_spin.valueChanged.connect(lambda: update_size_preview())
        step_spin.valueChanged.connect(lambda: update_size_preview())
        update_size_preview()

        layout.addSpacing(10)

        # --- Intensity Settings ---
        intensity_group = QLabel("<b>Intensity Filtering</b>")
        layout.addWidget(intensity_group)

        # Minimum intensity threshold checkbox
        min_intensity_check = QCheckBox("Use minimum intensity threshold")
        min_intensity_check.setChecked(self.use_min_intensity)
        min_intensity_check.setToolTip("Exclude pixels below this intensity from masks")
        layout.addWidget(min_intensity_check)

        # Minimum intensity slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("  Min intensity:"))
        min_intensity_slider = QSlider(Qt.Horizontal)
        min_intensity_slider.setRange(0, 100)
        min_intensity_slider.setValue(self.min_intensity_percent)
        slider_layout.addWidget(min_intensity_slider)
        min_intensity_label = QLabel(f"{self.min_intensity_percent}%")
        min_intensity_slider.valueChanged.connect(
            lambda v: min_intensity_label.setText(f"{v}%")
        )
        slider_layout.addWidget(min_intensity_label)
        layout.addLayout(slider_layout)

        layout.addSpacing(10)

        # Help text
        help_text = QLabel(
            "💡 Tip:\n"
            "• 0% = No minimum (include all pixels)\n"
            "• 30% = Default (good for most images)\n"
            "• 50% = Strict (exclude dimmer pixels)\n"
            "• 70%+ = Very strict (only bright pixels)"
        )
        help_text.setStyleSheet("padding: 10px; border-radius: 5px; border: 1px solid palette(mid);")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        layout.addSpacing(10)

        # Buttons
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)

        ok_btn = QPushButton("Generate Masks")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(dialog.accept)
        ok_btn.setStyleSheet(
            "QPushButton { border: 2px solid #4CAF50; font-weight: bold; padding: 5px; }")
        button_layout.addWidget(ok_btn)

        layout.addLayout(button_layout)

        dialog.setLayout(layout)
        dialog.setMinimumWidth(400)

        # Show dialog and get result
        if dialog.exec_() != QDialog.Accepted:
            return  # User cancelled

        # Save settings
        self.use_min_intensity = min_intensity_check.isChecked()
        self.min_intensity_percent = min_intensity_slider.value()
        self.mask_min_area = min_area_spin.value()
        self.mask_max_area = max_area_spin.value()
        self.mask_step_size = step_spin.value()

        # Now proceed with mask generation
        try:
            pixel_size = float(self.pixel_size_input.text())

            # Build area list from user settings
            area_list = list(range(self.mask_min_area, self.mask_max_area + 1, self.mask_step_size))
            if area_list[-1] != self.mask_max_area:
                area_list.append(self.mask_max_area)

            self.progress_bar.setVisible(True)
            self.progress_status_label.setVisible(True)

            total_outlines = sum(len(data['soma_outlines']) for data in self.images.values() if data['selected'])
            current_count = 0

            for img_name, img_data in self.images.items():
                if not img_data['selected'] or img_data['status'] != 'outlined':
                    continue

                self.log(f"Generating masks for {img_name}...")

                for soma_data in img_data['soma_outlines']:
                    centroid = soma_data['centroid']
                    soma_idx = soma_data['soma_idx']
                    soma_id = soma_data['soma_id']
                    soma_area_um2 = soma_data.get('soma_area_um2', 0)  # Get soma area from outline

                    self.progress_status_label.setText(f"Generating masks: {current_count + 1}/{total_outlines}")
                    QApplication.processEvents()

                    masks = self._create_annulus_masks(
                        centroid, area_list, pixel_size, soma_idx, soma_id,
                        img_data['processed'], img_name, soma_area_um2  # Pass soma area
                    )
                    img_data['masks'].extend(masks)

                    current_count += 1
                    self.progress_bar.setValue(int((current_count / total_outlines) * 100))

                img_data['status'] = 'masks_generated'
                self._update_file_list_item(img_name)

            self.progress_bar.setVisible(False)
            self.progress_status_label.setVisible(False)

            self.batch_qa_btn.setEnabled(True)
            self.clear_masks_btn.setEnabled(True)
            # self.update_workflow_status()

            total_masks = sum(len(data['masks']) for data in self.images.values() if data['selected'])

            self.log("=" * 50)
            self.log(f"✓ Generated {total_masks} masks total")
            self.log(f"✓ Mask sizes: {', '.join(str(a) for a in area_list)} µm²")
            if self.use_min_intensity:
                self.log(f"✓ Used minimum intensity: {self.min_intensity_percent}%")
            self.log("✓ Ready for QA")
            self.log("=" * 50)
            self._auto_save()

            QMessageBox.information(
                self, "Success",
                f"Generated {total_masks} masks!\n\nReady for QA."
            )

        except Exception as e:
            self.progress_bar.setVisible(False)
            self.progress_status_label.setVisible(False)
            self.log(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed: {e}")

    def regenerate_masks_current_image(self):
        """Regenerate masks for the image currently shown in QA with custom settings."""
        # Determine which image we're looking at
        if self.mask_qa_active and self.all_masks_flat and self.mask_qa_idx < len(self.all_masks_flat):
            img_name = self.all_masks_flat[self.mask_qa_idx]['image_name']
        elif self.current_image_name and self.current_image_name in self.images:
            img_name = self.current_image_name
        else:
            QMessageBox.warning(self, "Warning", "No image selected.")
            return

        img_data = self.images[img_name]

        if not img_data.get('soma_outlines'):
            QMessageBox.warning(self, "Warning", f"{img_name} has no soma outlines.")
            return

        # Show settings dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Redo Masks — {os.path.splitext(img_name)[0]}")
        dialog.setModal(True)

        layout = QVBoxLayout()

        title = QLabel(f"<b>{os.path.splitext(img_name)[0]}</b>")
        layout.addWidget(title)

        # Mask Size Settings
        size_grid = QHBoxLayout()
        size_grid.addWidget(QLabel("Min:"))
        min_area_spin = QSpinBox()
        min_area_spin.setRange(100, 2000)
        min_area_spin.setSingleStep(50)
        min_area_spin.setValue(self.mask_min_area)
        min_area_spin.setSuffix(" µm²")
        size_grid.addWidget(min_area_spin)

        size_grid.addWidget(QLabel("Max:"))
        max_area_spin = QSpinBox()
        max_area_spin.setRange(200, 5000)
        max_area_spin.setSingleStep(50)
        max_area_spin.setValue(self.mask_max_area)
        max_area_spin.setSuffix(" µm²")
        size_grid.addWidget(max_area_spin)

        size_grid.addWidget(QLabel("Step:"))
        step_spin = QSpinBox()
        step_spin.setRange(10, 500)
        step_spin.setSingleStep(10)
        step_spin.setValue(self.mask_step_size)
        step_spin.setSuffix(" µm²")
        size_grid.addWidget(step_spin)
        layout.addLayout(size_grid)

        # Preview of sizes
        preview_label = QLabel("")
        preview_label.setStyleSheet("color: palette(dark); font-size: 10px;")
        preview_label.setWordWrap(True)
        layout.addWidget(preview_label)

        def update_size_preview():
            mn = min_area_spin.value()
            mx = max_area_spin.value()
            st = step_spin.value()
            if mn > mx:
                preview_label.setText("Min must be <= Max")
                return
            sizes = list(range(mn, mx + 1, st))
            if sizes[-1] != mx:
                sizes.append(mx)
            preview_label.setText(f"Masks: {', '.join(str(s) for s in sizes)} µm²  ({len(sizes)} per cell)")

        min_area_spin.valueChanged.connect(lambda: update_size_preview())
        max_area_spin.valueChanged.connect(lambda: update_size_preview())
        step_spin.valueChanged.connect(lambda: update_size_preview())
        update_size_preview()

        layout.addSpacing(5)

        # Intensity Settings
        layout.addWidget(QLabel("<b>Intensity Floor</b>"))
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Min intensity:"))
        min_intensity_slider = QSlider(Qt.Horizontal)
        min_intensity_slider.setRange(0, 100)
        min_intensity_slider.setValue(self.min_intensity_percent)
        slider_layout.addWidget(min_intensity_slider)
        min_intensity_label = QLabel(f"{self.min_intensity_percent}%")
        min_intensity_slider.valueChanged.connect(
            lambda v: min_intensity_label.setText(f"{v}%")
        )
        slider_layout.addWidget(min_intensity_label)
        layout.addLayout(slider_layout)

        hint = QLabel("0% = no floor (grow freely)   |   Higher = only bright pixels")
        hint.setStyleSheet("color: palette(dark); font-size: 10px;")
        layout.addWidget(hint)

        layout.addSpacing(5)

        # Buttons
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)

        ok_btn = QPushButton("Redo Masks")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(dialog.accept)
        ok_btn.setStyleSheet(
            "QPushButton { border: 2px solid #4CAF50; font-weight: bold; padding: 5px; }")
        button_layout.addWidget(ok_btn)

        layout.addLayout(button_layout)
        dialog.setLayout(layout)
        dialog.setMinimumWidth(420)

        if dialog.exec_() != QDialog.Accepted:
            return

        # Read settings
        regen_min = min_area_spin.value()
        regen_max = max_area_spin.value()
        regen_step = step_spin.value()
        regen_intensity = min_intensity_slider.value()

        if regen_min > regen_max:
            QMessageBox.warning(self, "Warning", "Min area must be <= Max area.")
            return

        area_list = list(range(regen_min, regen_max + 1, regen_step))
        if area_list[-1] != regen_max:
            area_list.append(regen_max)

        try:
            pixel_size = float(self.pixel_size_input.text())
        except ValueError:
            pixel_size = 0.316

        # Delete old mask files from disk for this image
        if self.masks_dir and os.path.isdir(self.masks_dir):
            img_basename = os.path.splitext(img_name)[0]
            for f in os.listdir(self.masks_dir):
                if f.startswith(img_basename + "_") and f.endswith("_mask.tif"):
                    try:
                        os.remove(os.path.join(self.masks_dir, f))
                    except Exception:
                        pass

        # Remove old masks from all_masks_flat
        self.all_masks_flat = [flat for flat in self.all_masks_flat
                               if flat['image_name'] != img_name]

        # Clear existing masks for this image
        img_data['masks'] = []

        # Temporarily override intensity setting for this generation
        saved_intensity = self.min_intensity_percent
        saved_use_intensity = self.use_min_intensity
        self.min_intensity_percent = regen_intensity
        self.use_min_intensity = regen_intensity > 0

        # Generate new masks
        self.log(f"Redoing masks for {img_name}: {regen_min}-{regen_max} µm², "
                 f"step {regen_step}, intensity {regen_intensity}%")

        for soma_data in img_data['soma_outlines']:
            centroid = soma_data['centroid']
            soma_idx = soma_data['soma_idx']
            soma_id = soma_data['soma_id']
            soma_area_um2 = soma_data.get('soma_area_um2', 0)

            masks = self._create_annulus_masks(
                centroid, area_list, pixel_size, soma_idx, soma_id,
                img_data['processed'], img_name, soma_area_um2
            )
            img_data['masks'].extend(masks)

        # Restore global intensity settings
        self.min_intensity_percent = saved_intensity
        self.use_min_intensity = saved_use_intensity

        # Update status
        img_data['status'] = 'masks_generated'
        self._update_file_list_item(img_name)

        # Add new masks to all_masks_flat
        if img_data['selected']:
            for mask_data in img_data['masks']:
                self.all_masks_flat.append({
                    'image_name': img_name,
                    'mask_data': mask_data,
                    'processed_img': img_data['processed'],
                })

        total = len(img_data['masks'])
        self.log(f"Generated {total} new masks for {img_name}")
        self._auto_save()

        # If QA was active, jump to the first new mask for this image
        if self.mask_qa_active:
            for i, flat in enumerate(self.all_masks_flat):
                if flat['image_name'] == img_name and flat['mask_data']['approved'] is None:
                    self.mask_qa_idx = i
                    self._show_current_mask()
                    return
        else:
            self.batch_qa_btn.setEnabled(True)
            QMessageBox.information(self, "Done",
                f"Regenerated {total} masks for {os.path.splitext(img_name)[0]}.\n\nReady for QA.")

    def _create_annulus_masks(self, centroid, area_list_um2, pixel_size_um, soma_idx, soma_id, processed_img, img_name,
                              soma_area_um2):
        """Create nested cell masks using priority region growing from the centroid.

        Grows outward from the soma centroid, always adding the brightest
        neighboring pixel next. Each smaller mask is automatically a strict
        subset of the larger one because they share the same growth order.
        """
        import heapq

        masks = []
        cy, cx = int(centroid[0]), int(centroid[1])

        # Convert all target areas to pixels and find the largest
        sorted_areas = sorted(area_list_um2, reverse=True)
        largest_target_px = int(sorted_areas[0] / (pixel_size_um ** 2))

        # Size the ROI to guarantee enough room for the largest mask
        # Use radius of equivalent circle * 3 for safety (processes extend far)
        min_roi_radius = int(np.sqrt(largest_target_px / np.pi) * 3)
        roi_size = max(200, min_roi_radius)

        y_min = max(0, cy - roi_size)
        y_max = min(processed_img.shape[0], cy + roi_size)
        x_min = max(0, cx - roi_size)
        x_max = min(processed_img.shape[1], cx + roi_size)

        roi = processed_img[y_min:y_max, x_min:x_max].astype(np.float64)
        cy_roi, cx_roi = cy - y_min, cx - x_min
        h, w = roi.shape

        # Clamp centroid to ROI bounds
        cy_roi = max(0, min(h - 1, cy_roi))
        cx_roi = max(0, min(w - 1, cx_roi))

        # Priority region growing: grow from centroid, brightest neighbor first
        # Use a max-heap (negate intensity for min-heap)
        # The heap ordering ensures bright cell pixels are added first,
        # so smaller masks = bright core, larger masks extend outward naturally.
        visited = np.zeros((h, w), dtype=bool)
        growth_order = []  # list of (row, col) in the order pixels were added

        # Seed with the centroid
        heap = [(-roi[cy_roi, cx_roi], cy_roi, cx_roi)]
        visited[cy_roi, cx_roi] = True

        while heap and len(growth_order) < largest_target_px:
            neg_intensity, r, c = heapq.heappop(heap)

            growth_order.append((r, c))

            # Add 4-connected neighbors to the heap
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                    visited[nr, nc] = True
                    heapq.heappush(heap, (-roi[nr, nc], nr, nc))

        print(f"  {soma_id}: grew {len(growth_order)} pixels (target largest: {largest_target_px})")

        # Build masks for each target area from the growth order
        # Largest first (matches QA presentation order)
        for target_area_um2 in sorted_areas:
            target_px = int(target_area_um2 / (pixel_size_um ** 2))
            n_pixels = min(target_px, len(growth_order))

            mask_roi = np.zeros((h, w), dtype=np.uint8)
            for r, c in growth_order[:n_pixels]:
                mask_roi[r, c] = 1

            full_mask = np.zeros(processed_img.shape, dtype=np.uint8)
            full_mask[y_min:y_max, x_min:x_max] = mask_roi

            actual_area_um2 = n_pixels * (pixel_size_um ** 2)
            print(f"    {target_area_um2} um²: {n_pixels} px = {actual_area_um2:.1f} um² actual")

            masks.append({
                'image_name': img_name,
                'soma_idx': soma_idx,
                'soma_id': soma_id,
                'area_um2': target_area_um2,
                'mask': full_mask,
                'approved': None,
                'soma_area_um2': soma_area_um2
            })

        return masks

    def start_batch_qa(self):
        # Flatten all masks from all images
        self.all_masks_flat = []
        for img_name, img_data in self.images.items():
            if not img_data['selected']:
                continue
            for mask_data in img_data['masks']:
                self.all_masks_flat.append({
                    'image_name': img_name,
                    'mask_data': mask_data,
                    'processed_img': img_data['processed']
                })

        if not self.all_masks_flat:
            QMessageBox.warning(self, "Warning", "No masks to QA")
            return

        self.mask_qa_active = True
        self.mask_qa_idx = 0
        self.last_qa_decisions = []

        self.approve_mask_btn.setEnabled(True)
        self.reject_mask_btn.setEnabled(True)
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.done_btn.setEnabled(False)
        self.undo_qa_btn.setEnabled(True)
        self.regen_masks_btn.setVisible(True)

        self._show_current_mask()
        self.tabs.setCurrentIndex(3)

        self.log("=" * 50)
        self.log("🎯 BATCH MASK QA MODE")
        self.log(f"Total masks to review: {len(self.all_masks_flat)}")
        self.log("Keyboard: A=Approve, R=Reject, ←→=Navigate, Space=Approve&Next")
        self.log("=" * 50)

    def _show_current_mask(self):
        if not self.all_masks_flat or self.mask_qa_idx >= len(self.all_masks_flat):
            return

        flat_data = self.all_masks_flat[self.mask_qa_idx]
        mask_data = flat_data['mask_data']
        processed_img = flat_data['processed_img']
        img_name = flat_data['image_name']

        # Display in color or grayscale based on toggle
        img_data = self.images.get(img_name, {})
        if self.show_color_view and 'color_image' in img_data:
            proc_color = self._build_processed_color_image(img_data)
            if proc_color is not None:
                adjusted = self._apply_display_adjustments_color(proc_color)
            else:
                adjusted = self._apply_display_adjustments_color(img_data['color_image'])
            pixmap = self._array_to_pixmap_color(adjusted)
        else:
            adjusted = self._apply_display_adjustments(processed_img)
            pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
        self.mask_label.set_image(pixmap, mask_overlay=mask_data['mask'])

        # Auto-zoom to mask center
        mask_coords = np.argwhere(mask_data['mask'] > 0)
        if len(mask_coords) > 0:
            center_row = float(np.mean(mask_coords[:, 0]))
            center_col = float(np.mean(mask_coords[:, 1]))
            self.mask_label.zoom_to_point(center_row, center_col, zoom_level=3.0)

        status = mask_data.get('approved')
        status_text = "✓ Approved" if status is True else "✗ Rejected" if status is False else "⏳ Not reviewed"

        self.nav_status_label.setText(
            f"Mask {self.mask_qa_idx + 1}/{len(self.all_masks_flat)} | "
            f"{img_name} | {mask_data['soma_id']} | "
            f"Area: {mask_data['area_um2']} µm² | {status_text}"
        )

    def approve_current_mask(self):
        if not self.mask_qa_active or self.mask_qa_idx >= len(self.all_masks_flat):
            return

        flat_data = self.all_masks_flat[self.mask_qa_idx]
        mask_data = flat_data['mask_data']
        mask_data['approved'] = True

        current_soma_id = mask_data['soma_id']
        current_area = mask_data['area_um2']
        current_img = flat_data['image_name']

        self.log(f"✅ APPROVED | {current_img} | {current_soma_id} | Area: {current_area} µm²")

        # Record decision for undo
        self.last_qa_decisions.append({'flat_data': flat_data, 'was_approved': True})

        # Export mask immediately upon approval
        self._export_approved_mask(flat_data)

        # Auto-approve ALL smaller masks from the SAME soma in the SAME image
        auto_approved = []
        for i, other_flat in enumerate(self.all_masks_flat):
            other_mask = other_flat['mask_data']
            other_img = other_flat['image_name']

            # Must be: same image, same soma, smaller area, not yet reviewed
            if (other_img == current_img and
                    other_mask['soma_id'] == current_soma_id and
                    other_mask['area_um2'] < current_area and
                    other_mask['approved'] is None):
                other_mask['approved'] = True
                auto_approved.append((i + 1, other_mask['area_um2']))
                # Record auto-approval for undo
                self.last_qa_decisions.append({'flat_data': other_flat, 'was_approved': True})
                # Export auto-approved masks too
                self._export_approved_mask(other_flat)

        if auto_approved:
            self.log(f"   ⚡ Auto-approved {len(auto_approved)} smaller masks for {current_soma_id}:")
            for mask_num, area in auto_approved:
                self.log(f"      Mask #{mask_num} ({area} µm²)")

        # Auto-save every 5 QA decisions
        if len(self.last_qa_decisions) % 5 == 0:
            self._auto_save()

        # Move to next unreviewed mask
        self._advance_to_next_unreviewed()

    def _export_approved_mask(self, flat_data):
        """Export a single approved mask to TIFF file"""
        if not self.masks_dir:
            self.log("   ⚠️ No masks directory set - cannot export")
            return

        mask_data = flat_data['mask_data']
        img_name = flat_data['image_name']
        soma_id = mask_data['soma_id']
        area_um2 = mask_data.get('area_um2', 0)

        # Create unique filename with area to distinguish different masks for same soma
        img_basename = os.path.splitext(img_name)[0]
        mask_filename = f"{img_basename}_{soma_id}_area{int(area_um2)}_mask.tif"
        mask_path = os.path.join(self.masks_dir, mask_filename)

        # Get the mask array
        mask = mask_data.get('mask')
        if mask is None:
            self.log(f"   ⚠️ No mask data for {soma_id} - skipping export")
            return

        # Debug: Show mask properties
        self.log(
            f"   🔍 Mask {soma_id}: shape={mask.shape}, dtype={mask.dtype}, min={np.min(mask)}, max={np.max(mask)}, nonzero={np.count_nonzero(mask)}")

        # Check if mask has any data
        if not np.any(mask):
            self.log(f"   ⚠️ Mask {soma_id} is empty (all zeros) - skipping export")
            return

        # Convert mask to 8-bit (0 or 255)
        # Handle both boolean and integer masks
        if mask.dtype == bool:
            mask_8bit = mask.astype(np.uint8) * 255
        else:
            mask_8bit = (mask > 0).astype(np.uint8) * 255

        # Double-check the converted mask has values
        pixels_saved = np.count_nonzero(mask_8bit)
        if pixels_saved == 0:
            self.log(f"   ⚠️ Mask {soma_id} became empty after conversion")
            self.log(f"      Unique values in original: {np.unique(mask)}")
            return

        # Get pixel size
        try:
            pixel_size = float(self.pixel_size_input.text())
        except:
            pixel_size = 0.316  # default

        # Save as TIFF with calibration
        try:
            tifffile.imwrite(
                mask_path,
                mask_8bit,
                resolution=(1.0 / pixel_size, 1.0 / pixel_size),
                metadata={'unit': 'um'}
            )

            self.log(f"   💾 Exported: {mask_filename} ({pixels_saved} pixels)")

        except Exception as e:
            self.log(f"   ❌ Failed to export {mask_filename}: {e}")

    def _export_soma_outline(self, img_name, soma_id, mask, pixel_size, soma_area_um2):
        """Export a soma outline to TIFF file in the somas directory"""
        if not hasattr(self, 'somas_dir') or not self.somas_dir:
            self.log("   ⚠️ No somas directory set - cannot export")
            return

        # Create unique filename matching the mask naming convention
        img_basename = os.path.splitext(img_name)[0]
        soma_filename = f"{img_basename}_{soma_id}_soma.tif"
        soma_path = os.path.join(self.somas_dir, soma_filename)

        # Check if mask has any data
        if not np.any(mask):
            self.log(f"   ⚠️ Soma outline {soma_id} is empty - skipping export")
            return

        # Convert mask to 8-bit (0 or 255)
        if mask.dtype == bool:
            mask_8bit = mask.astype(np.uint8) * 255
        else:
            mask_8bit = (mask > 0).astype(np.uint8) * 255

        # Double-check the converted mask has values
        pixels_saved = np.count_nonzero(mask_8bit)
        if pixels_saved == 0:
            self.log(f"   ⚠️ Soma outline {soma_id} became empty after conversion")
            return

        # Save as TIFF with calibration
        try:
            tifffile.imwrite(
                soma_path,
                mask_8bit,
                resolution=(1.0 / pixel_size, 1.0 / pixel_size),
                metadata={'unit': 'um'}
            )

            self.log(f"   💾 Saved soma: {soma_filename} ({pixels_saved} pixels, {soma_area_um2:.1f} µm²)")

        except Exception as e:
            self.log(f"   ❌ Failed to save {soma_filename}: {e}")

    def _advance_to_next_unreviewed(self):
        """Skip to next unreviewed mask, or complete QA if all done"""
        original_idx = self.mask_qa_idx

        # Search forward for next unreviewed mask
        for idx in range(self.mask_qa_idx + 1, len(self.all_masks_flat)):
            if self.all_masks_flat[idx]['mask_data']['approved'] is None:
                self.mask_qa_idx = idx
                self._show_current_mask()
                return

        # If we get here, check if all masks are reviewed
        all_reviewed = all(flat['mask_data']['approved'] is not None for flat in self.all_masks_flat)

        if all_reviewed:
            self._check_qa_complete()
        else:
            # There are unreviewed masks before current position, stay at last mask
            self.mask_qa_idx = len(self.all_masks_flat) - 1
            self._show_current_mask()
            self.log("⚠️  Reached end. Use Previous to review any remaining masks.")

    def reject_current_mask(self):
        if not self.mask_qa_active or self.mask_qa_idx >= len(self.all_masks_flat):
            return

        flat_data = self.all_masks_flat[self.mask_qa_idx]
        mask_data = flat_data['mask_data']
        mask_data['approved'] = False

        # Record decision for undo
        self.last_qa_decisions.append({'flat_data': flat_data, 'was_approved': False})

        self.log(f"✗ Rejected: {mask_data['soma_id']} ({mask_data['area_um2']} µm²)")

        if self.mask_qa_idx < len(self.all_masks_flat) - 1:
            self.mask_qa_idx += 1
            self._show_current_mask()
        else:
            self._check_qa_complete()

    def next_mask(self):
        if not self.mask_qa_active:
            return
        if self.mask_qa_idx < len(self.all_masks_flat) - 1:
            self.mask_qa_idx += 1
            self._show_current_mask()

    def prev_mask(self):
        if not self.mask_qa_active:
            return
        if self.mask_qa_idx > 0:
            self.mask_qa_idx -= 1
            self._show_current_mask()

    def _check_qa_complete(self):
        all_reviewed = all(flat['mask_data']['approved'] is not None for flat in self.all_masks_flat)

        if all_reviewed:
            self.mask_qa_active = False
            self.approve_mask_btn.setEnabled(False)
            self.reject_mask_btn.setEnabled(False)
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.regen_masks_btn.setVisible(False)

            # Update image statuses
            for img_name, img_data in self.images.items():
                if img_data['selected'] and img_data['status'] == 'masks_generated':
                    img_data['status'] = 'qa_complete'
                    self._update_file_list_item(img_name)

            self.batch_calculate_btn.setEnabled(True)
            self.undo_qa_btn.setEnabled(True)

            approved_count = sum(1 for flat in self.all_masks_flat if flat['mask_data']['approved'])
            rejected_count = len(self.all_masks_flat) - approved_count

            self.log("=" * 50)
            self.log(f"✓ QA Complete!")
            self.log(f"Approved: {approved_count}, Rejected: {rejected_count}")
            self.log("=" * 50)
            self._auto_save()

            QMessageBox.information(
                self, "QA Complete",
                f"QA Complete!\n\nApproved: {approved_count}\nRejected: {rejected_count}"
            )

    def undo_last_qa(self):
        """Reset only the last QA session's approvals, delete its exported files, and revert statuses."""
        if not hasattr(self, 'last_qa_decisions') or not self.last_qa_decisions:
            QMessageBox.warning(self, "Nothing to Undo", "No recent QA decisions to undo.")
            return

        n_decisions = len(self.last_qa_decisions)
        n_approved = sum(1 for d in self.last_qa_decisions if d['was_approved'])
        n_rejected = n_decisions - n_approved

        reply = QMessageBox.question(
            self, "Undo Last QA",
            f"This will undo the last QA session:\n\n"
            f"  {n_approved} approved masks → reset to unreviewed\n"
            f"  {n_rejected} rejected masks → reset to unreviewed\n"
            f"  Exported mask files will be deleted\n\n"
            f"Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # Delete exported mask TIFFs and reset approval states
        deleted_count = 0
        reset_count = 0
        affected_images = set()

        for decision in self.last_qa_decisions:
            flat_data = decision['flat_data']
            mask_data = flat_data['mask_data']
            affected_images.add(flat_data['image_name'])

            # Delete exported file if it was approved
            if decision['was_approved'] and self.masks_dir and os.path.isdir(self.masks_dir):
                img_basename = os.path.splitext(flat_data['image_name'])[0]
                soma_id = mask_data['soma_id']
                area_um2 = mask_data.get('area_um2', 0)
                mask_filename = f"{img_basename}_{soma_id}_area{int(area_um2)}_mask.tif"
                mask_path = os.path.join(self.masks_dir, mask_filename)
                if os.path.exists(mask_path):
                    os.remove(mask_path)
                    deleted_count += 1

            # Reset approval state
            mask_data['approved'] = None
            reset_count += 1

        # Revert affected image statuses back to masks_generated
        for img_name in affected_images:
            if img_name in self.images:
                img_data = self.images[img_name]
                if img_data['status'] in ('qa_complete', 'analyzed'):
                    img_data['status'] = 'masks_generated'
                    self._update_file_list_item(img_name)

        # Clear the undo history
        self.last_qa_decisions = []

        # Update button states
        self.batch_qa_btn.setEnabled(True)
        self.batch_calculate_btn.setEnabled(False)
        self.undo_qa_btn.setEnabled(False)
        self.mask_qa_active = False
        self.approve_mask_btn.setEnabled(False)
        self.reject_mask_btn.setEnabled(False)

        self.log("=" * 50)
        self.log(f"↩ Last QA undone: {reset_count} masks reset, {deleted_count} exported files removed")
        self.log("You can now re-run QA All Masks.")
        self.log("=" * 50)

        QMessageBox.information(
            self, "QA Undone",
            f"Last QA session has been reset.\n\n"
            f"Masks reset: {reset_count}\n"
            f"Exported files deleted: {deleted_count}\n\n"
            f"You can now re-run QA."
        )

    def batch_calculate_morphology(self):
        try:
            pixel_size = float(self.pixel_size_input.text())

            # No ImageJ required for simple characteristics
            # Complex analysis (Sholl, Skeleton) will be done separately in ImageJ

            self.log(f"all_masks_flat has {len(self.all_masks_flat)} entries")
            approved_masks = [flat for flat in self.all_masks_flat if flat['mask_data']['approved']]
            total = len(approved_masks)

            if total == 0:
                self.log(f"DEBUG: all_masks_flat count = {len(self.all_masks_flat)}")
                for i, flat in enumerate(self.all_masks_flat[:5]):
                    self.log(f"  mask {i}: approved={flat['mask_data'].get('approved')}, "
                             f"soma_id={flat['mask_data'].get('soma_id')}, "
                             f"area={flat['mask_data'].get('area_um2')}")
                # Also check images directly
                for iname, idata in self.images.items():
                    self.log(f"  image '{iname}': status={idata['status']}, "
                             f"selected={idata['selected']}, masks={len(idata['masks'])}")
                QMessageBox.warning(self, "Warning", "No approved masks to analyze.\n\n"
                    f"Check the log for details.\n"
                    f"all_masks_flat: {len(self.all_masks_flat)} entries")
                return

            self.log("=" * 50)
            self.log("Calculating simple characteristics (Python)...")
            self.log("Note: Sholl & Skeleton analysis will be done in ImageJ")
            self.log("=" * 50)

            self.progress_bar.setVisible(True)
            self.progress_status_label.setVisible(True)
            self.progress_bar.setValue(0)
            self.start_timer()  # Start the timer

            # Disable the calculate button during processing
            self.batch_calculate_btn.setEnabled(False)

            # Create and start worker thread (no ImageJ needed)
            self.morph_thread = MorphologyCalculationThread(
                approved_masks, pixel_size, use_imagej=False, images=self.images
            )

            # Connect signals
            self.morph_thread.progress.connect(self._on_morph_progress)
            self.morph_thread.finished.connect(self._on_morph_finished)
            self.morph_thread.error_occurred.connect(self._on_morph_error)

            # Start the thread
            self.morph_thread.start()

        except Exception as e:
            self.log(f"Error starting morphology calculation: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start calculation: {e}")
            self.progress_bar.setVisible(False)
            self.progress_status_label.setVisible(False)
            self.stop_timer()

    def _on_morph_progress(self, percentage, status):
        """Update progress bar and status during morphology calculation"""
        self.progress_bar.setValue(percentage)
        self.progress_status_label.setText(status)

    def _on_morph_finished(self, all_results):
        """Handle completion of morphology calculations"""
        self.stop_timer()
        self.progress_bar.setVisible(False)
        self.progress_status_label.setVisible(False)
        self.batch_calculate_btn.setEnabled(True)
        self.timer_label.setVisible(False)

        # Calculate colocalization if in colocalization mode
        if self.colocalization_mode:
            self.log("=" * 50)
            self.log("🎨 Calculating colocalization metrics...")
            coloc_count = 0
            for i, result in enumerate(all_results):
                img_name = result.get('image_name', '')
                # Find the matching mask data
                for flat_data in self.all_masks_flat:
                    if (flat_data['mask_data'].get('approved') and
                        flat_data['mask_data'].get('soma_id') == result.get('soma_id') and
                        os.path.splitext(flat_data['image_name'])[0] == img_name):
                        mask = flat_data['mask_data'].get('mask')
                        full_img_name = flat_data['image_name']
                        if mask is not None:
                            coloc_results = self.calculate_colocalization(mask, full_img_name)
                            result.update(coloc_results)
                            if coloc_results.get('coloc_status') == 'ok':
                                coloc_count += 1
                        break
            self.log(f"✓ Colocalization calculated for {coloc_count}/{len(all_results)} cells")

        # Collect metadata for images
        self.log("=" * 50)
        self.log("Checking metadata... (any missing info will be requested)")
        if not self.collect_metadata_for_images():
            self.log("⚠️ Metadata entry cancelled - analysis aborted")
            return

        self._save_batch_results(all_results)

        # Update statuses
        for img_name, img_data in self.images.items():
            if img_data['selected'] and img_data['status'] == 'qa_complete':
                img_data['status'] = 'analyzed'
                self._update_file_list_item(img_name)

        self.log("=" * 50)
        self.log(f"✓ Simple characteristics calculated for {len(all_results)} cells")
        if self.colocalization_mode:
            coloc_ok = sum(1 for r in all_results if r.get('coloc_status') == 'ok')
            self.log(f"✓ Colocalization metrics calculated for {coloc_ok} cells")
        self.log(f"✓ Masks exported to: {self.masks_dir}")
        self.log("")
        self.log("NEXT STEP: Run ImageJ batch analysis")
        self.log("  1. Open Fiji/ImageJ")
        self.log("  2. Run imagej_batch_analysis.ijm macro")
        self.log("  3. Select the masks folder when prompted")
        self.log("=" * 50)
        self._auto_save()

        # Build success message
        success_msg = f"Simple characteristics calculated for {len(all_results)} cells!\n\n"
        if self.colocalization_mode:
            coloc_ok = sum(1 for r in all_results if r.get('coloc_status') == 'ok')
            success_msg += f"Colocalization metrics: {coloc_ok} cells\n\n"
        success_msg += f"Masks exported to:\n{self.masks_dir}\n\n"
        success_msg += "NEXT STEP:\nRun the ImageJ macro (imagej_batch_analysis.ijm)\nto calculate Sholl & Skeleton parameters."

        QMessageBox.information(self, "Success", success_msg)

    def _on_morph_error(self, error_msg):
        """Handle errors during morphology calculation"""
        self.stop_timer()
        self.progress_bar.setVisible(False)
        self.progress_status_label.setVisible(False)
        self.timer_label.setVisible(False)
        self.batch_calculate_btn.setEnabled(True)

        self.log(f"ERROR: {error_msg}")
        QMessageBox.critical(self, "Error", f"Morphology calculation failed:\n{error_msg}")

    def open_image_labeling(self):
        """Open the image labeling interface"""
        selected_images = [name for name, data in self.images.items() if data['selected']]

        if not selected_images:
            QMessageBox.warning(self, "No Images Selected",
                                "Please select at least one image first.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Image Labeling")
        dialog.setMinimumWidth(800)
        dialog.setMinimumHeight(600)

        layout = QVBoxLayout(dialog)

        # Instructions
        info_label = QLabel(
            "<b>Enter Animal ID and Treatment for each selected image:</b><br>"
            "You can edit these values anytime before running 'Calculate All Parameters'"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Create table-like layout
        from PyQt5.QtWidgets import QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView

        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Image Name", "Animal ID", "Treatment"])
        table.setRowCount(len(selected_images))

        # Make first column read-only, stretch to fit
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Interactive)
        header.setSectionResizeMode(2, QHeaderView.Interactive)
        table.setColumnWidth(1, 200)
        table.setColumnWidth(2, 200)

        # Populate table
        for row, img_name in enumerate(sorted(selected_images)):
            # Image name (read-only)
            name_item = QTableWidgetItem(img_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 0, name_item)

            # Animal ID (editable)
            animal_item = QTableWidgetItem(self.images[img_name].get('animal_id', ''))
            animal_item.setPlaceholderText = "e.g., Mouse_001"
            table.setItem(row, 1, animal_item)

            # Treatment (editable)
            treatment_item = QTableWidgetItem(self.images[img_name].get('treatment', ''))
            treatment_item.setPlaceholderText = "e.g., Control"
            table.setItem(row, 2, treatment_item)

        layout.addWidget(table)

        # Buttons
        button_layout = QHBoxLayout()

        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        cancel_btn = QPushButton("Cancel")

        def save_labels():
            # Save all the labels back to the images
            for row in range(table.rowCount()):
                img_name = table.item(row, 0).text()
                animal_id = table.item(row, 1).text().strip()
                treatment = table.item(row, 2).text().strip()

                self.images[img_name]['animal_id'] = animal_id
                self.images[img_name]['treatment'] = treatment

            self.log(f"✓ Labels saved for {table.rowCount()} images")
            dialog.accept()

        save_btn.clicked.connect(save_labels)
        cancel_btn.clicked.connect(dialog.reject)

        button_layout.addStretch()
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

        dialog.exec_()

    def collect_metadata_for_images(self):
        """Collect AnimalID and Treatment for each processed image"""
        from PyQt5.QtWidgets import QScrollArea

        # Only show dialog for images that don't have metadata yet
        selected_images = [
            name for name, data in self.images.items()
            if data['selected'] and (not data.get('animal_id') or not data.get('treatment'))
        ]

        if not selected_images:
            self.log("✓ All images already have metadata")
            return True

        dialog = QDialog(self)
        dialog.setWindowTitle("Enter Image Metadata")
        dialog.setMinimumWidth(600)
        dialog.setMinimumHeight(400)

        layout = QVBoxLayout(dialog)

        info_label = QLabel("<b>Enter Animal ID and Treatment for each image:</b>")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Scrollable area for multiple images
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Create input fields for each image
        image_inputs = {}
        for img_name in selected_images:
            img_group = QGroupBox(img_name)
            img_layout = QFormLayout()

            animal_id_input = QLineEdit()
            animal_id_input.setPlaceholderText("e.g., Mouse_001, Rat_A1")
            animal_id_input.setText(self.images[img_name].get('animal_id', ''))

            treatment_input = QLineEdit()
            treatment_input.setPlaceholderText("e.g., Control, Treatment_A, Saline")
            treatment_input.setText(self.images[img_name].get('treatment', ''))

            img_layout.addRow("Animal ID:", animal_id_input)
            img_layout.addRow("Treatment:", treatment_input)

            img_group.setLayout(img_layout)
            scroll_layout.addWidget(img_group)

            image_inputs[img_name] = {
                'animal_id': animal_id_input,
                'treatment': treatment_input
            }

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # Add buttons
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        cancel_btn = QPushButton("Cancel")

        def save_metadata():
            for img_name, inputs in image_inputs.items():
                self.images[img_name]['animal_id'] = inputs['animal_id'].text().strip()
                self.images[img_name]['treatment'] = inputs['treatment'].text().strip()
            dialog.accept()

        ok_btn.clicked.connect(save_metadata)
        cancel_btn.clicked.connect(dialog.reject)

        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

        return dialog.exec_() == QDialog.Accepted

    def _save_batch_results(self, results):
        if not self.output_dir or not results:
            return

        import csv

        # Add animal_id and treatment to each result
        for result in results:
            img_name_base = result['image_name']
            # Find the matching image data
            for full_name, img_data in self.images.items():
                if os.path.splitext(full_name)[0] == img_name_base:
                    result['animal_id'] = img_data.get('animal_id', '')
                    result['treatment'] = img_data.get('treatment', '')
                    break
            else:
                # If not found, set empty values
                result['animal_id'] = ''
                result['treatment'] = ''

        # Combined results file
        combined_path = os.path.join(self.output_dir, "combined_morphology_results.csv")

        keys = list(results[0].keys())
        # Remove these to reorder them
        for key in ['soma_id', 'image_name', 'animal_id', 'treatment', 'soma_idx']:
            if key in keys:
                keys.remove(key)

        # Separate colocalization keys from morphology keys for better organization
        coloc_keys = ['coloc_status', 'coloc_ch1', 'coloc_ch2', 'pearson_r',
                      'n_mask_pixels', 'n_ch1_signal', 'n_ch2_signal', 'n_coloc_pixels',
                      'ch1_coloc_percent', 'ch2_coloc_percent',
                      'ch1_has_signal', 'ch2_has_signal',
                      'ch1_threshold', 'ch1_min', 'ch1_max',
                      'ch2_threshold', 'ch2_min', 'ch2_max']
        morph_keys = [k for k in keys if k not in coloc_keys]
        coloc_present = [k for k in coloc_keys if k in keys]

        # Put them in the desired order: identifiers, morphology, colocalization
        ordered_keys = ['image_name', 'animal_id', 'treatment', 'soma_id', 'soma_idx'] + sorted(morph_keys) + coloc_present

        with open(combined_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_keys)
            writer.writeheader()
            writer.writerows(results)

        self.log(f"Combined results saved to: {combined_path}")

        # Export metadata CSV for ImageJ analysis matching
        metadata_path = os.path.join(self.masks_dir, "mask_metadata.csv")
        with open(metadata_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['mask_filename', 'soma_filename', 'image_name', 'soma_id', 'soma_idx',
                             'soma_x', 'soma_y', 'soma_area_um2', 'cell_area_um2',
                             'pixel_size_um', 'perimeter', 'eccentricity', 'roundness',
                             'cell_spread', 'polarity_index', 'principal_angle',
                             'major_axis_um', 'minor_axis_um', 'animal_id', 'treatment'])

            for result in results:
                img_name = result['image_name']
                soma_id = result['soma_id']
                soma_idx = result.get('soma_idx', 0)

                # Get soma position
                soma_x, soma_y = 0, 0
                for full_name, img_data in self.images.items():
                    if os.path.splitext(full_name)[0] == img_name:
                        if soma_idx < len(img_data['somas']):
                            soma_x, soma_y = img_data['somas'][soma_idx]
                        break

                mask_filename = f"{img_name}_{soma_id}_mask.tif"
                soma_filename = f"{img_name}_{soma_id}_soma.tif"

                writer.writerow([
                    mask_filename,
                    soma_filename,
                    img_name,
                    soma_id,
                    soma_idx,
                    f"{soma_x:.2f}",
                    f"{soma_y:.2f}",
                    result.get('soma_area', 0),
                    result.get('area_um2', 0),
                    result.get('mask_area', 0),
                    result.get('perimeter', 0),
                    result.get('eccentricity', 0),
                    result.get('roundness', 0),
                    result.get('cell_spread', 0),
                    result.get('polarity_index', 0),
                    result.get('principal_angle', 0),
                    result.get('major_axis_um', 0),
                    result.get('minor_axis_um', 0),
                    result.get('animal_id', ''),
                    result.get('treatment', '')
                ])

        self.log(f"Metadata saved to: {metadata_path}")

        # Per-image results files
        by_image = {}
        for result in results:
            img_name = result['image_name']
            if img_name not in by_image:
                by_image[img_name] = []
            by_image[img_name].append(result)

        for img_name, img_results in by_image.items():
            img_path = os.path.join(self.output_dir, f"{img_name}_morphology_results.csv")
            with open(img_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=ordered_keys)
                writer.writeheader()
                writer.writerows(img_results)

        self.log(f"Per-image results saved for {len(by_image)} images")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MicrogliaAnalysisGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
