# -*- coding: utf-8 -*-
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
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QBrush, QKeySequence, QIcon
from PyQt5.QtWidgets import QShortcut
from PIL import Image
import tifffile
from skimage import restoration, color, measure
from scipy import ndimage, stats
from matplotlib.path import Path as mplPath
import cv2
import glob
import json
import csv
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


# --- Global exception hook: prevent PyQt5 from silently swallowing crashes ---
def _global_exception_hook(exc_type, exc_value, exc_tb):
    import traceback
    traceback.print_exception(exc_type, exc_value, exc_tb)
    try:
        from PyQt5.QtWidgets import QMessageBox
        msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        QMessageBox.critical(None, "Unhandled Error", msg[:2000])
    except Exception:
        pass

sys.excepthook = _global_exception_hook


# --- 3D Z-stack functions (imported from 3DMicroglia.py) ---
try:
    import importlib.util as _ilu
    _3d_script = os.path.join(
        getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__))),
        "3DMicroglia.py",
    )
    _spec3d = _ilu.spec_from_file_location("ThreeDMicroglia", _3d_script)
    _mod3d = _ilu.module_from_spec(_spec3d)
    _spec3d.loader.exec_module(_mod3d)
    # Pull standalone 3D functions into module scope
    load_zstack = _mod3d.load_zstack
    ensure_grayscale_3d = _mod3d.ensure_grayscale_3d
    extract_channel_3d = _mod3d.extract_channel_3d
    preprocess_zstack = _mod3d.preprocess_zstack
    detect_soma_3d = _mod3d.detect_soma_3d
    create_spherical_annulus_masks = _mod3d.create_spherical_annulus_masks
    create_competitive_masks_3d = _mod3d.create_competitive_masks_3d
    Morphology3DCalculator = _mod3d.Morphology3DCalculator
    skeletonize_3d_mask = _mod3d.skeletonize_3d_mask
    sholl_analysis_3d = _mod3d.sholl_analysis_3d
    fractal_dimension_3d = _mod3d.fractal_dimension_3d
    export_mask_3d = _mod3d.export_mask_3d
    _HAS_3D = True
except Exception:
    _HAS_3D = False


def safe_tiff_read(filepath):
    """Read a TIFF via tifffile, falling back to PIL if imagecodecs is missing."""
    try:
        return tifffile.imread(filepath)
    except Exception:
        img = Image.open(filepath)
        # For multi-page TIFFs (Z-stacks), read all frames
        frames = []
        try:
            while True:
                frames.append(np.array(img))
                img.seek(img.tell() + 1)
        except EOFError:
            pass
        if len(frames) == 1:
            return frames[0]
        return np.stack(frames, axis=0)


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
# STANDALONE FUNCTIONS FOR MULTIPROCESSING
# (must be module-level to be picklable by ProcessPoolExecutor)
# ============================================================================

def _enforce_mask_subset_invariant(masks):
    """Ensure smaller masks are strict subsets of larger masks for the same soma.

    After region-growing, the growth_order prefix approach should guarantee
    this, but rounding and edge cases can occasionally violate it.
    Walks from the largest mask to the smallest and intersects each smaller
    mask with its next-larger sibling so that every pixel in the small mask
    is also present in the large mask.

    Modifies the mask arrays **in place** and returns the number of pixels
    that were removed to enforce the invariant.
    """
    if len(masks) < 2:
        return 0

    # Sort masks by descending actual pixel count (largest first)
    indexed = [(i, np.count_nonzero(m['mask'])) for i, m in enumerate(masks)
               if m.get('mask') is not None and not m.get('duplicate', False)]
    indexed.sort(key=lambda t: t[1], reverse=True)

    total_removed = 0
    for pos in range(1, len(indexed)):
        larger_idx = indexed[pos - 1][0]
        smaller_idx = indexed[pos][0]
        larger_mask = masks[larger_idx]['mask']
        smaller_mask = masks[smaller_idx]['mask']
        if larger_mask is None or smaller_mask is None:
            continue

        # Pixels in smaller but NOT in larger => violation
        violation = (smaller_mask > 0) & (larger_mask == 0)
        n_bad = int(np.count_nonzero(violation))
        if n_bad > 0:
            smaller_mask[violation] = 0
            masks[smaller_idx]['mask'] = smaller_mask
            total_removed += n_bad
            print(f"    ⚠️ Subset fix: removed {n_bad} px from "
                  f"{masks[smaller_idx].get('area_um2', '?')} µm² mask "
                  f"(not in {masks[larger_idx].get('area_um2', '?')} µm² mask)")

    return total_removed


def _grow_masks_for_soma(args):
    """Standalone region-growing mask generation for a single soma.

    Extracted from _create_annulus_masks so it can run in a subprocess.
    Returns a list of mask dicts (with 'mask' as None — only growth_order
    and metadata are returned to keep IPC lightweight).
    """
    import heapq

    (centroid, area_list_um2, pixel_size_um, soma_idx, soma_id,
     processed_img_shape, roi_data, roi_bounds,
     soma_area_um2, soma_outline_roi,
     territory_roi_data, my_territory_label,
     use_circular_constraint, circular_buffer_um2,
     use_min_intensity, min_intensity_percent, img_name) = args

    y_min, y_max, x_min, x_max = roi_bounds
    roi = roi_data  # already float64
    h, w = roi.shape
    cy, cx = int(centroid[0]), int(centroid[1])
    cy_roi, cx_roi = cy - y_min, cx - x_min
    cy_roi = max(0, min(h - 1, cy_roi))
    cx_roi = max(0, min(w - 1, cx_roi))

    sorted_areas = sorted(area_list_um2, reverse=True)
    largest_target_px = int(sorted_areas[0] / (pixel_size_um ** 2))

    # Circular constraint
    max_radius_px_sq = None
    if use_circular_constraint:
        constraint_area_um2 = sorted_areas[0] + circular_buffer_um2
        constraint_area_px = constraint_area_um2 / (pixel_size_um ** 2)
        max_radius_px = np.sqrt(constraint_area_px / np.pi)
        max_radius_px_sq = max_radius_px ** 2

    # Intensity floor
    intensity_floor = 0.0
    if use_min_intensity and min_intensity_percent > 0:
        roi_max = roi.max()
        if roi_max > 0:
            intensity_floor = roi_max * (min_intensity_percent / 100.0)

    # Territory constraint
    territory_roi = territory_roi_data
    my_label = my_territory_label

    def _in_territory(r, c):
        if territory_roi is None:
            return True
        return territory_roi[r, c] == my_label or territory_roi[r, c] <= 0

    def _in_circle(r, c):
        if max_radius_px_sq is None:
            return True
        dy = r - cy_roi
        dx = c - cx_roi
        return (dy * dy + dx * dx) <= max_radius_px_sq

    visited = np.zeros((h, w), dtype=bool)
    growth_order = []
    heap = []

    soma_seed_count = 0
    if soma_outline_roi is not None:
        soma_ys, soma_xs = np.where(soma_outline_roi > 0)
        for sr, sc in zip(soma_ys, soma_xs):
            if not visited[sr, sc]:
                visited[sr, sc] = True
                growth_order.append((sr, sc))
                soma_seed_count += 1
        for sr, sc in zip(soma_ys, soma_xs):
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = sr + dr, sc + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                    if roi[nr, nc] >= intensity_floor and _in_territory(nr, nc) and _in_circle(nr, nc):
                        visited[nr, nc] = True
                        heapq.heappush(heap, (-roi[nr, nc], nr, nc))

    if soma_seed_count == 0:
        visited[cy_roi, cx_roi] = True
        growth_order.append((cy_roi, cx_roi))
        soma_seed_count = 1
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cy_roi + dr, cx_roi + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                if roi[nr, nc] >= intensity_floor and _in_territory(nr, nc) and _in_circle(nr, nc):
                    visited[nr, nc] = True
                    heapq.heappush(heap, (-roi[nr, nc], nr, nc))

    while heap and len(growth_order) < largest_target_px:
        neg_intensity, r, c = heapq.heappop(heap)
        growth_order.append((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                if roi[nr, nc] >= intensity_floor and _in_territory(nr, nc) and _in_circle(nr, nc):
                    visited[nr, nc] = True
                    heapq.heappush(heap, (-roi[nr, nc], nr, nc))

    # Build masks for each target area
    soma_area_px = soma_seed_count
    masks = []
    mask_pixel_counts = []
    for target_area_um2 in sorted_areas:
        target_px = int(target_area_um2 / (pixel_size_um ** 2))

        # Per-step circular constraint: ring grows with each target
        if use_circular_constraint:
            step_constraint_um2 = target_area_um2 + circular_buffer_um2
            step_constraint_px = step_constraint_um2 / (pixel_size_um ** 2)
            step_radius_sq = step_constraint_px / np.pi
            step_order = [(r, c) for r, c in growth_order
                          if (r - cy_roi) ** 2 + (c - cx_roi) ** 2 <= step_radius_sq]
        else:
            step_order = growth_order

        n_pixels = min(target_px, len(step_order))
        n_pixels = max(n_pixels, soma_area_px)
        n_pixels = min(n_pixels, len(step_order))
        mask_pixel_counts.append(n_pixels)

        mask_roi = np.zeros((h, w), dtype=np.uint8)
        for r, c in step_order[:n_pixels]:
            mask_roi[r, c] = 1

        full_mask = np.zeros(processed_img_shape, dtype=np.uint8)
        full_mask[y_min:y_max, x_min:x_max] = mask_roi

        masks.append({
            'image_name': img_name,
            'soma_idx': soma_idx,
            'soma_id': soma_id,
            'area_um2': target_area_um2,
            'mask': full_mask,
            'approved': None,
            'soma_area_um2': soma_area_um2
        })

    # Auto-reject duplicates
    pixel_count_groups = {}
    for i, n_px in enumerate(mask_pixel_counts):
        pixel_count_groups.setdefault(n_px, []).append(i)
    for n_px, indices in pixel_count_groups.items():
        if len(indices) > 1:
            for idx in indices[:-1]:
                masks[idx]['approved'] = False
                masks[idx]['duplicate'] = True

    # Enforce subset invariant: every smaller mask ⊆ every larger mask
    _enforce_mask_subset_invariant(masks)

    return masks


def _compute_morphology_single(args):
    """Standalone morphology computation for a single mask.

    Runs in a subprocess via ProcessPoolExecutor.
    """
    mask_path, pixel_size, soma_area_um2 = args

    mask = safe_tiff_read(mask_path)
    mask = (mask > 0).astype(np.uint8)

    if not np.any(mask):
        return None

    props = measure.regionprops(mask.astype(int))
    if not props:
        return None

    p = props[0]
    params = {}

    params['perimeter'] = p.perimeter * pixel_size
    params['mask_area'] = p.area * (pixel_size ** 2)

    major_axis = p.major_axis_length
    minor_axis = p.minor_axis_length

    if major_axis > 0:
        axis_ratio = minor_axis / major_axis
        params['eccentricity'] = np.sqrt(1 - axis_ratio ** 2)
        params['roundness'] = axis_ratio ** 2
    else:
        params['eccentricity'] = 0
        params['roundness'] = 0

    centroid = np.array(p.centroid)
    coords = np.array(p.coords)

    top_point = coords[coords[:, 0].argmin()]
    bottom_point = coords[coords[:, 0].argmax()]
    left_point = coords[coords[:, 1].argmin()]
    right_point = coords[coords[:, 1].argmax()]

    extremities = np.array([top_point, bottom_point, left_point, right_point])
    distances = np.sqrt(np.sum((extremities - centroid) ** 2, axis=1))
    params['avg_centroid_distance'] = np.mean(distances) * pixel_size

    if soma_area_um2 is not None:
        params['soma_area'] = soma_area_um2
    else:
        params['soma_area'] = p.area * 0.1 * (pixel_size ** 2)

    # Polarity via PCA
    centered = coords - centroid
    if len(centered) >= 3:
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        major_val = eigenvalues[-1]
        minor_val = eigenvalues[0]
        major_vec = eigenvectors[:, -1]

        if major_val > 0:
            params['polarity_index'] = round(1.0 - (minor_val / major_val), 4)
        else:
            params['polarity_index'] = 0

        angle_rad = np.arctan2(major_vec[0], major_vec[1])
        params['principal_angle'] = round(np.degrees(angle_rad) % 180, 2)
        params['major_axis_um'] = round(2 * np.sqrt(major_val) * pixel_size, 4)
        params['minor_axis_um'] = round(2 * np.sqrt(max(minor_val, 0)) * pixel_size, 4)
    else:
        params['polarity_index'] = 0
        params['principal_angle'] = 0
        params['major_axis_um'] = 0
        params['minor_axis_um'] = 0

    return params


def _export_mask_to_disk(args):
    """Write a single mask TIFF to disk. For use with ThreadPoolExecutor."""
    mask_path, mask_8bit, pixel_size = args
    try:
        tifffile.imwrite(
            mask_path,
            mask_8bit,
            resolution=(1.0 / pixel_size, 1.0 / pixel_size),
            metadata={'unit': 'um'}
        )
        return True
    except Exception:
        return False


# ============================================================================
# AUTO-OUTLINE ALGORITHMS
# ============================================================================

MIN_OUTLINE_POINTS = 10


def _remove_branch_juts(points, centroid, max_iterations=3):
    """Remove narrow branch-like protrusions from a soma outline.

    Uses convexity defect analysis to detect pairs of deep concavities
    that indicate a branch root, then cuts them off with a chord.
    Also applies morphological opening on the filled mask to erode
    thin structures before re-extracting the contour.

    Args:
        points: list of (row, col) tuples - the polygon outline
        centroid: (row, col) of the soma center
        max_iterations: max rounds of defect removal

    Returns:
        list of (row, col) tuples - cleaned polygon outline
    """
    if points is None or len(points) < MIN_OUTLINE_POINTS:
        return points

    pts = np.array(points, dtype=np.float64)
    cy, cx = float(centroid[0]), float(centroid[1])

    # --- Phase 1: morphological opening on filled mask ---
    # Build a tight bounding box around the outline
    rows, cols = pts[:, 0], pts[:, 1]
    pad = 10
    r_min, r_max = int(rows.min()) - pad, int(rows.max()) + pad
    c_min, c_max = int(cols.min()) - pad, int(cols.max()) + pad

    # Create local-coordinate contour for cv2 (x, y format)
    local_pts = np.array([[[int(c - c_min), int(r - r_min)]]
                          for r, c in points], dtype=np.int32)
    h = r_max - r_min
    w = c_max - c_min
    if h <= 0 or w <= 0:
        return points

    # Draw filled polygon into a mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [local_pts.reshape(-1, 1, 2)], 255)

    # Estimate soma radius from mask area to scale the opening kernel
    area_px = cv2.countNonZero(mask)
    equiv_radius = max(3, int(np.sqrt(area_px / np.pi) * 0.45))

    # Morphological opening with circular kernel removes thin branches
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (equiv_radius, equiv_radius))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Make sure the centroid is still inside; if opening erased too much, relax
    local_cy = int(cy - r_min)
    local_cx = int(cx - c_min)
    if (0 <= local_cy < h and 0 <= local_cx < w
            and opened[local_cy, local_cx] == 0):
        # Opening was too aggressive — try a smaller kernel
        smaller = max(3, equiv_radius // 2)
        kernel_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (smaller, smaller))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_sm)
        if (0 <= local_cy < h and 0 <= local_cx < w
                and opened[local_cy, local_cx] == 0):
            # Even smaller kernel lost the centroid; skip opening
            opened = mask

    # Re-extract contour from opened mask
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return points

    # Pick contour nearest centroid
    best_contour = None
    best_dist = float('inf')
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] < 20:
            continue
        mcx = M['m10'] / M['m00']
        mcy = M['m01'] / M['m00']
        d = (mcx - local_cx) ** 2 + (mcy - local_cy) ** 2
        if d < best_dist:
            best_dist = d
            best_contour = cnt

    if best_contour is None:
        best_contour = max(contours, key=cv2.contourArea)

    # --- Phase 2: convexity defect pruning ---
    for _ in range(max_iterations):
        if len(best_contour) < 5:
            break
        hull_idx = cv2.convexHull(best_contour, returnPoints=False)
        if hull_idx is None or len(hull_idx) < 3:
            break

        # Sort hull indices for convexityDefects (must be ascending)
        hull_idx = np.sort(hull_idx, axis=0)
        try:
            defects = cv2.convexityDefects(best_contour, hull_idx)
        except cv2.error:
            break

        if defects is None or len(defects) == 0:
            break

        # Contour perimeter for relative depth threshold
        perimeter = cv2.arcLength(best_contour, True)
        depth_threshold = perimeter * 0.03  # defect must be >3% of perimeter

        # Collect deep defects sorted by depth (deepest first)
        deep = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            depth = d / 256.0
            if depth > depth_threshold:
                deep.append((s, e, f, depth))

        if len(deep) < 2:
            break

        deep.sort(key=lambda x: -x[3])

        # Try to find pairs of defects that form a narrow jut
        pruned = False
        used = set()
        for i in range(len(deep)):
            if i in used:
                continue
            s1, e1, f1, d1 = deep[i]
            start1 = tuple(best_contour[s1][0])
            end1 = tuple(best_contour[e1][0])
            far1 = tuple(best_contour[f1][0])

            for j in range(i + 1, len(deep)):
                if j in used:
                    continue
                s2, e2, f2, d2 = deep[j]
                start2 = tuple(best_contour[s2][0])
                end2 = tuple(best_contour[e2][0])
                far2 = tuple(best_contour[f2][0])

                # Check if the two defects' far points are close to each
                # other relative to their depth — that means a narrow jut
                # sits between them
                gap = np.sqrt((far1[0] - far2[0]) ** 2 +
                              (far1[1] - far2[1]) ** 2)
                avg_depth = (d1 + d2) / 2

                # Narrow jut: the gap between defect bases is small
                # relative to how deep they are
                if gap < avg_depth * 2.5:
                    # Cut the jut: connect the two far points, removing
                    # the contour segment between them
                    idx1, idx2 = sorted([f1, f2])
                    n = len(best_contour)

                    # Determine which arc to keep (the one containing
                    # the centroid)
                    arc_a = list(range(idx2, n)) + list(range(0, idx1 + 1))
                    arc_b = list(range(idx1, idx2 + 1))

                    # Check which arc contains points closer to centroid
                    def arc_centroid_dist(arc):
                        if not arc:
                            return float('inf')
                        apts = best_contour[arc].reshape(-1, 2)
                        dists = (apts[:, 0] - local_cx) ** 2 + \
                                (apts[:, 1] - local_cy) ** 2
                        return float(np.min(dists))

                    if arc_centroid_dist(arc_a) <= arc_centroid_dist(arc_b):
                        keep = arc_a
                    else:
                        keep = arc_b

                    if len(keep) >= MIN_OUTLINE_POINTS:
                        best_contour = best_contour[keep]
                        pruned = True
                        used.add(i)
                        used.add(j)
                        break  # restart defect analysis on new contour

            if pruned:
                break

        if not pruned:
            break

    # --- Convert back to (row, col) image coordinates ---
    approx = _simplify_contour(best_contour)
    result = []
    for pt in approx:
        px, py = pt[0]
        img_col = px + c_min
        img_row = py + r_min
        result.append((img_row, img_col))

    if len(result) < MIN_OUTLINE_POINTS:
        return points

    return result


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
    mask area, perimeter, average centroid distance, polarity index, principal angle,
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
            
            # Roundness: (minor/major)^2  — 0 = elongated, 1 = circular
            if major_axis > 0:
                params['roundness'] = axis_ratio ** 2
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
            params['avg_centroid_distance'] = np.mean(distances) * self.pixel_size

            if soma_area_um2 is not None:
                params['soma_area'] = soma_area_um2
            else:
                params['soma_area'] = props.area * 0.1 * (self.pixel_size ** 2)

            # Directional polarity via PCA on mask coordinates
            params.update(self._calculate_polarity(coords, centroid))
        else:
            params = {k: 0 for k in ['perimeter', 'mask_area', 'eccentricity',
                                     'roundness', 'avg_centroid_distance', 'soma_area',
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
    """Thread for calculating morphology parameters in the background.

    Uses ProcessPoolExecutor to parallelize across CPU cores when masks
    are on disk (the common case for 22k+ masks).
    """
    progress = pyqtSignal(int, str)  # progress percentage, status message
    finished = pyqtSignal(list)  # list of results
    error_occurred = pyqtSignal(str)

    def __init__(self, approved_masks, pixel_size, use_imagej, images, output_dir=None, pixel_size_map=None, masks_dir=None, soma_group_map=None):
        super().__init__()
        self.approved_masks = approved_masks
        self.pixel_size = pixel_size
        self.use_imagej = use_imagej
        self.images = images
        self.output_dir = output_dir
        self.pixel_size_map = pixel_size_map or {}
        self.masks_dir = masks_dir
        self.soma_group_map = soma_group_map or {}

    def run(self):
        import sys
        import os
        import time

        old_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')

        try:
            total = len(self.approved_masks)
            batch_start = time.time()
            n_workers = max(1, multiprocessing.cpu_count() - 1)

            # Build task list: separate disk-based masks (parallelizable)
            # from in-memory masks (serial)
            parallel_tasks = []  # (index, mask_path, pixel_size, soma_area_um2)
            serial_indices = []  # indices needing serial processing
            task_metadata = {}   # index -> (img_name, soma_id, soma_idx, area_um2)

            for i, flat_data in enumerate(self.approved_masks):
                mask_data = flat_data['mask_data']
                img_name = flat_data['image_name']
                img_pixel_size = self.pixel_size_map.get(img_name, self.pixel_size)
                soma_area_um2 = mask_data.get('soma_area_um2', None)

                task_metadata[i] = (
                    os.path.splitext(img_name)[0],
                    mask_data['soma_id'],
                    mask_data['soma_idx'],
                    mask_data['area_um2']
                )

                # Check if mask is on disk
                if self.masks_dir:
                    img_basename = os.path.splitext(img_name)[0]
                    soma_id = mask_data['soma_id']
                    area_um2 = mask_data.get('area_um2', 0)
                    mask_filename = f"{img_basename}_{soma_id}_area{int(area_um2)}_mask.tif"
                    mask_path = os.path.join(self.masks_dir, mask_filename)
                    if os.path.exists(mask_path):
                        parallel_tasks.append((i, mask_path, img_pixel_size, soma_area_um2))
                        continue

                # Mask in memory or not on disk — process serially
                serial_indices.append(i)

            all_results = [None] * total
            completed = 0

            # Process all disk-based masks serially
            serial_indices.extend([idx for idx, _, _, _ in parallel_tasks])

            # Process remaining masks serially (in-memory)
            calculator_cache = {}
            for i in serial_indices:
                flat_data = self.approved_masks[i]
                mask_data = flat_data['mask_data']
                img_name = flat_data['image_name']
                img_data = self.images[img_name]

                processed_img = img_data.get('processed')
                if processed_img is None and self.output_dir:
                    name_stem = os.path.splitext(img_name)[0]
                    processed_path = os.path.join(self.output_dir, f"{name_stem}_processed.tif")
                    if os.path.exists(processed_path):
                        processed_img = safe_tiff_read(processed_path)
                        img_data['processed'] = processed_img

                soma_centroid = img_data['somas'][mask_data['soma_idx']]
                soma_area_um2 = mask_data.get('soma_area_um2', None)

                if mask_data.get('mask') is None and self.masks_dir:
                    img_basename = os.path.splitext(img_name)[0]
                    soma_id = mask_data['soma_id']
                    area_um2 = mask_data.get('area_um2', 0)
                    mask_filename = f"{img_basename}_{soma_id}_area{int(area_um2)}_mask.tif"
                    mask_path = os.path.join(self.masks_dir, mask_filename)
                    if os.path.exists(mask_path):
                        mask_arr = safe_tiff_read(mask_path)
                        mask_data['mask'] = (mask_arr > 0).astype(np.uint8)
                if mask_data.get('mask') is None:
                    completed += 1
                    continue

                img_pixel_size = self.pixel_size_map.get(img_name, self.pixel_size)
                # Support per-image (x, y) tuple or single float
                if isinstance(img_pixel_size, (list, tuple)):
                    ps_for_calc = math.sqrt(img_pixel_size[0] * img_pixel_size[1])
                    ps_x, ps_y = img_pixel_size
                else:
                    ps_for_calc = img_pixel_size
                    ps_x = ps_y = img_pixel_size
                cache_key = (id(processed_img), ps_for_calc)
                if cache_key not in calculator_cache:
                    calculator_cache[cache_key] = MorphologyCalculator(processed_img, ps_for_calc, use_imagej=self.use_imagej)
                calculator = calculator_cache[cache_key]
                params = calculator.calculate_all_parameters(mask_data['mask'], soma_centroid, soma_area_um2)

                meta = task_metadata[i]
                params['image_name'] = meta[0]
                params['soma_id'] = meta[1]
                params['soma_idx'] = meta[2]
                params['area_um2'] = meta[3]
                params['soma_group'] = self.soma_group_map.get((meta[0], meta[1]), '')
                all_results[i] = params
                mask_data['mask'] = None

                completed += 1
                if completed % 50 == 0:
                    elapsed = time.time() - batch_start
                    speed = completed / max(elapsed, 0.001)
                    remaining = (total - completed) / max(speed, 0.001)
                    if remaining >= 60:
                        eta_str = f"~{remaining / 60:.1f}min left"
                    else:
                        eta_str = f"~{remaining:.0f}s left"
                    self.progress.emit(
                        int(completed / total * 100),
                        f"Processing ({completed}/{total}) [{speed:.1f}/s {eta_str}]"
                    )

            # Filter out None entries (skipped masks)
            all_results = [r for r in all_results if r is not None]

            total_time = time.time() - batch_start
            sys.stderr = old_stderr
            print(f"  Calculated {len(all_results)} masks in {total_time:.1f}s ({len(all_results)/max(total_time,0.001):.1f} masks/s)")
            sys.stderr = open(os.devnull, 'w')

            self.finished.emit(all_results)

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            if sys.stderr != old_stderr:
                sys.stderr.close()
                sys.stderr = old_stderr


class BackgroundRemovalThread(QThread):
    progress = pyqtSignal(int)
    status_update = pyqtSignal(str)
    finished_image = pyqtSignal(str, str, object)
    # Signal for extra cleaned channels: (img_name, {ch_idx: cleaned_array})
    finished_extra_channels = pyqtSignal(str, object)
    error_occurred = pyqtSignal(str)

    def __init__(self, image_data_list, output_dir):
        super().__init__()
        self.image_data_list = image_data_list
        self.output_dir = output_dir

    def _clean_single_channel(self, raw_img, ch_idx, radius, rb_enabled,
                               denoise_enabled, denoise_size,
                               sharpen_enabled, sharpen_amount):
        """Apply the cleaning pipeline to one channel and return the result."""
        if raw_img.ndim == 3:
            # Extract channel WITHOUT normalization to preserve original
            # intensity range. extract_channel() rescales to 0-255 uint8
            # which destroys relative intensities between channels.
            img = raw_img[:, :, ch_idx].copy()
        else:
            img = raw_img
        img_dtype = img.dtype
        result = img.copy()

        if rb_enabled:
            background = restoration.rolling_ball(img, radius=radius)
            result = img - background
            result = np.clip(result, 0, np.iinfo(img_dtype).max)

        if denoise_enabled:
            result = ndimage.median_filter(result, size=denoise_size)

        if sharpen_enabled:
            blurred = ndimage.gaussian_filter(result.astype(np.float32), sigma=2)
            result_float = result.astype(np.float32)
            sharpened = result_float + sharpen_amount * (result_float - blurred)
            result = np.clip(sharpened, 0, np.iinfo(img_dtype).max).astype(img_dtype)

        return result

    def run(self):
        try:
            total = len(self.image_data_list)
            for i, (
                    img_path, img_name, radius, rb_enabled, denoise_enabled, denoise_size,
                    sharpen_enabled, sharpen_amount, process_channels) in enumerate(
                self.image_data_list):
                try:
                    self.status_update.emit(f"Processing: {img_name}")
                    raw_img = load_tiff_image(img_path)

                    # process_channels is a list; first element is the primary channel
                    # Support legacy callers passing a single int
                    if isinstance(process_channels, int):
                        process_channels = [process_channels]

                    primary_ch = process_channels[0]
                    extra_channels = process_channels[1:] if len(process_channels) > 1 else []

                    # Clean the primary channel
                    result = self._clean_single_channel(
                        raw_img, primary_ch, radius, rb_enabled,
                        denoise_enabled, denoise_size,
                        sharpen_enabled, sharpen_amount)

                    name = os.path.splitext(img_name)[0]
                    out_path = os.path.join(self.output_dir, f"{name}_processed.tif")
                    tifffile.imwrite(out_path, result.astype(result.dtype))
                    self.finished_image.emit(out_path, img_name, result)

                    # Clean extra channels if requested
                    if extra_channels and raw_img.ndim == 3:
                        extra_results = {}
                        for ch_idx in extra_channels:
                            if ch_idx < raw_img.shape[2]:
                                self.status_update.emit(
                                    f"Processing: {img_name} (Ch {ch_idx + 1})")
                                ch_result = self._clean_single_channel(
                                    raw_img, ch_idx, radius, rb_enabled,
                                    denoise_enabled, denoise_size,
                                    sharpen_enabled, sharpen_amount)
                                # Save extra channel to disk
                                ch_path = os.path.join(
                                    self.output_dir,
                                    f"{name}_processed_ch{ch_idx + 1}.tif")
                                tifffile.imwrite(ch_path, ch_result.astype(ch_result.dtype))
                                extra_results[ch_idx] = ch_result
                        if extra_results:
                            self.finished_extra_channels.emit(img_name, extra_results)

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
        self.locked_centroids = []  # Pass 1 somas shown during Pass 2 (cyan)
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
        # Pixel intensity picker
        self.pixel_picker_mode = False

    def set_image(self, qpix, centroids=None, mask_overlay=None, polygon_pts=None, locked_centroids=None):
        self.pix_source = qpix
        self.centroids = centroids or []
        self.locked_centroids = locked_centroids or []
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

        # Draw locked centroids (Pass 1 somas during Pass 2) in cyan
        if self.locked_centroids:
            pen = QPen(QColor(0, 255, 255), 3)
            painter.setPen(pen)
            for centroid in self.locked_centroids:
                x, y = self._to_display_coords(centroid)
                if 0 <= x <= self.width() and 0 <= y <= self.height():
                    painter.drawEllipse(int(x - 6), int(y - 6), 12, 12)

        # Draw active soma markers (centroids) in red
        if self.centroids:
            pen = QPen(QColor(255, 0, 0), 3)
            painter.setPen(pen)
            for centroid in self.centroids:
                x, y = self._to_display_coords(centroid)
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

    def _find_nearest_point(self, click_pos, threshold=8):
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

        # Pixel intensity picker mode
        if self.pixel_picker_mode and event.button() == Qt.LeftButton:
            if self.parent_widget and hasattr(self.parent_widget, '_show_pixel_intensity'):
                self.parent_widget._show_pixel_intensity(coords)
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
            # Shift+click to drag existing points (prevents accidental grabs)
            if self.polygon_pts and event.button() == Qt.LeftButton and (event.modifiers() & Qt.ShiftModifier):
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


class CSVMergeDialog(QDialog):
    """Dialog to select which ImageJ CSV files to merge (Sholl, Skeleton, Fractal, Combined)."""
    def __init__(self, parent=None, initial_dir=None):
        super().__init__(parent)
        self.setWindowTitle("Merge ImageJ CSV Results")
        self.setModal(True)
        self.setMinimumWidth(600)
        self._initial_dir = initial_dir or os.path.expanduser("~")
        self._file_paths = {}  # csv_type -> file_path

        layout = QVBoxLayout(self)

        instructions = QLabel(
            "Select which ImageJ result CSVs to merge with your morphology results.\n"
            "Files found in your output directory are pre-filled below."
        )
        layout.addWidget(instructions)

        # Auto-detect known CSV files
        auto_detected = self._auto_detect(initial_dir) if initial_dir else {}

        # One row per CSV type: checkbox + path display + browse button
        self._checks = {}
        self._path_labels = {}
        csv_types = [
            ('simple', 'Simple Morphology Results'),
            ('combined', 'ImageJ Combined Results'),
            ('sholl', 'Sholl Results'),
            ('skeleton', 'Skeleton Results'),
            ('fractal', 'Fractal / Convex Hull Results'),
        ]
        for csv_type, display_name in csv_types:
            group = QGroupBox(display_name)
            row_layout = QHBoxLayout(group)

            check = QCheckBox("Include")
            self._checks[csv_type] = check
            row_layout.addWidget(check)

            path_label = QLabel("No file selected")
            path_label.setStyleSheet("color: gray; font-style: italic;")
            path_label.setMinimumWidth(300)
            self._path_labels[csv_type] = path_label
            row_layout.addWidget(path_label, 1)

            browse_btn = QPushButton("Browse...")
            browse_btn.clicked.connect(lambda checked, t=csv_type: self._browse(t))
            row_layout.addWidget(browse_btn)

            # Pre-fill if auto-detected
            if csv_type in auto_detected:
                self._file_paths[csv_type] = auto_detected[csv_type]
                path_label.setText(os.path.basename(auto_detected[csv_type]))
                path_label.setStyleSheet("color: black; font-style: normal;")
                check.setChecked(True)
            else:
                check.setChecked(False)

            layout.addWidget(group)

        # OK / Cancel
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("Merge")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _auto_detect(self, base_dir):
        """Search output directory for known ImageJ CSV files."""
        found = {}
        if not base_dir or not os.path.isdir(base_dir):
            return found
        # Check for simple morphology results
        simple = os.path.join(base_dir, "combined_morphology_results.csv")
        if os.path.isfile(simple):
            found['simple'] = simple
        # Check for ImageJ_Combined_Results.csv (in output root)
        combined = os.path.join(base_dir, "ImageJ_Combined_Results.csv")
        if os.path.isfile(combined):
            found['combined'] = combined
        # Check subdirectories for individual analysis results
        skel = os.path.join(base_dir, "skeleton_results", "Skeleton_Analysis_Results.csv")
        if os.path.isfile(skel):
            found['skeleton'] = skel
        frac = os.path.join(base_dir, "fractal_results", "Fractal_Analysis_Results.csv")
        if os.path.isfile(frac):
            found['fractal'] = frac
        sholl = os.path.join(base_dir, "sholl_results", "Sholl_All_Results.csv")
        if os.path.isfile(sholl):
            found['sholl'] = sholl
        return found

    def _browse(self, csv_type):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select {csv_type.title()} CSV",
            self._initial_dir,
            "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self._file_paths[csv_type] = path
            self._path_labels[csv_type].setText(os.path.basename(path))
            self._path_labels[csv_type].setStyleSheet("color: black; font-style: normal;")
            self._checks[csv_type].setChecked(True)
            # Update browse directory for convenience
            self._initial_dir = os.path.dirname(path)

    def get_selected_files(self):
        """Return dict of {csv_type: file_path} for checked types that have a file."""
        result = {}
        for csv_type, check in self._checks.items():
            if check.isChecked() and csv_type in self._file_paths:
                result[csv_type] = self._file_paths[csv_type]
        return result


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
        self.pixel_size = 0.316
        self.default_rolling_ball_radius = 50
        self.all_masks_flat = []
        self.mask_qa_idx = 0
        self.mask_qa_active = False
        self.last_qa_decisions = []
        # Sliding window for mask QA memory management:
        # Only keep masks for current + last 10 somas in memory.
        self._qa_soma_order = []        # ordered list of (img_name, soma_id) as encountered
        self._qa_finalized_somas = set()  # somas evicted from memory
        self._qa_soma_window_size = 10    # keep last N reviewed somas in memory
        self.soma_mode = False  # Initialize soma_mode to prevent crashes
        # Initialize display adjustment values
        self.brightness_value = 0
        self.contrast_value = 0
        # Per-channel brightness for colocalization mode
        self.channel_brightness = {'R': 0, 'G': 0, 'B': 0}
        # Mask generation settings (defaults)
        self.use_min_intensity = True
        self.min_intensity_percent = 5
        self.mask_min_area = 100
        self.mask_max_area = 800
        self.mask_step_size = 100
        self.mask_segmentation_method = 'none'  # 'none', 'competitive', 'watershed'
        self.use_circular_constraint = False
        self.circular_buffer_um2 = 200  # extra area (µm²) beyond target for circular boundary
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
        # Pixel intensity picker state
        self.pixel_picker_mode = False
        self.measure_point1 = None  # (row, col) image coords
        self.measure_point2 = None
        # --- 3D Z-stack mode state ---
        self.mode_3d = False
        self.current_z_slice = 0
        self.voxel_size_z = 1.0
        self.mask_min_volume = 500
        self.mask_max_volume = 5000
        self.soma_intensity_tolerance = 30
        self.soma_max_radius_um = 8.0
        self.init_ui()

    def keyPressEvent(self, event):
        key = event.key()

        # Track Z key for zoom functionality
        if key == Qt.Key_Z:
            self.z_key_held = True
            return  # Don't process further, Z is for zoom

        # ? shows help regardless of mode
        if key == Qt.Key_Question:
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
                        if 'soma_groups' in img_data:
                            img_data['soma_groups'].clear()
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
        # Set window icon (picks up MMPS.icns/.ico/.png next to the script)
        icon = _get_app_icon()
        if icon:
            self.setWindowIcon(icon)

        # Menu bar — Session management lives here instead of crowding the left panel
        menu_bar = self.menuBar()
        session_menu = menu_bar.addMenu("Session")
        save_action = session_menu.addAction("Save Session")
        save_action.setShortcut("Ctrl+S")
        save_action.setToolTip("Save current project state to resume later")
        save_action.triggered.connect(self.save_session)
        load_action = session_menu.addAction("Load Session")
        load_action.setShortcut("Ctrl+O")
        load_action.setToolTip("Resume a previously saved session")
        load_action.triggered.connect(self.load_session)

        # Mode menu
        mode_menu = menu_bar.addMenu("Mode")

        self.coloc_action = mode_menu.addAction("Colocalization")
        self.coloc_action.setCheckable(True)
        self.coloc_action.setChecked(self.colocalization_mode)
        self.coloc_action.setToolTip("Enable colocalization analysis (multi-channel color images)")
        self.coloc_action.triggered.connect(self._toggle_colocalization_mode)

        self.mode_3d_action = mode_menu.addAction("3D Z-Stack Analysis")
        self.mode_3d_action.setCheckable(True)
        self.mode_3d_action.setChecked(False)
        self.mode_3d_action.setToolTip("Open the 3D Z-stack analysis window")
        self.mode_3d_action.triggered.connect(self._toggle_3d_mode)

        # Advanced menu
        advanced_menu = menu_bar.addMenu("Advanced")
        per_image_px_action = advanced_menu.addAction("Set Per-Image Pixel Size...")
        per_image_px_action.setToolTip("Override pixel size for individual images")
        per_image_px_action.triggered.connect(self._set_per_image_pixel_size)

        # Cluster menu
        cluster_menu = menu_bar.addMenu("Cluster")
        mask_gen_action = cluster_menu.addAction("Generate Mask Generation Script...")
        mask_gen_action.setToolTip("Export a standalone Python script for mask generation on a compute cluster")
        mask_gen_action.triggered.connect(self.export_cluster_script)
        imagej_action = cluster_menu.addAction("Generate ImageJ Analysis Script...")
        imagej_action.setToolTip("Export Fiji scripts for Skeleton, Fractal/Hull, and Sholl analysis on a cluster")
        imagej_action.triggered.connect(self._open_imagej_cluster_dialog)
        spread_action = cluster_menu.addAction("Generate Spread Analysis Script...")
        spread_action.setToolTip("Export a Python script for cell spread / morphology analysis on a cluster")
        spread_action.triggered.connect(self._open_spread_cluster_dialog)

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
        self.shortcut_help = QShortcut(QKeySequence('?'), self)
        self.shortcut_help.setContext(Qt.ApplicationShortcut)
        self.shortcut_help.activated.connect(self.show_shortcut_help)
        self.shortcut_measure = QShortcut(QKeySequence('M'), self)
        self.shortcut_measure.activated.connect(self.toggle_measure_mode)
        self.shortcut_pixel_picker = QShortcut(QKeySequence('I'), self)
        self.shortcut_pixel_picker.activated.connect(self.toggle_pixel_picker_mode)
        self.shortcut_undo_qa = QShortcut(QKeySequence('B'), self)
        self.shortcut_undo_qa.activated.connect(self.undo_last_qa)

    def _create_left_panel(self):
        scroll = QScrollArea()
        scroll.setFixedWidth(450)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFocusPolicy(Qt.NoFocus)  # Don't steal focus from child widgets

        panel = QWidget()
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

        # --- Pixel size inputs (X, Y with link toggle) ---
        pixel_size_widget = QWidget()
        px_layout = QHBoxLayout(pixel_size_widget)
        px_layout.setContentsMargins(0, 0, 0, 0)

        self.pixel_size_x_input = QLineEdit(str(self.pixel_size))
        self.pixel_size_x_input.setFixedWidth(70)
        self.pixel_size_y_input = QLineEdit(str(self.pixel_size))
        self.pixel_size_y_input.setFixedWidth(70)

        self.pixel_size_link_btn = QPushButton("🔗")
        self.pixel_size_link_btn.setFixedWidth(30)
        self.pixel_size_link_btn.setCheckable(True)
        self.pixel_size_link_btn.setChecked(True)
        self.pixel_size_link_btn.setToolTip("Link X and Y pixel sizes (isotropic)")
        self.pixel_size_link_btn.toggled.connect(self._on_pixel_size_link_toggled)

        px_layout.addWidget(QLabel("X:"))
        px_layout.addWidget(self.pixel_size_x_input)
        px_layout.addWidget(self.pixel_size_link_btn)
        px_layout.addWidget(QLabel("Y:"))
        px_layout.addWidget(self.pixel_size_y_input)
        px_layout.addStretch()

        # When linked, typing in X updates Y automatically
        self.pixel_size_x_input.textChanged.connect(self._on_pixel_size_x_changed)
        self.pixel_size_y_input.textChanged.connect(self._on_pixel_size_y_changed)

        self.pixel_size_label = QLabel("Pixel size (μm/px):")
        form_layout.addRow(self.pixel_size_label, pixel_size_widget)

        # Backward-compat alias: self.pixel_size_input → self.pixel_size_x_input
        self.pixel_size_input = self.pixel_size_x_input

        # 3D voxel Z input (hidden until 3D mode enabled)
        self.voxel_z_input = QLineEdit(str(self.voxel_size_z))
        self.voxel_z_label = QLabel("Z step size (μm/slice):")
        form_layout.addRow(self.voxel_z_label, self.voxel_z_input)
        self.voxel_z_label.setVisible(False)
        self.voxel_z_input.setVisible(False)
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

        # Multi-channel cleaning option
        self.multi_clean_check = QCheckBox("Also clean additional channels")
        self.multi_clean_check.setChecked(False)
        self.multi_clean_check.setToolTip("Apply the same cleaning pipeline to extra channels (for color composite display)")
        self.multi_clean_check.toggled.connect(self._toggle_multi_channel_ui)
        param_layout.addWidget(self.multi_clean_check)

        self.extra_channel_widget = QWidget()
        extra_ch_layout = QHBoxLayout(self.extra_channel_widget)
        extra_ch_layout.setContentsMargins(20, 0, 0, 0)
        extra_ch_layout.addWidget(QLabel("Clean:"))
        self.clean_ch_checks = []
        for i in range(3):
            ch_check = QCheckBox(f"Ch {i + 1}")
            ch_check.setChecked(False)
            extra_ch_layout.addWidget(ch_check)
            self.clean_ch_checks.append(ch_check)
        extra_ch_layout.addStretch()
        self.extra_channel_widget.setVisible(False)
        param_layout.addWidget(self.extra_channel_widget)


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
        self.auto_outline_sensitivity.setRange(1, 90)
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
        self.outline_sens_display.setRange(1, 90)
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

        self.outline_controls_widget.setVisible(False)
        batch_layout.addWidget(self.outline_controls_widget)
        
        self.batch_generate_masks_btn = QPushButton("Generate All Masks")
        self.batch_generate_masks_btn.clicked.connect(self.batch_generate_masks)
        self.batch_generate_masks_btn.setEnabled(False)
        batch_layout.addWidget(self.batch_generate_masks_btn)

        # Clear All Masks — created here but placed in right panel (visible only during QA)
        self.clear_masks_btn = QPushButton("Clear All Masks")
        self.clear_masks_btn.clicked.connect(self.clear_all_masks)
        self.clear_masks_btn.setEnabled(False)
        self.clear_masks_btn.setVisible(False)
        self.clear_masks_btn.setStyleSheet("border: 2px solid #F44336; font-weight: bold; padding: 4px 10px;")

        self.batch_qa_btn = QPushButton("QA All Masks")
        self.batch_qa_btn.clicked.connect(self.start_batch_qa)
        self.batch_qa_btn.setEnabled(False)
        batch_layout.addWidget(self.batch_qa_btn)
        self.undo_qa_btn = QPushButton("Undo QA")
        self.undo_qa_btn.clicked.connect(self.undo_last_qa)
        self.undo_qa_btn.setEnabled(False)
        self.undo_qa_btn.setToolTip("Reset all mask approvals and restart QA")
        self.undo_qa_btn.setStyleSheet("border: 2px solid #FF9800;")
        self.undo_qa_btn.setVisible(False)

        # Approve All button — created here but placed in right panel zoom bar
        self.approve_all_btn = QPushButton("Approve All Remaining")
        self.approve_all_btn.clicked.connect(self._approve_all_remaining)
        self.approve_all_btn.setEnabled(False)
        self.approve_all_btn.setVisible(False)
        self.approve_all_btn.setStyleSheet("QPushButton { border: 2px solid #4CAF50; font-weight: bold; }")
        self.approve_all_btn.setToolTip(
            "Approve all remaining unreviewed masks at once.\n"
            "Useful for large datasets (22k+ images) where manual QA is impractical.")

        self.batch_calculate_btn = QPushButton("Calculate Simple Characteristics")
        self.batch_calculate_btn.clicked.connect(self.batch_calculate_morphology)
        self.batch_calculate_btn.setEnabled(False)
        batch_layout.addWidget(self.batch_calculate_btn)

        batch_group.setLayout(batch_layout)
        layout.addWidget(batch_group)

        # --- Step 4: ImageJ Plugin ---
        imagej_group = QGroupBox("4. ImageJ Plugin")
        imagej_group_layout = QVBoxLayout()
        self.import_imagej_btn = QPushButton("Import ImageJ Results")
        self.import_imagej_btn.clicked.connect(self.import_imagej_results)
        self.import_imagej_btn.setEnabled(True)
        self.import_imagej_btn.setToolTip("Import Sholl, Skeleton, & Fractal CSVs and merge into combined CSV")
        imagej_group_layout.addWidget(self.import_imagej_btn)
        imagej_group.setLayout(imagej_group_layout)
        layout.addWidget(imagej_group)
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
        scroll.setWidget(panel)
        scroll.viewport().setFocusPolicy(Qt.NoFocus)
        return scroll

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

        # Z-slice slider (hidden until 3D mode enabled)
        self.z_slider_widget = QWidget()
        z_layout = QHBoxLayout(self.z_slider_widget)
        z_layout.setContentsMargins(0, 0, 0, 0)
        z_layout.addWidget(QLabel("Z-Slice:"))
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setRange(0, 0)
        self.z_slider.setValue(0)
        self.z_slider.valueChanged.connect(self._on_z_slider_changed)
        z_layout.addWidget(self.z_slider)
        self.z_label = QLabel("0 / 0")
        self.z_label.setFixedWidth(80)
        z_layout.addWidget(self.z_label)
        self.z_slider_widget.setVisible(False)
        layout.addWidget(self.z_slider_widget)

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

        # Calibrate pixel size from measurement
        self.calibrate_btn = QPushButton("Calibrate")
        self.calibrate_btn.clicked.connect(self._calibrate_from_measurement)
        self.calibrate_btn.setToolTip("Calculate pixel size from a measured known distance")
        display_btn_layout.addWidget(self.calibrate_btn)

        # Pixel intensity picker button
        self.pixel_picker_btn = QPushButton("Pick Intensity (I)")
        self.pixel_picker_btn.clicked.connect(self.toggle_pixel_picker_mode)
        self.pixel_picker_btn.setToolTip("Click a pixel to see its intensity in each channel")
        self.pixel_picker_btn.setCheckable(True)
        display_btn_layout.addWidget(self.pixel_picker_btn)

        # Help button
        help_btn = QPushButton("?")
        help_btn.setFixedWidth(35)
        help_btn.clicked.connect(self.show_shortcut_help)
        help_btn.setToolTip("Show keyboard shortcuts for current mode")
        display_btn_layout.addWidget(help_btn)

        layout.addLayout(display_btn_layout)

        # Mask overlay opacity slider — hidden until masks are generated
        self.opacity_widget = QWidget()
        opacity_layout = QHBoxLayout(self.opacity_widget)
        opacity_layout.setContentsMargins(0, 0, 0, 0)
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
        self.opacity_widget.setVisible(False)
        layout.addWidget(self.opacity_widget)

        # Zoom hint row
        zoom_layout = QHBoxLayout()
        zoom_hint = QLabel("Z + Left-click: zoom in, Z + Right-click: zoom out, U: reset, M: measure, ?: help")
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

        # Clear All Masks + Undo QA + Approve All — next to Redo, only visible during QA
        zoom_layout.addWidget(self.clear_masks_btn)
        zoom_layout.addWidget(self.undo_qa_btn)
        zoom_layout.addWidget(self.approve_all_btn)

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

        # QA Autozoom level setting
        from PyQt5.QtWidgets import QDoubleSpinBox
        qa_zoom_layout = QHBoxLayout()
        qa_zoom_label = QLabel("QA Autozoom:")
        qa_zoom_label.setToolTip("Zoom level used when auto-centering on masks and somas during QA")
        self.qa_autozoom_spin = QDoubleSpinBox()
        self.qa_autozoom_spin.setRange(1.0, 10.0)
        self.qa_autozoom_spin.setSingleStep(0.5)
        self.qa_autozoom_spin.setValue(3.0)
        self.qa_autozoom_spin.setDecimals(1)
        self.qa_autozoom_spin.setSuffix("x")
        self.qa_autozoom_spin.setToolTip("Set the auto-zoom level for mask/soma QA (default 3.0x)")
        self.qa_autozoom_spin.setFixedWidth(80)
        qa_zoom_layout.addWidget(qa_zoom_label)
        qa_zoom_layout.addWidget(self.qa_autozoom_spin)
        qa_zoom_layout.addStretch()
        layout.addLayout(qa_zoom_layout)

        # Mask QA progress bar
        self.mask_qa_progress_bar = QProgressBar()
        self.mask_qa_progress_bar.setVisible(False)
        self.mask_qa_progress_bar.setMinimumHeight(20)
        self.mask_qa_progress_bar.setFormat("%v / %m masks reviewed")
        self.mask_qa_progress_bar.setStyleSheet("""
            QProgressBar { text-align: center; font-weight: bold; }
        """)
        layout.addWidget(self.mask_qa_progress_bar)

        return panel

    # ------------------------------------------------------------------
    # Cluster Tab
    # ------------------------------------------------------------------

    def _open_imagej_cluster_dialog(self):
        """Open a dialog for generating ImageJ HPC cluster scripts."""
        from PyQt5.QtWidgets import QDialog, QScrollArea

        dlg = QDialog(self)
        dlg.setWindowTitle("Generate ImageJ Cluster Scripts")
        dlg.setMinimumSize(500, 650)
        dlg_layout = QVBoxLayout(dlg)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)

        desc = QLabel(
            "Generate a self-contained Jython script and SLURM array job to run\n"
            "Skeleton, Fractal/Hull, and Sholl analyses on an HPC cluster using\n"
            "headless Fiji. One SLURM task per image, all submitted at once."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: palette(dark); padding-bottom: 8px;")
        layout.addWidget(desc)

        # --- Fiji Path ---
        fiji_group = QGroupBox("Fiji Installation")
        fiji_layout = QVBoxLayout()
        fiji_desc = QLabel("Path to Fiji on the cluster:")
        fiji_layout.addWidget(fiji_desc)
        self.cluster_fiji_path = QLineEdit("$SCRATCH/ImageJFiji/Fiji.app/ImageJ-linux64")
        fiji_layout.addWidget(self.cluster_fiji_path)
        fiji_group.setLayout(fiji_layout)
        layout.addWidget(fiji_group)

        # --- Analysis Selection ---
        analysis_group = QGroupBox("Analyses to Run")
        analysis_layout = QVBoxLayout()
        self.cluster_do_skeleton = QCheckBox("Skeleton Analysis")
        self.cluster_do_skeleton.setChecked(True)
        analysis_layout.addWidget(self.cluster_do_skeleton)
        self.cluster_do_fractal = QCheckBox("Fractal / Convex Hull Analysis")
        self.cluster_do_fractal.setChecked(True)
        analysis_layout.addWidget(self.cluster_do_fractal)
        self.cluster_do_sholl = QCheckBox("Sholl Analysis")
        self.cluster_do_sholl.setChecked(True)
        analysis_layout.addWidget(self.cluster_do_sholl)
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

        # --- Parameters ---
        param_group = QGroupBox("Parameters")
        param_layout = QFormLayout()
        try:
            px = str(self._get_pixel_size())
        except Exception:
            px = "0.316"
        self.cluster_pixel_size = QLineEdit(px)
        param_layout.addRow("Pixel size (um/px):", self.cluster_pixel_size)
        self.cluster_upscale_factor = QLineEdit("2")
        self.cluster_upscale_factor.setToolTip("2 for 20x, 1 for 40x")
        param_layout.addRow("Upscale factor:", self.cluster_upscale_factor)
        self.cluster_sholl_step = QLineEdit("0")
        self.cluster_sholl_step.setToolTip("0 = continuous / pixel-level")
        param_layout.addRow("Sholl step size (um):", self.cluster_sholl_step)
        self.cluster_largest_only = QCheckBox("Only analyze largest mask per cell")
        self.cluster_largest_only.setChecked(True)
        param_layout.addRow("", self.cluster_largest_only)
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # --- SLURM Settings ---
        slurm_group = QGroupBox("SLURM Job Settings (per image)")
        slurm_layout = QFormLayout()
        self.cluster_partition = QLineEdit("comm_small_day")
        slurm_layout.addRow("Partition:", self.cluster_partition)
        self.cluster_time = QLineEdit("24:00:00")
        slurm_layout.addRow("Wall time:", self.cluster_time)
        self.cluster_mem = QLineEdit("8G")
        slurm_layout.addRow("Memory:", self.cluster_mem)
        self.cluster_cpus = QLineEdit("1")
        slurm_layout.addRow("CPUs per task:", self.cluster_cpus)
        self.cluster_job_name = QLineEdit("mmps_imagej")
        slurm_layout.addRow("Job name:", self.cluster_job_name)
        self.cluster_module_load = QLineEdit("module load lang/python/cpython_3.11.3_gcc122 && export JAVA_HOME=$SCRATCH/jdk-21 && export PATH=$SCRATCH/jdk-21/bin:$PATH")
        self.cluster_module_load.setToolTip("Loads Python 3.11 for merge scripts; SNT (Sholl analysis) requires Java 21+.")
        slurm_layout.addRow("Module load:", self.cluster_module_load)
        slurm_group.setLayout(slurm_layout)
        layout.addWidget(slurm_group)

        # --- Generate Button ---
        gen_btn = QPushButton("Generate Scripts")
        gen_btn.setStyleSheet(
            "font-size: 13pt; font-weight: bold; padding: 10px; "
            "background-color: #4CAF50; color: white; border-radius: 5px;"
        )
        gen_btn.clicked.connect(lambda: self._do_export_imagej_cluster(dlg))
        layout.addWidget(gen_btn)

        layout.addStretch()
        scroll.setWidget(container)
        dlg_layout.addWidget(scroll)
        dlg.exec_()

    def _do_export_imagej_cluster(self, dlg):
        """Called from the dialog Generate button."""
        self.export_imagej_cluster_scripts()
        dlg.accept()

    def export_imagej_cluster_scripts(self):
        """Export cluster scripts for running ImageJ analyses on HPC."""
        try:
            self._export_imagej_cluster_scripts_impl()
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.log(f"ERROR generating ImageJ cluster scripts: {e}\n{tb}")
            QMessageBox.critical(self, "Error",
                f"Failed to generate cluster scripts:\n{e}\n\nSee log for details.")

    def _export_imagej_cluster_scripts_impl(self):
        """Internal implementation for generating ImageJ cluster scripts."""
        do_skeleton = self.cluster_do_skeleton.isChecked()
        do_fractal = self.cluster_do_fractal.isChecked()
        do_sholl = self.cluster_do_sholl.isChecked()

        if not do_skeleton and not do_fractal and not do_sholl:
            QMessageBox.warning(self, "Warning", "Select at least one analysis to run.")
            return

        # Validate parameters
        try:
            pixel_size = float(self.cluster_pixel_size.text())
            upscale_factor = int(self.cluster_upscale_factor.text())
            sholl_step = float(self.cluster_sholl_step.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Invalid numeric parameter. Check pixel size, upscale factor, and sholl step.")
            return

        largest_only = self.cluster_largest_only.isChecked()
        fiji_path = self.cluster_fiji_path.text().strip()

        # Warn if Fiji path looks like a directory instead of an executable
        if fiji_path.endswith('Fiji.app') or fiji_path.endswith('Fiji.app/'):
            reply = QMessageBox.warning(self, "Fiji Path Warning",
                "The Fiji path appears to point to the Fiji.app directory, "
                "not the executable.\n\n"
                f"Current: {fiji_path}\n\n"
                "It should typically end with:\n"
                "  .../Fiji.app/ImageJ-linux64\n\n"
                "Continue anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

        partition = self.cluster_partition.text().strip() or "comm_small_day"
        wall_time = self.cluster_time.text().strip() or "24:00:00"
        mem = self.cluster_mem.text().strip() or "8G"
        cpus = self.cluster_cpus.text().strip() or "1"
        job_name = self.cluster_job_name.text().strip() or "mmps_imagej"
        module_load = self.cluster_module_load.text().strip()

        # Ask where to save — scripts go into an "ImageJ plugin" subfolder
        parent_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory for Cluster Scripts")
        if not parent_dir:
            return
        save_dir = os.path.join(parent_dir, "ImageJ plugin")
        os.makedirs(save_dir, exist_ok=True)

        analyses = []
        if do_skeleton:
            analyses.append("skeleton")
        if do_fractal:
            analyses.append("fractal")
        if do_sholl:
            analyses.append("sholl")

        # Build per-image pixel size map from session data
        pixel_size_map = {}
        for img_name, img_data in self.images.items():
            per_img_px = img_data.get('pixel_size')
            if per_img_px is not None:
                # Use image name without extension to match mask filename prefixes
                pixel_size_map[os.path.splitext(img_name)[0]] = per_img_px

        # --- Generate the headless Jython wrapper script ---
        wrapper_script = self._build_imagej_wrapper_script(
            pixel_size, upscale_factor, sholl_step, largest_only, analyses, pixel_size_map
        )
        wrapper_path = os.path.join(save_dir, "mmps_imagej_cluster.py")
        with open(wrapper_path, 'w') as f:
            f.write(wrapper_script)

        # --- Generate the SLURM job script ---
        slurm_script = self._build_slurm_script(
            fiji_path, wrapper_path, partition, wall_time, mem, cpus, job_name, module_load
        )
        slurm_path = os.path.join(save_dir, "submit_imagej.sh")
        with open(slurm_path, 'w') as f:
            f.write(slurm_script)
        os.chmod(slurm_path, 0o755)

        # --- Generate the merge script ---
        merge_script = self._build_merge_script(analyses)
        merge_path = os.path.join(save_dir, "merge_results.py")
        with open(merge_path, 'w') as f:
            f.write(merge_script)

        # --- Generate a README ---
        readme = self._build_cluster_readme(analyses, fiji_path)
        readme_path = os.path.join(save_dir, "CLUSTER_README.txt")
        with open(readme_path, 'w') as f:
            f.write(readme)

        self.log(f"Cluster scripts saved to: {save_dir}")
        self.log(f"  - mmps_imagej_cluster.py  (Jython analysis script - runs per image)")
        self.log(f"  - submit_imagej.sh        (SLURM array job launcher)")
        self.log(f"  - merge_results.py        (combines per-image CSVs)")
        self.log(f"  - CLUSTER_README.txt      (instructions)")
        if pixel_size_map:
            self.log(f"  Per-image pixel sizes embedded for {len(pixel_size_map)} image(s):")
            for name, px in pixel_size_map.items():
                self.log(f"    {name}: {px} µm/px")

        analyses_str = ", ".join(a.title() for a in analyses)

        QMessageBox.information(self, "Cluster Scripts Generated",
            f"Scripts saved to:\n{save_dir}\n\n"
            f"Analyses: {analyses_str}\n\n"
            f"Upload to your cluster with your Fiji installation,\n"
            f"masks/ and somas/ folders, then run:\n\n"
            f"  bash submit_imagej.sh /path/to/mmps_output\n\n"
            f"This submits a SLURM array job (one task per image)\n"
            f"plus a merge job that combines results when done.")

    def _build_imagej_wrapper_script(self, pixel_size, upscale_factor, sholl_step, largest_only, analyses, pixel_size_map=None):
        """Build the Jython script that runs all selected analyses headlessly."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        script = f'''"""
MMPS ImageJ Cluster Analysis Script
Generated by MMPS on {timestamp}

This Jython script runs inside headless Fiji on an HPC cluster.
It performs batch Skeleton, Fractal/Hull, and/or Sholl analyses
on MMPS-exported mask files.

Usage (called by the SLURM job script):
    export MMPS_OUTPUT_DIR=/path/to/mmps_output
    ImageJ-linux64 --java-home $SCRATCH/jdk-21 --ij2 --headless --console --mem=3072m --run mmps_imagej_cluster.py

The MMPS_OUTPUT_DIR should contain:
    masks/   - *_mask.tif files
    somas/   - *_soma.tif files (for Sholl)
"""

from ij import IJ
from ij.measure import Calibration, ResultsTable
from ij.process import ImageProcessor

import os
import sys
import csv
import re
import math
import time


# ============================================================================
# PARAMETERS (embedded from MMPS session)
# ============================================================================

PIXEL_SIZE = {pixel_size}
PIXEL_SIZE_MAP = {repr(pixel_size_map or {})}
UPSCALE_FACTOR = {upscale_factor}
SHOLL_STEP = {sholl_step}
LARGEST_ONLY = {largest_only}
ANALYSES = {analyses}


def getPixelSize(imageName):
    \"\"\"Get per-image pixel size if set, otherwise fall back to global PIXEL_SIZE.\"\"\"
    return PIXEL_SIZE_MAP.get(imageName, PIXEL_SIZE)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def openImageQuiet(path):
    """Open an image without the Bio-Formats options dialog."""
    try:
        from loci.plugins import BF
        import loci.plugins
        ImporterOptions = getattr(loci.plugins, 'in').ImporterOptions
        opts = ImporterOptions()
        opts.setId(path)
        opts.setWindowless(True)
        imps = BF.openImagePlus(opts)
        if imps and len(imps) > 0:
            return imps[0]
        return None
    except Exception:
        return IJ.openImage(path)


def parseMaskInfo(maskFilename):
    """Extract image name, soma ID, and area from mask filename."""
    m = re.match(r'^(.+?)_(soma_\\d+_\\d+(?:_\\d+)?)_area(\\d+)_mask\\.tif$', maskFilename)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return maskFilename, 'unknown', 0


def getCellName(maskFilename):
    """Get a clean cell name from mask filename."""
    name = re.sub(r'_area[3-8]\\d{{2}}_mask\\.tif$', '', maskFilename)
    if name == maskFilename:
        name = re.sub(r'_area\\d+_mask\\.tif$', '', maskFilename)
    if name == maskFilename or name.endswith('_mask'):
        name = re.sub(r'_mask\\.tif$', '', maskFilename)
    return name


def loadMaskMetadata(masksDir):
    \"\"\"Load mask_metadata.csv from the masks directory.
    Returns a dict keyed by (image_name, soma_id) -> row dict.\"\"\"
    meta_path = os.path.join(masksDir, "mask_metadata.csv")
    lookup = {{}}
    if not os.path.isfile(meta_path):
        return lookup
    try:
        with open(meta_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row.get('image_name', ''), row.get('soma_id', ''))
                lookup[key] = row
        print("Loaded mask_metadata.csv: " + str(len(lookup)) + " entries")
    except Exception as e:
        print("WARNING: Could not load mask_metadata.csv: " + str(e))
    return lookup


def getMetaIds(metadata, imgName, somaId):
    \"\"\"Get standard identifier fields from metadata lookup.
    Returns (animal_id, treatment, soma_idx).\"\"\"
    row = metadata.get((imgName, somaId), {{}})
    animal_id = row.get('animal_id', '')
    treatment = row.get('treatment', '')
    soma_idx = row.get('soma_idx', '')
    if not treatment and imgName:
        treatment = imgName.split('_')[0]
    return animal_id, treatment, soma_idx


def filterLargestMasks(maskFiles):
    """Keep only the largest area mask per cell."""
    best = {{}}
    for f in maskFiles:
        imgName, somaId, area = parseMaskInfo(f)
        key = (imgName, somaId)
        if key not in best or area > best[key][0]:
            best[key] = (area, f)
    kept = set(v[1] for v in best.values())
    return [f for f in maskFiles if f in kept]


def formatTime(seconds):
    """Format elapsed time as human-readable string."""
    if seconds < 60:
        return str(int(seconds)) + "s"
    elif seconds < 3600:
        return str(int(seconds // 60)) + "m " + str(int(seconds % 60)).zfill(2) + "s"
    else:
        return str(int(seconds // 3600)) + "h " + str(int((seconds % 3600) // 60)).zfill(2) + "m"


# ============================================================================
# SKELETON ANALYSIS
# ============================================================================

def analyzeSkeleton(maskPath, pixelSize, scaleFactor, outputDirPath):
    """Analyze skeleton of a single mask image."""
    from sc.fiji.analyzeSkeleton import AnalyzeSkeleton_

    mask = openImageQuiet(maskPath)
    if mask is None:
        print("  ERROR: Could not open mask")
        return None

    cal = Calibration(mask)
    cal.pixelWidth = pixelSize
    cal.pixelHeight = pixelSize
    cal.setUnit("micron")
    mask.setCalibration(cal)

    maskProcessor = mask.getProcessor()
    maskWidth = mask.getWidth()
    maskHeight = mask.getHeight()
    maskPixelCount = 0
    for y in range(maskHeight):
        for x in range(maskWidth):
            if maskProcessor.getPixel(x, y) > 0:
                maskPixelCount += 1
    maskArea = maskPixelCount * (pixelSize * pixelSize)

    IJ.setThreshold(mask, 1, 255)
    IJ.run(mask, "Set Measurements...", "area perimeter shape redirect=None decimal=3")
    IJ.run(mask, "Measure", "")
    rt = ResultsTable.getResultsTable()
    maskPerimeter = rt.getValue("Perim.", 0)
    maskCircularity = rt.getValue("Circ.", 0)
    maskAR = rt.getValue("AR", 0)
    maskRound = rt.getValue("Round", 0)
    maskSolidity = rt.getValue("Solidity", 0)
    rt.reset()

    skel = mask.duplicate()
    if scaleFactor > 1:
        newWidth = int(mask.getWidth() * scaleFactor)
        newHeight = int(mask.getHeight() * scaleFactor)
        IJ.run(skel, "Size...", "width=" + str(newWidth) + " height=" + str(newHeight) + " interpolation=None")
        scaledPixelSize = pixelSize / float(scaleFactor)
        scaledCal = Calibration(skel)
        scaledCal.pixelWidth = scaledPixelSize
        scaledCal.pixelHeight = scaledPixelSize
        scaledCal.setUnit("micron")
        skel.setCalibration(scaledCal)

    IJ.setThreshold(skel, 1, 255)
    IJ.run(skel, "Convert to Mask", "")
    IJ.run(skel, "Skeletonize (2D/3D)", "")
    IJ.run(skel, "Select None", "")

    baseName = os.path.basename(maskPath)
    cellName = getCellName(baseName)

    skelPath = os.path.join(outputDirPath, cellName + "_skeleton.tif")
    IJ.save(skel, skelPath)

    analyzer = AnalyzeSkeleton_()
    analyzer.setup("", skel)
    result = analyzer.run(AnalyzeSkeleton_.SHORTEST_BRANCH, True, True, None, True, False)

    branches = result.getBranches()
    junctions = result.getJunctions()
    endPoints = result.getEndPoints()
    junctionVoxels = result.getJunctionVoxels()
    slabVoxels = result.getSlabs()
    triplePoints = result.getTriples()
    quadruplePoints = result.getQuadruples()
    maxBranchLength = result.getMaximumBranchLength()
    shortestPathList = result.getShortestPathList()

    numBranches = int(branches[0]) if len(branches) > 0 else 0
    numSlabVoxels = int(slabVoxels[0]) if len(slabVoxels) > 0 else 0

    try:
        avgBranchLengthArray = result.getAverageBranchLength()
        if avgBranchLengthArray is not None and len(avgBranchLengthArray) > 0:
            avgBranchLength = float(avgBranchLengthArray[0])
        else:
            avgBranchLength = 0.0
    except:
        if numBranches > 0 and numSlabVoxels > 0:
            effectivePx = pixelSize / float(scaleFactor) if scaleFactor > 1 else pixelSize
            avgBranchLength = (numSlabVoxels * effectivePx) / float(numBranches)
        else:
            avgBranchLength = 0.0

    longestShortestPath = 0.0
    try:
        if shortestPathList and len(shortestPathList) > 0:
            if hasattr(shortestPathList[0], '__len__') and len(shortestPathList[0]) > 0:
                longestShortestPath = float(max(shortestPathList[0]))
            elif shortestPathList[0]:
                longestShortestPath = float(shortestPathList[0])
    except:
        longestShortestPath = 0.0

    if avgBranchLength > 0 and numBranches > 0:
        totalSkeletonLength = avgBranchLength * numBranches
    else:
        effectivePx = pixelSize / float(scaleFactor) if scaleFactor > 1 else pixelSize
        totalSkeletonLength = numSlabVoxels * effectivePx

    skelProcessor = skel.getProcessor()
    skelWidth = skel.getWidth()
    skelHeight = skel.getHeight()
    skelPixelCount = 0
    for y in range(skelHeight):
        for x in range(skelWidth):
            if skelProcessor.getPixel(x, y) > 0:
                skelPixelCount += 1
    effectivePx = pixelSize / float(scaleFactor) if scaleFactor > 1 else pixelSize
    skeletonArea = skelPixelCount * (effectivePx * effectivePx)

    imgName, somaId, areaUm2 = parseMaskInfo(os.path.basename(maskPath))

    metrics = {{
        'image_name': imgName,
        'animal_id': '',
        'treatment': '',
        'soma_id': somaId,
        'soma_idx': '',
        'area_um2': areaUm2,
        'mask_file': os.path.basename(maskPath),
        'cell_name': cellName,
        'skeleton_file': os.path.basename(skelPath),
        'mask_area_um2': maskArea,
        'mask_perimeter_um': maskPerimeter,
        'mask_circularity': maskCircularity,
        'mask_aspect_ratio': maskAR,
        'mask_roundness': maskRound,
        'mask_solidity': maskSolidity,
        'num_branches': numBranches,
        'num_junctions': int(junctions[0]) if len(junctions) > 0 else 0,
        'num_end_points': int(endPoints[0]) if len(endPoints) > 0 else 0,
        'num_junction_voxels': int(junctionVoxels[0]) if len(junctionVoxels) > 0 else 0,
        'num_slab_voxels': numSlabVoxels,
        'num_triple_points': int(triplePoints[0]) if len(triplePoints) > 0 else 0,
        'num_quadruple_points': int(quadruplePoints[0]) if len(quadruplePoints) > 0 else 0,
        'max_branch_length_um': float(maxBranchLength[0]) if len(maxBranchLength) > 0 else 0,
        'avg_branch_length_um': avgBranchLength,
        'longest_shortest_path_um': longestShortestPath,
        'total_skeleton_length_um': totalSkeletonLength,
        'skeleton_area_um2': skeletonArea,
        'branching_density': skeletonArea / maskArea if maskArea > 0 else 0,
    }}

    mask.close()
    skel.close()
    return metrics


def runSkeletonBatch(masksDir, outputDir, maskFiles, imageName="all", pixelSize=None, metadata=None):
    """Run skeleton analysis on all mask files."""
    if pixelSize is None:
        pixelSize = PIXEL_SIZE
    if metadata is None:
        metadata = {{}}
    print("=" * 60)
    print("SKELETON ANALYSIS - " + imageName)
    print("=" * 60)

    skelDir = os.path.join(outputDir, "skeleton_results")
    if not os.path.exists(skelDir):
        os.makedirs(skelDir)

    allResults = []
    batchStart = time.time()
    total = len(maskFiles)

    for idx, maskFile in enumerate(maskFiles):
        maskPath = os.path.join(masksDir, maskFile)
        elapsed = time.time() - batchStart
        if idx > 0:
            eta = formatTime(elapsed / idx * (total - idx))
        else:
            eta = "estimating..."
        print("[" + str(idx + 1) + "/" + str(total) + "] " + maskFile + "  ETA: " + eta)

        metrics = analyzeSkeleton(maskPath, pixelSize, UPSCALE_FACTOR, skelDir)
        if metrics is not None:
            # Populate standard identifiers from metadata
            imgN, somaI, _ = parseMaskInfo(maskFile)
            aid, treat, sidx = getMetaIds(metadata, imgN, somaI)
            metrics['animal_id'] = aid
            metrics['treatment'] = treat
            metrics['soma_idx'] = sidx
            allResults.append(metrics)

    if allResults:
        outputPath = os.path.join(skelDir, "Skeleton_Results_" + imageName + ".csv")
        stdCols = ['image_name', 'animal_id', 'treatment', 'soma_id', 'soma_idx', 'area_um2']
        idCols = ['cell_name', 'mask_file', 'skeleton_file']
        maskCols = ['mask_area_um2', 'mask_perimeter_um', 'mask_circularity',
                    'mask_aspect_ratio', 'mask_roundness', 'mask_solidity']
        skelCols = ['num_branches', 'num_junctions', 'num_end_points',
                    'num_junction_voxels', 'num_slab_voxels', 'num_triple_points',
                    'num_quadruple_points', 'max_branch_length_um',
                    'avg_branch_length_um', 'longest_shortest_path_um',
                    'total_skeleton_length_um', 'skeleton_area_um2', 'branching_density']
        columns = stdCols + idCols + maskCols + skelCols
        f = open(outputPath, 'wb')
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(allResults)
        f.close()
        print("Skeleton results saved: " + outputPath)

    print("Skeleton analysis done: " + str(len(allResults)) + " masks in " + formatTime(time.time() - batchStart))
    return allResults


# ============================================================================
# FRACTAL / CONVEX HULL ANALYSIS
# ============================================================================

def runFractalAnalysis(maskPath, pixelSize):
    """Box-counting fractal dimension and lacunarity."""
    imp = openImageQuiet(maskPath)
    if imp is None:
        return None

    ip = imp.getProcessor()
    w = imp.getWidth()
    h = imp.getHeight()

    foreground = []
    totalFG = 0
    for y in range(h):
        row = []
        for x in range(w):
            val = ip.getPixel(x, y) > 0
            row.append(val)
            if val:
                totalFG += 1
        foreground.append(row)
    imp.close()

    if totalFG == 0:
        return None

    maxDim = min(w, h)
    boxSizes = []
    s = 2
    while s <= maxDim // 2:
        boxSizes.append(s)
        s *= 2
    extra = []
    for i in range(len(boxSizes) - 1):
        mid = int(round((boxSizes[i] + boxSizes[i + 1]) / 2.0))
        if mid not in boxSizes:
            extra.append(mid)
    boxSizes = sorted(set(boxSizes + extra))

    if len(boxSizes) < 3:
        return None

    logInvS = []
    logN = []
    lacunarities = []

    for s in boxSizes:
        nBoxes = 0
        counts = []
        for by in range(0, h, s):
            for bx in range(0, w, s):
                cnt = 0
                for dy in range(min(s, h - by)):
                    for dx in range(min(s, w - bx)):
                        if foreground[by + dy][bx + dx]:
                            cnt += 1
                counts.append(cnt)
                if cnt > 0:
                    nBoxes += 1
        if nBoxes == 0:
            continue
        logInvS.append(math.log(1.0 / s))
        logN.append(math.log(nBoxes))
        n = len(counts)
        meanC = sum(counts) / float(n)
        if meanC > 0:
            varC = sum((c - meanC) ** 2 for c in counts) / float(n)
            lac = varC / (meanC ** 2) + 1.0
        else:
            lac = float('nan')
        lacunarities.append(lac)

    if len(logInvS) < 3:
        return None

    n = len(logInvS)
    sx = sum(logInvS)
    sy = sum(logN)
    sxx = sum(x * x for x in logInvS)
    sxy = sum(logInvS[i] * logN[i] for i in range(n))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        fractalDim = float('nan')
        rSquared = float('nan')
    else:
        fractalDim = (n * sxy - sx * sy) / denom
        intercept = (sy - fractalDim * sx) / n
        yMean = sy / n
        ssTot = sum((y - yMean) ** 2 for y in logN)
        ssRes = sum((logN[i] - (fractalDim * logInvS[i] + intercept)) ** 2 for i in range(n))
        rSquared = 1.0 - ssRes / ssTot if ssTot > 0 else float('nan')

    validLac = [l for l in lacunarities if l == l]
    avgLacunarity = sum(validLac) / len(validLac) if validLac else float('nan')
    fgArea = totalFG * (pixelSize ** 2)

    metrics = {{
        'fractal_dimension': round(fractalDim, 6) if fractalDim == fractalDim else 'NaN',
        'fractal_r_squared': round(rSquared, 6) if rSquared == rSquared else 'NaN',
        'fractal_lacunarity_mean': round(avgLacunarity, 6) if avgLacunarity == avgLacunarity else 'NaN',
        'fractal_num_scales': len(boxSizes),
        'fractal_foreground_pixels': totalFG,
        'fractal_foreground_area_um2': round(fgArea, 4),
    }}
    if len(lacunarities) >= 2:
        metrics['fractal_lacunarity_small'] = round(lacunarities[0], 6) if lacunarities[0] == lacunarities[0] else 'NaN'
        metrics['fractal_lacunarity_large'] = round(lacunarities[-1], 6) if lacunarities[-1] == lacunarities[-1] else 'NaN'
    return metrics


def _grahamScanHull(points):
    """Graham scan convex hull."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    points = sorted(set(points))
    if len(points) <= 1:
        return list(points)
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def runConvexHullAnalysis(maskPath, pixelSize):
    """Convex hull metrics for a binary cell mask."""
    from ij.plugin.filter import ThresholdToSelection

    imp = openImageQuiet(maskPath)
    if imp is None:
        return None

    ip = imp.getProcessor()
    w = imp.getWidth()
    h = imp.getHeight()

    totalFG = 0
    boundary = []
    for y in range(h):
        for x in range(w):
            if ip.getPixel(x, y) > 0:
                totalFG += 1
                isBoundary = False
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        isBoundary = True
                        break
                    if ip.getPixel(nx, ny) == 0:
                        isBoundary = True
                        break
                if isBoundary:
                    boundary.append((x, y))

    if totalFG == 0:
        imp.close()
        return None

    hullX = None
    hullY = None
    try:
        ip.setThreshold(1, 255, ip.NO_LUT_UPDATE)
        roi = ThresholdToSelection.run(imp)
        if roi is not None:
            hullPoly = roi.getConvexHull()
            if hullPoly is not None:
                nPts = hullPoly.npoints
                if nPts >= 3:
                    hullX = [float(hullPoly.xpoints[i]) for i in range(nPts)]
                    hullY = [float(hullPoly.ypoints[i]) for i in range(nPts)]
    except Exception:
        pass

    if hullX is None:
        if len(boundary) < 3:
            imp.close()
            return None
        hullVerts = _grahamScanHull(boundary)
        if len(hullVerts) < 3:
            imp.close()
            return None
        hullX = [float(v[0]) for v in hullVerts]
        hullY = [float(v[1]) for v in hullVerts]

    imp.close()
    nPoints = len(hullX)

    hullArea = 0.0
    for i in range(nPoints):
        j = (i + 1) % nPoints
        hullArea += hullX[i] * hullY[j]
        hullArea -= hullX[j] * hullY[i]
    hullArea = abs(hullArea) / 2.0
    if hullArea == 0:
        return None

    hullPerimeter = 0.0
    for i in range(nPoints):
        j = (i + 1) % nPoints
        dx = hullX[j] - hullX[i]
        dy = hullY[j] - hullY[i]
        hullPerimeter += math.sqrt(dx * dx + dy * dy)

    hullCircularity = 4.0 * math.pi * hullArea / (hullPerimeter * hullPerimeter) if hullPerimeter > 0 else float('nan')
    density = totalFG / hullArea

    maxSpan = 0.0
    maxI, maxJ = 0, 0
    for i in range(nPoints):
        for j in range(i + 1, nPoints):
            dx = hullX[j] - hullX[i]
            dy = hullY[j] - hullY[i]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > maxSpan:
                maxSpan = dist
                maxI = i
                maxJ = j

    if maxSpan > 0:
        axDx = hullX[maxJ] - hullX[maxI]
        axDy = hullY[maxJ] - hullY[maxI]
        axLen = math.sqrt(axDx * axDx + axDy * axDy)
        perpX = -axDy / axLen
        perpY = axDx / axLen
        minProj = float('inf')
        maxProj = float('-inf')
        for i in range(nPoints):
            proj = (hullX[i] - hullX[maxI]) * perpX + (hullY[i] - hullY[maxI]) * perpY
            if proj < minProj:
                minProj = proj
            if proj > maxProj:
                maxProj = proj
        perpWidth = maxProj - minProj
        spanRatio = maxSpan / perpWidth if perpWidth > 0 else float('nan')
    else:
        spanRatio = float('nan')

    return {{
        'hull_area_px': round(hullArea, 4),
        'hull_area_um2': round(hullArea * pixelSize * pixelSize, 4),
        'hull_perimeter_px': round(hullPerimeter, 4),
        'hull_perimeter_um': round(hullPerimeter * pixelSize, 4),
        'hull_circularity': round(hullCircularity, 6) if hullCircularity == hullCircularity else 'NaN',
        'hull_density': round(density, 6),
        'hull_max_span_px': round(maxSpan, 4),
        'hull_max_span_um': round(maxSpan * pixelSize, 4),
        'hull_span_ratio': round(spanRatio, 6) if spanRatio == spanRatio else 'NaN',
    }}


def runFractalBatch(masksDir, outputDir, maskFiles, imageName="all", pixelSize=None, metadata=None):
    """Run fractal + hull analysis on all mask files."""
    if pixelSize is None:
        pixelSize = PIXEL_SIZE
    if metadata is None:
        metadata = {{}}
    print("=" * 60)
    print("FRACTAL / CONVEX HULL ANALYSIS - " + imageName)
    print("=" * 60)

    resultsDir = os.path.join(outputDir, "fractal_results")
    if not os.path.exists(resultsDir):
        os.makedirs(resultsDir)

    allResults = []
    batchStart = time.time()
    total = len(maskFiles)

    for idx, maskFile in enumerate(maskFiles):
        maskPath = os.path.join(masksDir, maskFile)
        imgName, somaId, areaUm2 = parseMaskInfo(maskFile)
        cellName = getCellName(maskFile)
        aid, treat, sidx = getMetaIds(metadata, imgName, somaId)
        elapsed = time.time() - batchStart
        if idx > 0:
            eta = formatTime(elapsed / idx * (total - idx))
        else:
            eta = "estimating..."
        print("[" + str(idx + 1) + "/" + str(total) + "] " + maskFile + "  ETA: " + eta)

        try:
            fracMetrics = runFractalAnalysis(maskPath, pixelSize)
            hullMetrics = runConvexHullAnalysis(maskPath, pixelSize)
            if fracMetrics is not None:
                row = {{
                    'image_name': imgName,
                    'animal_id': aid,
                    'treatment': treat,
                    'soma_id': somaId,
                    'soma_idx': sidx,
                    'area_um2': areaUm2,
                    'cell_name': cellName,
                    'mask_file': maskFile,
                }}
                row.update(fracMetrics)
                if hullMetrics is not None:
                    row.update(hullMetrics)
                allResults.append(row)
                print("  OK (D=" + str(fracMetrics['fractal_dimension']) + ")")
            else:
                print("  FAILED (empty mask or too few scales)")
        except Exception as e:
            print("  ERROR: " + str(e))

    if allResults:
        outputPath = os.path.join(resultsDir, "Fractal_Results_" + imageName + ".csv")
        stdCols = ['image_name', 'animal_id', 'treatment', 'soma_id', 'soma_idx', 'area_um2']
        idCols = ['cell_name', 'mask_file']
        fracCols = ['fractal_dimension', 'fractal_r_squared',
                    'fractal_lacunarity_mean', 'fractal_lacunarity_small',
                    'fractal_lacunarity_large', 'fractal_num_scales',
                    'fractal_foreground_pixels', 'fractal_foreground_area_um2']
        hullCols = ['hull_area_px', 'hull_area_um2', 'hull_perimeter_px', 'hull_perimeter_um',
                    'hull_circularity', 'hull_density', 'hull_max_span_px', 'hull_max_span_um',
                    'hull_span_ratio']
        columns = stdCols + idCols + fracCols + hullCols
        f = open(outputPath, 'wb')
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(allResults)
        f.close()
        print("Fractal results saved: " + outputPath)

    print("Fractal analysis done: " + str(len(allResults)) + " masks in " + formatTime(time.time() - batchStart))
    return allResults


# ============================================================================
# SHOLL ANALYSIS
# ============================================================================

def getSomaCentroid(somaPath):
    """Calculate centroid from a binary soma mask TIFF."""
    somaImp = openImageQuiet(somaPath)
    if somaImp is None:
        return None
    ip = somaImp.getProcessor()
    # Binarize: anything >= 1 becomes 255 (native Java, instant)
    ip.threshold(1)
    stats = ip.getStatistics()
    somaImp.close()
    if stats.mean == 0:
        return None
    # xCenterOfMass/yCenterOfMass are intensity-weighted; on a binary
    # image (0/255) this equals the geometric centroid of foreground pixels.
    return (int(round(stats.xCenterOfMass)), int(round(stats.yCenterOfMass)))


def getSomaRadius(somaPath, centroid, pixelSize):
    """Estimate soma radius in calibrated units from the soma mask."""
    somaImp = openImageQuiet(somaPath)
    if somaImp is None or centroid is None:
        return 0.0, 0.0
    ip = somaImp.getProcessor()
    # Use native Java histogram to count foreground pixels (any value > 0)
    hist = ip.getHistogram()
    count = sum(hist[1:])  # all non-zero bins
    somaImp.close()
    areaUm2 = count * (pixelSize ** 2)
    radiusUm = math.sqrt(areaUm2 / math.pi) if count > 0 else 0.0
    return radiusUm, areaUm2


def findSomaFile(somasDir, maskFilename):
    """Find the soma file corresponding to a mask file.
    Tries exact match first, then falls back to glob-based search."""
    base = re.sub(r'_area\\d+_mask\\.tif$', '', maskFilename)
    somaFilename = base + '_soma.tif'
    somaPath = os.path.join(somasDir, somaFilename)
    if os.path.exists(somaPath):
        return somaPath
    # Fallback: search for any soma file with the same soma ID
    info = parseMaskInfo(maskFilename)
    imgName = info[0]
    somaId = info[1]
    if somaId != 'unknown':
        for f in os.listdir(somasDir):
            if f.endswith('_soma.tif') and somaId in f:
                return os.path.join(somasDir, f)
    # Last resort: list available soma files for debugging
    try:
        available = [f for f in os.listdir(somasDir) if f.endswith('_soma.tif')]
        if available:
            print("  Available soma files: " + str(available[:5]))
        else:
            print("  WARNING: somas/ directory is EMPTY - no soma files found")
    except Exception:
        pass
    return None


_sholl_debug_count = [0]  # how many masks have had debug output

def analyzeOneSholl(maskPath, centroid, startRad, stepSize, pixelSize, saveLoc, maskName, somaAreaUm2):
    """Run Sholl analysis on one cell mask."""
    from sc.fiji.snt.analysis.sholl import Profile, ShollUtils
    from sc.fiji.snt.analysis.sholl.math import LinearProfileStats
    from sc.fiji.snt.analysis.sholl.math import NormalizedProfileStats
    from sc.fiji.snt.analysis.sholl.math import ShollStats
    from sc.fiji.snt.analysis.sholl.parsers import ImageParser2D

    debug = _sholl_debug_count[0] < 5  # verbose debug for first 5 masks

    cellName = os.path.splitext(maskName)[0]

    imp = openImageQuiet(maskPath)
    if imp is None:
        if debug:
            print("  [DEBUG] openImageQuiet returned None!")
        return None

    if debug:
        print("  [DEBUG] Image: " + str(imp.getWidth()) + "x" + str(imp.getHeight())
              + " type=" + str(imp.getType()) + " bitDepth=" + str(imp.getBitDepth()))
        ip = imp.getProcessor()
        hist = ip.getHistogram()
        fg_pixels = sum(hist[1:])
        print("  [DEBUG] Foreground pixels (>0): " + str(fg_pixels) + " / " + str(imp.getWidth() * imp.getHeight()))
        print("  [DEBUG] Min=" + str(ip.getMin()) + " Max=" + str(ip.getMax()))

    cal = Calibration(imp)
    cal.pixelWidth = pixelSize
    cal.pixelHeight = pixelSize
    cal.setUnit("um")
    imp.setCalibration(cal)

    # Set threshold directly on processor with NO_LUT_UPDATE for headless compatibility.
    # IJ.setThreshold() uses RED_LUT which can interfere in headless mode.
    imp.getProcessor().setThreshold(1, 255, ImageProcessor.NO_LUT_UPDATE)

    if debug:
        print("  [DEBUG] Threshold set: minThreshold=" + str(imp.getProcessor().getMinThreshold())
              + " maxThreshold=" + str(imp.getProcessor().getMaxThreshold()))

    parser = ImageParser2D(imp)
    parser.setRadiiSpan(0, ImageParser2D.MEAN)
    parser.setPosition(1, 1, 1)

    cx, cy = centroid
    # setCenter expects calibrated (um) coordinates, not pixel coordinates
    cx_cal = cx * pixelSize
    cy_cal = cy * pixelSize
    parser.setCenter(cx_cal, cy_cal)

    if debug:
        print("  [DEBUG] Center pixel: (" + str(cx) + ", " + str(cy)
              + ") -> calibrated: (" + str(round(cx_cal, 2)) + ", " + str(round(cy_cal, 2)) + ") um")
        # Check if center pixel is foreground
        pxVal = imp.getProcessor().getPixel(cx, cy)
        print("  [DEBUG] Pixel value at center: " + str(pxVal))

    # stepSize=0 means "continuous" (pixel-level); SNT hangs with step=0,
    # so convert to 1 pixel in calibrated units.
    effectiveStep = stepSize if stepSize > 0 else pixelSize
    # Use mask bounding box for end radius instead of maxPossibleRadius()
    # which extends to image corners and freezes the parser.
    from ij.plugin.filter import ThresholdToSelection
    roi = ThresholdToSelection.run(imp)
    if roi is not None:
        bounds = roi.getBounds()
        if debug:
            print("  [DEBUG] ROI bounds: x=" + str(bounds.x) + " y=" + str(bounds.y)
                  + " w=" + str(bounds.width) + " h=" + str(bounds.height))
        corners = [(bounds.x, bounds.y), (bounds.x + bounds.width, bounds.y),
                    (bounds.x, bounds.y + bounds.height), (bounds.x + bounds.width, bounds.y + bounds.height)]
        endRad = max(((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5 for bx, by in corners) * pixelSize
    else:
        if debug:
            print("  [DEBUG] ThresholdToSelection returned None! Threshold may not be set properly.")
        endRad = parser.maxPossibleRadius()

    numRadii = int((endRad - startRad) / effectiveStep) if effectiveStep > 0 else 0
    print("  Radii: start=" + str(round(startRad, 2)) + " end=" + str(round(endRad, 2))
          + " step=" + str(round(effectiveStep, 3)) + " (" + str(numRadii) + " samples)")

    parser.setRadii(startRad, effectiveStep, endRad)
    parser.setHemiShells('none')

    # Run parser in a thread with timeout so one slow mask doesn't block the whole job
    from java.lang import Thread, Runnable
    class _ParserRunner(Runnable):
        def __init__(self, p):
            self.parser = p
            self.finished = False
        def run(self):
            self.parser.parse()
            self.finished = True

    PARSE_TIMEOUT_SEC = 600  # 10 minutes per mask
    runner = _ParserRunner(parser)
    parseThread = Thread(runner)
    parseThread.setDaemon(True)
    parseThread.start()
    parseThread.join(long(PARSE_TIMEOUT_SEC * 1000))

    if not runner.finished:
        print("  TIMEOUT after " + str(PARSE_TIMEOUT_SEC) + "s (end=" + str(round(endRad, 1))
              + "um, " + str(numRadii) + " samples) - moving on")
        imp.close()
        _sholl_debug_count[0] += 1
        return 'TIMEOUT'

    if debug:
        print("  [DEBUG] parser.successful() = " + str(parser.successful()))

    if not parser.successful():
        if debug:
            print("  [DEBUG] Parser FAILED. Trying to get error info...")
            try:
                profile = parser.getProfile()
                if profile is not None:
                    print("  [DEBUG] Profile exists but isEmpty=" + str(profile.isEmpty())
                          + " size=" + str(profile.size()))
                else:
                    print("  [DEBUG] Profile is None")
            except Exception as e:
                print("  [DEBUG] Error getting profile: " + str(e))
        _sholl_debug_count[0] += 1
        imp.close()
        return None

    profile = parser.getProfile()
    if profile.isEmpty():
        if debug:
            print("  [DEBUG] Profile is empty after successful parse!")
        _sholl_debug_count[0] += 1
        imp.close()
        return None

    _sholl_debug_count[0] += 1

    profile.trimZeroCounts()
    lStats = LinearProfileStats(profile)
    nStatsSemiLog = NormalizedProfileStats(profile, ShollStats.AREA, 128)
    nStatsLogLog = NormalizedProfileStats(profile, ShollStats.AREA, 256)

    cal = Calibration(imp)

    maskMetrics = {{
        'mask_name': maskName,
        'soma_area_um2': somaAreaUm2,
        'primary_branches': lStats.getPrimaryBranches(False),
        'intersecting_radii': lStats.getIntersectingRadii(False),
        'sum_of_intersections': lStats.getSum(False),
        'mean_of_intersections': lStats.getMean(False),
        'median_of_intersections': lStats.getMedian(False),
        'skewness_sampled': lStats.getSkewness(False),
        'kurtosis_sampled': lStats.getKurtosis(False),
        'max_intersections': lStats.getMax(False),
        'max_intersection_radius': lStats.getXvalues()[lStats.getIndexOfInters(False, float(lStats.getMax(False)))],
        'ramification_index_sampled': lStats.getRamificationIndex(False),
        'centroid_radius': lStats.getCentroid(False).rawX(cal),
        'centroid_value': lStats.getCentroid(False).rawY(cal),
        'enclosing_radius': lStats.getEnclosingRadius(False),
        'regression_coeff_semi_log': nStatsSemiLog.getSlope(),
        'regression_coeff_log_log': nStatsLogLog.getSlope(),
        'regression_intercept_semi_log': nStatsSemiLog.getIntercept(),
        'regression_intercept_log_log': nStatsLogLog.getIntercept(),
    }}

    # P10-P90
    nStatsSemiLog.restrictRegToPercentile(10, 90)
    nStatsLogLog.restrictRegToPercentile(10, 90)
    maskMetrics['regression_coeff_semi_log_p10_p90'] = nStatsSemiLog.getSlope()
    maskMetrics['regression_coeff_log_log_p10_p90'] = nStatsLogLog.getSlope()
    maskMetrics['regression_intercept_semi_log_p10_p90'] = nStatsSemiLog.getIntercept()
    maskMetrics['regression_intercept_log_log_p10_p90'] = nStatsLogLog.getIntercept()

    # Polynomial fit
    bestDegree = lStats.findBestFit(1, 30, 0.7, 0.05)
    if bestDegree != -1:
        lStats.fitPolynomial(bestDegree)
        try:
            maskMetrics['kurtosis_fit'] = lStats.getKurtosis(True)
        except:
            maskMetrics['kurtosis_fit'] = 'NaN'
        try:
            maskMetrics['ramification_index_fit'] = lStats.getRamificationIndex(True)
        except:
            maskMetrics['ramification_index_fit'] = 'NaN'
        try:
            maskMetrics['mean_value'] = lStats.getMean(True)
        except:
            maskMetrics['mean_value'] = 'NaN'
        maskMetrics['polynomial_degree'] = bestDegree

        critVals = []
        critRadii = []
        try:
            trial = lStats.getPolynomialMaxima(0.0, 100.0, 50.0)
            if trial is not None:
                for curr in trial.toArray():
                    critVals.append(curr.rawY(cal))
                    critRadii.append(curr.rawX(cal))
        except:
            pass
        maskMetrics['critical_value'] = sum(critVals) / len(critVals) if critVals else 'NaN'
        maskMetrics['critical_radius'] = sum(critRadii) / len(critRadii) if critRadii else 'NaN'

    imp.close()
    return maskMetrics


def runShollBatch(masksDir, somasDir, outputDir, maskFiles, imageName="all", pixelSize=None, metadata=None):
    """Run Sholl analysis on all mask files."""
    if pixelSize is None:
        pixelSize = PIXEL_SIZE
    if metadata is None:
        metadata = {{}}
    print("=" * 60)
    print("SHOLL ANALYSIS - " + imageName)
    print("=" * 60)

    shollDir = os.path.join(outputDir, "sholl_results")
    if not os.path.exists(shollDir):
        os.makedirs(shollDir)

    allResults = []
    processed = 0
    skipped = 0
    timedOut = 0
    batchStart = time.time()
    total = len(maskFiles)

    # Write results incrementally so completed masks survive job kills
    combinedPath = os.path.join(shollDir, "Sholl_Results_" + imageName + ".csv")
    csvFile = None
    csvWriter = None
    csvKeys = None

    for idx, maskFile in enumerate(maskFiles):
        maskPath = os.path.join(masksDir, maskFile)
        imgName, somaId, areaUm2 = parseMaskInfo(maskFile)

        elapsed = time.time() - batchStart
        if idx > 0:
            eta = formatTime(elapsed / idx * (total - idx))
        else:
            eta = "estimating..."
        print("[" + str(idx + 1) + "/" + str(total) + "] " + maskFile + "  ETA: " + eta)

        somaPath = findSomaFile(somasDir, maskFile)
        if somaPath is None:
            print("  WARNING: No soma file found - skipping")
            skipped += 1
            continue

        print("  Finding soma centroid...")
        centroid = getSomaCentroid(somaPath)
        if centroid is None:
            print("  WARNING: Could not calculate centroid - skipping")
            skipped += 1
            continue
        print("  Centroid: " + str(centroid))

        somaResult = getSomaRadius(somaPath, centroid, pixelSize)
        somaRadiusUm = somaResult[0]
        somaAreaUm2 = somaResult[1]
        startRad = somaRadiusUm
        print("  Start radius: " + str(round(startRad, 2)) + " um, area: " + str(round(somaAreaUm2, 1)) + " um2")

        try:
            print("  Running Sholl parser...")
            t0 = time.time()
            metrics = analyzeOneSholl(
                maskPath, centroid, startRad, SHOLL_STEP, pixelSize,
                shollDir, maskFile, somaAreaUm2
            )
            dt = time.time() - t0
            if metrics is not None:
                aid, treat, sidx = getMetaIds(metadata, imgName, somaId)
                metrics['image_name'] = imgName
                metrics['animal_id'] = aid
                metrics['treatment'] = treat
                metrics['soma_id'] = somaId
                metrics['soma_idx'] = sidx
                metrics['area_um2'] = areaUm2
                metrics['centroid_x_px'] = centroid[0]
                metrics['centroid_y_px'] = centroid[1]
                metrics['start_radius_um'] = startRad
                allResults.append(metrics)
                processed += 1
                print("  Done (" + str(round(dt, 1)) + "s)")

                # Write to CSV immediately (create header on first result)
                if csvFile is None:
                    csvKeys = list(metrics.keys())
                    csvFile = open(combinedPath, 'wb')
                    csvWriter = csv.writer(csvFile)
                    csvWriter.writerow(csvKeys)
                else:
                    # Add any new keys from this result
                    for k in metrics.keys():
                        if k not in csvKeys:
                            csvKeys.append(k)
                csvWriter.writerow([metrics.get(k, '') for k in csvKeys])
                csvFile.flush()
            elif metrics == 'TIMEOUT':
                timedOut += 1
            else:
                skipped += 1
                print("  Skipped - no results (" + str(round(dt, 1)) + "s)")
        except Exception as e:
            skipped += 1
            print("  ERROR (" + str(round(time.time() - t0, 1)) + "s): " + str(e))

    if csvFile is not None:
        csvFile.close()
        print("Sholl results saved: " + combinedPath)

    summary = "Sholl analysis done: " + str(processed) + " processed, " + str(skipped) + " skipped"
    if timedOut > 0:
        summary += ", " + str(timedOut) + " timed out"
    summary += " in " + formatTime(time.time() - batchStart)
    if skipped > 0 and processed == 0 and timedOut == 0:
        print("WARNING: ALL " + str(skipped) + " masks were skipped!")
        print("  Check that soma files exist and match mask naming convention.")
        print("  Expected: <ImageName>_<soma_Y_X>_soma.tif in the somas/ folder")
    print(summary)
    return allResults


# ============================================================================
# IMAGE DISCOVERY
# ============================================================================

def discoverImages(masksDir):
    """Discover unique image names from mask filenames.
    Returns a sorted list of unique image name prefixes."""
    images = set()
    for f in os.listdir(masksDir):
        if not f.endswith('_mask.tif') or f.startswith('.'):
            continue
        m = re.match(r'^(.+?)_(soma_\\d+_\\d+(?:_\\d+)?)_area(\\d+)_mask\\.tif$', f)
        if m:
            images.add(m.group(1))
    return sorted(images)


def getMasksForImage(allMaskFiles, imageName):
    """Filter mask files belonging to a specific image."""
    prefix = imageName + "_soma_"
    return [f for f in allMaskFiles if f.startswith(prefix)]


# ============================================================================
# MAIN
# ============================================================================

# Get parameters from environment variables (set by SLURM --wrap)
mmps_output_dir = os.environ.get('MMPS_OUTPUT_DIR', '')
if not mmps_output_dir:
    print("ERROR: MMPS_OUTPUT_DIR environment variable not set.")
    sys.exit(1)

image_index = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))

masksDir = os.path.join(mmps_output_dir, "masks")
somasDir = os.path.join(mmps_output_dir, "somas")

if not os.path.isdir(masksDir):
    print("ERROR: No 'masks' folder found in: " + mmps_output_dir)
    sys.exit(1)

# Discover all unique images and select this task's image
allImages = discoverImages(masksDir)
if len(allImages) == 0:
    print("ERROR: No mask files found in: " + masksDir)
    sys.exit(1)

if image_index < 0 or image_index >= len(allImages):
    print("ERROR: image_index " + str(image_index) + " out of range (0-" + str(len(allImages) - 1) + ")")
    sys.exit(1)

imageName = allImages[image_index]

# Collect mask files for this image only
allMaskFiles = sorted([f for f in os.listdir(masksDir)
                       if f.endswith('_mask.tif') and not f.startswith('.')])
maskFiles = getMasksForImage(allMaskFiles, imageName)

if LARGEST_ONLY:
    totalBefore = len(maskFiles)
    maskFiles = filterLargestMasks(maskFiles)
    print("Largest-only filter: " + str(totalBefore) + " -> " + str(len(maskFiles)) + " masks")

imgPixelSize = getPixelSize(imageName)

print("=" * 60)
print("MMPS CLUSTER ImageJ ANALYSIS")
print("Image " + str(image_index + 1) + "/" + str(len(allImages)) + ": " + imageName)
print("Masks for this image: " + str(len(maskFiles)))
print("Analyses: " + str(ANALYSES))
if imageName in PIXEL_SIZE_MAP:
    print("Pixel size: " + str(imgPixelSize) + " um/px (per-image override)")
else:
    print("Pixel size: " + str(imgPixelSize) + " um/px (global)")
print("=" * 60)

if len(maskFiles) == 0:
    print("No masks for this image, nothing to do.")
    sys.exit(0)

# Load metadata for standard identifiers (animal_id, treatment, soma_idx)
metadata = loadMaskMetadata(masksDir)

totalStart = time.time()

if "skeleton" in ANALYSES:
    runSkeletonBatch(masksDir, mmps_output_dir, maskFiles, imageName, imgPixelSize, metadata)
    print("")

if "fractal" in ANALYSES:
    runFractalBatch(masksDir, mmps_output_dir, maskFiles, imageName, imgPixelSize, metadata)
    print("")

if "sholl" in ANALYSES:
    if not os.path.isdir(somasDir):
        print("ERROR: No 'somas' folder found at: " + somasDir)
        print("  Sholl analysis REQUIRES soma mask files.")
        print("  Make sure you uploaded the somas/ folder from your MMPS output.")
        print("  SKIPPING Sholl analysis.")
    else:
        somaFiles = [f for f in os.listdir(somasDir) if f.endswith('_soma.tif')]
        print("Found " + str(len(somaFiles)) + " soma files in: " + somasDir)
        if len(somaFiles) == 0:
            print("ERROR: somas/ folder exists but is EMPTY - no *_soma.tif files found.")
            print("  SKIPPING Sholl analysis.")
        else:
            runShollBatch(masksDir, somasDir, mmps_output_dir, maskFiles, imageName, imgPixelSize, metadata)
    print("")

print("=" * 60)
print("IMAGE COMPLETE: " + imageName + " in " + formatTime(time.time() - totalStart))
print("=" * 60)
'''
        return script

    def _build_slurm_script(self, fiji_path, wrapper_path, partition, wall_time, mem, cpus, job_name, module_load):
        """Build the SLURM array job submission script."""
        wrapper_basename = os.path.basename(wrapper_path)
        module_line = f"\n{module_load}" if module_load else ""

        script = f'''#!/bin/bash
# =============================================================================
# MMPS ImageJ Cluster Analysis - SLURM Array Job Launcher
# Generated by MMPS on {time.strftime("%Y-%m-%d %H:%M:%S")}
#
# This script discovers how many images you have, then submits:
#   1. A SLURM array job (one task per image, all submitted together)
#   2. A merge job that combines per-image CSVs after all tasks finish
#
# Usage:
#   bash submit_imagej.sh /path/to/mmps_output
# =============================================================================

set -e

if [ $# -lt 1 ]; then
    echo "Usage: bash submit_imagej.sh /path/to/mmps_output"
    exit 1
fi

export MMPS_OUTPUT_DIR="$(cd "$1" && pwd)"
FIJI="{fiji_path}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Set up Java 21 (required for SNT/Sholl analysis)
# Fiji's native launcher ignores JAVA_HOME and --java-home, so we must
# replace its bundled Java directory with a symlink to JDK 21.
FIJI_DIR="$(dirname "$FIJI")"
FIJI_JAVA_LINUX="$FIJI_DIR/java/linux64"

if [ -d "$SCRATCH/jdk-21" ]; then
    export JAVA_HOME="$SCRATCH/jdk-21"
    export PATH="$JAVA_HOME/bin:$PATH"

    if [ -d "$FIJI_JAVA_LINUX" ]; then
        # Find the bundled JDK directory (e.g. jdk1.8.0_172)
        BUNDLED_JDK=$(ls -d "$FIJI_JAVA_LINUX"/jdk* 2>/dev/null | head -1)
        if [ -n "$BUNDLED_JDK" ] && [ ! -L "$BUNDLED_JDK" ]; then
            # Back up bundled Java 8 and symlink to JDK 21
            echo "Replacing Fiji bundled Java with JDK 21..."
            mv "$BUNDLED_JDK" "${{BUNDLED_JDK}}.java8bak"
            ln -s "$SCRATCH/jdk-21" "$BUNDLED_JDK"
            echo "  $BUNDLED_JDK -> $SCRATCH/jdk-21"
        elif [ -L "$BUNDLED_JDK" ]; then
            echo "Fiji Java already symlinked: $(readlink "$BUNDLED_JDK")"
        fi
    fi
    echo "Java version: $(java -version 2>&1 | head -1)"
else
    echo "ERROR: $SCRATCH/jdk-21 not found. SNT/Sholl analysis requires Java 21+."
    echo "Install with: cd \\$SCRATCH && wget https://download.java.net/openjdk/jdk21/ri/openjdk-21+35_linux-x64_bin.tar.gz && tar -xzf openjdk-21+35_linux-x64_bin.tar.gz"
    exit 1
fi

# Verify Fiji exists
if [ ! -x "$FIJI" ]; then
    echo "ERROR: Fiji not found or not executable at: $FIJI"
    echo "Please set the correct path in this script."
    exit 1
fi

# Verify masks directory exists
if [ ! -d "$MMPS_OUTPUT_DIR/masks" ]; then
    echo "ERROR: No masks/ directory found in $MMPS_OUTPUT_DIR"
    exit 1
fi

# Discover unique image names from mask filenames
# Pattern handles both 2D (soma_Y_X) and 3D (soma_Z_Y_X) naming
NUM_IMAGES=$(find "$MMPS_OUTPUT_DIR/masks/" -maxdepth 1 -name '*_mask.tif' -printf '%f\\n' 2>/dev/null \\
    | sed 's/_soma_[0-9][0-9]*_[0-9][0-9]*\\(_[0-9][0-9]*\\)\\{{0,1\\}}_area[0-9][0-9]*_mask\\.tif$//' \\
    | sort -u \\
    | wc -l)

if [ "$NUM_IMAGES" -eq 0 ]; then
    echo "ERROR: No mask files found matching *_mask.tif in $MMPS_OUTPUT_DIR/masks/"
    echo ""
    echo "Contents of masks/ directory:"
    ls -la "$MMPS_OUTPUT_DIR/masks/" 2>/dev/null | head -20
    echo ""
    echo "Expected mask filename format: ImageName_soma_Y_X_area123_mask.tif"
    exit 1
fi

MAX_INDEX=$((NUM_IMAGES - 1))

echo "======================================"
echo "MMPS ImageJ Cluster Analysis"
echo "Output dir: $MMPS_OUTPUT_DIR"
echo "Images found: $NUM_IMAGES"
echo "Fiji: $FIJI"
echo "======================================"

# --- Submit the array job (one task per image) ---
ARRAY_JOB_ID=$(sbatch --parsable \\
    --job-name={job_name} \\
    --partition={partition} \\
    --time={wall_time} \\
    --mem={mem} \\
    --cpus-per-task={cpus} \\
    --array=0-${{MAX_INDEX}} \\
    --output=mmps_imagej_%A_%a.out \\
    --error=mmps_imagej_%A_%a.err \\
    --export=MMPS_OUTPUT_DIR \\
    --wrap="{module_line}
\\"$FIJI\\" --ij2 --headless --console --mem=3072m --run \\"$SCRIPT_DIR/{wrapper_basename}\\"
")

echo "Submitted array job: $ARRAY_JOB_ID (tasks 0-$MAX_INDEX)"

# --- Submit the merge job (runs after all array tasks complete) ---
MERGE_JOB_ID=$(sbatch --parsable \\
    --job-name={job_name}_merge \\
    --partition={partition} \\
    --time=00:10:00 \\
    --mem=2G \\
    --cpus-per-task=1 \\
    --dependency=afterok:$ARRAY_JOB_ID \\
    --output=mmps_imagej_merge_%j.out \\
    --error=mmps_imagej_merge_%j.err \\
    --wrap="{module_line}
python \\"$SCRIPT_DIR/merge_results.py\\" \\"$MMPS_OUTPUT_DIR\\""
)

echo "Submitted merge job:  $MERGE_JOB_ID (runs after array completes)"
echo ""
echo "Monitor with:  squeue -u \\$USER"
echo "Cancel with:   scancel $ARRAY_JOB_ID $MERGE_JOB_ID"
'''
        return script

    def _build_merge_script(self, analyses):
        """Build the Python3 script that merges per-image CSVs after all array tasks complete."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        script = f'''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MMPS Merge Results Script
Generated by MMPS on {timestamp}

Combines per-image CSV files from array job tasks into single result files.
Called automatically by the SLURM merge job after all array tasks complete.

Usage:
    python merge_results.py /path/to/mmps_output
"""

import os
import sys
import glob
import csv


def merge_csvs(results_dir, pattern, output_name):
    """Merge multiple CSVs matching pattern into one combined CSV."""
    csv_files = sorted(glob.glob(os.path.join(results_dir, pattern)))
    if not csv_files:
        print(f"  No files matching {{pattern}} in {{results_dir}}")
        return 0

    all_rows = []
    all_headers = []

    for csv_file in csv_files:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            for h in headers:
                if h not in all_headers:
                    all_headers.append(h)
            for row in reader:
                all_rows.append(row)

    output_path = os.path.join(results_dir, output_name)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_headers, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  Merged {{len(csv_files)}} files -> {{output_path}} ({{len(all_rows)}} rows)")
    return len(all_rows)


def main():
    if len(sys.argv) < 2:
        print("Usage: python merge_results.py /path/to/mmps_output")
        sys.exit(1)

    mmps_output = sys.argv[1]
    print("=" * 60)
    print("MMPS MERGE RESULTS")
    print(f"Output directory: {{mmps_output}}")
    print("=" * 60)

    total_rows = 0
'''
        if "skeleton" in analyses:
            script += '''
    skel_dir = os.path.join(mmps_output, "skeleton_results")
    if os.path.isdir(skel_dir):
        print("\\nMerging Skeleton results...")
        total_rows += merge_csvs(skel_dir, "Skeleton_Results_*.csv", "Skeleton_Analysis_Results.csv")
'''
        if "fractal" in analyses:
            script += '''
    frac_dir = os.path.join(mmps_output, "fractal_results")
    if os.path.isdir(frac_dir):
        print("\\nMerging Fractal results...")
        total_rows += merge_csvs(frac_dir, "Fractal_Results_*.csv", "Fractal_Analysis_Results.csv")
'''
        if "sholl" in analyses:
            script += '''
    sholl_dir = os.path.join(mmps_output, "sholl_results")
    if os.path.isdir(sholl_dir):
        print("\\nMerging Sholl results...")
        total_rows += merge_csvs(sholl_dir, "Sholl_Results_*.csv", "Sholl_All_Results.csv")
'''
        script += '''
    # --- Combine all analysis types into one CSV keyed by image_name + soma_id + mask_area_um2 ---
    combined = {}  # cell_key -> merged row dict
    merge_files = []
    skel_path = os.path.join(mmps_output, "skeleton_results", "Skeleton_Analysis_Results.csv")
    frac_path = os.path.join(mmps_output, "fractal_results", "Fractal_Analysis_Results.csv")
    sholl_path = os.path.join(mmps_output, "sholl_results", "Sholl_All_Results.csv")
    for csv_path in [skel_path, frac_path, sholl_path]:
        if not os.path.isfile(csv_path):
            continue
        merge_files.append(csv_path)
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img = row.get('image_name', '')
                soma = row.get('soma_id', '')
                area = row.get('area_um2', row.get('mask_area_um2', ''))
                if area:
                    try:
                        area = str(int(float(area)))
                    except (ValueError, TypeError):
                        area = ''
                cell_key = f"{img}_{soma}_area{area}" if img and soma else row.get('cell_name', f"row_{len(combined)}")
                if cell_key not in combined:
                    combined[cell_key] = {}
                # Merge columns, skip duplicates already present
                for k, v in row.items():
                    if k not in combined[cell_key] or not combined[cell_key][k]:
                        combined[cell_key][k] = v

    if combined:
        # Ensure standard ID columns come first
        std_cols = ['image_name', 'animal_id', 'treatment', 'soma_id', 'soma_idx', 'area_um2']
        all_cols = list(std_cols)
        for row in combined.values():
            for k in row.keys():
                if k not in all_cols:
                    all_cols.append(k)
        combined_path = os.path.join(mmps_output, "ImageJ_Combined_Results.csv")
        with open(combined_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction='ignore')
            writer.writeheader()
            for row in combined.values():
                writer.writerows([row])
        print(f"\\nCOMBINED all results -> {combined_path} ({len(combined)} cells)")
    elif merge_files:
        print("\\nWARNING: Could not combine results - no matching cell data found")
    else:
        print("\\nWARNING: No merged result files found to combine")

    print("")
    print("=" * 60)
    print(f"MERGE COMPLETE: {total_rows} total rows")
    if combined:
        print(f"Combined CSV: {os.path.join(mmps_output, 'ImageJ_Combined_Results.csv')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
'''
        return script

    def _build_cluster_readme(self, analyses, fiji_path):
        """Build the README instructions file."""
        analyses_str = ", ".join(a.title() for a in analyses)
        return f"""MMPS ImageJ Cluster Analysis - Instructions
============================================
Generated by MMPS on {time.strftime("%Y-%m-%d %H:%M:%S")}

Analyses: {analyses_str}

FILES GENERATED:
  mmps_imagej_cluster.py  - Jython script (runs per image inside headless Fiji)
  submit_imagej.sh        - SLURM array job launcher
  merge_results.py        - Merges per-image CSVs into combined results
  CLUSTER_README.txt      - This file

HOW IT WORKS:
  The launcher script automatically:
  1. Counts the unique images in your masks/ folder
  2. Submits a SLURM array job (one task per image, all at once)
  3. Submits a merge job that waits for all tasks to finish,
     then combines per-image CSVs into final result files

  Each image runs as an independent SLURM task — if you have 50 images
  and 50 nodes available, all 50 run simultaneously.

SETUP:
  1. Upload these 3 scripts to your cluster (same directory)
  2. Upload your Fiji.app installation to the cluster
     (or use an existing installation)
  3. Upload your MMPS output folder containing:
     - masks/  (the mask TIFF files + mask_metadata.csv)
     - somas/  (the soma TIFF files, needed for Sholl analysis)
     Note: mask_metadata.csv (in masks/) provides animal_id, treatment,
     and soma_idx for consistent CSV identifiers across all analyses.
  4. Make sure the Fiji path in submit_imagej.sh is correct
     Current setting: {fiji_path}

RUNNING:
  bash submit_imagej.sh /path/to/mmps_output

  This submits all jobs at once. Monitor with:
    squeue -u $USER

OUTPUT:
  Per-image results (one CSV per image):
  - skeleton_results/Skeleton_Results_<image>.csv
  - fractal_results/Fractal_Results_<image>.csv
  - sholl_results/Sholl_Results_<image>.csv

  Combined results (merged after all tasks complete):
  - skeleton_results/Skeleton_Analysis_Results.csv
  - fractal_results/Fractal_Analysis_Results.csv
  - sholl_results/Sholl_All_Results.csv

TROUBLESHOOTING:
  - If Fiji can't find Java, add "module load java/11" (or similar)
    to the Module Load field before generating scripts
  - If you get memory errors, increase the --mem SLURM setting
  - Check mmps_imagej_<jobid>_<taskid>.out for per-image logs
  - Check mmps_imagej_merge_<jobid>.out for merge job logs
  - Sholl analysis requires the SNT plugin (included in standard Fiji)
  - Skeleton analysis requires the AnalyzeSkeleton plugin (included in standard Fiji)
  - To re-run the merge manually:
      python merge_results.py /path/to/mmps_output
"""

    # ------------------------------------------------------------------
    # Spread / Morphology Cluster Job
    # ------------------------------------------------------------------

    def _open_spread_cluster_dialog(self):
        """Open a dialog for generating spread / morphology HPC cluster scripts."""
        from PyQt5.QtWidgets import QDialog, QScrollArea

        dlg = QDialog(self)
        dlg.setWindowTitle("Generate Spread Analysis Cluster Scripts")
        dlg.setMinimumSize(500, 550)
        dlg_layout = QVBoxLayout(dlg)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)

        desc = QLabel(
            "Generate a self-contained Python script and SLURM array job to compute\n"
            "cell spread and morphology metrics (perimeter, eccentricity, roundness,\n"
            "polarity, etc.) from MMPS mask files on an HPC cluster.\n"
            "One SLURM task per image, all submitted at once."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: palette(dark); padding-bottom: 8px;")
        layout.addWidget(desc)

        # --- Parameters ---
        param_group = QGroupBox("Parameters")
        param_layout = QFormLayout()
        try:
            px = str(self._get_pixel_size())
        except Exception:
            px = "0.316"
        self.spread_pixel_size = QLineEdit(px)
        param_layout.addRow("Pixel size (um/px):", self.spread_pixel_size)
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # --- SLURM Settings ---
        slurm_group = QGroupBox("SLURM Job Settings (per image)")
        slurm_layout = QFormLayout()
        self.spread_partition = QLineEdit("comm_small_day")
        slurm_layout.addRow("Partition:", self.spread_partition)
        self.spread_time = QLineEdit("01:00:00")
        slurm_layout.addRow("Wall time:", self.spread_time)
        self.spread_mem = QLineEdit("4G")
        slurm_layout.addRow("Memory:", self.spread_mem)
        self.spread_cpus = QLineEdit("1")
        slurm_layout.addRow("CPUs per task:", self.spread_cpus)
        self.spread_job_name = QLineEdit("mmps_spread")
        slurm_layout.addRow("Job name:", self.spread_job_name)
        self.spread_module_load = QLineEdit("module load lang/python/cpython_3.11.3_gcc122")
        self.spread_module_load.setToolTip("Python 3.11 for spread analysis; requires numpy, tifffile, scikit-image")
        slurm_layout.addRow("Module load:", self.spread_module_load)
        slurm_group.setLayout(slurm_layout)
        layout.addWidget(slurm_group)

        # --- Generate Button ---
        gen_btn = QPushButton("Generate Scripts")
        gen_btn.setStyleSheet(
            "font-size: 13pt; font-weight: bold; padding: 10px; "
            "background-color: #4CAF50; color: white; border-radius: 5px;"
        )
        gen_btn.clicked.connect(lambda: self._do_export_spread_cluster(dlg))
        layout.addWidget(gen_btn)

        layout.addStretch()
        scroll.setWidget(container)
        dlg_layout.addWidget(scroll)
        dlg.exec_()

    def _do_export_spread_cluster(self, dlg):
        """Called from the spread dialog Generate button."""
        try:
            self._export_spread_cluster_impl()
            dlg.accept()
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.log(f"ERROR generating spread cluster scripts: {e}\n{tb}")
            QMessageBox.critical(self, "Error",
                f"Failed to generate cluster scripts:\n{e}\n\nSee log for details.")

    def _export_spread_cluster_impl(self):
        """Generate the spread / morphology cluster scripts."""
        try:
            pixel_size = float(self.spread_pixel_size.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Invalid pixel size.")
            return

        partition = self.spread_partition.text().strip() or "comm_small_day"
        wall_time = self.spread_time.text().strip() or "01:00:00"
        mem = self.spread_mem.text().strip() or "4G"
        cpus = self.spread_cpus.text().strip() or "1"
        job_name = self.spread_job_name.text().strip() or "mmps_spread"
        module_load = self.spread_module_load.text().strip()

        parent_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory for Spread Scripts")
        if not parent_dir:
            return
        save_dir = os.path.join(parent_dir, "SpreadAnalysis")
        os.makedirs(save_dir, exist_ok=True)

        # Build per-image pixel size map from session data
        pixel_size_map = {}
        for img_name, img_data in self.images.items():
            per_img_px = img_data.get('pixel_size')
            if per_img_px is not None:
                pixel_size_map[os.path.splitext(img_name)[0]] = per_img_px

        # --- Generate the spread analysis Python script ---
        analysis_script = self._build_spread_analysis_script(pixel_size, pixel_size_map)
        analysis_path = os.path.join(save_dir, "mmps_spread_analysis.py")
        with open(analysis_path, 'w') as f:
            f.write(analysis_script)

        # --- Generate the SLURM job script ---
        slurm_script = self._build_spread_slurm_script(
            partition, wall_time, mem, cpus, job_name, module_load
        )
        slurm_path = os.path.join(save_dir, "submit_spread.sh")
        with open(slurm_path, 'w') as f:
            f.write(slurm_script)
        os.chmod(slurm_path, 0o755)

        self.log(f"Spread cluster scripts saved to: {save_dir}")
        self.log(f"  - mmps_spread_analysis.py  (Python analysis script - runs per image)")
        self.log(f"  - submit_spread.sh         (SLURM array job launcher)")
        if pixel_size_map:
            self.log(f"  Per-image pixel sizes embedded for {len(pixel_size_map)} image(s):")
            for name, px in pixel_size_map.items():
                self.log(f"    {name}: {px} µm/px")

        QMessageBox.information(self, "Spread Cluster Scripts Generated",
            f"Scripts saved to:\n{save_dir}\n\n"
            f"Pixel size: {pixel_size} um/px\n\n"
            f"Upload to your cluster with your masks/ and somas/ folders,\n"
            f"then run:\n\n"
            f"  bash submit_spread.sh /path/to/mmps_output\n\n"
            f"This submits a SLURM array job (one task per image)\n"
            f"plus a merge job that combines results when done.\n\n"
            f"Requirements: numpy, tifffile, scikit-image")

    def _build_spread_analysis_script(self, pixel_size, pixel_size_map=None):
        """Build the standalone Python script for cell spread / morphology analysis."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        script = f'''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MMPS Cell Spread / Morphology Cluster Analysis Script
Generated by MMPS on {timestamp}

Computes per-mask morphology metrics from MMPS-exported mask TIFF files:
  - cell_spread, perimeter, mask_area, eccentricity, roundness
  - polarity_index, principal_angle, major_axis_um, minor_axis_um
  - soma_area (from somas/ folder if available)

Usage:
    # Process all images:
    python mmps_spread_analysis.py /path/to/mmps_output

    # Process a single image by index (for SLURM array jobs):
    python mmps_spread_analysis.py /path/to/mmps_output --image-index 0

    # SLURM array job:
    #   export SLURM_ARRAY_TASK_ID=0
    #   python mmps_spread_analysis.py /path/to/mmps_output

Required files in mmps_output:
    masks/  - *_mask.tif files
    somas/  - *_soma.tif files (optional, for soma area)

Output:
    spread_results/  - per-image CSVs
    spread_results/Spread_Analysis_Results.csv  (combined, after merge)

Requirements: numpy, tifffile, scikit-image
"""

import os
import sys
import csv
import re
import argparse
import numpy as np
import tifffile
from skimage import measure

PIXEL_SIZE = {pixel_size}
PIXEL_SIZE_MAP = {repr(pixel_size_map or {})}


def get_pixel_size(image_name):
    \"\"\"Get per-image pixel size if set, otherwise fall back to global PIXEL_SIZE.\"\"\"
    return PIXEL_SIZE_MAP.get(image_name, PIXEL_SIZE)


MASK_RE = re.compile(r'^(.+?)_(soma_\\d+_\\d+)_area(\\d+)_mask\\.tif$')


def parse_mask_filename(filename):
    """Extract image_name, soma_id, area from a mask filename."""
    m = MASK_RE.match(filename)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return None, None, None


def load_mask_metadata(masks_dir):
    \"\"\"Load mask_metadata.csv for standard identifiers.
    Returns dict keyed by (image_name, soma_id) -> row dict.\"\"\"
    meta_path = os.path.join(masks_dir, "mask_metadata.csv")
    lookup = {{}}
    if not os.path.isfile(meta_path):
        return lookup
    try:
        with open(meta_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row.get('image_name', ''), row.get('soma_id', ''))
                lookup[key] = row
        print(f"Loaded mask_metadata.csv: {{len(lookup)}} entries")
    except Exception as e:
        print(f"WARNING: Could not load mask_metadata.csv: {{e}}")
    return lookup


def get_meta_ids(metadata, image_name, soma_id):
    \"\"\"Get standard identifier fields from metadata lookup.
    Returns (animal_id, treatment, soma_idx).\"\"\"
    row = metadata.get((image_name, soma_id), {{}})
    animal_id = row.get('animal_id', '')
    treatment = row.get('treatment', '')
    soma_idx = row.get('soma_idx', '')
    if not treatment and image_name:
        treatment = image_name.split('_')[0]
    return animal_id, treatment, soma_idx


def get_soma_area(somas_dir, image_name, soma_id, pixel_size):
    """Try to find the soma outline TIFF and compute its area."""
    if not somas_dir or not os.path.isdir(somas_dir):
        return None
    candidates = [
        f"{{image_name}}_{{soma_id}}_soma.tif",
        f"{{image_name}}_{{soma_id}}.tif",
    ]
    for c in candidates:
        path = os.path.join(somas_dir, c)
        if os.path.exists(path):
            soma = tifffile.imread(path)
            soma = (soma > 0).astype(np.uint8)
            if np.any(soma):
                return np.sum(soma) * (pixel_size ** 2)
            return None
    for f in os.listdir(somas_dir):
        if soma_id in f and f.endswith(".tif"):
            path = os.path.join(somas_dir, f)
            soma = tifffile.imread(path)
            soma = (soma > 0).astype(np.uint8)
            if np.any(soma):
                return np.sum(soma) * (pixel_size ** 2)
            return None
    return None


def compute_metrics(mask_path, pixel_size, soma_area_um2=None):
    """Load a mask TIFF and compute all morphology metrics."""
    mask = tifffile.imread(mask_path)
    mask = (mask > 0).astype(np.uint8)

    if not np.any(mask):
        return None

    props = measure.regionprops(mask.astype(int))
    if not props:
        return None

    p = props[0]
    params = {{}}

    # Basic shape metrics
    params['perimeter'] = round(p.perimeter * pixel_size, 4)
    params['mask_area'] = round(p.area * (pixel_size ** 2), 4)

    major_axis = p.major_axis_length
    minor_axis = p.minor_axis_length

    if major_axis > 0:
        axis_ratio = minor_axis / major_axis
        params['eccentricity'] = round(np.sqrt(1 - axis_ratio ** 2), 6)
        params['roundness'] = round(axis_ratio ** 2, 6)
    else:
        params['eccentricity'] = 0.0
        params['roundness'] = 0.0

    # Cell spread
    centroid = np.array(p.centroid)
    coords = np.array(p.coords)

    top_point = coords[coords[:, 0].argmin()]
    bottom_point = coords[coords[:, 0].argmax()]
    left_point = coords[coords[:, 1].argmin()]
    right_point = coords[coords[:, 1].argmax()]

    extremities = np.array([top_point, bottom_point, left_point, right_point])
    distances = np.sqrt(np.sum((extremities - centroid) ** 2, axis=1))
    params['cell_spread'] = round(np.mean(distances) * pixel_size, 4)

    # Soma area
    if soma_area_um2 is not None:
        params['soma_area'] = round(soma_area_um2, 4)
    else:
        params['soma_area'] = round(p.area * 0.1 * (pixel_size ** 2), 4)

    # Polarity via PCA
    centered = coords - centroid
    if len(centered) >= 3:
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        major_val = eigenvalues[-1]
        minor_val = eigenvalues[0]
        major_vec = eigenvectors[:, -1]

        if major_val > 0:
            params['polarity_index'] = round(1.0 - (minor_val / major_val), 4)
        else:
            params['polarity_index'] = 0

        angle_rad = np.arctan2(major_vec[0], major_vec[1])
        params['principal_angle'] = round(np.degrees(angle_rad) % 180, 2)
        params['major_axis_um'] = round(2 * np.sqrt(major_val) * pixel_size, 4)
        params['minor_axis_um'] = round(2 * np.sqrt(max(minor_val, 0)) * pixel_size, 4)
    else:
        params['polarity_index'] = 0
        params['principal_angle'] = 0
        params['major_axis_um'] = 0
        params['minor_axis_um'] = 0

    return params


def get_unique_images(masks_dir):
    """Get sorted list of unique image names from mask filenames."""
    images = set()
    for f in os.listdir(masks_dir):
        img_name, _, _ = parse_mask_filename(f)
        if img_name:
            images.add(img_name)
    return sorted(images)


def process_image(image_name, masks_dir, somas_dir, pixel_size, output_dir, metadata=None):
    """Process all masks for a single image and write a per-image CSV."""
    if metadata is None:
        metadata = {{}}
    all_files = sorted(os.listdir(masks_dir))
    mask_files = []
    for f in all_files:
        img, soma_id, area = parse_mask_filename(f)
        if img == image_name:
            mask_files.append((f, soma_id, area))

    if not mask_files:
        print(f"  No masks found for image: {{image_name}}")
        return 0

    results = []
    for filename, soma_id, area in mask_files:
        mask_path = os.path.join(masks_dir, filename)
        soma_area = get_soma_area(somas_dir, image_name, soma_id, pixel_size)
        aid, treat, sidx = get_meta_ids(metadata, image_name, soma_id)

        try:
            metrics = compute_metrics(mask_path, pixel_size, soma_area)
        except Exception as e:
            print(f"  ERROR: {{filename}}: {{e}}")
            continue

        if metrics is None:
            print(f"  SKIP (empty mask): {{filename}}")
            continue

        row = {{
            'image_name': image_name,
            'animal_id': aid,
            'treatment': treat,
            'soma_id': soma_id,
            'soma_idx': sidx,
            'area_um2': area,
        }}
        row.update(metrics)
        results.append(row)

    if not results:
        return 0

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"Spread_Results_{{image_name}}.csv")

    fieldnames = [
        'image_name', 'animal_id', 'treatment', 'soma_id', 'soma_idx',
        'area_um2',
        'mask_area', 'perimeter', 'roundness', 'eccentricity',
        'cell_spread', 'soma_area',
        'polarity_index', 'principal_angle',
        'major_axis_um', 'minor_axis_um',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"  Wrote {{len(results)}} rows to {{csv_path}}")
    return len(results)


def merge_results(output_dir):
    """Merge all per-image Spread CSVs into a single combined file."""
    csv_files = sorted(f for f in os.listdir(output_dir)
                       if f.startswith("Spread_Results_") and f.endswith(".csv"))
    if not csv_files:
        print("No per-image spread CSVs found to merge.")
        return

    all_rows = []
    for csv_file in csv_files:
        path = os.path.join(output_dir, csv_file)
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(row)

    combined_path = os.path.join(output_dir, "Spread_Analysis_Results.csv")
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(combined_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Merged {{len(all_rows)}} rows from {{len(csv_files)}} files into {{combined_path}}")


def main():
    parser = argparse.ArgumentParser(description="MMPS Cell Spread / Morphology Analysis")
    parser.add_argument("mmps_output_dir", help="Path to MMPS output directory (contains masks/ and somas/)")
    parser.add_argument("--image-index", type=int, default=None,
                        help="Process only this image index (for SLURM array jobs)")
    parser.add_argument("--merge-only", action="store_true",
                        help="Only merge existing per-image CSVs")
    parser.add_argument("--pixel-size", type=float, default=PIXEL_SIZE,
                        help=f"Pixel size in um/px (default: {{PIXEL_SIZE}})")
    args = parser.parse_args()

    mmps_dir = os.path.abspath(args.mmps_output_dir)
    masks_dir = os.path.join(mmps_dir, "masks")
    somas_dir = os.path.join(mmps_dir, "somas")
    output_dir = os.path.join(mmps_dir, "spread_results")
    pixel_size = args.pixel_size

    if not os.path.isdir(somas_dir):
        somas_dir = None

    if args.merge_only:
        merge_results(output_dir)
        return

    if not os.path.isdir(masks_dir):
        print(f"ERROR: masks/ directory not found in {{mmps_dir}}")
        sys.exit(1)

    unique_images = get_unique_images(masks_dir)
    if not unique_images:
        print("ERROR: No mask files found matching expected pattern.")
        sys.exit(1)

    # Determine which image(s) to process
    image_index = args.image_index
    if image_index is None:
        # Check for SLURM array task ID
        slurm_idx = os.environ.get("SLURM_ARRAY_TASK_ID")
        if slurm_idx is not None:
            image_index = int(slurm_idx)

    if image_index is not None:
        if image_index < 0 or image_index >= len(unique_images):
            print(f"ERROR: image-index {{image_index}} out of range (0-{{len(unique_images)-1}})")
            sys.exit(1)
        images_to_process = [unique_images[image_index]]
        print(f"Processing image {{image_index}}/{{len(unique_images)-1}}: {{images_to_process[0]}}")
    else:
        images_to_process = unique_images
        print(f"Processing all {{len(unique_images)}} images...")

    # Load metadata for standard identifiers
    metadata = load_mask_metadata(masks_dir)

    total = 0
    for img_name in images_to_process:
        img_pixel_size = get_pixel_size(img_name)
        if img_name in PIXEL_SIZE_MAP:
            print(f"\\nImage: {{img_name}}  (pixel size: {{img_pixel_size}} um/px, per-image override)")
        else:
            print(f"\\nImage: {{img_name}}  (pixel size: {{img_pixel_size}} um/px)")
        n = process_image(img_name, masks_dir, somas_dir, img_pixel_size, output_dir, metadata)
        total += n

    print(f"\\nDone. Total masks processed: {{total}}")

    # If processing all images, also merge
    if image_index is None:
        merge_results(output_dir)


if __name__ == "__main__":
    main()
'''
        return script

    def _build_spread_slurm_script(self, partition, wall_time, mem, cpus, job_name, module_load):
        """Build the SLURM array job submission script for spread analysis."""
        module_line = f"\n{module_load}" if module_load else ""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        script = f'''#!/bin/bash
# =============================================================================
# MMPS Cell Spread / Morphology - SLURM Array Job Launcher
# Generated by MMPS on {timestamp}
#
# This script discovers how many images you have, then submits:
#   1. A SLURM array job (one task per image, all submitted together)
#   2. A merge job that combines per-image CSVs after all tasks finish
#
# Usage:
#   bash submit_spread.sh /path/to/mmps_output
#
# Requirements: numpy, tifffile, scikit-image
# =============================================================================

set -e

if [ $# -lt 1 ]; then
    echo "Usage: bash submit_spread.sh /path/to/mmps_output"
    exit 1
fi

MMPS_OUTPUT_DIR="$(cd "$1" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Verify masks directory exists
if [ ! -d "$MMPS_OUTPUT_DIR/masks" ]; then
    echo "ERROR: No masks/ directory found in $MMPS_OUTPUT_DIR"
    exit 1
fi

# Discover unique image names from mask filenames
# Pattern handles both 2D (soma_Y_X) and 3D (soma_Z_Y_X) naming
NUM_IMAGES=$(find "$MMPS_OUTPUT_DIR/masks/" -maxdepth 1 -name '*_mask.tif' -printf '%f\\n' 2>/dev/null \\
    | sed 's/_soma_[0-9][0-9]*_[0-9][0-9]*\\(_[0-9][0-9]*\\)\\{{0,1\\}}_area[0-9][0-9]*_mask\\.tif$//' \\
    | sort -u \\
    | wc -l)

if [ "$NUM_IMAGES" -eq 0 ]; then
    echo "ERROR: No mask files found matching *_mask.tif in $MMPS_OUTPUT_DIR/masks/"
    echo ""
    echo "Contents of masks/ directory:"
    ls -la "$MMPS_OUTPUT_DIR/masks/" 2>/dev/null | head -20
    echo ""
    echo "Expected mask filename format: ImageName_soma_Y_X_area123_mask.tif"
    exit 1
fi

MAX_INDEX=$((NUM_IMAGES - 1))

echo "======================================"
echo "MMPS Cell Spread / Morphology Analysis"
echo "Output dir: $MMPS_OUTPUT_DIR"
echo "Images found: $NUM_IMAGES"
echo "======================================"

# --- Submit the array job (one task per image) ---
ARRAY_JOB_ID=$(sbatch --parsable \\
    --job-name={job_name} \\
    --partition={partition} \\
    --time={wall_time} \\
    --mem={mem} \\
    --cpus-per-task={cpus} \\
    --array=0-${{MAX_INDEX}} \\
    --output=mmps_spread_%A_%a.out \\
    --error=mmps_spread_%A_%a.err \\
    --wrap="{module_line}
python \\"$SCRIPT_DIR/mmps_spread_analysis.py\\" \\"$MMPS_OUTPUT_DIR\\"
")

echo "Submitted array job: $ARRAY_JOB_ID (tasks 0-$MAX_INDEX)"

# --- Submit the merge job (runs after all array tasks complete) ---
MERGE_JOB_ID=$(sbatch --parsable \\
    --job-name={job_name}_merge \\
    --partition={partition} \\
    --time=00:10:00 \\
    --mem=2G \\
    --cpus-per-task=1 \\
    --dependency=afterok:$ARRAY_JOB_ID \\
    --output=mmps_spread_merge_%j.out \\
    --error=mmps_spread_merge_%j.err \\
    --wrap="{module_line}
python \\"$SCRIPT_DIR/mmps_spread_analysis.py\\" \\"$MMPS_OUTPUT_DIR\\" --merge-only"
)

echo "Submitted merge job:  $MERGE_JOB_ID (runs after array completes)"
echo ""
echo "Monitor with:  squeue -u \\$USER"
echo "Cancel with:   scancel $ARRAY_JOB_ID $MERGE_JOB_ID"
'''
        return script

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

        Uses the processed (cleaned) channel intensities directly within
        each cell mask.  No thresholding — any pixel with intensity > 0
        in the processed image is considered signal.
        """
        if not self.colocalization_mode:
            return {}

        img_data = self.images.get(img_name)
        if img_data is None:
            return {'coloc_status': 'no_color_data'}

        # Lazy-load color image from raw file if not in memory
        if 'color_image' not in img_data or img_data['color_image'] is None:
            raw_path = img_data.get('raw_path')
            if raw_path and os.path.exists(raw_path):
                try:
                    raw_img = load_tiff_image(raw_path)
                    if raw_img is not None and raw_img.ndim == 3:
                        img_data['color_image'] = raw_img
                        img_data['num_channels'] = raw_img.shape[2]
                except Exception:
                    pass

        if 'color_image' not in img_data or img_data['color_image'] is None:
            return {'coloc_status': 'no_color_data'}

        color_img = img_data['color_image']
        if color_img.ndim != 3:
            return {'coloc_status': 'not_multichannel'}

        n_channels = color_img.shape[2]
        if self.coloc_channel_1 >= n_channels or self.coloc_channel_2 >= n_channels:
            return {'coloc_status': 'invalid_channels'}

        # Apply mask
        mask_bool = mask > 0
        if not np.any(mask_bool):
            return {'coloc_status': 'empty_mask'}

        n_mask_pixels = int(np.sum(mask_bool))

        # Get processed channel data (cleaned), falling back to raw
        processed_channels = img_data.get('processed_channels', {})
        processed_primary = img_data.get('processed')

        def _get_channel(ch_idx):
            if ch_idx in processed_channels:
                return processed_channels[ch_idx].astype(np.float64)
            if ch_idx == self.grayscale_channel and processed_primary is not None:
                return processed_primary.astype(np.float64)
            return color_img[:, :, ch_idx].astype(np.float64)

        ch1_full = _get_channel(self.coloc_channel_1)
        ch2_full = _get_channel(self.coloc_channel_2)

        ch1_masked = ch1_full[mask_bool]
        ch2_masked = ch2_full[mask_bool]

        # Signal = any pixel with intensity > 0 in the processed image
        ch1_signal = ch1_masked > 0
        ch2_signal = ch2_masked > 0

        n_ch1_signal = int(np.sum(ch1_signal))
        n_ch2_signal = int(np.sum(ch2_signal))

        results = {
            'coloc_status': 'ok',
            'coloc_ch1': self.coloc_channel_1 + 1,
            'coloc_ch2': self.coloc_channel_2 + 1,
            'n_mask_pixels': n_mask_pixels,
            'n_ch1_signal': n_ch1_signal,
            'n_ch2_signal': n_ch2_signal,
            'ch1_mean_intensity': round(float(np.mean(ch1_masked)), 4),
            'ch2_mean_intensity': round(float(np.mean(ch2_masked)), 4),
        }

        if n_ch1_signal == 0 or n_ch2_signal == 0:
            results['n_coloc_pixels'] = 0
            results['ch1_coloc_percent'] = 0.0
            results['ch2_coloc_percent'] = 0.0
            results['pearson_r'] = 0.0
            return results

        # Colocalized = both channels have signal (> 0) at same pixel
        colocalized = ch1_signal & ch2_signal
        n_coloc = int(np.sum(colocalized))

        results['n_coloc_pixels'] = n_coloc
        results['ch1_coloc_percent'] = round((n_coloc / n_ch1_signal) * 100, 2)
        results['ch2_coloc_percent'] = round((n_coloc / n_ch2_signal) * 100, 2)

        # Pearson's R on all mask pixels (both channels)
        if np.std(ch1_masked) > 0 and np.std(ch2_masked) > 0:
            pearson_r, _ = stats.pearsonr(ch1_masked, ch2_masked)
            results['pearson_r'] = round(pearson_r, 4)
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
            ("?", "Show this help"),
            ("C", "Toggle color / grayscale"),
            ("U", "Reset zoom"),
            ("Z + Left-click", "Zoom in"),
            ("Z + Right-click", "Zoom out"),
            ("M", "Toggle measure tool"),
            ("I", "Toggle pixel intensity picker"),
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
                ("Left-click", "Place outline point"),
                ("Shift+Left-click", "Drag existing point"),
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
                ("B", "Undo last QA decision"),
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

    def toggle_pixel_picker_mode(self):
        """Toggle pixel intensity picker on/off"""
        self.pixel_picker_mode = not getattr(self, 'pixel_picker_mode', False)
        self.pixel_picker_btn.setChecked(self.pixel_picker_mode)

        for label in [self.original_label, self.preview_label, self.processed_label, self.mask_label]:
            label.pixel_picker_mode = self.pixel_picker_mode

        if self.pixel_picker_mode:
            self.log("Pixel picker ON - click any pixel to see channel intensities")
        else:
            self.log("Pixel picker OFF")

    def _show_pixel_intensity(self, coords):
        """Show pixel intensity for each channel at the clicked coordinates."""
        row, col = int(coords[0]), int(coords[1])
        img_name = self.current_image_name
        if not img_name:
            return

        img_data = self.images.get(img_name)
        if img_data is None:
            return

        parts = [f"Pixel ({row}, {col})"]

        # Show raw / color image channel values
        color_img = img_data.get('color_image')
        if color_img is None and img_data.get('raw_path'):
            try:
                raw = load_tiff_image(img_data['raw_path'])
                if raw is not None and raw.ndim == 3:
                    img_data['color_image'] = raw.copy()
                    img_data['num_channels'] = raw.shape[2]
                    color_img = raw
            except Exception:
                pass

        if color_img is not None and color_img.ndim == 3:
            h, w, nc = color_img.shape
            if 0 <= row < h and 0 <= col < w:
                for ch in range(nc):
                    name = self.channel_names.get(ch, f'Ch{ch + 1}')
                    if not name:
                        name = f'Ch{ch + 1}'
                    val = int(color_img[row, col, ch])
                    parts.append(f"{name}={val}")
        elif color_img is not None and color_img.ndim == 2:
            h, w = color_img.shape
            if 0 <= row < h and 0 <= col < w:
                val = int(color_img[row, col])
                parts.append(f"Gray={val}")

        # Also show processed image value if available
        processed = img_data.get('processed')
        if processed is not None:
            if processed.ndim == 2:
                ph, pw = processed.shape
                if 0 <= row < ph and 0 <= col < pw:
                    parts.append(f"Processed={int(processed[row, col])}")
            elif processed.ndim == 3 and processed.shape[0] > 1:
                # 3D stack - show current Z slice
                z = getattr(self, 'current_z_slice', 0)
                if 0 <= z < processed.shape[0] and 0 <= row < processed.shape[1] and 0 <= col < processed.shape[2]:
                    parts.append(f"Processed(z={z})={int(processed[z, row, col])}")

        self.log(" | ".join(parts))

    def _get_measure_text(self):
        """Get formatted measurement text for display overlay"""
        label = self._get_active_label()
        if not label or label.measure_pt1 is None or label.measure_pt2 is None:
            return ""
        px_x, px_y = self._get_pixel_size_xy(self.current_image_name)

        pt1 = label.measure_pt1
        pt2 = label.measure_pt2
        dx = (pt2[1] - pt1[1])  # columns = X
        dy = (pt2[0] - pt1[0])  # rows = Y
        dist_px = math.sqrt(dx * dx + dy * dy)
        dist_um = math.sqrt((dx * px_x) ** 2 + (dy * px_y) ** 2)
        return f"{dist_um:.2f} um ({dist_px:.0f} px)"

    def _show_measurement(self):
        """Log the measurement result"""
        text = self._get_measure_text()
        if text:
            self.log(f"Measurement: {text}")

    def _calibrate_from_measurement(self):
        """Use the current measurement to calculate pixel size from a known distance."""
        label = self._get_active_label()
        if not label or label.measure_pt1 is None or label.measure_pt2 is None:
            QMessageBox.information(self, "Calibrate",
                                   "Draw a measurement line first (click two points with the Measure tool).")
            return

        pt1 = label.measure_pt1
        pt2 = label.measure_pt2
        dx = abs(pt2[1] - pt1[1])  # columns = X
        dy = abs(pt2[0] - pt1[0])  # rows = Y
        dist_px = math.sqrt(dx * dx + dy * dy)

        if dist_px < 1:
            QMessageBox.warning(self, "Calibrate", "Measurement line is too short.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Calibrate Pixel Size")
        dialog.setModal(True)
        layout = QVBoxLayout()

        layout.addWidget(QLabel(f"Measured distance: {dist_px:.1f} pixels  (dx={dx:.0f}, dy={dy:.0f})"))
        layout.addSpacing(5)

        known_input = QLineEdit()
        known_input.setPlaceholderText("e.g. 100")
        form = QFormLayout()
        form.addRow("Known distance (μm):", known_input)
        layout.addLayout(form)

        layout.addSpacing(5)

        # Result preview label
        result_label = QLabel("")
        result_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(result_label)

        def _update_preview():
            try:
                known_um = float(known_input.text())
                if known_um <= 0:
                    result_label.setText("")
                    return
                ps = known_um / dist_px
                result_label.setText(f"Pixel size = {ps:.6f} μm/px")
            except ValueError:
                result_label.setText("")

        known_input.textChanged.connect(_update_preview)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(cancel_btn)
        apply_btn = QPushButton("Apply")
        apply_btn.setDefault(True)
        apply_btn.setStyleSheet("QPushButton { font-weight: bold; }")
        apply_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(apply_btn)
        layout.addLayout(btn_layout)

        dialog.setLayout(layout)

        if dialog.exec_() != QDialog.Accepted:
            return

        try:
            known_um = float(known_input.text())
            if known_um <= 0:
                return
        except ValueError:
            return

        pixel_size = known_um / dist_px
        ps_str = f"{pixel_size:.6f}"

        # Set both X and Y
        self.pixel_size_x_input.setText(ps_str)
        self.pixel_size_y_input.setText(ps_str)
        self.pixel_size_link_btn.setChecked(True)
        self.log(f"Pixel size calibrated: {ps_str} μm/px  (from {known_um} μm / {dist_px:.1f} px)")

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

    def _determine_last_completed_step(self):
        """Determine the furthest completed workflow step across all images."""
        statuses = [d['status'] for d in self.images.values() if d['selected']]
        if not statuses:
            return 'none'
        # Priority order (highest = furthest along)
        priority = ['analyzed', 'qa_complete', 'masks_generated', 'outlined',
                     'somas_picked', 'processed', 'loaded']
        for step in priority:
            if any(s == step for s in statuses):
                return step
        return 'loaded'

    def _get_step_display_name(self, step):
        """Return a human-readable name for a workflow step."""
        names = {
            'none': 'No images loaded',
            'loaded': 'Images loaded',
            'processed': 'Image processing',
            'somas_picked': 'Soma selection',
            'outlined': 'Soma outlining',
            'masks_generated': 'Mask generation',
            'qa_complete': 'Mask QA',
            'analyzed': 'Morphology analysis',
        }
        return names.get(step, step)

    def _get_next_step_hint(self, step):
        """Return a hint about what the user should do next."""
        hints = {
            'none': 'Load images and select an output folder to begin.',
            'loaded': 'Select images and click "Process Selected Images".',
            'processed': 'Click "Pick Somas" to mark cell bodies.',
            'somas_picked': 'Click "Outline Somas" to trace soma boundaries.',
            'outlined': 'Click "Generate All Masks" to create analysis masks.',
            'masks_generated': 'Click "QA All Masks" to review masks.',
            'qa_complete': 'Click "Calculate Simple Characteristics" or import ImageJ results.',
            'analyzed': 'Analysis complete. Export results or import ImageJ results.',
        }
        return hints.get(step, '')

    # --- Checklist CSV helpers for tracking progress ---

    def _get_checklist_path(self, name):
        """Get path for a checklist CSV in the output directory."""
        if not self.output_dir:
            return None
        return os.path.join(self.output_dir, name)

    def _write_checklist(self, path, rows, header):
        """Write a checklist CSV with the given header and rows."""
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    def _read_checklist(self, path):
        """Read a checklist CSV and return rows as list of lists."""
        if not path or not os.path.exists(path):
            return []
        with open(path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            return [row for row in reader]

    def _update_checklist_row(self, path, key_col_idx, key_value, status_col_idx, new_status):
        """Update a single row in a checklist CSV by matching key column."""
        if not path or not os.path.exists(path):
            return
        rows = []
        header = None
        with open(path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            rows = [row for row in reader]
        for row in rows:
            if row[key_col_idx] == key_value:
                row[status_col_idx] = str(new_status)
        if header:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)

    def _flush_qa_checklist(self):
        """Flush deferred QA checklist updates to disk in one batch write."""
        if not hasattr(self, '_qa_checklist_dirty') or not self._qa_checklist_dirty:
            return
        qa_cl_path = self._get_checklist_path('mask_qa_checklist.csv')
        if not qa_cl_path or not os.path.exists(qa_cl_path):
            self._qa_checklist_dirty.clear()
            return
        rows = []
        header = None
        with open(qa_cl_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            rows = [row for row in reader]
        for row in rows:
            if row[0] in self._qa_checklist_dirty:
                row[1] = str(self._qa_checklist_dirty[row[0]])
        if header:
            with open(qa_cl_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
        self._qa_checklist_dirty.clear()

    def _delete_checklist(self, path):
        """Delete a checklist CSV if it exists."""
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

    def _make_relative(self, abs_path, session_dir):
        """Convert an absolute path to a relative path from session_dir, if possible."""
        if not abs_path or not session_dir:
            return abs_path
        try:
            return os.path.relpath(abs_path, session_dir)
        except ValueError:
            # On Windows, relpath fails across drives (e.g. C: vs D:)
            return abs_path

    def _build_session_dict(self, session_file_path=None):
        """Build a serializable session dictionary from current state."""
        # Determine the last completed workflow step
        last_step = self._determine_last_completed_step()

        # Compute relative paths from the session file location
        session_dir = os.path.dirname(os.path.abspath(session_file_path)) if session_file_path else None

        session = {
            'version': 3,
            'output_dir': self.output_dir,
            'masks_dir': self.masks_dir,
            'output_dir_rel': self._make_relative(self.output_dir, session_dir) if session_dir else None,
            'masks_dir_rel': self._make_relative(self.masks_dir, session_dir) if session_dir else None,
            'colocalization_mode': self.colocalization_mode,
            'pixel_size': self.pixel_size_x_input.text(),
            'pixel_size_y': self.pixel_size_y_input.text(),
            'pixel_size_linked': self.pixel_size_link_btn.isChecked(),
            'rolling_ball_radius': self.default_rolling_ball_radius,
            'use_min_intensity': self.use_min_intensity,
            'min_intensity_percent': self.min_intensity_percent,
            'mask_min_area': self.mask_min_area,
            'mask_max_area': self.mask_max_area,
            'mask_step_size': self.mask_step_size,
            'mask_segmentation_method': self.mask_segmentation_method,
            'use_circular_constraint': self.use_circular_constraint,
            'circular_buffer_um2': self.circular_buffer_um2,
            'coloc_channel_1': self.coloc_channel_1,
            'coloc_channel_2': self.coloc_channel_2,
            'grayscale_channel': self.grayscale_channel,
            'last_completed_step': last_step,
            'last_image_name': self.current_image_name,
            'mode_3d': self.mode_3d,
            'voxel_size_z': self.voxel_z_input.text() if self.mode_3d else None,
            'mask_min_volume': self.mask_min_volume,
            'mask_max_volume': self.mask_max_volume,
            'soma_intensity_tolerance': self.soma_intensity_tolerance,
            'soma_max_radius_um': self.soma_max_radius_um,
            'images': {}
        }

        for img_name, img_data in self.images.items():
            # Build path to processed TIFF if it exists on disk
            processed_path = None
            name_stem = os.path.splitext(img_name)[0]
            if self.output_dir:
                candidate = os.path.join(
                    self.output_dir, f"{name_stem}_processed.tif"
                )
                if os.path.exists(candidate):
                    processed_path = candidate

            # Build paths for extra cleaned channel TIFFs
            extra_channel_paths = {}
            if self.output_dir:
                for ch_idx in range(3):
                    ch_candidate = os.path.join(
                        self.output_dir, f"{name_stem}_processed_ch{ch_idx + 1}.tif"
                    )
                    if os.path.exists(ch_candidate):
                        extra_channel_paths[str(ch_idx)] = ch_candidate

            # Compute relative paths for portability across machines
            extra_channel_paths_rel = {}
            if session_dir:
                for ch_str, ch_path in extra_channel_paths.items():
                    extra_channel_paths_rel[ch_str] = self._make_relative(ch_path, session_dir)

            img_session = {
                'raw_path': img_data['raw_path'],
                'raw_path_rel': self._make_relative(img_data['raw_path'], session_dir) if session_dir else None,
                'processed_path': processed_path,
                'processed_path_rel': self._make_relative(processed_path, session_dir) if session_dir and processed_path else None,
                'extra_channel_paths': extra_channel_paths,
                'extra_channel_paths_rel': extra_channel_paths_rel if session_dir else {},
                'status': img_data['status'],
                'selected': img_data['selected'],
                'animal_id': img_data.get('animal_id', ''),
                'treatment': img_data.get('treatment', ''),
                'rolling_ball_radius': img_data.get('rolling_ball_radius', 50),
                'pixel_size': img_data.get('pixel_size'),
                'somas': [tuple(float(c) for c in s) for s in img_data.get('somas', [])],
                'soma_ids': img_data.get('soma_ids', []),
                'soma_groups': img_data.get('soma_groups', []),
                'soma_outlines': [],
            }
            # Save soma outlines with full metadata (2D only)
            for outline in img_data.get('soma_outlines', []):
                outline_data = {
                    'soma_idx': outline.get('soma_idx', 0),
                    'soma_id': outline.get('soma_id', ''),
                    'centroid': [float(outline['centroid'][0]), float(outline['centroid'][1])] if 'centroid' in outline else None,
                    'soma_area_um2': outline.get('soma_area_um2', 0),
                    'polygon_points': [(float(pt[0]), float(pt[1])) for pt in outline.get('polygon_points', [])],
                }
                img_session['soma_outlines'].append(outline_data)

            # Record which masks exist on disk and their approval state
            mask_qa_state = []
            for mask in img_data.get('masks', []):
                mqa = {
                    'soma_id': mask.get('soma_id', ''),
                    'approved': mask.get('approved'),
                    'soma_idx': mask.get('soma_idx', 0),
                    'duplicate': mask.get('duplicate', False),
                }
                if self.mode_3d:
                    mqa['volume_um3'] = mask.get('volume_um3', 0)
                else:
                    mqa['area_um2'] = mask.get('area_um2', 0)
                mask_qa_state.append(mqa)
            img_session['mask_qa_state'] = mask_qa_state

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
        ext = '.mmps3d_session' if self.mode_3d else '.mmps_session'
        filt = f"Session Files (*{ext});;All Files (*)"
        path, _ = QFileDialog.getSaveFileName(self, "Save Session", "", filt)
        if not path:
            return

        if not path.endswith(ext):
            path += ext

        try:
            session = self._build_session_dict(session_file_path=path)
            with open(path, 'w') as f:
                json.dump(session, f, separators=(',', ':'))

            last_step = session.get('last_completed_step', 'none')
            step_name = self._get_step_display_name(last_step)
            next_hint = self._get_next_step_hint(last_step)

            self.log(f"Session saved to: {path}")
            self.log(f"Last completed step: {step_name}")
            QMessageBox.information(self, "Session Saved",
                f"Session saved to:\n{path}\n\n"
                f"Last completed step: {step_name}\n"
                f"Next: {next_hint}")

        except Exception as e:
            self.log(f"ERROR saving session: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save session:\n{e}")

    def export_cluster_script(self):
        """Export a standalone Python script for mask generation on a compute cluster."""
        try:
            self._export_cluster_script_impl()
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.log(f"ERROR in cluster script export: {e}\n{tb}")
            QMessageBox.critical(self, "Error",
                f"Failed to generate cluster script:\n{e}\n\nSee log for details.")

    def _export_cluster_script_impl(self):
        """Internal implementation of cluster script export."""
        # Validate that we have enough data
        has_outlines = any(
            img_data.get('soma_outlines') and img_data['selected']
            for img_data in self.images.values()
        )
        if not self.images:
            QMessageBox.warning(self, "Warning", "No images loaded. Load images first.")
            return
        if not has_outlines:
            QMessageBox.warning(self, "Warning",
                "No soma outlines found.\n\n"
                "Complete soma picking and outlining before generating a cluster script.")
            return

        # --- Settings dialog ---
        dialog = QDialog(self)
        dialog.setWindowTitle("Cluster Mask Generation Settings")
        dialog.setModal(True)
        layout = QVBoxLayout()

        title = QLabel("Configure Mask Generation for Cluster")
        title_font = title.font()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        layout.addWidget(title)

        # --- Pixel Size (read-only summary) ---
        px_group = QGroupBox("Pixel Calibration")
        px_layout = QVBoxLayout()

        per_image_overrides = {name: data.get('pixel_size')
                               for name, data in self.images.items()
                               if data.get('pixel_size') is not None}

        try:
            global_px = self._get_pixel_size()
        except Exception:
            global_px = 0.316

        # Build summary of what each image will use
        summary_lines = []
        for img_name in sorted(self.images.keys()):
            img_px = self._get_pixel_size(img_name)
            display_name = os.path.splitext(img_name)[0]
            if img_name in per_image_overrides:
                summary_lines.append(f"  {display_name}: {img_px} µm/px (per-image)")
            else:
                summary_lines.append(f"  {display_name}: {img_px} µm/px")

        global_label = QLabel(f"Global: {global_px} µm/px")
        global_label.setStyleSheet("font-weight: bold;")
        px_layout.addWidget(global_label)

        if per_image_overrides:
            info_label = QLabel(
                f"{len(per_image_overrides)} image(s) have per-image overrides.\n"
                + "\n".join(summary_lines))
        else:
            info_label = QLabel("All images use the global pixel size.\n" + "\n".join(summary_lines))
        info_label.setStyleSheet("color: palette(dark); font-size: 10px;")
        info_label.setWordWrap(True)
        px_layout.addWidget(info_label)

        hint_label = QLabel("Change via Advanced > Set Per-Image Pixel Size or the main pixel size field.")
        hint_label.setStyleSheet("color: palette(dark); font-style: italic; font-size: 10px;")
        px_layout.addWidget(hint_label)

        px_group.setLayout(px_layout)
        layout.addWidget(px_group)

        # --- Mask Size Settings ---
        size_group = QGroupBox("Mask Sizes (µm²)")
        size_layout = QHBoxLayout()

        size_layout.addWidget(QLabel("Min:"))
        min_area_spin = QSpinBox()
        min_area_spin.setRange(10, 2000)
        min_area_spin.setSingleStep(50)
        min_area_spin.setValue(self.mask_min_area)
        min_area_spin.setSuffix(" µm²")
        size_layout.addWidget(min_area_spin)

        size_layout.addWidget(QLabel("Max:"))
        max_area_spin = QSpinBox()
        max_area_spin.setRange(200, 5000)
        max_area_spin.setSingleStep(50)
        max_area_spin.setValue(self.mask_max_area)
        max_area_spin.setSuffix(" µm²")
        size_layout.addWidget(max_area_spin)

        size_layout.addWidget(QLabel("Step:"))
        step_spin = QSpinBox()
        step_spin.setRange(10, 500)
        step_spin.setSingleStep(10)
        step_spin.setValue(self.mask_step_size)
        step_spin.setSuffix(" µm²")
        size_layout.addWidget(step_spin)

        size_group.setLayout(size_layout)
        layout.addWidget(size_group)

        # Preview of mask sizes
        preview_label = QLabel("")
        preview_label.setStyleSheet("color: palette(dark); font-size: 10px;")
        preview_label.setWordWrap(True)
        layout.addWidget(preview_label)

        def update_size_preview():
            mn = min_area_spin.value()
            mx = max_area_spin.value()
            st = step_spin.value()
            if mn > mx:
                preview_label.setText("Warning: Min must be <= Max")
                return
            sizes = list(range(mn, mx + 1, st))
            if sizes[-1] != mx:
                sizes.append(mx)
            preview_label.setText(f"Masks: {', '.join(str(s) for s in sizes)} µm²  ({len(sizes)} per cell)")

        min_area_spin.valueChanged.connect(lambda: update_size_preview())
        max_area_spin.valueChanged.connect(lambda: update_size_preview())
        step_spin.valueChanged.connect(lambda: update_size_preview())
        update_size_preview()

        # --- Intensity Settings ---
        intensity_group = QGroupBox("Intensity Filtering")
        intensity_layout = QVBoxLayout()

        min_intensity_check = QCheckBox("Use minimum intensity threshold")
        min_intensity_check.setChecked(self.use_min_intensity)
        intensity_layout.addWidget(min_intensity_check)

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Min intensity:"))
        min_intensity_slider = QSlider(Qt.Horizontal)
        min_intensity_slider.setRange(0, 100)
        min_intensity_slider.setValue(self.min_intensity_percent)
        slider_layout.addWidget(min_intensity_slider)
        min_intensity_label = QLabel(f"{self.min_intensity_percent}%")
        min_intensity_slider.valueChanged.connect(
            lambda v: min_intensity_label.setText(f"{v}%"))
        slider_layout.addWidget(min_intensity_label)
        intensity_layout.addLayout(slider_layout)

        intensity_group.setLayout(intensity_layout)
        layout.addWidget(intensity_group)

        # --- Segmentation Method ---
        seg_group = QGroupBox("Cell Boundary Segmentation")
        seg_layout = QVBoxLayout()

        seg_combo = QComboBox()
        seg_combo.addItem("None (independent growth)", "none")
        seg_combo.addItem("Competitive Growth (shared priority queue)", "competitive")
        seg_combo.addItem("Watershed Territories (pre-computed basins)", "watershed")
        for idx in range(seg_combo.count()):
            if seg_combo.itemData(idx) == self.mask_segmentation_method:
                seg_combo.setCurrentIndex(idx)
                break
        seg_layout.addWidget(seg_combo)

        seg_help = QLabel("")
        seg_help.setWordWrap(True)
        seg_help.setStyleSheet("color: palette(dark); font-size: 10px;")
        seg_layout.addWidget(seg_help)

        def update_seg_help(index):
            method = seg_combo.itemData(index)
            if method == 'none':
                seg_help.setText("Each cell grows independently. Masks may overlap.")
            elif method == 'competitive':
                seg_help.setText("All cells grow simultaneously. Pixels claimed by whichever cell reaches first.")
            elif method == 'watershed':
                seg_help.setText("Watershed basins confine each cell's growth to its territory.")

        seg_combo.currentIndexChanged.connect(update_seg_help)
        update_seg_help(seg_combo.currentIndex())

        seg_group.setLayout(seg_layout)
        layout.addWidget(seg_group)

        # --- Circular Growth Constraint ---
        circ_group = QGroupBox("Circular Growth Constraint")
        circ_layout = QVBoxLayout()

        circular_check = QCheckBox("Limit growth to circular boundary around soma")
        circular_check.setChecked(self.use_circular_constraint)
        circ_layout.addWidget(circular_check)

        buffer_layout = QHBoxLayout()
        buffer_layout.addWidget(QLabel("Buffer:"))
        buffer_spin = QSpinBox()
        buffer_spin.setRange(0, 2000)
        buffer_spin.setSingleStep(50)
        buffer_spin.setValue(self.circular_buffer_um2)
        buffer_spin.setSuffix(" µm²")
        buffer_layout.addWidget(buffer_spin)
        circ_layout.addLayout(buffer_layout)

        circ_group.setLayout(circ_layout)
        layout.addWidget(circ_group)

        layout.addSpacing(10)

        # Buttons
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        button_layout.addStretch()
        ok_btn = QPushButton("Generate Cluster Script")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(dialog.accept)
        ok_btn.setStyleSheet(
            "QPushButton { border: 2px solid #4CAF50; font-weight: bold; padding: 8px; }")
        button_layout.addWidget(ok_btn)
        layout.addLayout(button_layout)

        dialog.setLayout(layout)
        dialog.setMinimumWidth(500)

        if dialog.exec_() != QDialog.Accepted:
            return

        # --- Read settings from dialog ---
        pixel_size = self._get_pixel_size()

        use_min_intensity = min_intensity_check.isChecked()
        min_intensity_percent = min_intensity_slider.value()
        mask_min_area = min_area_spin.value()
        mask_max_area = max_area_spin.value()
        mask_step_size = step_spin.value()
        seg_method = seg_combo.currentData()
        use_circular_constraint = circular_check.isChecked()
        circular_buffer_um2 = buffer_spin.value()

        if mask_min_area > mask_max_area:
            QMessageBox.warning(self, "Warning", "Min area must be <= Max area.")
            return

        # Save settings back to instance for next time
        self.use_min_intensity = use_min_intensity
        self.min_intensity_percent = min_intensity_percent
        self.mask_min_area = mask_min_area
        self.mask_max_area = mask_max_area
        self.mask_step_size = mask_step_size
        self.mask_segmentation_method = seg_method
        self.use_circular_constraint = use_circular_constraint
        self.circular_buffer_um2 = circular_buffer_um2

        # --- Build area list ---
        area_list = list(range(mask_min_area, mask_max_area + 1, mask_step_size))
        if area_list[-1] != mask_max_area:
            area_list.append(mask_max_area)

        # Build per-image soma data
        image_data = {}
        for img_name, img_data in self.images.items():
            if not img_data['selected'] or not img_data.get('soma_outlines'):
                continue
            somas = []
            for outline in img_data['soma_outlines']:
                somas.append({
                    'soma_idx': outline['soma_idx'],
                    'soma_id': outline['soma_id'],
                    'centroid': [float(outline['centroid'][0]), float(outline['centroid'][1])],
                    'soma_area_um2': outline.get('soma_area_um2', 0),
                    'polygon_points': [[float(p[0]), float(p[1])] for p in outline.get('polygon_points', [])],
                })
            # Include per-image pixel size if set
            img_entry = {
                'processed_filename': os.path.splitext(img_name)[0] + '_processed.tif',
                'somas': somas,
            }
            per_img_px = img_data.get('pixel_size')
            if per_img_px is not None:
                img_entry['pixel_size'] = per_img_px
            image_data[img_name] = img_entry

        # Ask where to save — script goes into a "MaskGeneration" subfolder
        parent_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory for Cluster Script")
        if not parent_dir:
            return
        save_dir = os.path.join(parent_dir, "MaskGeneration")
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "cluster_mask_generation.py")

        # Build the script content
        settings = {
            'pixel_size_um': pixel_size,
            'area_list_um2': area_list,
            'segmentation_method': seg_method,
            'use_min_intensity': use_min_intensity,
            'min_intensity_percent': min_intensity_percent,
            'use_circular_constraint': use_circular_constraint,
            'circular_buffer_um2': circular_buffer_um2,
        }

        script = self._build_cluster_script(settings, image_data, path)

        try:
            with open(path, 'w') as f:
                f.write(script)
            self.log(f"Cluster script saved to: {save_dir}/")
            self.log(f"  Global pixel size: {pixel_size} µm/px")
            if per_image_overrides:
                self.log(f"  Per-image pixel sizes embedded for {len(per_image_overrides)} image(s):")
                for name, px in per_image_overrides.items():
                    self.log(f"    {os.path.splitext(name)[0]}: {px} µm/px")
            self.log(f"  Segmentation: {seg_method}")
            self.log(f"  Min intensity: {'On (' + str(min_intensity_percent) + '%)' if use_min_intensity else 'Off'}")
            self.log(f"  Circular constraint: {'On (buffer=' + str(circular_buffer_um2) + ' µm²)' if use_circular_constraint else 'Off'}")

            n_images = len(image_data)
            n_somas = sum(len(d['somas']) for d in image_data.values())
            n_masks = n_somas * len(area_list)
            QMessageBox.information(self, "Cluster Script Exported",
                f"Script saved to:\n{save_dir}\n\n"
                f"Images: {n_images}\n"
                f"Somas: {n_somas}\n"
                f"Masks to generate: {n_masks}\n"
                f"Mask sizes: {', '.join(str(a) for a in area_list)} µm²\n"
                f"Segmentation: {seg_method}\n"
                f"{'Per-image pixel sizes: ' + str(len(per_image_overrides)) + ' overrides' if per_image_overrides else 'Pixel size: ' + str(pixel_size) + ' µm/px'}\n\n"
                f"Upload to your cluster along with the _processed.tif files,\n"
                f"then run:\n"
                f"  python {os.path.basename(path)} --input-dir /path/to/processed/tifs")
        except Exception as e:
            self.log(f"ERROR saving cluster script: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save script:\n{e}")

    def _build_cluster_script(self, settings, image_data, path=None):
        """Build the standalone Python script string for cluster mask generation."""
        # Convert JSON to valid Python literals (true->True, false->False, null->None)
        settings_json = json.dumps(settings, indent=4).replace(': true', ': True').replace(': false', ': False').replace(': null', ': None')
        image_data_json = json.dumps(image_data, indent=4).replace(': true', ': True').replace(': false', ': False').replace(': null', ': None')

        script = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cluster Mask Generation Script
Generated by MMPS on {timestamp}

This script generates cell masks from processed microscopy images.
It is self-contained and requires only: numpy, tifffile, scipy, opencv-python

Usage:
    # Process all images:
    python {script_name} --input-dir /path/to/processed/tifs

    # Process a single image (for SLURM array jobs):
    python {script_name} --input-dir /path/to/processed/tifs --image-index 0

    # SLURM array job example:
    # sbatch --array=0-{max_index} job.sh
    # In job.sh:
    #   python {script_name} --input-dir $INPUT_DIR --image-index $SLURM_ARRAY_TASK_ID

Required files in --input-dir:
    - *_processed.tif files (one per image)

Output:
    - masks/ folder with *_mask.tif files
    - masks_results.csv with mask metadata
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import tifffile
from matplotlib.path import Path as mplPath

try:
    import cv2
except ImportError:
    print("WARNING: opencv-python not available. Watershed segmentation will not work.")
    cv2 = None

# ============================================================================
# SETTINGS (from MMPS session)
# ============================================================================

SETTINGS = {settings_json}

IMAGE_DATA = {image_data_json}

# ============================================================================
# MASK GENERATION ALGORITHMS
# ============================================================================

def polygon_to_mask(polygon_points, shape):
    """Convert polygon points to a binary mask."""
    if len(polygon_points) < 3:
        return np.zeros(shape, dtype=np.uint8)
    poly_array = np.array([[p[1], p[0]] for p in polygon_points])
    h, w = shape[:2]
    yy, xx = np.mgrid[:h, :w]
    points = np.c_[xx.ravel(), yy.ravel()]
    path = mplPath(poly_array)
    mask = path.contains_points(points).reshape(h, w)
    return mask.astype(np.uint8)


def create_competitive_masks(processed_img, soma_outlines_data, area_list_um2,
                              pixel_size_um, img_name, use_min_intensity, min_intensity_percent,
                              use_circular_constraint=False, circular_buffer_um2=200):
    """Create masks for ALL somas using competitive priority region growing.

    All somas grow simultaneously from a single shared priority queue.
    Each pixel is claimed by whichever soma reaches it first (brightest-
    neighbor-first). This naturally creates territory boundaries along
    intensity valleys between cells.
    """
    import heapq

    h, w = processed_img.shape
    sorted_areas = sorted(area_list_um2, reverse=True)
    largest_target_px = int(sorted_areas[0] / (pixel_size_um ** 2))

    intensity_floor = 0.0
    if use_min_intensity and min_intensity_percent > 0:
        img_max = processed_img.max()
        if img_max > 0:
            intensity_floor = img_max * (min_intensity_percent / 100.0)

    roi = processed_img.astype(np.float64)

    # Circular constraint
    max_radius_px_sq = None
    if use_circular_constraint:
        constraint_area_um2 = sorted_areas[0] + circular_buffer_um2
        constraint_area_px = constraint_area_um2 / (pixel_size_um ** 2)
        max_radius_px = np.sqrt(constraint_area_px / np.pi)
        max_radius_px_sq = max_radius_px ** 2

    owner_map = np.full((h, w), -1, dtype=np.int32)
    visited = np.zeros((h, w), dtype=bool)
    heap = []

    n_somas = len(soma_outlines_data)
    growth_orders = [[] for _ in range(n_somas)]
    soma_seed_counts = [0] * n_somas
    soma_centroids = []

    for si, soma_data in enumerate(soma_outlines_data):
        centroid = soma_data['centroid']
        cy, cx = int(centroid[0]), int(centroid[1])
        cy = max(0, min(h - 1, cy))
        cx = max(0, min(w - 1, cx))
        soma_centroids.append((cy, cx))
        soma_outline = soma_data.get('outline')

        seeded = False
        if soma_outline is not None and soma_outline.shape == (h, w):
            soma_ys, soma_xs = np.where(soma_outline > 0)
            for sr, sc in zip(soma_ys, soma_xs):
                if not visited[sr, sc]:
                    visited[sr, sc] = True
                    owner_map[sr, sc] = si
                    growth_orders[si].append((sr, sc))
                    soma_seed_counts[si] += 1
            for sr, sc in zip(soma_ys, soma_xs):
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = sr + dr, sc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                        if roi[nr, nc] >= intensity_floor:
                            if max_radius_px_sq is not None:
                                dy, dx = nr - cy, nc - cx
                                if (dy * dy + dx * dx) > max_radius_px_sq:
                                    continue
                            visited[nr, nc] = True
                            owner_map[nr, nc] = si
                            heapq.heappush(heap, (-roi[nr, nc], nr, nc, si))
            seeded = True

        if not seeded:
            if not visited[cy, cx]:
                visited[cy, cx] = True
                owner_map[cy, cx] = si
                growth_orders[si].append((cy, cx))
                soma_seed_counts[si] = 1
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cy + dr, cx + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                        if roi[nr, nc] >= intensity_floor:
                            if max_radius_px_sq is not None:
                                dy, dx = nr - cy, nc - cx
                                if (dy * dy + dx * dx) > max_radius_px_sq:
                                    continue
                            visited[nr, nc] = True
                            owner_map[nr, nc] = si
                            heapq.heappush(heap, (-roi[nr, nc], nr, nc, si))

    soma_done = [False] * n_somas
    while heap:
        neg_intensity, r, c, si = heapq.heappop(heap)
        if owner_map[r, c] != si:
            continue
        if len(growth_orders[si]) >= largest_target_px:
            soma_done[si] = True
            if all(soma_done):
                break
            continue
        growth_orders[si].append((r, c))
        sc_cy, sc_cx = soma_centroids[si]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                if roi[nr, nc] >= intensity_floor:
                    if max_radius_px_sq is not None:
                        dy, dx = nr - sc_cy, nc - sc_cx
                        if (dy * dy + dx * dx) > max_radius_px_sq:
                            continue
                    visited[nr, nc] = True
                    owner_map[nr, nc] = si
                    heapq.heappush(heap, (-roi[nr, nc], nr, nc, si))

    all_masks = []
    for si, soma_data in enumerate(soma_outlines_data):
        soma_idx = soma_data['soma_idx']
        soma_id = soma_data['soma_id']
        soma_area_um2 = soma_data.get('soma_area_um2', 0)
        soma_area_px = soma_seed_counts[si]
        go = growth_orders[si]

        print(f"  {{soma_id}}: soma={{soma_area_px}}px, grew to {{len(go)}}px (target: {{largest_target_px}})")

        soma_masks_start = len(all_masks)
        mask_pixel_counts = []
        for target_area_um2 in sorted_areas:
            target_px = int(target_area_um2 / (pixel_size_um ** 2))
            n_pixels = min(target_px, len(go))
            n_pixels = max(n_pixels, soma_area_px)
            n_pixels = min(n_pixels, len(go))
            mask_pixel_counts.append(n_pixels)

            full_mask = np.zeros((h, w), dtype=np.uint8)
            for r, c in go[:n_pixels]:
                full_mask[r, c] = 1

            actual_area_um2 = n_pixels * (pixel_size_um ** 2)
            print(f"    {{target_area_um2}} um2: {{n_pixels}} px = {{actual_area_um2:.1f}} um2 actual")

            all_masks.append({{
                'image_name': img_name,
                'soma_idx': soma_idx,
                'soma_id': soma_id,
                'area_um2': target_area_um2,
                'mask': full_mask,
                'soma_area_um2': soma_area_um2,
            }})

        # Auto-reject duplicate masks
        pixel_count_groups = {{}}
        for i, n_px in enumerate(mask_pixel_counts):
            pixel_count_groups.setdefault(n_px, []).append(i)
        for n_px, indices in pixel_count_groups.items():
            if len(indices) > 1:
                keep_idx = indices[-1]
                for idx in indices[:-1]:
                    all_masks[soma_masks_start + idx]['duplicate'] = True
                    print(f"    Auto-rejected {{all_masks[soma_masks_start + idx]['area_um2']}} um2 "
                          f"(duplicate of {{all_masks[soma_masks_start + keep_idx]['area_um2']}} um2, both {{n_px}} px)")

        # Enforce subset invariant for this soma's masks
        soma_masks_slice = all_masks[soma_masks_start:]
        _enforce_mask_subset_invariant(soma_masks_slice)

    return all_masks


def create_annulus_masks(centroid, area_list_um2, pixel_size_um, soma_idx, soma_id,
                          processed_img, img_name, soma_area_um2,
                          soma_outline_mask=None, territory_map=None,
                          use_min_intensity=False, min_intensity_percent=0,
                          use_circular_constraint=False, circular_buffer_um2=200):
    """Create nested cell masks using priority region growing from the soma outline."""
    import heapq

    masks = []
    cy, cx = int(centroid[0]), int(centroid[1])

    sorted_areas = sorted(area_list_um2, reverse=True)
    largest_target_px = int(sorted_areas[0] / (pixel_size_um ** 2))

    min_roi_radius = int(np.sqrt(largest_target_px / np.pi) * 3)
    roi_size = max(200, min_roi_radius)

    y_min = max(0, cy - roi_size)
    y_max = min(processed_img.shape[0], cy + roi_size)
    x_min = max(0, cx - roi_size)
    x_max = min(processed_img.shape[1], cx + roi_size)

    roi = processed_img[y_min:y_max, x_min:x_max].astype(np.float64)
    cy_roi, cx_roi = cy - y_min, cx - x_min
    h, w = roi.shape

    cy_roi = max(0, min(h - 1, cy_roi))
    cx_roi = max(0, min(w - 1, cx_roi))

    # Circular constraint
    max_radius_px_sq = None
    if use_circular_constraint:
        constraint_area_um2 = sorted_areas[0] + circular_buffer_um2
        constraint_area_px = constraint_area_um2 / (pixel_size_um ** 2)
        max_radius_px = np.sqrt(constraint_area_px / np.pi)
        max_radius_px_sq = max_radius_px ** 2

    intensity_floor = 0.0
    if use_min_intensity and min_intensity_percent > 0:
        roi_max = roi.max()
        if roi_max > 0:
            intensity_floor = roi_max * (min_intensity_percent / 100.0)

    territory_roi = None
    my_label = 0
    if territory_map is not None:
        territory_roi = territory_map[y_min:y_max, x_min:x_max]
        my_label = territory_roi[cy_roi, cx_roi]
        if my_label <= 0:
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    nr, nc = cy_roi + dr, cx_roi + dc
                    if 0 <= nr < h and 0 <= nc < w and territory_roi[nr, nc] > 0:
                        my_label = territory_roi[nr, nc]
                        break
                if my_label > 0:
                    break

    def _in_territory(r, c):
        if territory_roi is None:
            return True
        return territory_roi[r, c] == my_label or territory_roi[r, c] <= 0

    def _in_circle(r, c):
        if max_radius_px_sq is None:
            return True
        dy = r - cy_roi
        dx = c - cx_roi
        return (dy * dy + dx * dx) <= max_radius_px_sq

    visited = np.zeros((h, w), dtype=bool)
    growth_order = []
    heap = []

    soma_seed_count = 0
    if soma_outline_mask is not None:
        outline_roi = soma_outline_mask[y_min:y_max, x_min:x_max]
        soma_ys, soma_xs = np.where(outline_roi > 0)
        for sr, sc in zip(soma_ys, soma_xs):
            if not visited[sr, sc]:
                visited[sr, sc] = True
                growth_order.append((sr, sc))
                soma_seed_count += 1
        for sr, sc in zip(soma_ys, soma_xs):
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = sr + dr, sc + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                    if roi[nr, nc] >= intensity_floor and _in_territory(nr, nc) and _in_circle(nr, nc):
                        visited[nr, nc] = True
                        heapq.heappush(heap, (-roi[nr, nc], nr, nc))

    if soma_seed_count == 0:
        visited[cy_roi, cx_roi] = True
        growth_order.append((cy_roi, cx_roi))
        soma_seed_count = 1
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cy_roi + dr, cx_roi + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                if roi[nr, nc] >= intensity_floor and _in_territory(nr, nc) and _in_circle(nr, nc):
                    visited[nr, nc] = True
                    heapq.heappush(heap, (-roi[nr, nc], nr, nc))

    while heap and len(growth_order) < largest_target_px:
        neg_intensity, r, c = heapq.heappop(heap)
        growth_order.append((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                if roi[nr, nc] >= intensity_floor and _in_territory(nr, nc) and _in_circle(nr, nc):
                    visited[nr, nc] = True
                    heapq.heappush(heap, (-roi[nr, nc], nr, nc))

    print(f"  {{soma_id}}: soma={{soma_seed_count}}px, grew to {{len(growth_order)}}px (target: {{largest_target_px}})")

    soma_area_px = soma_seed_count
    mask_pixel_counts = []
    for target_area_um2 in sorted_areas:
        target_px = int(target_area_um2 / (pixel_size_um ** 2))

        # Per-step circular constraint: ring grows with each target
        if use_circular_constraint:
            step_constraint_um2 = target_area_um2 + circular_buffer_um2
            step_constraint_px = step_constraint_um2 / (pixel_size_um ** 2)
            step_radius_sq = step_constraint_px / np.pi
            step_order = [(r, c) for r, c in growth_order
                          if (r - cy_roi) ** 2 + (c - cx_roi) ** 2 <= step_radius_sq]
        else:
            step_order = growth_order

        n_pixels = min(target_px, len(step_order))
        n_pixels = max(n_pixels, soma_area_px)
        n_pixels = min(n_pixels, len(step_order))
        mask_pixel_counts.append(n_pixels)

        mask_roi = np.zeros((h, w), dtype=np.uint8)
        for r, c in step_order[:n_pixels]:
            mask_roi[r, c] = 1

        full_mask = np.zeros(processed_img.shape, dtype=np.uint8)
        full_mask[y_min:y_max, x_min:x_max] = mask_roi

        actual_area_um2 = n_pixels * (pixel_size_um ** 2)
        print(f"    {{target_area_um2}} um2: {{n_pixels}} px = {{actual_area_um2:.1f}} um2 actual")

        masks.append({{
            'image_name': img_name,
            'soma_idx': soma_idx,
            'soma_id': soma_id,
            'area_um2': target_area_um2,
            'mask': full_mask,
            'soma_area_um2': soma_area_um2,
        }})

    # Auto-reject duplicates
    pixel_count_groups = {{}}
    for i, n_px in enumerate(mask_pixel_counts):
        pixel_count_groups.setdefault(n_px, []).append(i)
    for n_px, indices in pixel_count_groups.items():
        if len(indices) > 1:
            keep_idx = indices[-1]
            for idx in indices[:-1]:
                masks[idx]['duplicate'] = True
                print(f"    Auto-rejected {{masks[idx]['area_um2']}} um2 "
                      f"(duplicate of {{masks[keep_idx]['area_um2']}} um2, both {{n_px}} px)")

    # Enforce subset invariant
    _enforce_mask_subset_invariant(masks)

    return masks


def build_watershed_territory_map(processed_img, soma_outlines, pixel_size_um):
    """Build watershed territory map assigning each pixel to the nearest soma basin."""
    if cv2 is None:
        print("ERROR: opencv-python required for watershed segmentation")
        sys.exit(1)

    h, w = processed_img.shape
    img_norm = processed_img.astype(np.float64)
    imin, imax = img_norm.min(), img_norm.max()
    if imax > imin:
        img_norm = (img_norm - imin) / (imax - imin) * 255.0
    img_u8 = img_norm.astype(np.uint8)

    grad_x = cv2.Sobel(img_u8, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_u8, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)
    gradient = (gradient / (gradient.max() + 1e-10) * 255).astype(np.uint8)

    markers = np.zeros((h, w), dtype=np.int32)
    for i, soma_data in enumerate(soma_outlines):
        label = i + 1
        centroid = soma_data['centroid']
        cy, cx = int(centroid[0]), int(centroid[1])
        cy = max(0, min(h - 1, cy))
        cx = max(0, min(w - 1, cx))
        outline_mask = soma_data.get('outline')
        if outline_mask is not None and outline_mask.shape == (h, w):
            markers[outline_mask > 0] = label
        else:
            cv2.circle(markers, (cx, cy), max(3, int(5 / pixel_size_um)), label, -1)

    grad_color = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    cv2.watershed(grad_color, markers)

    territory = markers.copy()
    territory[territory < 0] = 0
    return territory


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_image(img_name, img_info, input_dir, output_dir, settings):
    """Process a single image: generate all masks and save to disk."""
    # Use per-image pixel size if available, otherwise global
    pixel_size = img_info.get('pixel_size', settings['pixel_size_um'])
    area_list = settings['area_list_um2']
    seg_method = settings['segmentation_method']
    use_min_intensity = settings['use_min_intensity']
    min_intensity_percent = settings['min_intensity_percent']
    use_circular_constraint = settings.get('use_circular_constraint', False)
    circular_buffer_um2 = settings.get('circular_buffer_um2', 200)

    # Load processed image
    processed_path = os.path.join(input_dir, img_info['processed_filename'])
    if not os.path.exists(processed_path):
        print(f"ERROR: Processed image not found: {{processed_path}}")
        return []

    print(f"Loading {{img_info['processed_filename']}}...")
    processed_img = tifffile.imread(processed_path)
    if processed_img.ndim == 3:
        from skimage import color
        processed_img = (color.rgb2gray(processed_img) * 255).astype(np.uint8)

    h, w = processed_img.shape
    print(f"  Image size: {{w}}x{{h}}")

    # Reconstruct soma outline masks from polygon points
    soma_outlines = []
    for soma in img_info['somas']:
        outline_mask = None
        if soma.get('polygon_points') and len(soma['polygon_points']) >= 3:
            outline_mask = polygon_to_mask(soma['polygon_points'], processed_img.shape)
        soma_outlines.append({{
            'soma_idx': soma['soma_idx'],
            'soma_id': soma['soma_id'],
            'centroid': soma['centroid'],
            'soma_area_um2': soma.get('soma_area_um2', 0),
            'outline': outline_mask,
        }})

    print(f"  Somas: {{len(soma_outlines)}}")
    print(f"  Segmentation: {{seg_method}}")

    # Generate masks
    if seg_method == 'competitive':
        print(f"  Using competitive growth for {{len(soma_outlines)}} cells")
        masks = create_competitive_masks(
            processed_img, soma_outlines, area_list, pixel_size, img_name,
            use_min_intensity, min_intensity_percent,
            use_circular_constraint=use_circular_constraint,
            circular_buffer_um2=circular_buffer_um2,
        )
    else:
        territory_map = None
        if seg_method == 'watershed':
            print(f"  Computing watershed territories...")
            territory_map = build_watershed_territory_map(
                processed_img, soma_outlines, pixel_size
            )

        masks = []
        for soma_data in soma_outlines:
            m = create_annulus_masks(
                soma_data['centroid'], area_list, pixel_size,
                soma_data['soma_idx'], soma_data['soma_id'],
                processed_img, img_name, soma_data.get('soma_area_um2', 0),
                soma_outline_mask=soma_data.get('outline'),
                territory_map=territory_map,
                use_min_intensity=use_min_intensity,
                min_intensity_percent=min_intensity_percent,
                use_circular_constraint=use_circular_constraint,
                circular_buffer_um2=circular_buffer_um2,
            )
            masks.extend(m)

    # Save masks to disk
    masks_dir = os.path.join(output_dir, 'masks')
    os.makedirs(masks_dir, exist_ok=True)

    img_basename = os.path.splitext(img_name)[0]
    saved_count = 0
    for mask_data in masks:
        if mask_data.get('duplicate'):
            continue
        mask = mask_data['mask']
        if mask is None or not np.any(mask):
            continue

        mask_filename = f"{{img_basename}}_{{mask_data['soma_id']}}_area{{int(mask_data['area_um2'])}}_mask.tif"
        mask_path = os.path.join(masks_dir, mask_filename)

        mask_8bit = (mask > 0).astype(np.uint8) * 255
        tifffile.imwrite(
            mask_path, mask_8bit,
            resolution=(1.0 / pixel_size, 1.0 / pixel_size),
            metadata={{'unit': 'um'}}
        )
        saved_count += 1

    print(f"  Saved {{saved_count}} masks to {{masks_dir}}")
    return masks


def main():
    parser = argparse.ArgumentParser(description='Cluster Mask Generation for MMPS')
    parser.add_argument('--input-dir', required=True,
                        help='Directory containing _processed.tif files')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: same as input-dir)')
    parser.add_argument('--image-index', type=int, default=None,
                        help='Process only this image index (for SLURM array jobs)')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dir

    image_names = list(IMAGE_DATA.keys())

    if args.image_index is not None:
        if args.image_index < 0 or args.image_index >= len(image_names):
            print(f"ERROR: --image-index {{args.image_index}} out of range (0-{{len(image_names) - 1}})")
            sys.exit(1)
        image_names = [image_names[args.image_index]]

    print("=" * 60)
    print("MMPS Cluster Mask Generation")
    print(f"Images to process: {{len(image_names)}}")
    print(f"Pixel size: {{SETTINGS['pixel_size_um']}} um")
    print(f"Mask sizes: {{SETTINGS['area_list_um2']}} um2")
    print(f"Segmentation: {{SETTINGS['segmentation_method']}}")
    print("=" * 60)

    total_masks = 0
    start_time = time.time()

    for img_name in image_names:
        img_info = IMAGE_DATA[img_name]
        print(f"\\nProcessing {{img_name}}...")
        t0 = time.time()
        masks = process_image(img_name, img_info, args.input_dir, args.output_dir, SETTINGS)
        elapsed = time.time() - t0
        total_masks += len([m for m in masks if not m.get('duplicate')])
        print(f"  Done in {{elapsed:.1f}}s")

    total_time = time.time() - start_time
    print("\\n" + "=" * 60)
    print(f"Complete! Generated {{total_masks}} masks in {{total_time:.1f}}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
'''.format(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            script_name=os.path.basename(path) if path else 'cluster_mask_generation.py',
            max_index=max(0, len(image_data) - 1),
            settings_json=settings_json,
            image_data_json=image_data_json,
        )

        return script

    def _toggle_colocalization_mode(self, checked):
        """Toggle colocalization mode on/off from the Mode menu."""
        self.colocalization_mode = checked
        self.coloc_action.setChecked(checked)

        if checked:
            self.log("=" * 50)
            self.log("COLOCALIZATION MODE ENABLED")
            self.log("Images will be displayed in color")
            self.log("Use 'Channel Display' button to select which channels to show")
            self.log("Press C to toggle between color and grayscale")
            self.log("=" * 50)
            self.show_color_view = True
            self.color_toggle_btn.setText("Show Grayscale (C)")
            self.channel_select_btn.setVisible(True)
        else:
            self.log("Colocalization mode disabled")
            self.show_color_view = False
            self.color_toggle_btn.setText("Show Color (C)")
            self.channel_select_btn.setVisible(False)

        # Refresh display if an image is loaded
        if self.current_image_name:
            self._display_current_image()

    def _toggle_3d_mode(self, checked):
        """Toggle 3D Z-stack analysis mode within the same GUI."""
        if checked and not _HAS_3D:
            QMessageBox.warning(self, "Error",
                "3DMicroglia.py not found.\n\n"
                "Make sure 3DMicroglia.py is in the same folder as MMPS.")
            self.mode_3d_action.setChecked(False)
            return

        if checked and self.images:
            reply = QMessageBox.question(
                self, "Switch to 3D Mode",
                "Switching to 3D mode will clear current images.\nContinue?",
                QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                self.mode_3d_action.setChecked(False)
                return
            self.images.clear()
            self.file_list.clear()
            self.current_image_name = None
            self.all_masks_flat.clear()

        self.mode_3d = checked

        # Show/hide 3D-specific UI
        self.z_slider_widget.setVisible(checked)
        self.voxel_z_label.setVisible(checked)
        self.voxel_z_input.setVisible(checked)
        if checked:
            self.pixel_size_label.setText("XY pixel size (μm/px):")
            self.setWindowTitle("Microglia Analysis - 3D Z-Stack Mode")
        else:
            self.pixel_size_label.setText("Pixel size (μm/px):")
            self.setWindowTitle("Microglia Analysis - Multi-Image Batch Processing")

        self.log(f"{'3D Z-Stack' if checked else '2D'} mode {'enabled' if checked else 'restored'}")

    # ----------------------------------------------------------------
    # 3D DISPLAY HELPERS
    # ----------------------------------------------------------------

    def _get_current_z_max(self):
        """Get the Z depth of the current image's stack."""
        if not self.current_image_name or self.current_image_name not in self.images:
            return 0
        img_data = self.images[self.current_image_name]
        stack = img_data.get('processed') or img_data.get('raw_stack')
        if stack is not None and stack.ndim >= 3:
            return stack.shape[0] - 1
        return 0

    def _get_slice_for_display(self, stack, z=None):
        """Get a 2D slice from a 3D stack for display."""
        if stack is None:
            return np.zeros((100, 100), dtype=np.uint8)
        if z is None:
            z = self.current_z_slice
        z = max(0, min(stack.shape[0] - 1, z))
        return stack[z]

    def _on_z_slider_changed(self, value):
        self.current_z_slice = value
        z_max = self.z_slider.maximum()
        self.z_label.setText(f"{value} / {z_max}")
        self._refresh_3d_view()

    def _refresh_3d_view(self):
        """Refresh the currently visible tab with the current Z-slice."""
        if not self.mode_3d or not self.current_image_name:
            return
        img_data = self.images.get(self.current_image_name)
        if not img_data:
            return

        if self.mask_qa_active:
            self._show_current_mask()
            return

        tab_idx = self.tabs.currentIndex()
        if tab_idx == 0:  # Original
            stack = img_data.get('raw_stack')
            if stack is not None:
                sl = self._get_slice_for_display(stack)
                adjusted = self._apply_display_adjustments(sl)
                pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
                self.original_label.set_image(pixmap)
        elif tab_idx == 1:  # Preview
            stack = img_data.get('preview_stack')
            if stack is not None:
                sl = self._get_slice_for_display(stack)
                adjusted = self._apply_display_adjustments(sl)
                pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
                self.preview_label.set_image(pixmap)
        elif tab_idx == 2:  # Processed
            stack = img_data.get('processed')
            if stack is not None:
                sl = self._get_slice_for_display(stack)
                adjusted = self._apply_display_adjustments(sl)
                centroids_2d = self._get_centroids_on_slice_3d(img_data)
                pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
                self.processed_label.set_image(pixmap, centroids=centroids_2d)
        elif tab_idx == 3:  # Masks
            if self.mask_qa_active:
                self._show_current_mask()

    def _get_centroids_on_slice_3d(self, img_data, z_tolerance=2):
        """Get soma centroids visible on the current Z-slice."""
        centroids_2d = []
        z = self.current_z_slice
        for soma in img_data.get('somas', []):
            if len(soma) == 3:
                sz, sy, sx = soma
                if abs(sz - z) <= z_tolerance:
                    centroids_2d.append((sy, sx))
        return centroids_2d

    def _update_z_slider_for_image(self, img_name=None):
        """Update Z-slider range based on current image stack depth."""
        if img_name is None:
            img_name = self.current_image_name
        if not img_name or img_name not in self.images:
            self.z_slider.setRange(0, 0)
            self.z_label.setText("0 / 0")
            return
        z_max = self._get_current_z_max()
        self.z_slider.setRange(0, z_max)
        if self.current_z_slice > z_max:
            self.current_z_slice = z_max // 2
        self.z_slider.setValue(self.current_z_slice)
        self.z_label.setText(f"{self.current_z_slice} / {z_max}")

    def _snap_to_brightest_3d(self, z, y, x):
        """Snap to brightest voxel within a small 3D radius."""
        if not self.current_image_name:
            return z, y, x
        img_data = self.images[self.current_image_name]
        stack = img_data.get('processed') or img_data.get('raw_stack')
        if stack is None:
            return z, y, x
        try:
            voxel_xy = float(self.pixel_size_input.text())
        except ValueError:
            voxel_xy = 0.3
        radius_px = max(1, int(round(2.0 / voxel_xy)))
        Z, H, W = stack.shape
        best_val = -1
        best_z, best_y, best_x = z, y, x
        for dz in range(-1, 2):
            nz = z + dz
            if nz < 0 or nz >= Z:
                continue
            for dy in range(-radius_px, radius_px + 1):
                ny = y + dy
                if ny < 0 or ny >= H:
                    continue
                for dx in range(-radius_px, radius_px + 1):
                    nx = x + dx
                    if nx < 0 or nx >= W:
                        continue
                    if dy ** 2 + dx ** 2 <= radius_px ** 2:
                        val = float(stack[nz, ny, nx])
                        if val > best_val:
                            best_val = val
                            best_z, best_y, best_x = nz, ny, nx
        return best_z, best_y, best_x

    # --- Pixel-size link callbacks ---

    def _on_pixel_size_link_toggled(self, linked):
        """Update the link button appearance and sync Y to X when re-linked."""
        self.pixel_size_link_btn.setText("🔗" if linked else "✂")
        self.pixel_size_link_btn.setToolTip(
            "Link X and Y pixel sizes (isotropic)" if linked
            else "X and Y pixel sizes are independent (anisotropic)"
        )
        if linked:
            # Sync Y to X when re-linking
            self.pixel_size_y_input.setText(self.pixel_size_x_input.text())

    def _on_pixel_size_x_changed(self, text):
        """When linked, mirror X value into Y."""
        if self.pixel_size_link_btn.isChecked():
            self.pixel_size_y_input.blockSignals(True)
            self.pixel_size_y_input.setText(text)
            self.pixel_size_y_input.blockSignals(False)

    def _on_pixel_size_y_changed(self, text):
        """When linked, mirror Y value into X."""
        if self.pixel_size_link_btn.isChecked():
            self.pixel_size_x_input.blockSignals(True)
            self.pixel_size_x_input.setText(text)
            self.pixel_size_x_input.blockSignals(False)

    # --- Pixel-size getters ---

    def _get_pixel_size(self, img_name=None):
        """Get the pixel size for an image, falling back to the global setting.

        For anisotropic pixels, returns the geometric mean sqrt(px_x * px_y)
        so that area conversions (pixel_size**2) remain correct.

        Args:
            img_name: Image name to look up. If None, returns the global pixel size.

        Returns:
            float pixel size in µm/px.
        """
        if img_name and img_name in self.images:
            per_image = self.images[img_name].get('pixel_size')
            if per_image is not None:
                # Per-image may store (x, y) tuple or single float
                if isinstance(per_image, (list, tuple)):
                    px, py = per_image
                    return math.sqrt(px * py)
                return per_image
        px_x, px_y = self._get_pixel_size_xy()
        return math.sqrt(px_x * px_y)

    def _get_pixel_size_xy(self, img_name=None):
        """Get separate X and Y pixel sizes.

        Args:
            img_name: Image name to look up. If None, returns the global pixel sizes.

        Returns:
            tuple (pixel_size_x, pixel_size_y) in µm/px.
        """
        if img_name and img_name in self.images:
            per_image = self.images[img_name].get('pixel_size')
            if per_image is not None:
                if isinstance(per_image, (list, tuple)):
                    return float(per_image[0]), float(per_image[1])
                return float(per_image), float(per_image)
        try:
            px_x = float(self.pixel_size_x_input.text())
        except (ValueError, AttributeError):
            px_x = 0.316
        try:
            px_y = float(self.pixel_size_y_input.text())
        except (ValueError, AttributeError):
            px_y = px_x
        return px_x, px_y

    def _set_per_image_pixel_size(self):
        """Show a dialog to set pixel size overrides for individual images."""
        if not self.images:
            QMessageBox.warning(self, "Warning", "No images loaded.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Per-Image Pixel Size")
        dialog.setModal(True)
        layout = QVBoxLayout()

        info = QLabel(
            "Override pixel size for individual images.\n"
            "Leave blank to use the global pixel size.\n"
            "For anisotropic pixels, enter X and Y separately."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Global pixel size display
        global_px_x = self.pixel_size_x_input.text()
        global_px_y = self.pixel_size_y_input.text()
        if global_px_x == global_px_y:
            global_label = QLabel(f"Global pixel size: {global_px_x} µm/px")
        else:
            global_label = QLabel(f"Global pixel size: X={global_px_x}, Y={global_px_y} µm/px")
        global_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(global_label)

        layout.addSpacing(5)

        # Table of images with pixel size inputs (X and Y columns)
        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Image", "X (µm/px)", "Y (µm/px)"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        table.setColumnWidth(1, 100)
        table.setColumnWidth(2, 100)

        image_names = list(self.images.keys())
        table.setRowCount(len(image_names))

        inputs = {}
        for row, img_name in enumerate(image_names):
            name_item = QTableWidgetItem(os.path.splitext(img_name)[0])
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 0, name_item)

            per_image = self.images[img_name].get('pixel_size')
            x_edit = QLineEdit()
            y_edit = QLineEdit()
            x_edit.setPlaceholderText(f"{global_px_x}")
            y_edit.setPlaceholderText(f"{global_px_y}")

            if per_image is not None:
                if isinstance(per_image, (list, tuple)):
                    x_edit.setText(str(per_image[0]))
                    y_edit.setText(str(per_image[1]))
                else:
                    x_edit.setText(str(per_image))
                    y_edit.setText(str(per_image))

            table.setCellWidget(row, 1, x_edit)
            table.setCellWidget(row, 2, y_edit)
            inputs[img_name] = (x_edit, y_edit)

        layout.addWidget(table)

        # Buttons
        btn_layout = QHBoxLayout()

        clear_btn = QPushButton("Clear All Overrides")
        def _clear_all():
            for x_inp, y_inp in inputs.values():
                x_inp.clear()
                y_inp.clear()
        clear_btn.clicked.connect(_clear_all)
        btn_layout.addWidget(clear_btn)

        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(cancel_btn)

        ok_btn = QPushButton("Apply")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(dialog.accept)
        ok_btn.setStyleSheet("QPushButton { font-weight: bold; }")
        btn_layout.addWidget(ok_btn)

        layout.addLayout(btn_layout)
        dialog.setLayout(layout)
        dialog.setMinimumWidth(600)
        dialog.setMinimumHeight(400)

        if dialog.exec_() != QDialog.Accepted:
            return

        # Apply the pixel sizes
        changed = 0
        for img_name, (x_edit, y_edit) in inputs.items():
            x_text = x_edit.text().strip()
            y_text = y_edit.text().strip()
            if x_text or y_text:
                try:
                    val_x = float(x_text) if x_text else None
                    val_y = float(y_text) if y_text else None
                    if val_x is not None and val_x > 0:
                        if val_y is not None and val_y > 0 and val_x != val_y:
                            self.images[img_name]['pixel_size'] = [val_x, val_y]
                        else:
                            self.images[img_name]['pixel_size'] = val_x
                        changed += 1
                    else:
                        self.images[img_name]['pixel_size'] = None
                except ValueError:
                    self.images[img_name]['pixel_size'] = None
            else:
                self.images[img_name]['pixel_size'] = None

        overrides = {name: data['pixel_size'] for name, data in self.images.items()
                     if data.get('pixel_size') is not None}
        if overrides:
            self.log(f"Per-image pixel sizes set for {len(overrides)} image(s):")
            for name, px in overrides.items():
                if isinstance(px, (list, tuple)):
                    self.log(f"  {os.path.splitext(name)[0]}: X={px[0]}, Y={px[1]} µm/px")
                else:
                    self.log(f"  {os.path.splitext(name)[0]}: {px} µm/px")
        elif changed == 0:
            self.log("All per-image pixel size overrides cleared (using global)")

    def _auto_save(self):
        """Silently auto-save the session to the output directory."""
        if not self.output_dir or not self.images:
            return
        try:
            path = os.path.join(self.output_dir, "autosave.mmps_session")
            session = self._build_session_dict(session_file_path=path)
            with open(path, 'w') as f:
                json.dump(session, f, separators=(',', ':'))
        except Exception:
            pass  # Silent - never interrupt the user's workflow

    def load_session(self):
        """Restore a previously saved session"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "",
            "Session Files (*.mmps_session *.mmps3d_session);;All Files (*)"
        )
        if not path:
            return

        try:
            # Use fastest available JSON parser
            self.log("Loading session file...")
            QApplication.processEvents()
            try:
                import orjson
                with open(path, 'rb') as f:
                    session = orjson.loads(f.read())
            except ImportError:
                try:
                    import ujson
                    with open(path, 'r') as f:
                        session = ujson.load(f)
                except ImportError:
                    with open(path, 'r') as f:
                        session = json.load(f)

            if session.get('version', 1) < 2:
                QMessageBox.warning(self, "Warning", "Incompatible session file version.")
                return

            # Resolve paths: try absolute first, then relative to session file
            session_dir = os.path.dirname(os.path.abspath(path))

            def _resolve_path(abs_path, rel_path):
                """Try absolute path first, then resolve relative path from session dir."""
                if abs_path and os.path.exists(abs_path):
                    return abs_path
                if rel_path:
                    resolved = os.path.normpath(os.path.join(session_dir, rel_path))
                    if os.path.exists(resolved):
                        return resolved
                return abs_path  # return original even if not found

            def _resolve_dir(abs_path, rel_path):
                """Like _resolve_path but for directories."""
                if abs_path and os.path.isdir(abs_path):
                    return abs_path
                if rel_path:
                    resolved = os.path.normpath(os.path.join(session_dir, rel_path))
                    if os.path.isdir(resolved):
                        return resolved
                return abs_path

            # Resolve top-level directories
            resolved_output_dir = _resolve_dir(
                session.get('output_dir'), session.get('output_dir_rel'))
            resolved_masks_dir = _resolve_dir(
                session.get('masks_dir'), session.get('masks_dir_rel'))

            # Resolve per-image paths
            for img_name, img_session in session['images'].items():
                img_session['raw_path'] = _resolve_path(
                    img_session.get('raw_path'), img_session.get('raw_path_rel'))
                img_session['processed_path'] = _resolve_path(
                    img_session.get('processed_path'), img_session.get('processed_path_rel'))
                # Resolve extra channel paths
                extra_rel = img_session.get('extra_channel_paths_rel', {})
                resolved_extra = {}
                for ch_str, ch_path in img_session.get('extra_channel_paths', {}).items():
                    resolved_extra[ch_str] = _resolve_path(ch_path, extra_rel.get(ch_str))
                img_session['extra_channel_paths'] = resolved_extra

            # Verify image files still exist (check both raw and processed)
            missing = []
            for img_name, img_session in session['images'].items():
                raw_exists = os.path.exists(img_session['raw_path'])
                proc_exists = img_session.get('processed_path') and os.path.exists(img_session['processed_path'])
                if not raw_exists and not proc_exists:
                    missing.append(img_name)

            # If many files are missing, offer to remap the base directory
            if missing and len(missing) > len(session['images']) * 0.5:
                reply = QMessageBox.question(
                    self, "Files Not Found",
                    f"{len(missing)} of {len(session['images'])} image(s) not found.\n\n"
                    "This session may have been created on a different computer.\n"
                    "Would you like to select the folder containing your images?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    remap_dir = QFileDialog.getExistingDirectory(
                        self, "Select Folder Containing Images")
                    if remap_dir:
                        # Try to find each missing file by name in the remap directory
                        remap_found = 0
                        for img_name in list(missing):
                            # Search recursively for the file
                            candidates = glob.glob(os.path.join(remap_dir, '**', img_name), recursive=True)
                            if candidates:
                                session['images'][img_name]['raw_path'] = candidates[0]
                                missing.remove(img_name)
                                remap_found += 1
                        # Also remap output/masks dirs if they're under the same tree
                        if resolved_output_dir and not os.path.isdir(resolved_output_dir):
                            # Try to find "Output" folder near images
                            out_candidates = glob.glob(os.path.join(remap_dir, '**', 'Output'), recursive=True)
                            if out_candidates:
                                resolved_output_dir = out_candidates[0]
                                masks_candidate = os.path.join(out_candidates[0], 'masks')
                                if os.path.isdir(masks_candidate):
                                    resolved_masks_dir = masks_candidate
                                # Re-resolve processed paths
                                for img_name, img_session in session['images'].items():
                                    if img_session.get('processed_path') and not os.path.exists(img_session['processed_path']):
                                        name_stem = os.path.splitext(img_name)[0]
                                        candidate = os.path.join(resolved_output_dir, f"{name_stem}_processed.tif")
                                        if os.path.exists(candidate):
                                            img_session['processed_path'] = candidate
                        self.log(f"Remapped {remap_found} image path(s) from selected folder")

            if missing:
                reply = QMessageBox.question(
                    self, "Missing Files",
                    f"{len(missing)} image(s) not found:\n\n" +
                    "\n".join(missing[:5]) +
                    ("\n..." if len(missing) > 5 else "") +
                    "\n\nContinue loading available images?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return

            # Restore settings
            self.output_dir = resolved_output_dir
            self.masks_dir = resolved_masks_dir
            self.colocalization_mode = session.get('colocalization_mode', False)
            self.coloc_action.setChecked(self.colocalization_mode)
            self.use_min_intensity = session.get('use_min_intensity', True)
            self.min_intensity_percent = session.get('min_intensity_percent', 5)
            self.mask_min_area = session.get('mask_min_area', 100)
            self.mask_max_area = session.get('mask_max_area', 800)
            self.mask_step_size = session.get('mask_step_size', 100)
            self.mask_segmentation_method = session.get('mask_segmentation_method', 'none')
            self.use_circular_constraint = session.get('use_circular_constraint', False)
            self.circular_buffer_um2 = session.get('circular_buffer_um2', 200)
            self.coloc_channel_1 = session.get('coloc_channel_1', 0)
            self.coloc_channel_2 = session.get('coloc_channel_2', 1)
            self.grayscale_channel = session.get('grayscale_channel', 0)

            pixel_size = session.get('pixel_size', '0.316')
            pixel_size_y = session.get('pixel_size_y', pixel_size)
            pixel_size_linked = session.get('pixel_size_linked', True)
            self.pixel_size_link_btn.setChecked(pixel_size_linked)
            self.pixel_size_x_input.setText(str(pixel_size))
            self.pixel_size_y_input.setText(str(pixel_size_y))
            self.default_rolling_ball_radius = session.get('rolling_ball_radius', 50)

            # Restore 3D mode state
            is_3d = session.get('mode_3d', False)
            if is_3d and not _HAS_3D:
                QMessageBox.warning(self, "Warning", "This is a 3D session but 3DMicroglia.py was not found.")
                return
            self.mode_3d = is_3d
            self.mode_3d_action.setChecked(is_3d)
            self.z_slider_widget.setVisible(is_3d)
            self.voxel_z_label.setVisible(is_3d)
            self.voxel_z_input.setVisible(is_3d)
            if is_3d:
                vz = session.get('voxel_size_z', '1.0')
                self.voxel_z_input.setText(str(vz))
                self.pixel_size_label.setText("XY pixel size (um/px):")
                self.setWindowTitle("Microglia Analysis - 3D Z-Stack Mode")
                self.mask_min_volume = session.get('mask_min_volume', 500)
                self.mask_max_volume = session.get('mask_max_volume', 5000)
                self.soma_intensity_tolerance = session.get('soma_intensity_tolerance', 30)
                self.soma_max_radius_um = session.get('soma_max_radius_um', 8.0)
            else:
                self.pixel_size_label.setText("Pixel size (um/px):")
                self.setWindowTitle("Microglia Analysis - Multi-Image Batch Processing")

            if self.colocalization_mode:
                self.show_color_view = True
                self.color_toggle_btn.setText("Show Grayscale (C)")
                self.channel_select_btn.setVisible(True)

            # Ensure output dirs exist — if the saved path is unreachable
            # (e.g. session created on a different computer), ask the user
            # to pick a new output folder instead of failing with PermissionError.
            def _dir_is_usable(d):
                """Check if a directory exists or can be created."""
                if not d:
                    return False
                if os.path.isdir(d):
                    return True
                try:
                    os.makedirs(d, exist_ok=True)
                    return True
                except OSError:
                    return False

            if self.output_dir and not _dir_is_usable(self.output_dir):
                reply = QMessageBox.question(
                    self, "Output Folder Not Found",
                    f"The saved output folder is not accessible:\n"
                    f"{self.output_dir}\n\n"
                    "This session may have been created on a different computer.\n"
                    "Would you like to select a new output folder?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    new_output = QFileDialog.getExistingDirectory(
                        self, "Select Output Folder")
                    if new_output:
                        self.output_dir = new_output
                        if is_3d:
                            self.masks_dir = os.path.join(new_output, "masks_3d")
                        else:
                            self.masks_dir = os.path.join(new_output, "masks")
                        resolved_output_dir = new_output
                        resolved_masks_dir = self.masks_dir
                        # Re-resolve processed_path for every image so lazy-load
                        # picks up the new output folder instead of stale paths.
                        for img_name, img_session in session['images'].items():
                            name_stem = os.path.splitext(img_name)[0]
                            candidate = os.path.join(new_output, f"{name_stem}_processed.tif")
                            if os.path.exists(candidate):
                                img_session['processed_path'] = candidate
                else:
                    # User declined — clear the invalid paths so we don't
                    # error out later trying to write to them.
                    self.output_dir = None
                    self.masks_dir = None

            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
            if self.masks_dir:
                os.makedirs(self.masks_dir, exist_ok=True)
                self.somas_dir = os.path.join(self.output_dir, "somas") if self.output_dir else None
                if self.somas_dir:
                    os.makedirs(self.somas_dir, exist_ok=True)

            # Check if we can skip the expensive directory scan by using mask_qa_state
            # from the session JSON (which already has all mask metadata)
            has_qa_state = any(
                img_s.get('mask_qa_state') for img_s in session['images'].values()
                if img_s.get('status', 'loaded') in ('masks_generated', 'qa_complete', 'analyzed')
            )
            all_mask_files = None  # lazy: only scan if needed
            if not has_qa_state and self.masks_dir and os.path.isdir(self.masks_dir):
                self.log("Scanning masks directory (no QA state in session)...")
                QApplication.processEvents()
                all_mask_files = sorted(os.listdir(self.masks_dir))

            # Restore images
            from PyQt5.QtGui import QColor, QBrush
            from PyQt5.QtWidgets import QProgressDialog
            self.images = {}
            self.file_list.clear()

            n_total = len(session['images']) - len(missing)
            progress = QProgressDialog("Restoring images...", "Cancel", 0, max(n_total, 1), self)
            progress.setWindowTitle("Loading Session")
            progress.setMinimumDuration(500)  # only show if load takes > 500ms
            progress.setWindowModality(Qt.WindowModal)
            img_counter = 0

            for img_name, img_session in session['images'].items():
                if img_name in missing:
                    continue

                img_counter += 1
                progress.setValue(img_counter)
                progress.setLabelText(f"Restoring {img_name}... ({img_counter}/{n_total})")
                QApplication.processEvents()
                if progress.wasCanceled():
                    self.log("Session load cancelled by user.")
                    break

                # Try to reload processed image from disk
                processed_data = None
                processed_path = img_session.get('processed_path')
                if processed_path and os.path.exists(processed_path):
                    try:
                        processed_data = safe_tiff_read(processed_path)
                    except Exception:
                        processed_data = None

                # Reload extra cleaned channel TIFFs
                processed_channels = {}
                for ch_str, ch_path in img_session.get('extra_channel_paths', {}).items():
                    if os.path.exists(ch_path):
                        try:
                            processed_channels[int(ch_str)] = safe_tiff_read(ch_path)
                        except Exception:
                            pass

                # Reconstruct soma outlines with full metadata
                restored_outlines = []
                for outline_data in img_session.get('soma_outlines', []):
                    if isinstance(outline_data, dict) and 'soma_idx' in outline_data:
                        # New format: full outline dict
                        polygon_pts = [tuple(pt) for pt in outline_data.get('polygon_points', [])]
                        centroid = tuple(outline_data['centroid']) if outline_data.get('centroid') else None
                        # Defer outline mask reconstruction — will be recomputed on
                        # demand from polygon_points when needed (mask generation, display)
                        restored_outlines.append({
                            'soma_idx': outline_data['soma_idx'],
                            'soma_id': outline_data['soma_id'],
                            'centroid': centroid,
                            'outline': None,
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

                img_dict = {
                    'raw_path': img_session['raw_path'],
                    'processed_path': img_session.get('processed_path'),
                    'processed': processed_data,
                    'rolling_ball_radius': img_session.get('rolling_ball_radius', 50),
                    'somas': [tuple(s) for s in img_session.get('somas', [])],
                    'soma_ids': img_session.get('soma_ids', []),
                    'soma_groups': img_session.get('soma_groups', []),
                    'masks': [],
                    'status': img_session.get('status', 'loaded'),
                    'selected': img_session.get('selected', False),
                    'animal_id': img_session.get('animal_id', ''),
                    'treatment': img_session.get('treatment', ''),
                    'pixel_size': img_session.get('pixel_size'),
                }
                if is_3d:
                    img_dict['raw_stack'] = None
                    img_dict['soma_masks'] = {}
                    # In 3D, processed is a stack - reload from disk
                    if processed_data is not None and processed_data.ndim >= 3:
                        img_dict['processed'] = processed_data
                    else:
                        img_dict['processed'] = None
                else:
                    img_dict['processed_channels'] = processed_channels
                    img_dict['soma_outlines'] = restored_outlines
                self.images[img_name] = img_dict

                # If processed image wasn't found, downgrade status
                # But don't downgrade qa_complete — masks on disk are still valid
                if processed_data is None and img_session.get('status') not in ('loaded', 'masks_generated', 'qa_complete', 'analyzed'):
                    self.images[img_name]['status'] = 'loaded'

                # Restore mask metadata WITHOUT reading TIFFs (lazy loading)
                # Mask pixel data is loaded on demand via _reload_mask_from_disk
                orig_status = img_session.get('status', 'loaded')
                if orig_status in ('masks_generated', 'qa_complete', 'analyzed') and self.masks_dir:
                    # Build a lookup of soma outlines for soma_area_um2
                    outline_lookup = {}
                    if not is_3d:
                        for ol in restored_outlines:
                            outline_lookup[ol.get('soma_id', '')] = ol.get('soma_area_um2', 0)
                    soma_ids_list = self.images[img_name]['soma_ids']

                    mask_qa_state = img_session.get('mask_qa_state', [])
                    if mask_qa_state:
                        # FAST PATH: reconstruct masks directly from session metadata
                        # No directory scan needed — all info is in the JSON
                        for qs in mask_qa_state:
                            qs_soma_id = qs.get('soma_id', '')
                            soma_idx = qs.get('soma_idx', 0)
                            if soma_idx == 0 and qs_soma_id in soma_ids_list:
                                soma_idx = soma_ids_list.index(qs_soma_id)
                            mask_entry = {
                                'image_name': img_name,
                                'soma_idx': soma_idx,
                                'soma_id': qs_soma_id,
                                'mask': None,  # Lazy: loaded on demand
                                'approved': qs.get('approved'),
                                'duplicate': qs.get('duplicate', False),
                                'soma_area_um2': outline_lookup.get(qs_soma_id, 0),
                            }
                            if is_3d:
                                mask_entry['volume_um3'] = qs.get('volume_um3', 0)
                            else:
                                mask_entry['area_um2'] = qs.get('area_um2', 0)
                            self.images[img_name]['masks'].append(mask_entry)
                    elif all_mask_files is not None and os.path.isdir(self.masks_dir):
                        # FALLBACK: old session without mask_qa_state — scan directory
                        img_basename = os.path.splitext(img_name)[0]
                        if is_3d:
                            mask_pattern = re.compile(
                                re.escape(img_basename) + r'_(soma_\d+_\d+_\d+)_vol(\d+)_mask3d\.tif$'
                            )
                        else:
                            mask_pattern = re.compile(
                                re.escape(img_basename) + r'_(soma_\d+_\d+)_area(\d+)_mask\.tif$'
                            )
                        # Group masks by soma_id to detect duplicates (same soma,
                        # different target areas that produced identical pixel masks).
                        # For each soma, keep only the largest-area mask per unique
                        # file-size on disk (a rough proxy for pixel count).
                        soma_masks_pending = {}  # soma_id -> [(size_val, mf, soma_idx)]
                        for mf in all_mask_files:
                            m = mask_pattern.match(mf)
                            if not m:
                                continue
                            soma_id = m.group(1)
                            size_val = int(m.group(2))
                            soma_idx = soma_ids_list.index(soma_id) if soma_id in soma_ids_list else 0
                            soma_masks_pending.setdefault(soma_id, []).append(
                                (size_val, mf, soma_idx))

                        approval = True if orig_status in ('qa_complete', 'analyzed') else None
                        for soma_id, entries in soma_masks_pending.items():
                            # Sort largest area first for consistency with QA ordering
                            entries.sort(key=lambda e: e[0], reverse=True)
                            # Detect duplicates: masks with the same file size on disk
                            # are likely pixel-identical (same growth_order prefix)
                            seen_file_sizes = {}
                            for size_val, mf, soma_idx in entries:
                                mf_path = os.path.join(self.masks_dir, mf)
                                try:
                                    fsize = os.path.getsize(mf_path)
                                except OSError:
                                    fsize = -1
                                is_dup = fsize in seen_file_sizes and fsize > 0
                                if not is_dup:
                                    seen_file_sizes[fsize] = size_val
                                mask_entry = {
                                    'image_name': img_name,
                                    'soma_idx': soma_idx,
                                    'soma_id': soma_id,
                                    'mask': None,
                                    'approved': False if is_dup else approval,
                                    'duplicate': is_dup,
                                }
                                if is_3d:
                                    mask_entry['volume_um3'] = size_val
                                else:
                                    mask_entry['area_um2'] = size_val
                                mask_entry['soma_area_um2'] = outline_lookup.get(soma_id, 0)
                                self.images[img_name]['masks'].append(mask_entry)

                    if len(self.images[img_name]['masks']) == 0:
                        print(f"Warning: No masks found for {img_name}")

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

            progress.close()

            # Load and display the last viewed image (or first if not saved)
            if self.images:
                last_img = session.get('last_image_name')
                if last_img and last_img in self.images:
                    self.current_image_name = last_img
                else:
                    self.current_image_name = sorted(self.images.keys())[0]
                self._display_current_image()
                # Set the file list selection to match
                for row_i in range(self.file_list.count()):
                    item = self.file_list.item(row_i)
                    if item and item.data(Qt.UserRole) == self.current_image_name:
                        self.file_list.setCurrentRow(row_i)
                        break
                self.process_selected_btn.setEnabled(True)

            # Rebuild all_masks_flat from loaded masks (sorted by soma pick order, then largest-first)
            self.all_masks_flat = []
            size_key = 'volume_um3' if is_3d else 'area_um2'
            for iname, idata in self.images.items():
                if not idata['selected']:
                    continue
                sorted_masks = sorted(idata['masks'],
                                      key=lambda m: (m.get('soma_idx', 0), -m.get(size_key, 0)))
                for mask_data in sorted_masks:
                    self.all_masks_flat.append({
                        'image_name': iname,
                        'mask_data': mask_data,
                    })

            # Rebuild soma ordering for sliding window memory management
            self._qa_soma_order = []
            self._qa_finalized_somas = set()
            seen_somas = set()
            for flat in self.all_masks_flat:
                key = (flat['image_name'], flat['mask_data']['soma_id'])
                if key not in seen_somas:
                    seen_somas.add(key)
                    self._qa_soma_order.append(key)

            # Enable buttons based on restored state
            self._update_buttons_after_session_load()

            n_loaded = len(self.images)
            n_with_somas = sum(1 for d in self.images.values() if d['somas'])
            n_with_outlines = sum(1 for d in self.images.values() if d['soma_outlines'])
            n_with_processed = sum(1 for d in self.images.values() if d['processed'] is not None)

            # Count masks from restored image data (no directory scan needed)
            n_mask_files = sum(len(d['masks']) for d in self.images.values())

            # Determine last completed step
            last_step = session.get('last_completed_step', self._determine_last_completed_step())
            step_name = self._get_step_display_name(last_step)
            next_hint = self._get_next_step_hint(last_step)

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
            self.log(f"  Last completed step: {step_name}")
            self.log(f"  Next: {next_hint}")

            # Check for in-progress checklist files
            soma_cl_path = self._get_checklist_path('soma_checklist.csv')
            mask_cl_path = self._get_checklist_path('mask_checklist.csv')
            if soma_cl_path and os.path.exists(soma_cl_path):
                rows = self._read_checklist(soma_cl_path)
                done = sum(1 for r in rows if len(r) > 1 and r[1] == '1')
                self.log(f"  Found soma_checklist.csv: {done}/{len(rows)} somas outlined")
            if mask_cl_path and os.path.exists(mask_cl_path):
                rows = self._read_checklist(mask_cl_path)
                done = sum(1 for r in rows if len(r) > 1 and r[1] == '1')
                self.log(f"  Found mask_checklist.csv: {done}/{len(rows)} masks generated")
            qa_cl_path = self._get_checklist_path('mask_qa_checklist.csv')
            if qa_cl_path and os.path.exists(qa_cl_path):
                rows = self._read_checklist(qa_cl_path)
                done = sum(1 for r in rows if len(r) > 1 and r[1] == '1')
                self.log(f"  Found mask_qa_checklist.csv: {done}/{len(rows)} masks QA'd")
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
                f"Mask files on disk: {n_mask_files}\n\n"
                f"Last completed step: {step_name}\n"
                f"Next: {next_hint}"
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
            self.opacity_widget.setVisible(True)
        if has_qa_complete:
            self.batch_calculate_btn.setEnabled(True)

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
                    return  # Don't check QA if resuming outlining

        # Check for partially-QA'd masks and offer to resume
        if has_masks:
            total_masks = sum(len(d['masks']) for d in self.images.values()
                              if d['selected'] and d['status'] in ('masks_generated', 'qa_complete', 'analyzed'))
            reviewed_masks = sum(
                1 for d in self.images.values()
                if d['selected'] and d['status'] in ('masks_generated', 'qa_complete', 'analyzed')
                for m in d['masks'] if m.get('approved') is not None
            )
            if total_masks > 0 and 0 < reviewed_masks < total_masks:
                reply = QMessageBox.question(
                    self, 'Resume Mask QA',
                    f"Found {reviewed_masks}/{total_masks} masks reviewed.\n\n"
                    f"Resume QA for the remaining {total_masks - reviewed_masks} masks?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    self.start_batch_qa()

    # ========================================================================
    # IMPORT IMAGEJ RESULTS
    # ========================================================================

    def _detect_imagej_csv_type(self, file_path):
        """Detect the type of ImageJ result CSV from its column headers.
        Returns one of: 'sholl', 'skeleton', 'fractal', 'combined', or None."""
        import csv
        try:
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader, [])
        except Exception:
            return None

        headers_lower = set(h.strip().lower() for h in headers)

        # Check for Sholl (uses 'Mask Name', 'Primary Branches', etc.)
        sholl_markers = {'mask name', 'primary branches', 'sum of intersections',
                         'intersecting radii', 'enclosing radius'}
        if len(sholl_markers & headers_lower) >= 2:
            return 'sholl'

        # Check for pre-prefixed combined data
        has_sholl_prefix = any(h.startswith('sholl_') for h in headers_lower)
        has_skel_prefix = any(h.startswith('skel_') for h in headers_lower)
        has_fractal_prefix = any(h.startswith('fractal_') for h in headers_lower)
        if sum([has_sholl_prefix, has_skel_prefix, has_fractal_prefix]) >= 2:
            return 'combined'

        # Check for Skeleton
        skel_markers = {'num_branches', 'num_junctions', 'skeleton_area_um2',
                        'num_end_points', 'avg_branch_length_um'}
        if len(skel_markers & headers_lower) >= 2:
            return 'skeleton'

        # Check for Fractal (includes hull columns from same script)
        fractal_markers = {'fractal_dimension', 'fractal_lacunarity_mean',
                           'fractal_r_squared', 'fractal_foreground_pixels'}
        hull_markers = {'hull_area_um2', 'hull_circularity', 'hull_density'}
        if len(fractal_markers & headers_lower) >= 1 or len(hull_markers & headers_lower) >= 2:
            return 'fractal'

        # Filename fallback
        basename = os.path.basename(file_path).lower()
        if 'sholl' in basename:
            return 'sholl'
        if 'skeleton' in basename or 'skel' in basename:
            return 'skeleton'
        if 'fractal' in basename:
            return 'fractal'
        if 'combined_analysis' in basename:
            return 'combined'

        return None

    def _load_imagej_csv(self, file_path, csv_type):
        """Load an ImageJ CSV, returning a dict of {cell_name: {'data': {...}, 'area': int|None}}.
        Applies correct column prefixes (sholl_, skel_, fractal_/hull_) and extracts
        the mask area for area-aware matching."""
        import csv
        import re as _re
        data = {}
        id_columns = {'cell_name', 'image_name', 'soma_id', 'mask_file',
                       'area_um2', 'mask_area_um2', 'cell', 'mask name', 'image name',
                       'soma id', 'mask area (um2)', 'centroid x (px)',
                       'centroid y (px)', 'start radius (um)',
                       'pixel_size_um', 'upscale_factor', 'skeleton_file',
                       'soma area (um2)'}
        try:
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Extract cell_name based on type
                    if csv_type == 'sholl':
                        cell_name = row.get('Mask Name', row.get('Cell', row.get('cell_name', '')))
                    else:
                        cell_name = row.get('cell_name', '')

                    if not cell_name:
                        continue

                    # Extract area for area-aware matching
                    area = None
                    if csv_type == 'sholl':
                        area_str = row.get('Mask Area (um2)', '')
                        if area_str:
                            try:
                                area = int(float(area_str))
                            except (ValueError, TypeError):
                                pass
                        if area is None:
                            m = _re.search(r'_area(\d+)_', cell_name)
                            if m:
                                area = int(m.group(1))
                    else:
                        area_str = row.get('area_um2', row.get('mask_area_um2', ''))
                        if area_str:
                            try:
                                area = int(float(area_str))
                            except (ValueError, TypeError):
                                pass

                    # Build prefixed data columns
                    prefix_map = {'sholl': 'sholl_', 'skeleton': 'skel_', 'fractal': 'fractal_'}
                    prefix = prefix_map.get(csv_type, '')

                    row_data = {}
                    for k, v in row.items():
                        if k.strip().lower() in id_columns:
                            continue
                        if csv_type == 'combined':
                            # Already prefixed from CombinedAnalysis script
                            row_data[k] = v
                        elif k.startswith(prefix) or k.startswith('hull_'):
                            # Already has correct prefix
                            row_data[k] = v
                        else:
                            row_data[prefix + k] = v

                    # Normalize Sholl cell_name: strip full mask filename down
                    # to match skeleton/fractal format (imgname_somaid)
                    if csv_type == 'sholl':
                        # Sholl uses 'Mask Name' which is the full filename
                        # e.g. "img_soma_1_0_area400_mask.tif" -> "img_soma_1_0"
                        import re as _re2
                        norm = _re2.sub(r'_area\d+_mask\.tif$', '', cell_name)
                        if norm == cell_name:
                            norm = _re2.sub(r'_mask\.tif$', '', cell_name)
                        cell_name = norm

                    # Use area-aware key so multiple areas per cell aren't overwritten
                    if area is not None:
                        dict_key = f"{cell_name}_area{area}"
                    else:
                        dict_key = cell_name
                    data[dict_key] = {'data': row_data, 'area': area}
        except Exception as e:
            self.log(f"  ERROR reading {os.path.basename(file_path)}: {e}")
        return data

    def import_imagej_results(self):
        """Import ImageJ result CSVs and merge with morphology results.
        Shows a dialog letting the user pick which CSV types to include."""
        import csv
        import re as _re

        # Determine initial directory for file dialog
        initial_dir = self.output_dir if self.output_dir else os.path.expanduser("~")

        # Show the CSV merge dialog
        dlg = CSVMergeDialog(self, initial_dir=initial_dir)
        if dlg.exec_() != QDialog.Accepted:
            return

        selected_files = dlg.get_selected_files()
        if not selected_files:
            QMessageBox.information(self, "No Files", "No CSV files were selected.")
            return

        # Set output_dir from the first selected file's directory
        if not self.output_dir:
            first_path = next(iter(selected_files.values()))
            self.output_dir = os.path.dirname(first_path)

        self.log("=" * 50)
        self.log("Importing ImageJ results...")

        # Load each selected CSV by its declared type
        sholl_data = {}   # {cell_key: {'data': {...}, 'area': int|None}}
        skeleton_data = {}
        fractal_data = {}
        simple_path = None  # Track user-selected simple morphology CSV

        for csv_type, fp in selected_files.items():
            if csv_type == 'simple':
                simple_path = fp
                self.log(f"  Simple morphology: {os.path.basename(fp)}")
                continue

            self.log(f"  Loading {csv_type}: {os.path.basename(fp)}")
            loaded = self._load_imagej_csv(fp, csv_type)

            if csv_type == 'combined':
                # Split combined data into categories
                for cell_name, entry in loaded.items():
                    rd = entry['data']
                    area = entry['area']
                    s = {k: v for k, v in rd.items() if k.startswith('sholl_')}
                    k_data = {k: v for k, v in rd.items() if k.startswith('skel_')}
                    fr = {k: v for k, v in rd.items() if k.startswith('fractal_') or k.startswith('hull_')}
                    if s:
                        sholl_data[cell_name] = {'data': s, 'area': area}
                    if k_data:
                        skeleton_data[cell_name] = {'data': k_data, 'area': area}
                    if fr:
                        fractal_data[cell_name] = {'data': fr, 'area': area}
            elif csv_type == 'sholl':
                sholl_data.update(loaded)
            elif csv_type == 'skeleton':
                skeleton_data.update(loaded)
            elif csv_type == 'fractal':
                fractal_data.update(loaded)

            self.log(f"    Loaded {len(loaded)} cells")

        if not sholl_data and not skeleton_data and not fractal_data and not simple_path:
            QMessageBox.information(self, "No Data", "No CSV files could be loaded.")
            return

        if sholl_data:
            self.log(f"  Sholl total: {len(sholl_data)} cells")
        if skeleton_data:
            self.log(f"  Skeleton total: {len(skeleton_data)} cells")
        if fractal_data:
            self.log(f"  Fractal/Hull total: {len(fractal_data)} cells")

        # Read simple morphology results (user-selected or auto-detected)
        morphology_path = simple_path or os.path.join(self.output_dir, "combined_morphology_results.csv")
        morphology_rows = []
        morph_fieldnames = []
        if os.path.exists(morphology_path):
            try:
                with open(morphology_path, 'r') as f:
                    reader = csv.DictReader(f)
                    morph_fieldnames = list(reader.fieldnames)
                    morphology_rows = list(reader)
                self.log(f"  Morphology: loaded {len(morphology_rows)} cells from {os.path.basename(morphology_path)}")
            except Exception as e:
                self.log(f"  ERROR reading morphology results: {e}")

        if not morphology_rows:
            self.log("  No morphology results to merge with. Saving ImageJ results separately.")
            # Save a combined ImageJ-only file
            all_ij_data = {}
            for cell_name, entry in sholl_data.items():
                all_ij_data.setdefault(cell_name, {'cell_name': cell_name}).update(entry['data'])
            for cell_name, entry in skeleton_data.items():
                all_ij_data.setdefault(cell_name, {'cell_name': cell_name}).update(entry['data'])
            for cell_name, entry in fractal_data.items():
                all_ij_data.setdefault(cell_name, {'cell_name': cell_name}).update(entry['data'])

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
            # Merge with morphology results using direct field matching
            matched_sholl = 0
            matched_skel = 0
            matched_fractal = 0

            # Collect all new column names
            new_sholl_keys = set()
            new_skel_keys = set()
            new_fractal_keys = set()
            for entry in sholl_data.values():
                new_sholl_keys.update(entry['data'].keys())
            for entry in skeleton_data.values():
                new_skel_keys.update(entry['data'].keys())
            for entry in fractal_data.values():
                new_fractal_keys.update(entry['data'].keys())

            # Rebuild ImageJ data into lookup dicts keyed by (image_name, soma_id, area)
            # so we match on actual fields, not substring hacks
            def _build_ij_lookup(ij_data):
                """Build {(image_name, soma_id, area): data_dict} from loaded ImageJ data."""
                import re as _re3
                lookup = {}
                for dict_key, entry in ij_data.items():
                    area = entry.get('area')
                    # Parse image_name and soma_id from dict_key format: "imagename_soma_Y_X_area123"
                    m = _re3.match(r'^(.+?)_(soma_\d+_\d+(?:_\d+)?)(?:_area(\d+))?$', dict_key)
                    if m:
                        img = m.group(1)
                        sid = m.group(2)
                        if area is None and m.group(3):
                            area = int(m.group(3))
                    else:
                        # Fallback: try without area suffix
                        m2 = _re3.match(r'^(.+?)_(soma_\d+_\d+(?:_\d+)?)$', dict_key)
                        if m2:
                            img = m2.group(1)
                            sid = m2.group(2)
                        else:
                            continue
                    lookup[(img, sid, area)] = entry['data']
                return lookup

            sholl_lookup = _build_ij_lookup(sholl_data) if sholl_data else {}
            skel_lookup = _build_ij_lookup(skeleton_data) if skeleton_data else {}
            fractal_lookup = _build_ij_lookup(fractal_data) if fractal_data else {}

            for row in morphology_rows:
                img_name = row.get('image_name', '')
                soma_id = row.get('soma_id', '')
                morph_area_str = row.get('area_um2', '')
                try:
                    morph_area = int(float(morph_area_str)) if morph_area_str else None
                except (ValueError, TypeError):
                    morph_area = None

                key = (img_name, soma_id, morph_area)
                key_no_area = (img_name, soma_id, None)

                for lookup, counter_name in [
                    (sholl_lookup, 'sholl'),
                    (skel_lookup, 'skeleton'),
                    (fractal_lookup, 'fractal'),
                ]:
                    if not lookup:
                        continue
                    # Try exact match with area first, then without area
                    match_data = lookup.get(key) or lookup.get(key_no_area)
                    if match_data:
                        row.update(match_data)
                        if counter_name == 'sholl':
                            matched_sholl += 1
                        elif counter_name == 'skeleton':
                            matched_skel += 1
                        elif counter_name == 'fractal':
                            matched_fractal += 1

            # Write merged results
            all_keys = morph_fieldnames + sorted(new_sholl_keys) + sorted(new_skel_keys) + sorted(new_fractal_keys)
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
            self.log(f"  Matched Fractal data: {matched_fractal}/{len(morphology_rows)} cells")
            self.log(f"  Merged results saved to: {merged_path}")

        self.log("=" * 50)

        summary = "ImageJ Results Import Complete\n\n"
        if sholl_data:
            summary += f"Sholl: {len(sholl_data)} cells\n"
        if skeleton_data:
            summary += f"Skeleton: {len(skeleton_data)} cells\n"
        if fractal_data:
            summary += f"Fractal/Hull: {len(fractal_data)} cells\n"
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
            if self.processed_label.polygon_mode and hasattr(self, 'outlining_queue') and self.outlining_queue:
                queue_idx = getattr(self, 'current_outline_idx', 0)
                if queue_idx < len(self.outlining_queue):
                    q_img_name, q_soma_idx = self.outlining_queue[queue_idx]
                    q_img_data = self.images[q_img_name]
                    soma = q_img_data['somas'][q_soma_idx]
                    pixmap = self._get_outlining_pixmap(q_img_data)
                    self.processed_label.set_image(pixmap, centroids=[soma],
                                                   polygon_pts=self.polygon_points)
            elif self.processed_label.soma_mode:
                self.processed_label.set_image(pixmap, centroids=img_data['somas'])
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
        # During soma picking, use the dedicated refresh to preserve picking state
        if self.processed_label.soma_mode:
            self._load_image_for_soma_picking()
        else:
            self.update_display()

    def _toggle_multi_channel_ui(self, checked):
        """Show/hide the extra channel checkboxes"""
        self.extra_channel_widget.setVisible(checked)
        if checked:
            # Auto-check the primary channel and uncheck others
            for i, ch_check in enumerate(self.clean_ch_checks):
                ch_check.setChecked(i == self.grayscale_channel)

    def _get_channels_to_clean(self):
        """Return list of channel indices to clean. Always includes the primary channel."""
        channels = [self.grayscale_channel]
        if self.multi_clean_check.isChecked():
            for i, ch_check in enumerate(self.clean_ch_checks):
                if ch_check.isChecked() and i != self.grayscale_channel:
                    channels.append(i)
        return sorted(set(channels))

    def _on_process_channel_changed(self, index):
        """Update the grayscale channel when user changes the dropdown"""
        self.grayscale_channel = index
        self.log(f"Processing channel set to: Channel {index + 1}")
        # Auto-check the primary channel in the multi-clean checkboxes
        if self.multi_clean_check.isChecked():
            for i, ch_check in enumerate(self.clean_ch_checks):
                if i == index:
                    ch_check.setChecked(True)
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
            # In 3D mode, delegate to the 3D display path
            if self.mode_3d:
                self._display_current_image_3d()
                return

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
                        if use_color and 'color_image' in img_data:
                            # Build color composite with preview channel replacing original
                            color_img = img_data['color_image']
                            if color_img.ndim == 3:
                                h, w = color_img.shape[:2]
                                c = min(color_img.shape[2], 3)
                                composite = np.zeros((h, w, 3), dtype=np.float32)
                                for i in range(c):
                                    if i == self.grayscale_channel:
                                        composite[:, :, i] = img_data['preview'].astype(np.float32)
                                    else:
                                        composite[:, :, i] = color_img[:, :, i].astype(np.float32)
                                adjusted = self._apply_display_adjustments_color(composite)
                                pixmap = self._array_to_pixmap_color(adjusted)
                            else:
                                adjusted = self._apply_display_adjustments(img_data['preview'])
                                pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
                        else:
                            adjusted = self._apply_display_adjustments(img_data['preview'])
                            pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
                        self.preview_label.set_image(pixmap)
                elif current_tab == 2:  # Processed
                    # In outlining mode, use the queue's image, not current_image_name
                    if self.processed_label.polygon_mode and hasattr(self, 'outlining_queue') and self.outlining_queue:
                        queue_idx = getattr(self, 'current_outline_idx', 0)
                        if queue_idx < len(self.outlining_queue):
                            q_img_name, q_soma_idx = self.outlining_queue[queue_idx]
                            q_img_data = self.images[q_img_name]
                            soma = q_img_data['somas'][q_soma_idx]
                            pixmap = self._get_outlining_pixmap(q_img_data)
                            self.processed_label.set_image(pixmap, centroids=[soma],
                                                           polygon_pts=self.polygon_points)
                    elif use_color and 'color_image' in img_data:
                        # Use processed channel in color composite if available
                        proc_color = self._build_processed_color_image(img_data)
                        if proc_color is not None:
                            adjusted = self._apply_display_adjustments_color(proc_color)
                        else:
                            adjusted = self._apply_display_adjustments_color(img_data['color_image'])
                        pixmap = self._array_to_pixmap_color(adjusted)
                        self.processed_label.set_image(pixmap, centroids=img_data['somas'])
                    elif img_data['processed'] is not None:
                        # Always use the processed image for grayscale display
                        gray_img = img_data['processed']
                        adjusted = self._apply_display_adjustments(gray_img)
                        pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
                        # Preserve soma markers if in soma picking mode
                        if self.soma_mode:
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
        """Build a color composite with processed channels replacing the originals.
        Uses cleaned versions for all channels that were processed."""
        if 'color_image' not in img_data or img_data['processed'] is None:
            return None

        color_img = img_data['color_image']
        processed = img_data['processed']
        processed_channels = img_data.get('processed_channels', {})

        # Start with a copy of the original color image
        if color_img.ndim != 3:
            return None

        # Build RGB composite
        h, w = color_img.shape[:2]
        c = min(color_img.shape[2], 3)
        composite = np.zeros((h, w, 3), dtype=np.float32)

        for i in range(c):
            if i in processed_channels:
                # Use the cleaned version for this channel
                composite[:, :, i] = processed_channels[i].astype(np.float32)
            elif i == self.grayscale_channel:
                # Primary channel always uses the processed result
                composite[:, :, i] = processed.astype(np.float32)
            else:
                # Use original channel
                composite[:, :, i] = color_img[:, :, i].astype(np.float32)

        return composite

    def select_folder(self):
        title = "Select Z-Stack Folder" if self.mode_3d else "Select Image Folder"
        folder = QFileDialog.getExistingDirectory(self, title)
        if not folder:
            return

        # Apply current colocalization mode settings to display
        if self.colocalization_mode and not self.mode_3d:
            self.show_color_view = True
            self.color_toggle_btn.setText("Show Grayscale (C)")
            self.channel_select_btn.setVisible(True)

        # Include both lowercase and uppercase extensions for macOS compatibility
        if self.mode_3d:
            exts = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
        else:
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
            img_dict = {
                'raw_path': f,
                'processed': None,
                'rolling_ball_radius': self.default_rolling_ball_radius,
                'somas': [],
                'soma_ids': [],
                'soma_groups': [],
                'masks': [],
                'status': 'loaded',
                'selected': False,
                'animal_id': '',
                'treatment': '',
                'pixel_size': None,
            }
            if self.mode_3d:
                img_dict['raw_stack'] = None
                img_dict['soma_masks'] = {}
            else:
                img_dict['processed_channels'] = {}
                img_dict['soma_outlines'] = []
            self.images[img_name] = img_dict
            item = QListWidgetItem(f"  {img_name} [loaded]")
            item.setData(Qt.UserRole, img_name)
            item.setCheckState(Qt.Unchecked)
            item.setForeground(QBrush(QColor(128, 128, 128)))  # Gray for loaded
            self.file_list.addItem(item)

        if self.images:
            self.process_selected_btn.setEnabled(True)
            kind = "Z-stack files" if self.mode_3d else "images"
            self.log(f"Loaded {len(self.images)} {kind}")

            # Automatically load and display the first image
            first_image_name = sorted(self.images.keys())[0]
            self.current_image_name = first_image_name

            if self.mode_3d:
                self._load_and_display_raw_3d(first_image_name)
            else:
                self._display_current_image()

            self.file_list.setCurrentRow(0)
            self.log(f"Displaying: {first_image_name}")

    def _to_grayscale_3d(self, stack):
        """Convert a Z-stack to grayscale using the selected channel."""
        if stack.ndim == 4 and _HAS_3D:
            return extract_channel_3d(stack, self.grayscale_channel)
        return ensure_grayscale_3d(stack)

    def _load_and_display_raw_3d(self, img_name):
        """Load raw Z-stack and display the middle slice."""
        img_data = self.images[img_name]
        if img_data.get('raw_stack') is None:
            try:
                stack = load_zstack(img_data['raw_path'])
                stack = self._to_grayscale_3d(stack)
                img_data['raw_stack'] = stack
            except Exception as e:
                self.log(f"ERROR loading {img_name}: {e}")
                return
        stack = img_data['raw_stack']
        self.current_z_slice = stack.shape[0] // 2
        self._update_z_slider_for_image(img_name)
        sl = self._get_slice_for_display(stack)
        adjusted = self._apply_display_adjustments(sl)
        pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
        self.original_label.set_image(pixmap)
        self.tabs.setCurrentIndex(0)

    def select_output(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder"
        )
        if folder:
            self.output_dir = folder
            if self.mode_3d:
                self.masks_dir = os.path.join(folder, "masks_3d")
            else:
                self.masks_dir = os.path.join(folder, "masks")
            self.somas_dir = os.path.join(folder, "somas")
            os.makedirs(self.masks_dir, exist_ok=True)
            os.makedirs(self.somas_dir, exist_ok=True)
            self.log(f"Output folder: {folder}")
            self.log(f"Masks will be saved to: {self.masks_dir}")
            self.log(f"Somas will be saved to: {self.somas_dir}")

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
        # During outlining, soma picking, or QA, don't switch images from file list clicks
        if self.processed_label.polygon_mode or self.processed_label.soma_mode or self.mask_qa_active:
            return
        img_name = item.data(Qt.UserRole)
        is_checked = item.checkState() == Qt.Checked
        self.images[img_name]['selected'] = is_checked
        self.current_image_name = img_name
        if self.mode_3d:
            self._display_current_image_3d()
        else:
            self._display_current_image()

    def _display_current_image(self):
        if not self.current_image_name or self.current_image_name not in self.images:
            return
        try:
            img_data = self.images[self.current_image_name]

            # Reload processed image from disk if it was freed to save RAM
            if img_data.get('processed') is None and img_data.get('status') not in (None, 'loaded'):
                self._ensure_processed_loaded(self.current_image_name)

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

    def _display_current_image_3d(self):
        """Display current Z-stack image in 3D mode."""
        if not self.current_image_name or self.current_image_name not in self.images:
            return
        try:
            img_data = self.images[self.current_image_name]
            # Load raw stack if needed
            if img_data.get('raw_stack') is None:
                try:
                    stack = load_zstack(img_data['raw_path'])
                    stack = self._to_grayscale_3d(stack)
                    img_data['raw_stack'] = stack
                except Exception as e:
                    self.log(f"ERROR loading: {e}")
                    return
            self._update_z_slider_for_image()
            # Show original slice
            raw_stack = img_data['raw_stack']
            sl = self._get_slice_for_display(raw_stack)
            adjusted = self._apply_display_adjustments(sl)
            pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
            self.original_label.set_image(pixmap)
            # Show processed slice if available
            if img_data['processed'] is not None:
                proc_sl = self._get_slice_for_display(img_data['processed'])
                adjusted_proc = self._apply_display_adjustments(proc_sl)
                centroids_2d = self._get_centroids_on_slice_3d(img_data)
                pixmap_proc = self._array_to_pixmap(adjusted_proc, skip_rescale=True)
                self.processed_label.set_image(pixmap_proc, centroids=centroids_2d)
        except Exception as e:
            self.log(f"ERROR displaying 3D image: {str(e)}")
            import traceback
            traceback.print_exc()

    def _create_blank_pixmap(self):
        blank = np.ones((500, 500), dtype=np.uint8) * 128
        return self._array_to_pixmap(blank)

    def _array_to_pixmap(self, arr, skip_rescale=False):
        arr_disp = arr.astype(float)

        # Ensure 2D — convert 3D+ arrays to grayscale
        if arr_disp.ndim > 2:
            arr_disp = arr_disp.squeeze()
        if arr_disp.ndim > 2:
            # Still 3D after squeeze: take first channel or convert
            arr_disp = arr_disp[:, :, 0] if arr_disp.ndim == 3 else arr_disp.reshape(arr_disp.shape[-2], arr_disp.shape[-1])

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

        if self.mode_3d:
            self._preview_current_image_3d()
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

    def _preview_current_image_3d(self):
        """Preview 3D preprocessing on the current Z-stack."""
        img_data = self.images[self.current_image_name]
        if img_data.get('raw_stack') is None:
            try:
                stack = load_zstack(img_data['raw_path'])
                stack = self._to_grayscale_3d(stack)
                img_data['raw_stack'] = stack
            except Exception as e:
                self.log(f"ERROR: {e}")
                return
        rb_r = self.rb_slider.value() if self.rb_check.isChecked() else 0
        dn_s = self.denoise_spin.value() if self.denoise_check.isChecked() else 0
        sh_a = self.sharpen_slider.value() / 10.0 if self.sharpen_check.isChecked() else 0.0
        self.log(f"Previewing 3D: {self.current_image_name}...")
        QApplication.processEvents()
        try:
            preview = preprocess_zstack(img_data['raw_stack'],
                                        rolling_ball_radius=rb_r,
                                        denoise_size=dn_s,
                                        sharpen_amount=sh_a)
            img_data['preview_stack'] = preview
            sl = self._get_slice_for_display(preview)
            adjusted = self._apply_display_adjustments(sl)
            pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
            self.preview_label.set_image(pixmap)
            self.tabs.setCurrentIndex(1)
            self.log("3D Preview complete")
        except Exception as e:
            self.log(f"ERROR in 3D preview: {e}")

    def _preview_intensity_threshold(self, intensity_percent, img_name=None):
        """Show a popup previewing which pixels fall below the intensity threshold.

        Pixels below threshold are shown in red on top of the grayscale image,
        so the user can see exactly what gets excluded from masks.
        """
        # Determine which image to preview
        if img_name is None:
            img_name = self.current_image_name
        if not img_name or img_name not in self.images:
            QMessageBox.warning(self, "Warning", "No image selected to preview.")
            return

        img_data = self.images[img_name]

        # Get the processed image (what masks are actually grown on)
        processed = img_data.get('processed')
        if processed is None:
            processed = self._ensure_processed_loaded(img_name)
        if processed is None:
            # Fall back to preview or raw
            processed = img_data.get('preview')
        if processed is None:
            QMessageBox.warning(self, "Warning",
                                "No processed image available.\nProcess images first, or use Preview.")
            return

        # For 3D, use the max-intensity projection or current slice
        if processed.ndim == 3:
            mid = processed.shape[0] // 2
            processed = processed[mid]

        # Compute threshold
        img_max = processed.max()
        if img_max == 0:
            QMessageBox.warning(self, "Warning", "Image is entirely black.")
            return
        threshold = img_max * (intensity_percent / 100.0)

        # Build RGB visualization: grayscale image with sub-threshold pixels in red
        norm = processed.astype(np.float64)
        norm = norm / img_max * 255.0
        gray8 = np.clip(norm, 0, 255).astype(np.uint8)

        # Create RGB from grayscale
        rgb = np.stack([gray8, gray8, gray8], axis=-1)

        # Mark sub-threshold pixels red
        below = processed < threshold
        rgb[below, 0] = 255   # R
        rgb[below, 1] = 40    # G (slight tint for visibility)
        rgb[below, 2] = 40    # B

        # Count statistics
        total_px = processed.size
        excluded_px = int(below.sum())
        excluded_pct = excluded_px / total_px * 100.0

        # Create popup dialog
        preview_dlg = QDialog(self)
        preview_dlg.setWindowTitle(
            f"Intensity Threshold Preview — {intensity_percent}%")
        preview_dlg.setModal(False)  # Non-modal so user can adjust slider

        dlg_layout = QVBoxLayout()

        # Info label
        info = QLabel(
            f"<b>Threshold: {intensity_percent}% of max</b> "
            f"(value {threshold:.1f} / {img_max:.1f})<br>"
            f"<span style='color:red'>Red pixels</span> = excluded from masks "
            f"({excluded_pct:.1f}% of image, {excluded_px:,} pixels)")
        info.setWordWrap(True)
        dlg_layout.addWidget(info)

        # Image display
        h, w = gray8.shape
        rgb_contiguous = np.ascontiguousarray(rgb)
        qimg = QImage(rgb_contiguous.data, w, h, 3 * w, QImage.Format_RGB888)
        qimg = qimg.copy()  # detach from numpy memory
        pixmap = QPixmap.fromImage(qimg)

        img_label = QLabel()
        # Scale to fit reasonably on screen
        max_dim = 800
        if max(w, h) > max_dim:
            pixmap = pixmap.scaled(max_dim, max_dim, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        img_label.setPixmap(pixmap)
        img_label.setAlignment(Qt.AlignCenter)

        scroll = QScrollArea()
        scroll.setWidget(img_label)
        scroll.setWidgetResizable(True)
        dlg_layout.addWidget(scroll)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(preview_dlg.close)
        dlg_layout.addWidget(close_btn)

        preview_dlg.setLayout(dlg_layout)
        preview_dlg.resize(min(w + 40, 850), min(h + 120, 700))
        preview_dlg.show()

    def process_selected_images(self):
        if not self.output_dir:
            QMessageBox.warning(self, "Warning", "Select output folder first")
            return
        selected_images = [(name, data) for name, data in self.images.items() if data['selected']]
        if not selected_images:
            QMessageBox.warning(self, "Warning", "No images selected")
            return

        if self.mode_3d:
            self._process_selected_images_3d(selected_images)
            return

        radius = self.rb_slider.value()
        rb_enabled = self.rb_check.isChecked()
        denoise_enabled = self.denoise_check.isChecked()
        denoise_size = self.denoise_spin.value()
        sharpen_enabled = self.sharpen_check.isChecked()
        sharpen_amount = self.sharpen_slider.value() / 10.0

        # Determine which channels to clean
        channels_to_clean = self._get_channels_to_clean()
        process_channel = self.grayscale_channel

        # Log what processing steps will be applied
        if len(channels_to_clean) > 1:
            ch_names = [f"Ch{c + 1}" for c in channels_to_clean]
            steps = [f"Channels {', '.join(ch_names)} (primary: Ch{process_channel + 1})"]
        else:
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
                                 channels_to_clean))
        self.thread = BackgroundRemovalThread(process_list, self.output_dir)
        self.thread.status_update.connect(self.log)
        self.thread.progress.connect(self._update_progress)
        self.thread.finished_image.connect(self._handle_processed_image)
        self.thread.finished_extra_channels.connect(self._handle_extra_channels)
        self.thread.finished.connect(self._background_removal_finished)
        self.thread.error_occurred.connect(lambda msg: self.log(f"ERROR: {msg}"))
        self.progress_bar.setVisible(True)
        self.progress_status_label.setVisible(True)
        if len(channels_to_clean) > 1:
            self.progress_status_label.setText(f"Processing {len(channels_to_clean)} channels...")
        elif len(steps) > 1:
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

    def _handle_extra_channels(self, img_name, extra_results):
        """Store extra cleaned channel arrays for color composite display."""
        if img_name in self.images:
            if 'processed_channels' not in self.images[img_name]:
                self.images[img_name]['processed_channels'] = {}
            self.images[img_name]['processed_channels'].update(extra_results)
            # Also store the primary channel in processed_channels for completeness
            if self.images[img_name].get('processed') is not None:
                self.images[img_name]['processed_channels'][self.grayscale_channel] = \
                    self.images[img_name]['processed']
            ch_names = [f"Ch{c + 1}" for c in sorted(extra_results.keys())]
            self.log(f"  Extra channels cleaned for {img_name}: {', '.join(ch_names)}")
            # Refresh color display if this is the current image
            if img_name == self.current_image_name and self.show_color_view:
                proc_color = self._build_processed_color_image(self.images[img_name])
                if proc_color is not None:
                    adjusted = self._apply_display_adjustments_color(proc_color)
                    pixmap = self._array_to_pixmap_color(adjusted)
                    self.processed_label.set_image(pixmap)

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

    # ----------------------------------------------------------------
    # 3D PROCESSING PIPELINE
    # ----------------------------------------------------------------

    def _process_selected_images_3d(self, selected_images):
        """Process selected Z-stacks using 3D preprocessing."""
        rb_r = self.rb_slider.value()
        rb_enabled = self.rb_check.isChecked()
        dn_enabled = self.denoise_check.isChecked()
        dn_size = self.denoise_spin.value()
        sh_enabled = self.sharpen_check.isChecked()
        sh_amount = self.sharpen_slider.value() / 10.0

        process_list = []
        for img_name, img_data in selected_images:
            process_list.append((img_data['raw_path'], img_name, rb_r, rb_enabled,
                                 dn_enabled, dn_size, sh_enabled, sh_amount))

        # Use the 3D preprocessing thread from 3DMicroglia
        from PyQt5.QtCore import QThread, pyqtSignal

        class _PreprocessThread3D(QThread):
            progress = pyqtSignal(int)
            status_update = pyqtSignal(str)
            finished_image = pyqtSignal(str, str, object)
            error_occurred = pyqtSignal(str)

            def __init__(self, image_data_list, output_dir, channel_idx=0):
                super().__init__()
                self.image_data_list = image_data_list
                self.output_dir = output_dir
                self.channel_idx = channel_idx

            def run(self):
                try:
                    total = len(self.image_data_list)
                    for i, (raw_path, img_name, rb_radius, rb_on,
                            dn_on, dn_sz, sh_on, sh_amt) in enumerate(self.image_data_list):
                        try:
                            self.status_update.emit(f"Processing {img_name}...")
                            stack = load_zstack(raw_path)
                            self.status_update.emit(f"  Loaded {img_name}: shape={stack.shape}, dtype={stack.dtype}")
                            if stack.ndim == 4 and _HAS_3D:
                                stack = extract_channel_3d(stack, self.channel_idx)
                                self.status_update.emit(f"  Extracted channel {self.channel_idx + 1}: shape={stack.shape}")
                            else:
                                stack = ensure_grayscale_3d(stack)
                            rb_r2 = rb_radius if rb_on else 0
                            dn_s2 = dn_sz if dn_on else 0
                            sh_a2 = sh_amt if sh_on else 0.0
                            processed = preprocess_zstack(stack, rolling_ball_radius=rb_r2,
                                                          denoise_size=dn_s2, sharpen_amount=sh_a2)
                            out_stem = os.path.splitext(img_name)[0]
                            out_path = os.path.join(self.output_dir, f"{out_stem}_processed.tif")
                            tifffile.imwrite(out_path, processed)
                            self.finished_image.emit(out_path, img_name, processed)
                        except Exception as e:
                            import traceback
                            tb = traceback.format_exc()
                            self.error_occurred.emit(f"{img_name}: {e}\n{tb}")
                        self.progress.emit(int((i + 1) / total * 100))
                except Exception as e:
                    import traceback
                    self.error_occurred.emit(f"Fatal 3D processing error: {e}\n{traceback.format_exc()}")

        self._preprocess_thread_3d = _PreprocessThread3D(process_list, self.output_dir, self.grayscale_channel)
        self._preprocess_thread_3d.status_update.connect(self.log)
        self._preprocess_thread_3d.progress.connect(lambda v: self.progress_bar.setValue(v))
        self._preprocess_thread_3d.finished_image.connect(self._handle_processed_image_3d)
        self._preprocess_thread_3d.error_occurred.connect(lambda msg: self.log(f"ERROR: {msg}"))
        self._preprocess_thread_3d.finished.connect(self._processing_finished_3d)

        self.progress_bar.setVisible(True)
        self.progress_status_label.setVisible(True)
        self.progress_status_label.setText("Processing Z-stacks...")
        self.process_selected_btn.setEnabled(False)
        self._preprocess_thread_3d.start()

    def _handle_processed_image_3d(self, output_path, img_name, processed_stack):
        try:
            if img_name in self.images:
                self.images[img_name]['processed'] = processed_stack
                self.images[img_name]['status'] = 'processed'
                if self.images[img_name].get('raw_stack') is None:
                    try:
                        raw = load_zstack(self.images[img_name]['raw_path'])
                        self.images[img_name]['raw_stack'] = self._to_grayscale_3d(raw)
                    except Exception:
                        pass
                self._update_file_list_item(img_name)
                if img_name == self.current_image_name:
                    self._update_z_slider_for_image()
                    sl = self._get_slice_for_display(processed_stack)
                    adjusted = self._apply_display_adjustments(sl)
                    pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
                    self.processed_label.set_image(pixmap)
                    self.tabs.setCurrentIndex(2)
        except Exception as e:
            import traceback
            self.log(f"ERROR handling processed 3D image {img_name}: {e}\n{traceback.format_exc()}")

    def _processing_finished_3d(self):
        self.progress_bar.setVisible(False)
        self.progress_status_label.setVisible(False)
        self.process_selected_btn.setEnabled(True)
        self.batch_pick_somas_btn.setEnabled(True)
        total = sum(1 for d in self.images.values() if d['status'] == 'processed')
        self.log("=" * 50)
        self.log(f"Processing complete! {total} Z-stacks processed.")
        self.log("Ready for soma picking.")
        self.log("=" * 50)
        self._auto_save()
        QMessageBox.information(self, "Complete",
                                f"Processed {total} Z-stacks!\n\nReady for soma picking.")

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
        try:
            self._start_batch_soma_picking_impl()
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.log(f"ERROR in soma picking: {e}\n{tb}")
            QMessageBox.critical(self, "Error", f"Failed to start soma picking:\n{e}\n\nSee log for details.")

    def _start_batch_soma_picking_impl(self):
        self.soma_picking_queue = [name for name, data in self.images.items()
                                   if data['selected'] and data['status'] in ('processed', 'somas_picked')]
        if not self.soma_picking_queue:
            QMessageBox.warning(self, "Warning", "No processed images to pick somas from")
            return

        # In colocalization mode, use two-pass soma picking
        if self.colocalization_mode:
            self._coloc_soma_pass = 1
            QMessageBox.information(self, "Colocalization — Pass 1 of 2",
                "PASS 1: Colocalized cells\n\n"
                "You will see the color composite (both channels).\n"
                "Click on cells that show signal in BOTH channels.\n\n"
                "After finishing all images, Pass 2 will start\n"
                "for single-channel-only cells.")
        else:
            self._coloc_soma_pass = 0

        self._begin_soma_picking_pass()

    def _begin_soma_picking_pass(self):
        """Start or restart soma picking for the current pass."""
        self.batch_mode = True

        if self._coloc_soma_pass == 2:
            # Pass 2: force single-channel view
            self._coloc_saved_color_view = self.show_color_view
            self.show_color_view = False
            self.color_toggle_btn.setText("Show Color (C)")
            self.current_image_name = self.soma_picking_queue[0]
        else:
            # Pass 1 or non-coloc: resume where we left off
            if self.current_image_name and self.current_image_name in self.soma_picking_queue:
                pass
            else:
                resume_name = None
                for name in self.soma_picking_queue:
                    if not self.images[name]['somas']:
                        resume_name = name
                        break
                self.current_image_name = resume_name or self.soma_picking_queue[0]

        self.processed_label.soma_mode = True
        self.original_label.soma_mode = False
        self.preview_label.soma_mode = False
        self.mask_label.soma_mode = False
        self._load_image_for_soma_picking()
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.done_btn.setEnabled(True)
        if self.mode_3d:
            self.log("=" * 50)
            self.log("BATCH SOMA PICKING MODE (3D)")
            self.log(f"Click somas on: {self.current_image_name}")
            self.log("Use Z-slider to find the soma's brightest slice")
            self.log("Click 'Done with Current' (Enter) when finished")
            self.log("Backspace = undo last, Escape = clear all on image")
            self.log("=" * 50)
        else:
            pass_label = ""
            if self._coloc_soma_pass == 1:
                pass_label = " — PASS 1: Coloc cells (both channels)"
            elif self._coloc_soma_pass == 2:
                pass_label = " — PASS 2: Single-channel cells (cyan = already picked)"
            self.log("=" * 50)
            self.log(f"BATCH SOMA PICKING MODE{pass_label}")
            self.log(f"Click somas on: {self.current_image_name}")
            self.log("Click 'Done with Current' when finished with this image")
            self.log("=" * 50)

    def _load_image_for_soma_picking(self):
        if not self.current_image_name:
            return
        img_data = self.images[self.current_image_name]

        if self.mode_3d:
            self._load_image_for_soma_picking_3d()
            return

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
                # No processed or color image available — load raw from disk
                try:
                    raw_img = load_tiff_image(img_data['raw_path'])
                    gray_img = ensure_grayscale(raw_img)
                except Exception:
                    self.log(f"ERROR: Could not load image for display")
                    return
            adjusted = self._apply_display_adjustments(gray_img)
            pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)

        # In Pass 2, split somas into locked (Pass 1) and active (Pass 2)
        if getattr(self, '_coloc_soma_pass', 0) == 2:
            groups = img_data.get('soma_groups', [])
            locked = [s for s, g in zip(img_data['somas'], groups) if g == 'coloc']
            active = [s for s, g in zip(img_data['somas'], groups) if g != 'coloc']
            self.processed_label.set_image(pixmap, centroids=active, locked_centroids=locked)
        else:
            self.processed_label.set_image(pixmap, centroids=img_data['somas'])

        self.tabs.setCurrentIndex(2)
        current_idx = self.soma_picking_queue.index(
            self.current_image_name) if self.current_image_name in self.soma_picking_queue else -1
        pass_label = ""
        if getattr(self, '_coloc_soma_pass', 0) == 1:
            pass_label = " [Pass 1: Coloc]"
        elif getattr(self, '_coloc_soma_pass', 0) == 2:
            pass_label = " [Pass 2: Single-channel]"
        self.nav_status_label.setText(
            f"Image {current_idx + 1}/{len(self.soma_picking_queue)}: {self.current_image_name} | "
            f"Somas: {len(img_data['somas'])}{pass_label}"
        )

    def _load_image_for_soma_picking_3d(self):
        """Load Z-stack for 3D soma picking."""
        img_data = self.images[self.current_image_name]
        # Ensure raw stack loaded
        if img_data.get('raw_stack') is None:
            try:
                raw = load_zstack(img_data['raw_path'])
                img_data['raw_stack'] = self._to_grayscale_3d(raw)
            except Exception:
                pass
        self._update_z_slider_for_image()
        stack = img_data.get('processed') or img_data.get('raw_stack')
        if stack is None:
            return
        sl = self._get_slice_for_display(stack)
        adjusted = self._apply_display_adjustments(sl)
        centroids_2d = self._get_centroids_on_slice_3d(img_data)
        pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)
        self.processed_label.set_image(pixmap, centroids=centroids_2d)
        self.tabs.setCurrentIndex(2)
        current_idx = self.soma_picking_queue.index(
            self.current_image_name) if self.current_image_name in self.soma_picking_queue else -1
        self.nav_status_label.setText(
            f"Z-Stack {current_idx + 1}/{len(self.soma_picking_queue)}: "
            f"{self.current_image_name} | "
            f"Somas: {len(img_data['somas'])} | "
            f"Z-Slice: {self.current_z_slice}"
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
        pixel_size = self._get_pixel_size(self.current_image_name)
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

        if self.mode_3d:
            row, col = coords
            z = self.current_z_slice
            z, row, col = self._snap_to_brightest_3d(z, row, col)
            soma_zyx = (z, row, col)
            img_data['somas'].append(soma_zyx)
            soma_id = f"soma_{z}_{row}_{col}"
            img_data['soma_ids'].append(soma_id)
            group = "coloc" if getattr(self, '_coloc_soma_pass', 0) == 1 else \
                    "single_channel" if getattr(self, '_coloc_soma_pass', 0) == 2 else ""
            img_data['soma_groups'].append(group)
            self.log(f"Soma {len(img_data['somas'])} added at Z={z}, Y={row}, X={col} | ID: {soma_id}")
            if z != self.current_z_slice:
                self.current_z_slice = z
                self.z_slider.setValue(z)
            self._load_image_for_soma_picking()
            return

        coords = self._snap_to_brightest(coords)
        img_data['somas'].append(coords)
        soma_id = f"soma_{coords[0]}_{coords[1]}"
        img_data['soma_ids'].append(soma_id)
        group = "coloc" if getattr(self, '_coloc_soma_pass', 0) == 1 else \
                "single_channel" if getattr(self, '_coloc_soma_pass', 0) == 2 else ""
        img_data['soma_groups'].append(group)

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
                # No processed or color image available — load raw from disk
                try:
                    raw_img = load_tiff_image(img_data['raw_path'])
                    gray_img = ensure_grayscale(raw_img)
                except Exception:
                    self.log(f"ERROR: Could not load image for display")
                    return
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

        # Remove the last soma, its ID, and its group
        removed_soma = img_data['somas'].pop()
        removed_id = img_data['soma_ids'].pop() if img_data['soma_ids'] else None
        if img_data.get('soma_groups'):
            img_data['soma_groups'].pop()

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
                # No processed or color image available — load raw from disk
                try:
                    raw_img = load_tiff_image(img_data['raw_path'])
                    gray_img = ensure_grayscale(raw_img)
                except Exception:
                    self.log(f"ERROR: Could not load image for display")
                    return
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
        # In coloc mode Pass 1 → transition to Pass 2
        if getattr(self, '_coloc_soma_pass', 0) == 1:
            pass1_count = sum(
                sum(1 for g in data.get('soma_groups', []) if g == 'coloc')
                for data in self.images.values() if data['selected']
            )
            self._coloc_soma_pass = 2
            QMessageBox.information(self, "Colocalization — Pass 2 of 2",
                f"Pass 1 complete! {pass1_count} coloc cells marked.\n\n"
                "PASS 2: Single-channel cells\n\n"
                "You will now see only the primary channel.\n"
                "Cyan circles show your Pass 1 (coloc) somas — do NOT re-click them.\n"
                "Click on cells that show signal in only ONE channel.")
            self._begin_soma_picking_pass()
            return

        # Restore color view if it was saved during Pass 2
        if getattr(self, '_coloc_soma_pass', 0) == 2:
            if hasattr(self, '_coloc_saved_color_view'):
                self.show_color_view = self._coloc_saved_color_view
                if self.show_color_view:
                    self.color_toggle_btn.setText("Show Grayscale (C)")
                del self._coloc_saved_color_view
            self._coloc_soma_pass = 0

        self.batch_mode = False
        self.processed_label.soma_mode = False
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.done_btn.setEnabled(False)
        total_somas = sum(len(data['somas']) for data in self.images.values() if data['selected'])
        self.batch_outline_btn.setEnabled(True)

        # Log group counts if coloc mode was used
        if self.colocalization_mode:
            coloc_count = sum(
                sum(1 for g in data.get('soma_groups', []) if g == 'coloc')
                for data in self.images.values() if data['selected']
            )
            single_count = sum(
                sum(1 for g in data.get('soma_groups', []) if g == 'single_channel')
                for data in self.images.values() if data['selected']
            )
            self.log("=" * 50)
            self.log(f"✓ Soma picking complete! Total: {total_somas} "
                     f"(coloc: {coloc_count}, single-channel: {single_count})")
            self.log("✓ Ready for outlining")
            self.log("=" * 50)
        else:
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
            if len(img_data.get('somas', [])) == 0:
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
        sens_spin.setRange(1, 90)
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
        self.processed_label.dragging_point = False
        self.original_label.polygon_mode = False
        self.preview_label.polygon_mode = False
        self.mask_label.polygon_mode = False
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.done_btn.setEnabled(False)
        # Reset stale interaction state
        self.z_key_held = False
        for label in [self.processed_label, self.original_label, self.preview_label, self.mask_label]:
            label.measure_mode = False
            label.measure_pt1 = None
            label.measure_pt2 = None
        self.measure_mode = False

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

        # Create soma_checklist.csv to track outline progress
        cl_path = self._get_checklist_path('soma_checklist.csv')
        if cl_path:
            rows = []
            for img_name, soma_idx in self.outlining_queue:
                soma_id = self.images[img_name]['soma_ids'][soma_idx]
                name = f"{img_name}_{soma_id}"
                passed = 1 if self._soma_has_outline(img_name, soma_idx) else 0
                rows.append([name, str(passed)])
            self._write_checklist(cl_path, rows, ['Soma', 'Passed QA'])

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
                # Remove branch juts from auto outline
                points = _remove_branch_juts(points, soma)
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

        # Reset stale UI state that could block interaction
        self.z_key_held = False
        for label in [self.processed_label, self.original_label, self.preview_label, self.mask_label]:
            label.measure_mode = False
            label.measure_pt1 = None
            label.measure_pt2 = None
        self.measure_mode = False

        # Find the first soma that actually needs review (skip already-outlined)
        start_idx = self._find_next_unoutlined_idx(0)
        if start_idx is None:
            # All somas already outlined — nothing to review
            self._finish_review_mode()
            return

        self.current_review_idx = start_idx

        self.log("")
        self.log("=" * 50)
        self.log("📋 REVIEW MODE - Check each outline")
        self.log("• Shift+drag points to adjust")
        self.log("• Press Enter or [Accept] to approve")
        self.log("• Click [Manual] to redraw from scratch")
        self.log("=" * 50)

        self._load_review_soma(start_idx)

    def _load_review_soma(self, review_idx):
        """Load a soma for review"""
        if review_idx >= len(self.outlining_queue):
            self._finish_review_mode()
            return

        # Skip somas that already have saved outlines (from previous session)
        img_name_check, soma_idx_check = self.outlining_queue[review_idx]
        if self._soma_has_outline(img_name_check, soma_idx_check):
            # This soma is already outlined — advance to the next unoutlined one
            next_idx = self._find_next_unoutlined_idx(start_from=review_idx + 1)
            if next_idx is None:
                self._finish_review_mode()
            else:
                self._load_review_soma(next_idx)
            return

        # Reset interaction state to prevent stale flags from blocking drag/click
        self.z_key_held = False
        self.processed_label.dragging_point = False
        self.processed_label.selected_point_idx = None

        self.current_review_idx = review_idx
        self.current_outline_idx = review_idx
        img_name, soma_idx = self.outlining_queue[review_idx]
        self.current_image_name = img_name
        img_data = self.images[img_name]

        # Lazy-load processed image from disk if missing
        if img_data.get('processed') is None:
            processed_path = img_data.get('processed_path')
            # If stored path is stale (e.g. session from another computer), fall back
            # to constructing the path from the current output_dir.
            if not processed_path or not os.path.exists(processed_path):
                if self.output_dir:
                    processed_path = os.path.join(self.output_dir, f"{os.path.splitext(img_name)[0]}_processed.tif")
                else:
                    processed_path = None
            if processed_path and os.path.exists(processed_path):
                try:
                    img_data['processed'] = safe_tiff_read(processed_path)
                    img_data['processed_path'] = processed_path  # update stale path
                except Exception:
                    pass

        # Ensure color_image is loaded for color toggle support
        if 'color_image' not in img_data and 'raw_path' in img_data:
            try:
                raw_img = load_tiff_image(img_data['raw_path'])
                if raw_img is not None and raw_img.ndim == 3:
                    img_data['color_image'] = raw_img.copy()
                    img_data['num_channels'] = raw_img.shape[2]
                elif raw_img is not None and raw_img.ndim == 2 and img_data.get('processed') is None:
                    img_data['processed'] = raw_img.copy()
            except Exception:
                pass

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
        self.processed_label.zoom_to_point(soma[0], soma[1], zoom_level=self.qa_autozoom_spin.value())
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
        proc = img_data.get('processed')
        if proc is None:
            # Fallback: try raw image as grayscale
            proc = self._get_image_for_outlining(img_data)
        if proc is None:
            # Return a blank pixmap to avoid crash
            blank = np.zeros((100, 100), dtype=np.uint8)
            return self._array_to_pixmap(blank)
        adjusted = self._apply_display_adjustments(proc)
        return self._array_to_pixmap(adjusted, skip_rescale=True)

    def _load_soma_for_outlining(self, queue_idx):
        if queue_idx >= len(self.outlining_queue):
            self._finish_outlining()
            return
        self.current_outline_idx = queue_idx
        img_name, soma_idx = self.outlining_queue[queue_idx]
        self.current_image_name = img_name
        img_data = self.images[img_name]

        # Lazy-load processed image from disk if missing
        if img_data.get('processed') is None:
            processed_path = img_data.get('processed_path')
            # If stored path is stale (e.g. session from another computer), fall back
            # to constructing the path from the current output_dir.
            if not processed_path or not os.path.exists(processed_path):
                if self.output_dir:
                    processed_path = os.path.join(self.output_dir, f"{os.path.splitext(img_name)[0]}_processed.tif")
                else:
                    processed_path = None
            if processed_path and os.path.exists(processed_path):
                try:
                    img_data['processed'] = safe_tiff_read(processed_path)
                    img_data['processed_path'] = processed_path  # update stale path
                except Exception:
                    pass

        # Ensure color_image is loaded for color toggle support
        if 'color_image' not in img_data and 'raw_path' in img_data:
            try:
                raw_img = load_tiff_image(img_data['raw_path'])
                if raw_img is not None and raw_img.ndim == 3:
                    img_data['color_image'] = raw_img.copy()
                    img_data['num_channels'] = raw_img.shape[2]
                elif raw_img is not None and raw_img.ndim == 2 and img_data.get('processed') is None:
                    # Fallback: use raw grayscale image as processed
                    img_data['processed'] = raw_img.copy()
            except Exception:
                pass

        soma = img_data['somas'][soma_idx]
        soma_id = img_data['soma_ids'][soma_idx]
        pixmap = self._get_outlining_pixmap(img_data)
        self.processed_label.set_image(pixmap, centroids=[soma], polygon_pts=self.polygon_points)
        self.processed_label.zoom_to_point(soma[0], soma[1], zoom_level=self.qa_autozoom_spin.value())
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
        try:
            self._finish_polygon_impl()
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.log(f"ERROR in finish_polygon: {e}\n{tb}")
            QMessageBox.critical(self, "Error", f"Failed to confirm outline:\n{e}\n\nSee log for details.")

    def _finish_polygon_impl(self):
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
        # Determine image shape for mask creation
        outline_img = self._get_image_for_outlining(img_data)
        if outline_img is not None:
            mask_shape = outline_img.shape[:2]
        elif img_data.get('processed') is not None:
            mask_shape = img_data['processed'].shape[:2]
        else:
            # Fallback: try loading raw image just for shape
            try:
                raw_img = load_tiff_image(img_data['raw_path'])
                mask_shape = raw_img.shape[:2]
            except Exception:
                QMessageBox.warning(self, "Error", "Cannot determine image dimensions for mask.")
                return
        mask = self._polygon_to_mask(self.polygon_points, mask_shape)
        soma_id = img_data['soma_ids'][soma_idx]

        # Calculate soma area from the outline
        pixel_size = self._get_pixel_size(img_name)
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

        # Update soma_checklist.csv
        cl_path = self._get_checklist_path('soma_checklist.csv')
        if cl_path and os.path.exists(cl_path):
            checklist_key = f"{img_name}_{soma_id}"
            self._update_checklist_row(cl_path, 0, checklist_key, 1, 1)

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
        if img_data.get('processed') is not None:
            return img_data['processed']
        elif 'color_image' in img_data:
            return extract_channel(img_data['color_image'], self.grayscale_channel)
        # Last resort: try loading the raw image
        raw_path = img_data.get('raw_path')
        if raw_path and os.path.exists(raw_path):
            try:
                raw_img = load_tiff_image(raw_path)
                if raw_img is not None:
                    if raw_img.ndim == 2:
                        return raw_img
                    elif raw_img.ndim == 3:
                        return raw_img[:, :, self.grayscale_channel] if raw_img.shape[2] > self.grayscale_channel else raw_img[:, :, 0]
            except Exception:
                pass
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

        # Remove branch juts from auto outline
        points = _remove_branch_juts(points, soma)

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
        self.log("  → Shift+drag points to adjust, then press Enter or [Accept]")

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

            # Remove branch juts from auto outline
            points = _remove_branch_juts(points, soma)

            # Create mask from points
            mask = self._polygon_to_mask(points, outline_img.shape)
            pixel_size = self._get_pixel_size(img_name)
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
        self.clear_masks_btn.setVisible(False)
        self.regen_masks_btn.setVisible(False)
        self.undo_qa_btn.setVisible(False)
        self.mask_qa_active = False
        self.mask_qa_progress_bar.setVisible(False)
        self.opacity_widget.setVisible(False)

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

    def _ensure_outline_masks(self, img_name, img_shape):
        """Reconstruct outline masks from polygon_points if they are None.

        After a session load, outline masks are set to None to save memory.
        This method lazily rebuilds them from the stored polygon_points
        before mask generation so that the soma outline properly seeds
        the region-growing algorithm.
        """
        img_data = self.images.get(img_name)
        if img_data is None:
            return
        for outline in img_data.get('soma_outlines', []):
            if outline.get('outline') is None and outline.get('polygon_points'):
                pts = outline['polygon_points']
                if len(pts) >= 3:
                    outline['outline'] = self._polygon_to_mask(pts, img_shape)

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
        """Update the main progress bar with current outline completion count."""
        if not hasattr(self, 'outlining_queue') or not self.outlining_queue:
            return
        completed = sum(1 for in_, si in self.outlining_queue if self._soma_has_outline(in_, si))
        total = len(self.outlining_queue)
        self.progress_bar.setFormat("%v / %m somas outlined")
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(completed)
        self.progress_bar.setVisible(True)

    def _finish_outlining(self):
        self.progress_bar.setVisible(False)
        self.progress_bar.setFormat("%p%")  # Reset to default format
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
            if img_data['selected'] and img_data['soma_outlines']:
                img_data['status'] = 'outlined'
                self._update_file_list_item(img_name)
        self.batch_generate_masks_btn.setEnabled(True)
        # self.update_workflow_status()
        # Delete soma checklist — outlining is done
        self._delete_checklist(self._get_checklist_path('soma_checklist.csv'))

        self.log("=" * 50)
        self.log("✓ All somas outlined!")
        self.log("✓ Ready to generate masks")
        self.log("=" * 50)
        self._auto_save()
        QMessageBox.information(self, "Complete", "All somas outlined!\n\nReady to generate masks.")

    def batch_generate_masks(self):
        if self.mode_3d:
            self._batch_generate_masks_3d()
            return
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
        min_area_spin.setRange(10, 2000)
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

        # Preview threshold button
        preview_thresh_btn = QPushButton("Preview Threshold on Current Image")
        preview_thresh_btn.setToolTip(
            "Opens a window showing which pixels would be excluded (red) at the current threshold")
        preview_thresh_btn.clicked.connect(
            lambda: self._preview_intensity_threshold(min_intensity_slider.value()))
        layout.addWidget(preview_thresh_btn)

        layout.addSpacing(10)

        # --- Cell Boundary Segmentation ---
        seg_group = QLabel("<b>Cell Boundary Segmentation</b>")
        layout.addWidget(seg_group)

        seg_desc = QLabel("Controls how neighboring cells share territory when masks overlap:")
        seg_desc.setWordWrap(True)
        layout.addWidget(seg_desc)

        seg_combo = QComboBox()
        seg_combo.addItem("None (independent growth)", "none")
        seg_combo.addItem("Competitive Growth (shared priority queue)", "competitive")
        seg_combo.addItem("Watershed Territories (pre-computed basins)", "watershed")
        # Set current selection
        for idx in range(seg_combo.count()):
            if seg_combo.itemData(idx) == self.mask_segmentation_method:
                seg_combo.setCurrentIndex(idx)
                break
        layout.addWidget(seg_combo)

        seg_help = QLabel("")
        seg_help.setWordWrap(True)
        seg_help.setStyleSheet("color: palette(dark); font-size: 10px;")
        layout.addWidget(seg_help)

        def update_seg_help(index):
            method = seg_combo.itemData(index)
            if method == 'none':
                seg_help.setText("Each cell grows independently. Masks may overlap for cells in close proximity.")
            elif method == 'competitive':
                seg_help.setText(
                    "All cells grow simultaneously in a single shared priority queue. "
                    "Each pixel is claimed by whichever cell reaches it first (brightest-neighbor-first). "
                    "Creates natural boundaries along intensity valleys between cells.")
            elif method == 'watershed':
                seg_help.setText(
                    "Computes watershed basins from the image gradient using soma centroids as seeds. "
                    "Each cell's growth is confined to its watershed territory. "
                    "Good for cells separated by clear intensity dips.")

        seg_combo.currentIndexChanged.connect(update_seg_help)
        update_seg_help(seg_combo.currentIndex())

        layout.addSpacing(10)

        # --- Circular Growth Constraint ---
        circ_group = QLabel("<b>Circular Growth Constraint</b>")
        layout.addWidget(circ_group)

        circular_check = QCheckBox("Limit growth to circular boundary around soma")
        circular_check.setChecked(self.use_circular_constraint)
        circular_check.setToolTip(
            "Constrains mask growth to a circle centered on the soma centroid.\n"
            "The circle's area = target mask area + buffer.\n"
            "This promotes even radial development and discourages\n"
            "one long branch from dominating the mask.")
        layout.addWidget(circular_check)

        buffer_layout = QHBoxLayout()
        buffer_layout.addWidget(QLabel("  Buffer:"))
        buffer_spin = QSpinBox()
        buffer_spin.setRange(0, 2000)
        buffer_spin.setSingleStep(50)
        buffer_spin.setValue(self.circular_buffer_um2)
        buffer_spin.setSuffix(" um2")
        buffer_spin.setToolTip("Extra area beyond target mask size for the circular boundary")
        buffer_layout.addWidget(buffer_spin)
        layout.addLayout(buffer_layout)

        circ_help = QLabel(
            "Radius = sqrt((target_area + buffer) / pi). "
            "Larger buffer = more room for branches; smaller = more compact masks.")
        circ_help.setWordWrap(True)
        circ_help.setStyleSheet("color: palette(dark); font-size: 10px;")
        layout.addWidget(circ_help)

        layout.addSpacing(10)

        # Help text
        help_text = QLabel(
            "💡 Tip:\n"
            "• 0% = No minimum (include all pixels)\n"
            "• 5% = Default (exclude background)\n"
            "• 30% = Moderate (exclude dimmer pixels)\n"
            "• 50%+ = Strict (only bright pixels)"
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
        self.mask_segmentation_method = seg_combo.currentData()
        self.use_circular_constraint = circular_check.isChecked()
        self.circular_buffer_um2 = buffer_spin.value()

        # Now proceed with mask generation
        try:
            pixel_size = self._get_pixel_size()

            # Build area list from user settings
            area_list = list(range(self.mask_min_area, self.mask_max_area + 1, self.mask_step_size))
            if area_list[-1] != self.mask_max_area:
                area_list.append(self.mask_max_area)

            self.progress_bar.setVisible(True)
            self.progress_status_label.setVisible(True)

            total_outlines = sum(len(data['soma_outlines']) for data in self.images.values()
                                 if data['selected'] and data['soma_outlines'])
            current_count = 0

            # Create TEMPMASKCHECKLIST folder for per-image checklists
            temp_cl_dir = None
            if self.output_dir:
                temp_cl_dir = os.path.join(self.output_dir, 'TEMPMASKCHECKLIST')
                os.makedirs(temp_cl_dir, exist_ok=True)

            seg_method = self.mask_segmentation_method

            for img_name, img_data in self.images.items():
                if not img_data['selected'] or not img_data['soma_outlines']:
                    continue

                self.log(f"Generating masks for {img_name}...")

                # Ensure processed image is in memory (may have been freed to save RAM)
                processed_img = self._ensure_processed_loaded(img_name)
                if processed_img is None:
                    self.log(f"  ⚠️ Skipping {img_name}: cannot load processed image")
                    continue

                # Reconstruct outline masks from polygon_points if needed
                # (after session load, outline masks are None)
                self._ensure_outline_masks(img_name, processed_img.shape[:2])

                # Use per-image pixel size if set
                img_pixel_size = self._get_pixel_size(img_name)
                if img_pixel_size != pixel_size:
                    self.log(f"  Using per-image pixel size: {img_pixel_size} µm/px")

                # Write per-image checklist CSV into TEMPMASKCHECKLIST
                img_cl_path = None
                if temp_cl_dir:
                    img_basename = os.path.splitext(img_name)[0]
                    img_cl_path = os.path.join(temp_cl_dir, f"{img_basename}_mask_checklist.csv")
                    cl_rows = []
                    for sd in img_data['soma_outlines']:
                        for area_val in area_list:
                            mask_key = f"{img_name}_{sd['soma_id']}_area{area_val}"
                            cl_rows.append([mask_key, '0'])
                    self._write_checklist(img_cl_path, cl_rows, ['Mask', 'Generated'])

                if seg_method == 'competitive':
                    # Competitive growth: all somas grow simultaneously
                    n_img_somas = len(img_data['soma_outlines'])
                    self.log(f"  Using competitive growth for {n_img_somas} cells")
                    self.progress_status_label.setText(f"Competitive growth: {img_name}")
                    QApplication.processEvents()

                    # Progress callback keeps the UI responsive during the long growth loop
                    base_pct = int((current_count / total_outlines) * 100)
                    img_pct_range = max(1, int((n_img_somas / total_outlines) * 100))

                    def _competitive_progress(growth_pct):
                        bar_val = base_pct + int(growth_pct / 100.0 * img_pct_range)
                        self.progress_bar.setValue(min(bar_val, 99))
                        self.progress_status_label.setText(
                            f"Competitive growth: {img_name} ({growth_pct}%)")
                        QApplication.processEvents()

                    masks = self._create_competitive_masks(
                        processed_img, img_data['soma_outlines'],
                        area_list, img_pixel_size, img_name,
                        progress_callback=_competitive_progress
                    )
                    img_data['masks'].extend(masks)

                    if img_cl_path and os.path.exists(img_cl_path):
                        for m in masks:
                            m_key = f"{img_name}_{m['soma_id']}_area{m['area_um2']}"
                            self._update_checklist_row(img_cl_path, 0, m_key, 1, 1)

                    current_count += n_img_somas
                    self.progress_bar.setValue(int((current_count / total_outlines) * 100))
                    QApplication.processEvents()
                else:
                    # Independent or watershed: per-soma growth — PARALLEL
                    territory_map = None
                    if seg_method == 'watershed':
                        self.log(f"  Computing watershed territories for {len(img_data['soma_outlines'])} cells")
                        self.progress_status_label.setText(f"Watershed: {img_name}")
                        QApplication.processEvents()
                        territory_map = self._build_watershed_territory_map(
                            processed_img, img_data['soma_outlines'], img_pixel_size
                        )

                    # processed_img already loaded above
                    n_somas = len(img_data['soma_outlines'])
                    n_workers = min(n_somas, max(1, multiprocessing.cpu_count() - 1))

                    self.log(f"  Generating masks for {n_somas} somas")
                    self.progress_status_label.setText(f"Generating masks: {img_name} ({n_somas} somas)")
                    QApplication.processEvents()

                    # Prepare arguments for parallel execution
                    task_args = []
                    for soma_data in img_data['soma_outlines']:
                        centroid = soma_data['centroid']
                        soma_idx = soma_data['soma_idx']
                        soma_id = soma_data['soma_id']
                        soma_area_um2 = soma_data.get('soma_area_um2', 0)
                        soma_outline = soma_data.get('outline')

                        cy, cx = int(centroid[0]), int(centroid[1])
                        sorted_areas = sorted(area_list, reverse=True)
                        largest_target_px = int(sorted_areas[0] / (img_pixel_size ** 2))
                        min_roi_radius = int(np.sqrt(largest_target_px / np.pi) * 3)
                        roi_size = max(200, min_roi_radius)

                        y_min = max(0, cy - roi_size)
                        y_max = min(processed_img.shape[0], cy + roi_size)
                        x_min = max(0, cx - roi_size)
                        x_max = min(processed_img.shape[1], cx + roi_size)

                        roi = processed_img[y_min:y_max, x_min:x_max].astype(np.float64)

                        # Extract soma outline ROI
                        soma_outline_roi = None
                        if soma_outline is not None:
                            soma_outline_roi = soma_outline[y_min:y_max, x_min:x_max]

                        # Extract territory ROI
                        territory_roi_data = None
                        my_territory_label = 0
                        if territory_map is not None:
                            territory_roi_data = territory_map[y_min:y_max, x_min:x_max]
                            cy_roi = max(0, min(roi.shape[0] - 1, cy - y_min))
                            cx_roi = max(0, min(roi.shape[1] - 1, cx - x_min))
                            my_territory_label = territory_roi_data[cy_roi, cx_roi]
                            if my_territory_label <= 0:
                                for dr in range(-3, 4):
                                    for dc in range(-3, 4):
                                        nr, nc = cy_roi + dr, cx_roi + dc
                                        if 0 <= nr < roi.shape[0] and 0 <= nc < roi.shape[1] and territory_roi_data[nr, nc] > 0:
                                            my_territory_label = territory_roi_data[nr, nc]
                                            break
                                    if my_territory_label > 0:
                                        break

                        task_args.append((
                            centroid, area_list, img_pixel_size, soma_idx, soma_id,
                            processed_img.shape, roi, (y_min, y_max, x_min, x_max),
                            soma_area_um2, soma_outline_roi,
                            territory_roi_data, my_territory_label,
                            self.use_circular_constraint, self.circular_buffer_um2,
                            self.use_min_intensity, self.min_intensity_percent, img_name
                        ))

                    # Run serially (desktops are not set up for parallel work)
                    for a in task_args:
                        masks = _grow_masks_for_soma(a)
                        img_data['masks'].extend(masks)
                        current_count += 1
                        self.progress_bar.setValue(int((current_count / total_outlines) * 100))
                        self.progress_status_label.setText(f"Generating masks: {current_count}/{total_outlines}")
                        QApplication.processEvents()

                    # Mark checklists after all somas done
                    if img_cl_path and os.path.exists(img_cl_path):
                        for m in img_data['masks']:
                            m_key = f"{img_name}_{m['soma_id']}_area{m['area_um2']}"
                            self._update_checklist_row(img_cl_path, 0, m_key, 1, 1)

                # Export ALL generated masks to disk immediately so they
                # survive save/load even before QA approval
                if self.masks_dir and os.path.isdir(self.masks_dir):
                    self._export_all_masks_to_disk(img_name, img_data['masks'])

                # Free mask arrays from RAM — they are now safely on disk
                for mask_data in img_data['masks']:
                    mask_data['mask'] = None

                # Free processed image and color image from RAM for this image
                # They can be reloaded from disk on demand
                img_data['processed'] = None
                img_data['color_image'] = None
                img_data.pop('color_image', None)

                # Delete per-image checklist — this image is done
                if img_cl_path and os.path.exists(img_cl_path):
                    self._delete_checklist(img_cl_path)

                img_data['status'] = 'masks_generated'
                self._update_file_list_item(img_name)
                self.log(f"  Freed RAM for {img_name} (masks saved to disk)")

            self.progress_bar.setVisible(False)
            self.progress_status_label.setVisible(False)

            # Delete TEMPMASKCHECKLIST folder — all images are done
            if temp_cl_dir and os.path.isdir(temp_cl_dir):
                import shutil
                try:
                    shutil.rmtree(temp_cl_dir)
                except Exception:
                    pass

            self.batch_qa_btn.setEnabled(True)
            self.clear_masks_btn.setEnabled(True)
            self.opacity_widget.setVisible(True)
            # self.update_workflow_status()

            total_masks = sum(len(data['masks']) for data in self.images.values() if data['selected'])

            seg_labels = {'none': 'None (independent)', 'competitive': 'Competitive growth', 'watershed': 'Watershed territories'}
            self.log("=" * 50)
            self.log(f"✓ Generated {total_masks} masks total")
            self.log(f"✓ Mask sizes: {', '.join(str(a) for a in area_list)} µm²")
            if self.use_min_intensity:
                self.log(f"✓ Used minimum intensity: {self.min_intensity_percent}%")
            if seg_method != 'none':
                self.log(f"✓ Cell boundary segmentation: {seg_labels.get(seg_method, seg_method)}")
            if self.use_circular_constraint:
                self.log(f"✓ Circular growth constraint: buffer {self.circular_buffer_um2} µm²")
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
        min_area_spin.setRange(10, 2000)
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

        # Preview threshold button
        preview_thresh_btn_qa = QPushButton("Preview Threshold on This Image")
        preview_thresh_btn_qa.setToolTip(
            "Opens a window showing which pixels would be excluded (red) at the current threshold")
        preview_thresh_btn_qa.clicked.connect(
            lambda: self._preview_intensity_threshold(min_intensity_slider.value(), img_name))
        layout.addWidget(preview_thresh_btn_qa)

        layout.addSpacing(5)

        # Circular constraint
        layout.addWidget(QLabel("<b>Circular Growth Constraint</b>"))
        regen_circular_check = QCheckBox("Limit growth to circular boundary")
        regen_circular_check.setChecked(self.use_circular_constraint)
        regen_circular_check.setToolTip(
            "Promotes even radial growth instead of one long branch dominating.")
        layout.addWidget(regen_circular_check)

        regen_buffer_layout = QHBoxLayout()
        regen_buffer_layout.addWidget(QLabel("  Buffer:"))
        regen_buffer_spin = QSpinBox()
        regen_buffer_spin.setRange(0, 2000)
        regen_buffer_spin.setSingleStep(50)
        regen_buffer_spin.setValue(self.circular_buffer_um2)
        regen_buffer_spin.setSuffix(" um2")
        regen_buffer_layout.addWidget(regen_buffer_spin)
        layout.addLayout(regen_buffer_layout)

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

        pixel_size = self._get_pixel_size(img_name)

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

        # Temporarily override settings for this generation
        saved_intensity = self.min_intensity_percent
        saved_use_intensity = self.use_min_intensity
        saved_circular = self.use_circular_constraint
        saved_buffer = self.circular_buffer_um2
        self.min_intensity_percent = regen_intensity
        self.use_min_intensity = regen_intensity > 0
        self.use_circular_constraint = regen_circular_check.isChecked()
        self.circular_buffer_um2 = regen_buffer_spin.value()

        # Generate new masks
        circ_info = f", circular buffer {self.circular_buffer_um2} µm²" if self.use_circular_constraint else ""
        self.log(f"Redoing masks for {img_name}: {regen_min}-{regen_max} µm², "
                 f"step {regen_step}, intensity {regen_intensity}%{circ_info}")

        # Ensure processed image is loaded (may have been freed to save RAM)
        processed_img = self._ensure_processed_loaded(img_name)
        if processed_img is None:
            QMessageBox.warning(self, "Error", f"Cannot reload processed image for {img_name}")
            return

        # Reconstruct outline masks from polygon_points if needed
        self._ensure_outline_masks(img_name, processed_img.shape[:2])

        for soma_data in img_data['soma_outlines']:
            centroid = soma_data['centroid']
            soma_idx = soma_data['soma_idx']
            soma_id = soma_data['soma_id']
            soma_area_um2 = soma_data.get('soma_area_um2', 0)
            soma_outline = soma_data.get('outline')

            masks = self._create_annulus_masks(
                centroid, area_list, pixel_size, soma_idx, soma_id,
                processed_img, img_name, soma_area_um2,
                soma_outline_mask=soma_outline
            )
            img_data['masks'].extend(masks)

        # Restore global settings
        self.min_intensity_percent = saved_intensity
        self.use_min_intensity = saved_use_intensity
        self.use_circular_constraint = saved_circular
        self.circular_buffer_um2 = saved_buffer

        # Export all regenerated masks to disk
        if self.masks_dir and os.path.isdir(self.masks_dir):
            self._export_all_masks_to_disk(img_name, img_data['masks'])

        # Update status
        img_data['status'] = 'masks_generated'
        self._update_file_list_item(img_name)

        # Add new masks to all_masks_flat
        if img_data['selected']:
            for mask_data in img_data['masks']:
                self.all_masks_flat.append({
                    'image_name': img_name,
                    'mask_data': mask_data,
                })

        # Rebuild soma ordering after regeneration
        self._qa_soma_order = []
        seen_somas = set()
        for flat in self.all_masks_flat:
            key = (flat['image_name'], flat['mask_data']['soma_id'])
            if key not in seen_somas:
                seen_somas.add(key)
                self._qa_soma_order.append(key)
        # Keep existing finalized set — only somas still present matter
        self._qa_finalized_somas = {k for k in self._qa_finalized_somas if k in seen_somas}

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
            self.opacity_widget.setVisible(True)
            QMessageBox.information(self, "Done",
                f"Regenerated {total} masks for {os.path.splitext(img_name)[0]}.\n\nReady for QA.")

    def _build_watershed_territory_map(self, processed_img, soma_outlines, pixel_size_um):
        """Build a watershed territory map assigning each pixel to the nearest soma basin.

        Uses soma centroids as seeds and the image gradient as the landscape.
        Returns an array of same shape as processed_img where each pixel value
        is the soma index (1-based) that owns it, 0 for background/boundary.
        """
        import cv2 as _cv2

        h, w = processed_img.shape
        img_norm = processed_img.astype(np.float64)
        imin, imax = img_norm.min(), img_norm.max()
        if imax > imin:
            img_norm = (img_norm - imin) / (imax - imin) * 255.0
        img_u8 = img_norm.astype(np.uint8)

        # Compute gradient magnitude for watershed landscape
        grad_x = _cv2.Sobel(img_u8, _cv2.CV_64F, 1, 0, ksize=3)
        grad_y = _cv2.Sobel(img_u8, _cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)
        gradient = (gradient / (gradient.max() + 1e-10) * 255).astype(np.uint8)

        # Build markers: each soma gets a unique label (1-based)
        markers = np.zeros((h, w), dtype=np.int32)
        for i, soma_data in enumerate(soma_outlines):
            label = i + 1
            centroid = soma_data['centroid']
            cy, cx = int(centroid[0]), int(centroid[1])
            cy = max(0, min(h - 1, cy))
            cx = max(0, min(w - 1, cx))
            # Seed with soma outline if available, otherwise a small circle
            outline_mask = soma_data.get('outline')
            if outline_mask is not None and outline_mask.shape == (h, w):
                markers[outline_mask > 0] = label
            else:
                _cv2.circle(markers, (cx, cy), max(3, int(5 / pixel_size_um)), label, -1)

        # Convert gradient to 3-channel for cv2.watershed
        grad_color = _cv2.cvtColor(gradient, _cv2.COLOR_GRAY2BGR)
        _cv2.watershed(grad_color, markers)

        # markers now has: -1 = boundary, 0 = bg, >0 = soma label
        # Convert to territory map: set boundaries to 0
        territory = markers.copy()
        territory[territory < 0] = 0
        return territory

    def _create_competitive_masks(self, processed_img, soma_outlines_data, area_list_um2,
                                   pixel_size_um, img_name, progress_callback=None):
        """Create masks for ALL somas in an image using competitive priority region growing.

        All somas grow simultaneously from a single shared priority queue.
        Each pixel is claimed by whichever soma reaches it first (brightest-
        neighbor-first).  This naturally creates territory boundaries along
        intensity valleys between cells.

        Returns a list of mask dicts (same format as _create_annulus_masks).
        """
        import heapq

        h, w = processed_img.shape
        sorted_areas = sorted(area_list_um2, reverse=True)
        largest_target_px = int(sorted_areas[0] / (pixel_size_um ** 2))

        # Compute intensity floor
        intensity_floor = 0.0
        if self.use_min_intensity and self.min_intensity_percent > 0:
            img_max = processed_img.max()
            if img_max > 0:
                intensity_floor = img_max * (self.min_intensity_percent / 100.0)

        roi = processed_img.astype(np.float64)

        # Circular constraint: limit growth to a circle centered on each soma's
        # centroid whose area = largest_target + buffer.  Promotes radial growth.
        max_radius_px_sq = None
        if self.use_circular_constraint:
            constraint_area_um2 = sorted_areas[0] + self.circular_buffer_um2
            constraint_area_px = constraint_area_um2 / (pixel_size_um ** 2)
            max_radius_px = np.sqrt(constraint_area_px / np.pi)
            max_radius_px_sq = max_radius_px ** 2

        # Shared state across all somas
        # owner_map: which soma owns each pixel (-1 = unclaimed)
        owner_map = np.full((h, w), -1, dtype=np.int32)
        visited = np.zeros((h, w), dtype=bool)
        heap = []  # shared priority queue: (-intensity, row, col, soma_index)

        # Per-soma growth tracking
        n_somas = len(soma_outlines_data)
        growth_orders = [[] for _ in range(n_somas)]  # growth_orders[i] = [(r,c), ...]
        soma_seed_counts = [0] * n_somas
        soma_centroids = []  # (cy, cx) per soma for circular constraint

        # Seed all somas into the shared heap
        for si, soma_data in enumerate(soma_outlines_data):
            centroid = soma_data['centroid']
            cy, cx = int(centroid[0]), int(centroid[1])
            cy = max(0, min(h - 1, cy))
            cx = max(0, min(w - 1, cx))
            soma_centroids.append((cy, cx))
            soma_outline = soma_data.get('outline')

            seeded = False
            if soma_outline is not None and soma_outline.shape == (h, w):
                soma_ys, soma_xs = np.where(soma_outline > 0)
                for sr, sc in zip(soma_ys, soma_xs):
                    if not visited[sr, sc]:
                        visited[sr, sc] = True
                        owner_map[sr, sc] = si
                        growth_orders[si].append((sr, sc))
                        soma_seed_counts[si] += 1
                # Push boundary neighbors
                for sr, sc in zip(soma_ys, soma_xs):
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = sr + dr, sc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                            if roi[nr, nc] >= intensity_floor:
                                if max_radius_px_sq is not None:
                                    dy = nr - cy
                                    dx = nc - cx
                                    if (dy * dy + dx * dx) > max_radius_px_sq:
                                        continue
                                visited[nr, nc] = True
                                owner_map[nr, nc] = si
                                heapq.heappush(heap, (-roi[nr, nc], nr, nc, si))
                seeded = True

            if not seeded:
                if not visited[cy, cx]:
                    visited[cy, cx] = True
                    owner_map[cy, cx] = si
                    growth_orders[si].append((cy, cx))
                    soma_seed_counts[si] = 1
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cy + dr, cx + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                            if roi[nr, nc] >= intensity_floor:
                                if max_radius_px_sq is not None:
                                    dy = nr - cy
                                    dx = nc - cx
                                    if (dy * dy + dx * dx) > max_radius_px_sq:
                                        continue
                                visited[nr, nc] = True
                                owner_map[nr, nc] = si
                                heapq.heappush(heap, (-roi[nr, nc], nr, nc, si))

        # Competitive growth: all somas grow simultaneously
        # Stop each soma when it reaches its largest target
        soma_done = [False] * n_somas
        total_target_px = largest_target_px * n_somas
        total_grown = sum(len(go) for go in growth_orders)
        progress_interval = max(1000, total_target_px // 50)  # ~50 updates
        pixels_since_update = 0
        while heap:
            neg_intensity, r, c, si = heapq.heappop(heap)

            # Check if this pixel was already claimed by another soma
            # (can happen if a neighbor was pushed by multiple somas before being popped)
            if owner_map[r, c] != si:
                continue

            # Check if this soma has reached its target
            if len(growth_orders[si]) >= largest_target_px:
                soma_done[si] = True
                if all(soma_done):
                    break
                continue

            growth_orders[si].append((r, c))
            pixels_since_update += 1
            if progress_callback and pixels_since_update >= progress_interval:
                total_grown += pixels_since_update
                pixels_since_update = 0
                pct = min(99, int(total_grown / total_target_px * 100))
                progress_callback(pct)

            # Push unclaimed 4-connected neighbors
            sc_cy, sc_cx = soma_centroids[si]
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                    if roi[nr, nc] >= intensity_floor:
                        if max_radius_px_sq is not None:
                            dy = nr - sc_cy
                            dx = nc - sc_cx
                            if (dy * dy + dx * dx) > max_radius_px_sq:
                                continue
                        visited[nr, nc] = True
                        owner_map[nr, nc] = si
                        heapq.heappush(heap, (-roi[nr, nc], nr, nc, si))

        # Build mask dicts for each soma at each target area
        all_masks = []
        for si, soma_data in enumerate(soma_outlines_data):
            soma_idx = soma_data['soma_idx']
            soma_id = soma_data['soma_id']
            soma_area_um2 = soma_data.get('soma_area_um2', 0)
            soma_area_px = soma_seed_counts[si]
            go = growth_orders[si]

            print(f"  {soma_id}: soma={soma_area_px}px, grew to {len(go)}px (target: {largest_target_px})")

            soma_masks_start = len(all_masks)
            mask_pixel_counts = []
            for target_area_um2 in sorted_areas:
                target_px = int(target_area_um2 / (pixel_size_um ** 2))
                n_pixels = min(target_px, len(go))
                n_pixels = max(n_pixels, soma_area_px)
                n_pixels = min(n_pixels, len(go))

                mask_pixel_counts.append(n_pixels)

                full_mask = np.zeros((h, w), dtype=np.uint8)
                for r, c in go[:n_pixels]:
                    full_mask[r, c] = 1

                actual_area_um2 = n_pixels * (pixel_size_um ** 2)
                print(f"    {target_area_um2} um2: {n_pixels} px = {actual_area_um2:.1f} um2 actual")

                all_masks.append({
                    'image_name': img_name,
                    'soma_idx': soma_idx,
                    'soma_id': soma_id,
                    'area_um2': target_area_um2,
                    'mask': full_mask,
                    'approved': None,
                    'soma_area_um2': soma_area_um2
                })

            # Auto-reject duplicate masks for this soma: when multiple target
            # areas produce the same pixel count (pixel-identical masks), reject
            # all except the one with the smallest target area.
            pixel_count_groups = {}
            for i, n_px in enumerate(mask_pixel_counts):
                pixel_count_groups.setdefault(n_px, []).append(i)
            for n_px, indices in pixel_count_groups.items():
                if len(indices) > 1:
                    keep_idx = indices[-1]
                    for idx in indices[:-1]:
                        all_masks[soma_masks_start + idx]['approved'] = False
                        all_masks[soma_masks_start + idx]['duplicate'] = True
                        print(f"    ⚠️ Auto-rejected {all_masks[soma_masks_start + idx]['area_um2']} µm² "
                              f"(duplicate of {all_masks[soma_masks_start + keep_idx]['area_um2']} µm², both {n_px} px)")

            # Enforce subset invariant for this soma's masks
            _enforce_mask_subset_invariant(all_masks[soma_masks_start:])

        return all_masks

    def _create_annulus_masks(self, centroid, area_list_um2, pixel_size_um, soma_idx, soma_id, processed_img, img_name,
                              soma_area_um2, soma_outline_mask=None, territory_map=None):
        """Create nested cell masks using priority region growing from the soma outline.

        Seeds the growth with the entire soma outline so the soma pixels are
        "free" and the mask pixel budget goes toward territory beyond the soma.
        Grows outward from the soma boundary, always adding the brightest
        neighboring pixel next.  Respects the minimum intensity floor setting.

        If territory_map is provided (from watershed), growth is constrained to
        pixels within this soma's territory.
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

        # Compute circular constraint radius (in pixels) if enabled
        # The constraint limits growth to a circle centered on the soma centroid
        # whose area = largest_target_area + buffer.  This encourages more radial
        # (circular) growth instead of one long branch dominating.
        max_radius_px_sq = None  # None = no constraint
        if self.use_circular_constraint:
            constraint_area_um2 = sorted_areas[0] + self.circular_buffer_um2
            constraint_area_px = constraint_area_um2 / (pixel_size_um ** 2)
            max_radius_px = np.sqrt(constraint_area_px / np.pi)
            max_radius_px_sq = max_radius_px ** 2

        # Compute intensity floor from user settings
        intensity_floor = 0.0
        if self.use_min_intensity and self.min_intensity_percent > 0:
            roi_max = roi.max()
            if roi_max > 0:
                intensity_floor = roi_max * (self.min_intensity_percent / 100.0)

        # Build territory constraint ROI if watershed territory_map is provided
        territory_roi = None
        if territory_map is not None:
            territory_roi = territory_map[y_min:y_max, x_min:x_max]
            # Find this soma's label from the centroid position
            my_label = territory_roi[cy_roi, cx_roi]
            if my_label <= 0:
                # Centroid fell on a boundary — search nearby for our label
                for dr in range(-3, 4):
                    for dc in range(-3, 4):
                        nr, nc = cy_roi + dr, cx_roi + dc
                        if 0 <= nr < h and 0 <= nc < w and territory_roi[nr, nc] > 0:
                            my_label = territory_roi[nr, nc]
                            break
                    if my_label > 0:
                        break

        def _in_territory(r, c):
            """Check if pixel (r,c) is within this soma's watershed territory."""
            if territory_roi is None:
                return True
            return territory_roi[r, c] == my_label or territory_roi[r, c] <= 0

        def _in_circle(r, c):
            """Check if pixel (r,c) is within the circular growth constraint."""
            if max_radius_px_sq is None:
                return True
            dy = r - cy_roi
            dx = c - cx_roi
            return (dy * dy + dx * dx) <= max_radius_px_sq

        # Priority region growing: grow from soma outline, brightest neighbor first
        # Use a max-heap (negate intensity for min-heap)
        visited = np.zeros((h, w), dtype=bool)
        growth_order = []  # list of (row, col) in the order pixels were added

        heap = []

        # Seed with all soma outline pixels (they are "free" — part of every mask)
        soma_seed_count = 0
        if soma_outline_mask is not None:
            outline_roi = soma_outline_mask[y_min:y_max, x_min:x_max]
            soma_ys, soma_xs = np.where(outline_roi > 0)
            for sr, sc in zip(soma_ys, soma_xs):
                if not visited[sr, sc]:
                    visited[sr, sc] = True
                    growth_order.append((sr, sc))
                    soma_seed_count += 1
            # Push boundary neighbors of the soma into the heap
            for sr, sc in zip(soma_ys, soma_xs):
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = sr + dr, sc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                        if roi[nr, nc] >= intensity_floor and _in_territory(nr, nc) and _in_circle(nr, nc):
                            visited[nr, nc] = True
                            heapq.heappush(heap, (-roi[nr, nc], nr, nc))

        # Fallback: if no soma outline available, seed with centroid
        if soma_seed_count == 0:
            visited[cy_roi, cx_roi] = True
            growth_order.append((cy_roi, cx_roi))
            soma_seed_count = 1
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cy_roi + dr, cx_roi + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                    if roi[nr, nc] >= intensity_floor and _in_territory(nr, nc) and _in_circle(nr, nc):
                        visited[nr, nc] = True
                        heapq.heappush(heap, (-roi[nr, nc], nr, nc))

        # Grow outward from the soma boundary up to largest target
        while heap and len(growth_order) < largest_target_px:
            neg_intensity, r, c = heapq.heappop(heap)

            growth_order.append((r, c))

            # Add 4-connected neighbors to the heap
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                    if roi[nr, nc] >= intensity_floor and _in_territory(nr, nc) and _in_circle(nr, nc):
                        visited[nr, nc] = True
                        heapq.heappush(heap, (-roi[nr, nc], nr, nc))

        print(f"  {soma_id}: soma={soma_seed_count}px, grew to {len(growth_order)}px (target: {largest_target_px})")

        # Build masks for each target area from the growth order
        # Largest first (matches QA presentation order)
        # Every mask always includes the full soma at minimum
        # Target area is a ceiling — if intensity floor stopped growth early,
        # min(target_px, len(growth_order)) naturally caps the mask smaller
        # If target < soma area, substitute the soma mask for that size
        soma_area_px = soma_seed_count
        mask_pixel_counts = []
        for target_area_um2 in sorted_areas:
            target_px = int(target_area_um2 / (pixel_size_um ** 2))

            # Per-step circular constraint: ring grows with each target
            if self.use_circular_constraint:
                step_constraint_um2 = target_area_um2 + self.circular_buffer_um2
                step_constraint_px = step_constraint_um2 / (pixel_size_um ** 2)
                step_radius_sq = step_constraint_px / np.pi
                step_order = [(r, c) for r, c in growth_order
                              if (r - cy_roi) ** 2 + (c - cx_roi) ** 2 <= step_radius_sq]
            else:
                step_order = growth_order

            n_pixels = min(target_px, len(step_order))
            n_pixels = max(n_pixels, soma_area_px)  # always include full soma
            n_pixels = min(n_pixels, len(step_order))

            mask_pixel_counts.append(n_pixels)

            mask_roi = np.zeros((h, w), dtype=np.uint8)
            for r, c in step_order[:n_pixels]:
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

        # Auto-reject duplicate masks: when multiple target areas produce the
        # same pixel count (and thus pixel-identical masks), reject all except
        # the one with the smallest target area.
        pixel_count_groups = {}
        for i, n_px in enumerate(mask_pixel_counts):
            pixel_count_groups.setdefault(n_px, []).append(i)
        for n_px, indices in pixel_count_groups.items():
            if len(indices) > 1:
                # sorted_areas is largest-first, so last index = smallest target area
                keep_idx = indices[-1]
                for idx in indices[:-1]:
                    masks[idx]['approved'] = False
                    masks[idx]['duplicate'] = True
                    print(f"    ⚠️ Auto-rejected {masks[idx]['area_um2']} µm² "
                          f"(duplicate of {masks[keep_idx]['area_um2']} µm², both {n_px} px)")

        # Enforce subset invariant
        _enforce_mask_subset_invariant(masks)

        return masks

    # ----------------------------------------------------------------
    # 3D MASK GENERATION
    # ----------------------------------------------------------------

    def _batch_generate_masks_3d(self):
        """Show 3D mask generation settings dialog then generate masks."""
        from PyQt5.QtWidgets import QDoubleSpinBox
        dialog = QDialog(self)
        dialog.setWindowTitle("3D Mask Generation Settings")
        dialog.setModal(True)
        layout = QVBoxLayout()

        title = QLabel("Configure 3D Mask Generation")
        title_font = title.font()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        layout.addWidget(title)

        # Soma detection settings
        layout.addWidget(QLabel("<b>Soma Detection (3D Region Growing)</b>"))
        soma_layout = QHBoxLayout()
        soma_layout.addWidget(QLabel("Intensity tolerance:"))
        tol_spin = QSpinBox()
        tol_spin.setRange(5, 100)
        tol_spin.setValue(self.soma_intensity_tolerance)
        soma_layout.addWidget(tol_spin)
        soma_layout.addWidget(QLabel("Max radius (um):"))
        rad_spin = QDoubleSpinBox()
        rad_spin.setRange(1.0, 50.0)
        rad_spin.setValue(self.soma_max_radius_um)
        rad_spin.setSingleStep(0.5)
        soma_layout.addWidget(rad_spin)
        layout.addLayout(soma_layout)
        layout.addSpacing(10)

        # Volume settings
        layout.addWidget(QLabel("<b>Mask Volumes (um^3)</b>"))
        size_grid = QHBoxLayout()
        size_grid.addWidget(QLabel("Min:"))
        min_vol_spin = QSpinBox()
        min_vol_spin.setRange(50, 50000)
        min_vol_spin.setSingleStep(100)
        min_vol_spin.setValue(self.mask_min_volume)
        min_vol_spin.setSuffix(" um^3")
        size_grid.addWidget(min_vol_spin)
        size_grid.addWidget(QLabel("Max:"))
        max_vol_spin = QSpinBox()
        max_vol_spin.setRange(100, 100000)
        max_vol_spin.setSingleStep(500)
        max_vol_spin.setValue(self.mask_max_volume)
        max_vol_spin.setSuffix(" um^3")
        size_grid.addWidget(max_vol_spin)
        size_grid.addWidget(QLabel("Step:"))
        step_spin = QSpinBox()
        step_spin.setRange(50, 10000)
        step_spin.setSingleStep(100)
        step_spin.setValue(self.mask_step_size)
        step_spin.setSuffix(" um^3")
        size_grid.addWidget(step_spin)
        layout.addLayout(size_grid)

        preview_label = QLabel("")
        preview_label.setStyleSheet("color: palette(dark); font-size: 10px;")
        preview_label.setWordWrap(True)
        layout.addWidget(preview_label)

        def update_size_preview():
            mn = min_vol_spin.value()
            mx = max_vol_spin.value()
            st = step_spin.value()
            if mn > mx:
                preview_label.setText("Min must be <= Max")
                return
            sizes = list(range(mn, mx + 1, st))
            if sizes[-1] != mx:
                sizes.append(mx)
            preview_label.setText(
                f"Masks: {', '.join(str(s) for s in sizes)} um^3  ({len(sizes)} masks per cell)")

        min_vol_spin.valueChanged.connect(lambda: update_size_preview())
        max_vol_spin.valueChanged.connect(lambda: update_size_preview())
        step_spin.valueChanged.connect(lambda: update_size_preview())
        update_size_preview()
        layout.addSpacing(10)

        # Intensity filtering
        layout.addWidget(QLabel("<b>Intensity Filtering</b>"))
        min_intensity_check = QCheckBox("Use minimum intensity threshold")
        min_intensity_check.setChecked(self.use_min_intensity)
        layout.addWidget(min_intensity_check)
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("  Min intensity:"))
        min_intensity_slider = QSlider(Qt.Horizontal)
        min_intensity_slider.setRange(0, 100)
        min_intensity_slider.setValue(self.min_intensity_percent)
        slider_layout.addWidget(min_intensity_slider)
        min_int_label = QLabel(f"{self.min_intensity_percent}%")
        min_intensity_slider.valueChanged.connect(lambda v: min_int_label.setText(f"{v}%"))
        slider_layout.addWidget(min_int_label)
        layout.addLayout(slider_layout)

        # Preview threshold button
        preview_thresh_btn_3d = QPushButton("Preview Threshold on Current Image")
        preview_thresh_btn_3d.setToolTip(
            "Opens a window showing which pixels would be excluded (red) at the current threshold")
        preview_thresh_btn_3d.clicked.connect(
            lambda: self._preview_intensity_threshold(min_intensity_slider.value()))
        layout.addWidget(preview_thresh_btn_3d)

        layout.addSpacing(10)

        # Segmentation method
        layout.addWidget(QLabel("<b>Cell Boundary Segmentation</b>"))
        seg_combo = QComboBox()
        seg_combo.addItem("None (independent growth)", "none")
        seg_combo.addItem("Competitive Growth (shared priority queue)", "competitive")
        for idx in range(seg_combo.count()):
            if seg_combo.itemData(idx) == self.mask_segmentation_method:
                seg_combo.setCurrentIndex(idx)
                break
        layout.addWidget(seg_combo)
        layout.addSpacing(10)

        # Buttons
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        ok_btn = QPushButton("Generate 3D Masks")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(dialog.accept)
        ok_btn.setStyleSheet("QPushButton { border: 2px solid #4CAF50; font-weight: bold; padding: 5px; }")
        button_layout.addWidget(ok_btn)
        layout.addLayout(button_layout)
        dialog.setLayout(layout)
        dialog.setMinimumWidth(450)

        if dialog.exec_() != QDialog.Accepted:
            return

        # Save settings
        self.soma_intensity_tolerance = tol_spin.value()
        self.soma_max_radius_um = rad_spin.value()
        self.use_min_intensity = min_intensity_check.isChecked()
        self.min_intensity_percent = min_intensity_slider.value()
        self.mask_min_volume = min_vol_spin.value()
        self.mask_max_volume = max_vol_spin.value()
        self.mask_step_size = step_spin.value()
        self.mask_segmentation_method = seg_combo.currentData()

        self._run_mask_generation_3d()

    def _run_mask_generation_3d(self):
        """Execute 3D mask generation with current settings."""
        try:
            voxel_xy = float(self.pixel_size_input.text())
            voxel_z = float(self.voxel_z_input.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Invalid voxel dimensions")
            return

        vol_list = list(range(self.mask_min_volume, self.mask_max_volume + 1, self.mask_step_size))
        if vol_list[-1] != self.mask_max_volume:
            vol_list.append(self.mask_max_volume)

        min_int_pct = self.min_intensity_percent if self.use_min_intensity else 0
        seg_method = self.mask_segmentation_method

        self.progress_bar.setVisible(True)
        self.progress_status_label.setVisible(True)

        total_somas = sum(len(data['somas']) for data in self.images.values()
                          if data['selected'] and data['somas'])
        current_count = 0

        try:
            for img_name, img_data in self.images.items():
                if not img_data['selected'] or not img_data['somas']:
                    continue

                stack = img_data.get('processed') or img_data.get('raw_stack')
                if stack is None:
                    self.log(f"SKIP {img_name}: no stack data")
                    continue

                self.log(f"Generating 3D masks for {img_name}...")

                # Detect somas in 3D
                soma_data_list = []
                for si, soma_zyx in enumerate(img_data['somas']):
                    soma_id = img_data['soma_ids'][si]
                    self.progress_status_label.setText(
                        f"Detecting soma {si + 1}/{len(img_data['somas'])}: {img_name}")
                    QApplication.processEvents()

                    soma_mask, centroid = detect_soma_3d(
                        stack, soma_zyx,
                        intensity_tolerance=self.soma_intensity_tolerance,
                        max_radius_um=self.soma_max_radius_um,
                        voxel_size_xy=voxel_xy,
                        voxel_size_z=voxel_z
                    )
                    if 'soma_masks' not in img_data:
                        img_data['soma_masks'] = {}
                    img_data['soma_masks'][soma_id] = soma_mask

                    soma_data_list.append({
                        'centroid': centroid,
                        'soma_mask': soma_mask,
                        'soma_id': soma_id,
                        'soma_idx': si,
                    })

                # Generate masks
                if seg_method == 'competitive' and len(soma_data_list) > 1:
                    self.log(f"  Competitive 3D growth for {len(soma_data_list)} cells")
                    self.progress_status_label.setText(f"Competitive 3D growth: {img_name}")
                    QApplication.processEvents()
                    masks = create_competitive_masks_3d(
                        stack, soma_data_list, vol_list,
                        voxel_xy, voxel_z, min_intensity_pct=min_int_pct
                    )
                    img_data['masks'].extend(masks)
                    current_count += len(soma_data_list)
                else:
                    for sdata in soma_data_list:
                        self.progress_status_label.setText(
                            f"3D masks: {current_count + 1}/{total_somas}")
                        QApplication.processEvents()
                        masks = create_spherical_annulus_masks(
                            stack, sdata['centroid'], vol_list,
                            voxel_xy, voxel_z,
                            soma_mask=sdata['soma_mask'],
                            min_intensity_pct=min_int_pct
                        )
                        for m in masks:
                            m['soma_idx'] = sdata['soma_idx']
                            m['soma_id'] = sdata['soma_id']
                        img_data['masks'].extend(masks)
                        current_count += 1
                    self.progress_bar.setValue(int(current_count / total_somas * 100))

                # Export masks to disk
                if self.masks_dir:
                    self._export_3d_masks_to_disk(img_name, img_data['masks'])

                img_data['status'] = 'masks_generated'
                self._update_file_list_item(img_name)

            self.progress_bar.setVisible(False)
            self.progress_status_label.setVisible(False)
            self.batch_qa_btn.setEnabled(True)
            self.clear_masks_btn.setEnabled(True)
            self.opacity_widget.setVisible(True)

            total_masks = sum(len(data['masks']) for data in self.images.values() if data['selected'])
            self.log("=" * 50)
            self.log(f"Generated {total_masks} 3D masks total")
            self.log(f"Mask volumes: {', '.join(str(v) for v in vol_list)} um^3")
            self.log("Ready for QA")
            self.log("=" * 50)
            self._auto_save()
            QMessageBox.information(self, "Success",
                                    f"Generated {total_masks} 3D masks!\n\nReady for QA.")

        except Exception as e:
            self.progress_bar.setVisible(False)
            self.progress_status_label.setVisible(False)
            self.log(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed: {e}")

    def _export_3d_masks_to_disk(self, img_name, masks):
        """Export 3D mask volumes to disk as multi-page TIFFs."""
        if not self.masks_dir:
            return
        os.makedirs(self.masks_dir, exist_ok=True)
        img_basename = os.path.splitext(img_name)[0]
        for mask_data in masks:
            mask_3d = mask_data.get('mask')
            if mask_3d is None:
                continue
            soma_id = mask_data.get('soma_id', 'soma_0')
            vol = mask_data.get('volume_um3', 0)
            fname = f"{img_basename}_{soma_id}_vol{int(vol)}_mask3d.tif"
            path = os.path.join(self.masks_dir, fname)
            try:
                tifffile.imwrite(path, (mask_3d * 255).astype(np.uint8))
            except Exception as e:
                self.log(f"  ERROR exporting {fname}: {e}")

    def _reload_3d_mask_from_disk(self, mask_data, img_name):
        """Reload a 3D mask from its TIFF file on disk."""
        if mask_data.get('mask') is not None:
            return True
        if not self.masks_dir or not os.path.isdir(self.masks_dir):
            return False
        img_basename = os.path.splitext(img_name)[0]
        soma_id = mask_data.get('soma_id', '')
        vol = mask_data.get('volume_um3', 0)
        fname = f"{img_basename}_{soma_id}_vol{int(vol)}_mask3d.tif"
        path = os.path.join(self.masks_dir, fname)
        if os.path.exists(path):
            try:
                arr = safe_tiff_read(path)
                mask_data['mask'] = (arr > 0).astype(np.uint8)
                return True
            except Exception as e:
                self.log(f"  Could not reload {fname}: {e}")
        return False

    def start_batch_qa(self):
        # Flatten all masks from all images
        self.all_masks_flat = []
        for img_name, img_data in self.images.items():
            if not img_data['selected']:
                continue
            # Sort masks: by soma pick order (soma_idx), then largest area first
            # within each soma so QA flows big→small
            size_key = 'volume_um3' if self.mode_3d else 'area_um2'
            sorted_masks = sorted(img_data['masks'],
                                  key=lambda m: (m.get('soma_idx', 0), -m.get(size_key, 0)))
            for mask_data in sorted_masks:
                self.all_masks_flat.append({
                    'image_name': img_name,
                    'mask_data': mask_data,
                })

        if not self.all_masks_flat:
            QMessageBox.warning(self, "Warning", "No masks to QA")
            return

        self.mask_qa_active = True
        # Keep existing undo history so user can undo back across QA sessions
        if not hasattr(self, 'last_qa_decisions'):
            self.last_qa_decisions = []

        # Build soma ordering for sliding window memory management
        # and soma->masks index for O(1) lookup instead of O(n) scans
        self._qa_soma_order = []
        self._qa_soma_order_index = {}  # soma_key -> position in _qa_soma_order
        self._qa_finalized_somas = set()
        self._qa_soma_mask_index = {}  # (img, soma_id) -> [list of flat indices]
        seen_somas = set()
        for i, flat in enumerate(self.all_masks_flat):
            key = (flat['image_name'], flat['mask_data'].get('soma_id', ''))
            if key not in seen_somas:
                seen_somas.add(key)
                self._qa_soma_order_index[key] = len(self._qa_soma_order)
                self._qa_soma_order.append(key)
            self._qa_soma_mask_index.setdefault(key, []).append(i)

        # Per-image soma count for fast "all somas finalized?" check
        self._qa_image_soma_count = {}
        for key in self._qa_soma_order:
            self._qa_image_soma_count[key[0]] = self._qa_image_soma_count.get(key[0], 0) + 1

        # Build running counters for O(1) status display (avoid O(n) scans)
        self._qa_auto_rejected_count = 0
        self._qa_approved_count = 0
        self._qa_user_rejected_count = 0
        for f in self.all_masks_flat:
            md = f['mask_data']
            if md.get('duplicate'):
                self._qa_auto_rejected_count += 1
            elif md.get('approved') is True:
                self._qa_approved_count += 1
            elif md.get('approved') is False:
                self._qa_user_rejected_count += 1

        # Deferred checklist: track updates in memory, flush with auto-save
        self._qa_checklist_dirty = {}

        # Find first unreviewed mask to resume from
        auto_rejected_count = self._qa_auto_rejected_count
        manually_reviewed_count = self._qa_approved_count + self._qa_user_rejected_count
        reviewed_count = auto_rejected_count + manually_reviewed_count
        first_unreviewed = 0
        for i, flat in enumerate(self.all_masks_flat):
            if flat['mask_data'].get('approved') is None:
                first_unreviewed = i
                break
        self.mask_qa_idx = first_unreviewed

        # If resuming, finalize somas that are already far behind
        if manually_reviewed_count > 0:
            self._evict_old_qa_masks()

        # Create mask_qa_checklist.csv to track QA progress
        qa_cl_path = self._get_checklist_path('mask_qa_checklist.csv')
        if qa_cl_path:
            cl_rows = []
            for flat in self.all_masks_flat:
                md = flat['mask_data']
                if self.mode_3d:
                    size_val = md.get('volume_um3', 0)
                    key = f"{flat['image_name']}_{md.get('soma_id', '')}_vol{size_val}"
                else:
                    size_val = md.get('area_um2', 0)
                    key = f"{flat['image_name']}_{md.get('soma_id', '')}_area{size_val}"
                passed = 1 if md.get('approved') is True else 0
                cl_rows.append([key, str(passed)])
            self._write_checklist(qa_cl_path, cl_rows, ['Mask', 'Passed QA'])

        self.approve_mask_btn.setEnabled(True)
        self.reject_mask_btn.setEnabled(True)
        self.approve_all_btn.setEnabled(True)
        self.approve_all_btn.setVisible(True)
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self.done_btn.setEnabled(False)
        self.undo_qa_btn.setEnabled(len(self.last_qa_decisions) > 0)
        self.undo_qa_btn.setVisible(True)
        self.regen_masks_btn.setVisible(True)
        self.clear_masks_btn.setEnabled(True)
        self.clear_masks_btn.setVisible(True)

        # Show and init progress bar — base it on masks that actually need
        # human review (total minus auto-rejected duplicates).
        masks_needing_review = len(self.all_masks_flat) - auto_rejected_count
        self.mask_qa_progress_bar.setMaximum(masks_needing_review)
        self.mask_qa_progress_bar.setValue(manually_reviewed_count)
        self.mask_qa_progress_bar.setVisible(True)

        self._show_current_mask()
        self.tabs.setCurrentIndex(3)

        self.log("=" * 50)
        self.log("🎯 BATCH MASK QA MODE")
        self.log(f"Total masks generated: {len(self.all_masks_flat)}")
        # Report auto-rejected duplicates
        if auto_rejected_count > 0:
            self.log(f"⚠️ {auto_rejected_count} duplicate masks auto-rejected (identical to a smaller target area)")
        if manually_reviewed_count > 0:
            self.log(f"Resuming: {manually_reviewed_count}/{masks_needing_review} manually reviewed")
        remaining = sum(1 for f in self.all_masks_flat if f['mask_data'].get('approved') is None)
        self.log(f"Masks needing review: {remaining}")
        self.log("Keyboard: A=Approve, R=Reject, ←→=Navigate, Space=Approve&Next")
        self.log("=" * 50)

    def _evict_old_qa_masks(self):
        """Evict mask arrays for somas that are beyond the sliding window.

        Keeps only the current soma + last _qa_soma_window_size reviewed somas
        in memory. Evicted approved masks are already on disk. Evicted rejected
        masks have their TIFF files deleted from disk.
        """
        if not self.all_masks_flat or self.mask_qa_idx >= len(self.all_masks_flat):
            return

        # Determine which soma we're currently on
        current_flat = self.all_masks_flat[self.mask_qa_idx]
        current_soma_key = (current_flat['image_name'], current_flat['mask_data']['soma_id'])

        current_soma_idx = self._qa_soma_order_index.get(current_soma_key)
        if current_soma_idx is None:
            return

        # Everything before this index should be evicted
        evict_before = current_soma_idx - self._qa_soma_window_size
        if evict_before <= 0:
            return

        for soma_idx in range(evict_before):
            soma_key = self._qa_soma_order[soma_idx]
            if soma_key in self._qa_finalized_somas:
                continue  # already evicted

            self._qa_finalized_somas.add(soma_key)
            img_name, soma_id = soma_key

            # Use soma index for O(masks_per_soma) instead of O(all_masks)
            for idx in self._qa_soma_mask_index.get(soma_key, []):
                mask_data = self.all_masks_flat[idx]['mask_data']
                if mask_data.get('approved') is False:
                    self._delete_rejected_mask_tiff(img_name, mask_data)
                mask_data['mask'] = None

            self.log(f"   💾 Finalized {soma_id} — freed mask arrays from memory")

        # Free processed images for images where all somas have been finalized
        # Only check images that had somas evicted in this call
        for soma_idx in range(max(0, evict_before)):
            soma_key = self._qa_soma_order[soma_idx]
            img_name = soma_key[0]
            if img_name not in self.images:
                continue
            img_data = self.images[img_name]
            if img_data.get('processed') is None:
                continue  # already freed
            # Check using per-image soma count
            img_somas = self._qa_image_soma_count.get(img_name, 0)
            img_finalized = sum(1 for k in self._qa_finalized_somas if k[0] == img_name)
            if img_finalized >= img_somas:
                img_data['processed'] = None
                img_data.pop('color_image', None)
                self.log(f"   💾 Freed processed image for {img_name}")

        # Trim undo history: remove decisions for finalized somas
        self.last_qa_decisions = [
            d for d in self.last_qa_decisions
            if (d['flat_data']['image_name'], d['flat_data']['mask_data']['soma_id'])
            not in self._qa_finalized_somas
        ]

    def _delete_rejected_mask_tiff(self, img_name, mask_data):
        """Delete the TIFF file for a rejected mask from disk."""
        if not self.masks_dir or not os.path.isdir(self.masks_dir):
            return
        img_basename = os.path.splitext(img_name)[0]
        soma_id = mask_data['soma_id']
        area_um2 = mask_data.get('area_um2', 0)
        mask_filename = f"{img_basename}_{soma_id}_area{int(area_um2)}_mask.tif"
        mask_path = os.path.join(self.masks_dir, mask_filename)
        if os.path.exists(mask_path):
            try:
                os.remove(mask_path)
                self.log(f"   🗑️ Deleted rejected mask: {mask_filename}")
            except Exception as e:
                self.log(f"   ⚠️ Could not delete {mask_filename}: {e}")

    def _reload_mask_from_disk(self, mask_data, img_name):
        """Reload a mask array from its TIFF file on disk.

        Returns True if successfully loaded, False otherwise.
        """
        if mask_data.get('mask') is not None:
            return True  # already in memory

        if not self.masks_dir or not os.path.isdir(self.masks_dir):
            return False

        img_basename = os.path.splitext(img_name)[0]
        soma_id = mask_data['soma_id']
        area_um2 = mask_data.get('area_um2', 0)
        mask_filename = f"{img_basename}_{soma_id}_area{int(area_um2)}_mask.tif"
        mask_path = os.path.join(self.masks_dir, mask_filename)

        if os.path.exists(mask_path):
            try:
                mask_arr = safe_tiff_read(mask_path)
                # Convert back from 0/255 to 0/1
                mask_data['mask'] = (mask_arr > 0).astype(np.uint8)
                return True
            except Exception as e:
                self.log(f"   ⚠️ Could not reload mask {mask_filename}: {e}")
        return False

    def _ensure_processed_loaded(self, img_name):
        """Ensure the processed image for img_name is loaded in memory.

        Reloads from disk if it was freed to save RAM. Returns the processed array
        or None if it cannot be loaded.
        """
        img_data = self.images.get(img_name)
        if img_data is None:
            return None
        if img_data.get('processed') is not None:
            return img_data['processed']

        # Try to reload from disk — check stored path first, then output_dir
        name_stem = os.path.splitext(img_name)[0]
        candidates = []
        stored_path = img_data.get('processed_path')
        if stored_path:
            candidates.append(stored_path)
        if self.output_dir:
            candidates.append(os.path.join(self.output_dir, f"{name_stem}_processed.tif"))

        for path in candidates:
            if path and os.path.exists(path):
                try:
                    img_data['processed'] = safe_tiff_read(path)
                    img_data['processed_path'] = path
                    self.log(f"   Reloaded processed image for {img_name}")
                    return img_data['processed']
                except Exception as e:
                    self.log(f"   ⚠️ Could not reload processed image {name_stem} from {path}: {e}")

        # Last resort: load from raw if it's a single-channel image
        raw_path = img_data.get('raw_path')
        if raw_path and os.path.exists(raw_path):
            try:
                raw_img = load_tiff_image(raw_path)
                if raw_img is not None and raw_img.ndim == 2:
                    img_data['processed'] = raw_img.copy()
                    self.log(f"   Loaded processed image from raw for {img_name}")
                    return img_data['processed']
            except Exception:
                pass

        self.log(f"   ⚠️ No processed image found on disk for {img_name}")
        return None

    def _show_current_mask(self):
        if not self.all_masks_flat or self.mask_qa_idx >= len(self.all_masks_flat):
            return

        # Skip auto-rejected duplicates — find the next non-duplicate mask
        while self.mask_qa_idx < len(self.all_masks_flat):
            candidate = self.all_masks_flat[self.mask_qa_idx]['mask_data']
            if candidate.get('duplicate', False) and candidate.get('approved') is False:
                self.mask_qa_idx += 1
            else:
                break
        if self.mask_qa_idx >= len(self.all_masks_flat):
            self._check_qa_complete()
            return

        flat_data = self.all_masks_flat[self.mask_qa_idx]
        mask_data = flat_data['mask_data']
        img_name = flat_data['image_name']
        img_data = self.images.get(img_name, {})

        # Keep current_image_name in sync when QA switches images
        self.current_image_name = img_name

        if self.mode_3d:
            self._show_current_mask_3d(flat_data, mask_data, img_name, img_data)
            return

        try:
            # Reload processed image from disk if it was freed to save RAM
            processed_img = self._ensure_processed_loaded(img_name)
            if processed_img is None:
                self.log(f"Cannot display mask - processed image not found for {img_name}")
                return

            # Reload mask from disk if it was evicted from memory
            if mask_data.get('mask') is None:
                if not self._reload_mask_from_disk(mask_data, img_name):
                    self.log(f"Cannot display mask - file not found on disk")
                    return

            # Ensure processed image is 2D for grayscale display
            if processed_img.ndim > 2:
                processed_img = ensure_grayscale(processed_img)

            # Display in color or grayscale based on toggle
            # Ensure color_image is loaded for color toggle
            if 'color_image' not in img_data and 'raw_path' in img_data:
                try:
                    raw_img = load_tiff_image(img_data['raw_path'])
                    if raw_img is not None and raw_img.ndim == 3:
                        img_data['color_image'] = raw_img.copy()
                        img_data['num_channels'] = raw_img.shape[2]
                except Exception:
                    pass
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
            # Show the clicked soma center as a centroid marker
            soma_centroid = []
            soma_idx = mask_data.get('soma_idx')
            if soma_idx is not None and soma_idx < len(img_data.get('somas', [])):
                soma_centroid = [img_data['somas'][soma_idx]]
            self.mask_label.set_image(pixmap, centroids=soma_centroid, mask_overlay=mask_data['mask'])

            # Auto-zoom to mask center
            mask_coords = np.argwhere(mask_data['mask'] > 0)
            if len(mask_coords) > 0:
                center_row = float(np.mean(mask_coords[:, 0]))
                center_col = float(np.mean(mask_coords[:, 1]))
                self.mask_label.zoom_to_point(center_row, center_col, zoom_level=self.qa_autozoom_spin.value())
        except Exception as e:
            self.log(f"ERROR displaying mask: {str(e)}")
            import traceback
            traceback.print_exc()

        status = mask_data.get('approved')
        status_text = "Approved" if status is True else "Rejected" if status is False else "Not reviewed"
        auto_rejected = self._qa_auto_rejected_count
        reviewed = self._qa_approved_count + self._qa_user_rejected_count
        masks_needing_review = len(self.all_masks_flat) - auto_rejected

        self.nav_status_label.setText(
            f"Mask {self.mask_qa_idx + 1 - auto_rejected}/{masks_needing_review} | "
            f"Reviewed: {reviewed}/{masks_needing_review} | "
            f"{img_name} | {mask_data.get('soma_id', '')} | "
            f"Area: {mask_data.get('area_um2', 0)} um^2 | {status_text}"
        )

        # Update progress bar
        self.mask_qa_progress_bar.setValue(reviewed)

    def _show_current_mask_3d(self, flat_data, mask_data, img_name, img_data):
        """Display current mask in 3D QA mode."""
        mask_3d = mask_data.get('mask')
        if mask_3d is None:
            if not self._reload_3d_mask_from_disk(mask_data, img_name):
                self.log("Cannot display mask - file not found on disk")
                return
            mask_3d = mask_data.get('mask')

        # Switch current image if needed
        if self.current_image_name != img_name:
            self.current_image_name = img_name
            self._update_z_slider_for_image()

        # Find Z-slice with most mask voxels
        if mask_3d is not None and mask_3d.ndim == 3:
            z_sums = mask_3d.sum(axis=(1, 2))
            best_z = int(np.argmax(z_sums))
            self.current_z_slice = best_z
            self.z_slider.setValue(best_z)

        # Get processed stack for background
        stack = img_data.get('processed') or img_data.get('raw_stack')
        if stack is None:
            return

        z = self.current_z_slice
        sl = self._get_slice_for_display(stack, z)
        adjusted = self._apply_display_adjustments(sl)
        pixmap = self._array_to_pixmap(adjusted, skip_rescale=True)

        # Get mask slice
        mask_slice = None
        if mask_3d is not None and mask_3d.ndim == 3:
            z_clamped = max(0, min(mask_3d.shape[0] - 1, z))
            mask_slice = mask_3d[z_clamped]

        # Get soma centroid on this slice
        soma_centroid = []
        soma_idx = mask_data.get('soma_idx')
        if soma_idx is not None and soma_idx < len(img_data.get('somas', [])):
            soma = img_data['somas'][soma_idx]
            if len(soma) == 3:
                sz, sy, sx = soma
                if abs(sz - z) <= 2:
                    soma_centroid = [(sy, sx)]

        self.mask_label.set_image(pixmap, centroids=soma_centroid, mask_overlay=mask_slice)

        # Auto-zoom to mask center on this slice
        if mask_slice is not None:
            mask_coords = np.argwhere(mask_slice > 0)
            if len(mask_coords) > 0:
                center_row = float(np.mean(mask_coords[:, 0]))
                center_col = float(np.mean(mask_coords[:, 1]))
                self.mask_label.zoom_to_point(center_row, center_col, zoom_level=self.qa_autozoom_spin.value())

        status = mask_data.get('approved')
        status_text = "Approved" if status is True else "Rejected" if status is False else "Not reviewed"
        auto_rejected = self._qa_auto_rejected_count
        reviewed = self._qa_approved_count + self._qa_user_rejected_count
        masks_needing_review = len(self.all_masks_flat) - auto_rejected
        vol = mask_data.get('volume_um3', mask_data.get('actual_volume_um3', 0))
        self.nav_status_label.setText(
            f"Mask {self.mask_qa_idx + 1 - auto_rejected}/{masks_needing_review} | "
            f"Reviewed: {reviewed}/{masks_needing_review} | "
            f"{img_name} | {mask_data.get('soma_id', '')} | "
            f"Vol: {vol} um^3 | {status_text}"
        )
        self.mask_qa_progress_bar.setValue(reviewed)

    def approve_current_mask(self):
        if not self.mask_qa_active or self.mask_qa_idx >= len(self.all_masks_flat):
            return

        flat_data = self.all_masks_flat[self.mask_qa_idx]
        mask_data = flat_data['mask_data']
        mask_data['approved'] = True

        current_soma_id = mask_data.get('soma_id', '')
        current_img = flat_data['image_name']

        # Use volume for 3D, area for 2D
        if self.mode_3d:
            current_size = mask_data.get('volume_um3', 0)
            size_key = 'volume_um3'
            size_unit = 'um^3'
        else:
            current_size = mask_data.get('area_um2', 0)
            size_key = 'area_um2'
            size_unit = 'um^2'

        self.log(f"APPROVED | {current_img} | {current_soma_id} | {size_key}: {current_size} {size_unit}")

        # Record decision for undo
        self.last_qa_decisions.append({'flat_data': flat_data, 'was_approved': True})
        self._qa_approved_count += 1

        # Mark in deferred checklist
        if self.mode_3d:
            cl_key = f"{current_img}_{current_soma_id}_vol{current_size}"
        else:
            cl_key = f"{current_img}_{current_soma_id}_area{current_size}"
        self._qa_checklist_dirty[cl_key] = 1

        # Auto-approve ALL smaller masks from the SAME soma in the SAME image
        # Uses soma index for O(masks_per_soma) instead of O(all_masks)
        # Collect masks needing export — actual I/O is deferred below
        pending_exports = []
        if not self.mode_3d:
            pending_exports.append(flat_data)

        auto_approved = []
        soma_key = (current_img, current_soma_id)
        for idx in self._qa_soma_mask_index.get(soma_key, []):
            other_flat = self.all_masks_flat[idx]
            other_mask = other_flat['mask_data']
            other_size = other_mask.get(size_key, 0)

            if other_size < current_size and other_mask.get('approved') is None:
                other_mask['approved'] = True
                auto_approved.append((idx + 1, other_size))
                self.last_qa_decisions.append({'flat_data': other_flat, 'was_approved': True})
                self._qa_approved_count += 1
                if not self.mode_3d:
                    pending_exports.append(other_flat)
                # Mark in deferred checklist
                if self.mode_3d:
                    auto_key = f"{current_img}_{current_soma_id}_vol{other_size}"
                else:
                    auto_key = f"{current_img}_{current_soma_id}_area{other_size}"
                self._qa_checklist_dirty[auto_key] = 1

        if auto_approved:
            self.log(f"   Auto-approved {len(auto_approved)} smaller masks for {current_soma_id}")

        # Show next mask FIRST so UI feels instant
        self._advance_to_next_unreviewed()

        # Defer all disk I/O (TIFF exports, eviction, auto-save) so UI stays responsive
        should_autosave = len(self.last_qa_decisions) % 50 == 0
        QTimer.singleShot(0, lambda: self._deferred_approve_io(
            pending_exports, should_autosave))

    def _approve_all_remaining(self):
        """Approve ALL remaining unreviewed masks at once.

        For large datasets (22k+ images) where manual QA of every mask
        is impractical. Marks all unreviewed masks as approved and
        completes QA immediately.
        """
        if not self.mask_qa_active or not self.all_masks_flat:
            return

        remaining = sum(1 for f in self.all_masks_flat if f['mask_data'].get('approved') is None)
        if remaining == 0:
            QMessageBox.information(self, "Info", "All masks have already been reviewed.")
            return

        reply = QMessageBox.question(
            self, "Approve All Remaining",
            f"This will approve {remaining} unreviewed masks.\n\n"
            "Are you sure? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        self.log(f"Approving all {remaining} remaining masks...")
        approved_count = 0

        for flat_data in self.all_masks_flat:
            mask_data = flat_data['mask_data']
            if mask_data.get('approved') is None:
                mask_data['approved'] = True
                approved_count += 1
                self._qa_approved_count += 1

                # Mark in deferred checklist
                if self.mode_3d:
                    size_val = mask_data.get('volume_um3', 0)
                    cl_key = f"{flat_data['image_name']}_{mask_data.get('soma_id', '')}_vol{size_val}"
                else:
                    size_val = mask_data.get('area_um2', 0)
                    cl_key = f"{flat_data['image_name']}_{mask_data.get('soma_id', '')}_area{size_val}"
                self._qa_checklist_dirty[cl_key] = 1

        self.log(f"Approved {approved_count} masks in bulk")

        # Flush and save
        self._flush_qa_checklist()
        self._auto_save()

        # Mark QA complete
        self._check_qa_complete()

    def _deferred_approve_io(self, pending_exports, should_autosave):
        """Run disk I/O deferred from approve_current_mask via QTimer.

        This keeps the UI responsive — the next mask is already displayed
        before any TIFF writes, eviction, or auto-save happen.
        """
        for flat_data in pending_exports:
            self._export_approved_mask(flat_data)

        self._evict_old_qa_masks()

        if should_autosave:
            self._flush_qa_checklist()
            self._auto_save()

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
        px_x, px_y = self._get_pixel_size_xy(img_name)

        # Save as TIFF with calibration
        try:
            tifffile.imwrite(
                mask_path,
                mask_8bit,
                resolution=(1.0 / px_x, 1.0 / px_y),
                metadata={'unit': 'um'}
            )

            self.log(f"   💾 Exported: {mask_filename} ({pixels_saved} pixels)")

        except Exception as e:
            self.log(f"   ❌ Failed to export {mask_filename}: {e}")

    def _export_all_masks_to_disk(self, img_name, masks):
        """Export all masks for an image to disk (for session persistence).

        Called after mask generation so that ALL masks exist on disk before QA.
        This ensures masks survive a save/load cycle even if QA is incomplete.
        Uses ThreadPoolExecutor for parallel I/O to speed up writing many masks.
        """
        if not self.masks_dir:
            return

        img_basename = os.path.splitext(img_name)[0]

        pixel_size = self._get_pixel_size(img_name)

        # Prepare all write tasks (skip auto-rejected duplicates)
        write_tasks = []
        for mask_data in masks:
            mask = mask_data.get('mask')
            if mask is None or not np.any(mask):
                continue

            soma_id = mask_data['soma_id']
            area_um2 = mask_data.get('area_um2', 0)
            mask_filename = f"{img_basename}_{soma_id}_area{int(area_um2)}_mask.tif"
            mask_path = os.path.join(self.masks_dir, mask_filename)

            # Don't write auto-rejected duplicates to disk — delete if stale
            if mask_data.get('duplicate', False):
                if os.path.exists(mask_path):
                    try:
                        os.remove(mask_path)
                    except Exception:
                        pass
                continue

            # Always write the current in-memory mask to disk — stale masks
            # from a previous run with different settings must be overwritten.

            if mask.dtype == bool:
                mask_8bit = mask.astype(np.uint8) * 255
            else:
                mask_8bit = (mask > 0).astype(np.uint8) * 255

            if np.count_nonzero(mask_8bit) == 0:
                continue

            write_tasks.append((mask_path, mask_8bit, pixel_size))

        if not write_tasks:
            return

        # Write in parallel using threads (I/O bound)
        n_writers = min(len(write_tasks), 8)
        exported = 0
        if n_writers > 1:
            with ThreadPoolExecutor(max_workers=n_writers) as executor:
                results = executor.map(_export_mask_to_disk, write_tasks)
                exported = sum(1 for r in results if r)
        else:
            for task in write_tasks:
                if _export_mask_to_disk(task):
                    exported += 1

        if exported > 0:
            self.log(f"   💾 Saved {exported} masks to disk for {img_name}")

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

        # If we get here, check if all masks are reviewed (O(1) via counters)
        total_decided = self._qa_auto_rejected_count + self._qa_approved_count + self._qa_user_rejected_count
        all_reviewed = total_decided >= len(self.all_masks_flat)

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

        # Guard against double-counting: if this mask was already auto-rejected
        # as a duplicate, don't count it again as a user rejection
        was_already_rejected = mask_data.get('duplicate', False) and mask_data.get('approved') is False
        mask_data['approved'] = False

        # Record decision for undo
        self.last_qa_decisions.append({'flat_data': flat_data, 'was_approved': False,
                                       'was_already_rejected': was_already_rejected})
        if not was_already_rejected:
            self._qa_user_rejected_count += 1

        if not self.mode_3d:
            self.log(f"Rejected: {mask_data.get('soma_id', '')} ({mask_data.get('area_um2', 0)} um^2)")
        else:
            vol = mask_data.get('volume_um3', 0)
            self.log(f"Rejected: {mask_data.get('soma_id', '')} ({vol} um^3)")

        # Mark in deferred checklist
        if self.mode_3d:
            key = f"{flat_data['image_name']}_{mask_data.get('soma_id', '')}_vol{mask_data.get('volume_um3', 0)}"
        else:
            key = f"{flat_data['image_name']}_{mask_data.get('soma_id', '')}_area{mask_data.get('area_um2', 0)}"
        self._qa_checklist_dirty[key] = 1

        # Show next mask FIRST so UI feels instant
        self._advance_to_next_unreviewed()

        # Defer disk I/O (3D mask deletion, eviction) so UI stays responsive
        delete_3d = self.mode_3d
        reject_img = flat_data['image_name']
        reject_mask_data = mask_data
        QTimer.singleShot(0, lambda: self._deferred_reject_io(
            delete_3d, reject_img, reject_mask_data))

    def _deferred_reject_io(self, delete_3d, img_name, mask_data):
        """Run disk I/O deferred from reject_current_mask via QTimer."""
        if delete_3d:
            self._delete_rejected_3d_mask(img_name, mask_data)
        else:
            # Delete rejected 2D mask TIFF from disk immediately
            self._delete_rejected_mask_tiff(img_name, mask_data)
        self._evict_old_qa_masks()

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
            # Skip backwards past auto-rejected duplicates
            target = self.mask_qa_idx - 1
            while target > 0:
                candidate = self.all_masks_flat[target]['mask_data']
                if candidate.get('duplicate', False) and candidate.get('approved') is False:
                    target -= 1
                else:
                    break
            # Check if the target mask's soma has been finalized
            prev_flat = self.all_masks_flat[target]
            prev_key = (prev_flat['image_name'], prev_flat['mask_data']['soma_id'])
            if prev_key in self._qa_finalized_somas:
                self.log("Cannot go back further - those masks have been finalized to save memory")
                return
            self.mask_qa_idx = target
            self._show_current_mask()

    def _delete_rejected_3d_mask(self, img_name, mask_data):
        """Delete a rejected 3D mask TIFF from disk."""
        if not self.masks_dir or not os.path.isdir(self.masks_dir):
            return
        img_basename = os.path.splitext(img_name)[0]
        soma_id = mask_data.get('soma_id', '')
        vol = mask_data.get('volume_um3', 0)
        fname = f"{img_basename}_{soma_id}_vol{int(vol)}_mask3d.tif"
        path = os.path.join(self.masks_dir, fname)
        if os.path.exists(path):
            try:
                os.remove(path)
                self.log(f"   Deleted rejected: {fname}")
            except Exception as e:
                self.log(f"   Could not delete {fname}: {e}")

    def _check_qa_complete(self):
        # Flush any pending checklist updates before finishing
        self._flush_qa_checklist()

        total_decided = self._qa_auto_rejected_count + self._qa_approved_count + self._qa_user_rejected_count
        all_reviewed = total_decided >= len(self.all_masks_flat)

        if all_reviewed:
            self.mask_qa_active = False
            self.approve_mask_btn.setEnabled(False)
            self.reject_mask_btn.setEnabled(False)
            self.approve_all_btn.setEnabled(False)
            self.approve_all_btn.setVisible(False)
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.regen_masks_btn.setVisible(False)
            self.clear_masks_btn.setVisible(False)
            self.clear_masks_btn.setEnabled(False)
            self.undo_qa_btn.setVisible(False)
            self.mask_qa_progress_bar.setVisible(False)

            # Update image statuses
            for img_name, img_data in self.images.items():
                if img_data['selected'] and img_data['status'] == 'masks_generated':
                    img_data['status'] = 'qa_complete'
                    self._update_file_list_item(img_name)

            self.batch_calculate_btn.setEnabled(True)
            self.undo_qa_btn.setEnabled(True)
            self.undo_qa_btn.setVisible(True)

            approved_count = sum(1 for flat in self.all_masks_flat if flat['mask_data']['approved'])
            rejected_count = len(self.all_masks_flat) - approved_count

            # Count somas that have at least one approved mask
            somas_with_masks = set()
            total_somas = set()
            for flat in self.all_masks_flat:
                key = (flat['image_name'], flat['mask_data']['soma_id'])
                total_somas.add(key)
                if flat['mask_data']['approved']:
                    somas_with_masks.add(key)
            cells_used = len(somas_with_masks)
            cells_total = len(total_somas)

            # Delete mask QA checklist — QA is done
            self._delete_checklist(self._get_checklist_path('mask_qa_checklist.csv'))

            self.log("=" * 50)
            self.log(f"✓ QA Complete!")
            self.log(f"Approved: {approved_count}, Rejected: {rejected_count}")
            self.log(f"Cells used: {cells_used}/{cells_total}")
            self.log("=" * 50)
            self._auto_save()

            QMessageBox.information(
                self, "QA Complete",
                f"QA Complete!\n\n"
                f"Approved masks: {approved_count}\n"
                f"Rejected masks: {rejected_count}\n\n"
                f"Cells used: {cells_used} / {cells_total}\n"
                f"({cells_total - cells_used} cells had all masks rejected)"
            )

    def undo_last_qa(self):
        """Undo the single most recent QA decision and jump back to that mask."""
        if not hasattr(self, 'last_qa_decisions') or not self.last_qa_decisions:
            QMessageBox.warning(self, "Nothing to Undo", "No recent QA decisions to undo.")
            return

        # Pop the last decision
        decision = self.last_qa_decisions.pop()
        flat_data = decision['flat_data']
        mask_data = flat_data['mask_data']
        img_name = flat_data['image_name']

        # Mask files are kept on disk (written during generation) —
        # only the in-memory approval state is reset.

        # Reset approval state and decrement running counters
        was = "approved" if decision['was_approved'] else "rejected"
        if decision['was_approved']:
            self._qa_approved_count = max(0, self._qa_approved_count - 1)
        else:
            # Don't decrement if this was an already-auto-rejected duplicate
            if not decision.get('was_already_rejected', False):
                self._qa_user_rejected_count = max(0, self._qa_user_rejected_count - 1)
        mask_data['approved'] = None

        # Revert image status if needed
        if img_name in self.images:
            img_data = self.images[img_name]
            if img_data['status'] in ('qa_complete', 'analyzed'):
                img_data['status'] = 'masks_generated'
                self._update_file_list_item(img_name)

        self.log(f"↩ Undid {was}: {mask_data['soma_id']} ({mask_data['area_um2']} µm²)")

        # Jump QA back to that mask
        if hasattr(self, 'all_masks_flat') and self.all_masks_flat:
            for i, flat in enumerate(self.all_masks_flat):
                if flat is flat_data:
                    self.mask_qa_idx = i
                    break

            self.mask_qa_active = True
            self.approve_mask_btn.setEnabled(True)
            self.reject_mask_btn.setEnabled(True)
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
            self.undo_qa_btn.setEnabled(len(self.last_qa_decisions) > 0)
            self.undo_qa_btn.setVisible(True)
            self.regen_masks_btn.setVisible(True)
            self.clear_masks_btn.setEnabled(True)
            self.clear_masks_btn.setVisible(True)
            self._show_current_mask()
            self.tabs.setCurrentIndex(3)

    def batch_calculate_morphology(self):
        if self.mode_3d:
            self._batch_calculate_morphology_3d()
            return
        try:
            pixel_size = self._get_pixel_size()

            # No ImageJ required for simple characteristics
            # Complex analysis (Sholl, Skeleton) will be done separately in ImageJ

            # Count mask states for diagnostics
            n_approved = sum(1 for f in self.all_masks_flat if f['mask_data'].get('approved') is True)
            n_rejected = sum(1 for f in self.all_masks_flat if f['mask_data'].get('approved') is False)
            n_unreviewed = sum(1 for f in self.all_masks_flat if f['mask_data'].get('approved') is None)
            n_duplicates = sum(1 for f in self.all_masks_flat if f['mask_data'].get('duplicate'))
            self.log(f"Mask inventory: {len(self.all_masks_flat)} total | "
                     f"{n_approved} approved | {n_rejected} rejected | "
                     f"{n_unreviewed} unreviewed | {n_duplicates} duplicates")

            # Filter to only approved, non-duplicate masks
            approved_masks = [flat for flat in self.all_masks_flat
                              if flat['mask_data'].get('approved') is True
                              and not flat['mask_data'].get('duplicate', False)]
            total = len(approved_masks)

            self.log(f"Running morphology on {total} approved (non-duplicate) masks")

            # Only reload processed images on main thread (fast, one per image)
            # Mask reloads are deferred to the worker thread to avoid freezing the UI
            reloaded_images = set()
            processed_reload_count = 0
            for flat in approved_masks:
                if flat['image_name'] not in reloaded_images:
                    reloaded_images.add(flat['image_name'])
                    if self._ensure_processed_loaded(flat['image_name']) is not None:
                        if self.images[flat['image_name']].get('processed') is not None:
                            processed_reload_count += 1
            if processed_reload_count > 0:
                self.log(f"   Reloaded {processed_reload_count} processed images from disk for analysis")

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

            # Build per-image pixel size map (stores (x, y) tuples)
            pixel_size_map = {}
            for img_name, img_data in self.images.items():
                pixel_size_map[img_name] = self._get_pixel_size_xy(img_name)

            # Build soma_group lookup: (img_basename, soma_id) -> group string
            soma_group_map = {}
            for img_name, img_data in self.images.items():
                basename = os.path.splitext(img_name)[0]
                groups = img_data.get('soma_groups', [])
                for i, sid in enumerate(img_data.get('soma_ids', [])):
                    group = groups[i] if i < len(groups) else ''
                    soma_group_map[(basename, sid)] = group

            # Create and start worker thread (no ImageJ needed)
            self.morph_thread = MorphologyCalculationThread(
                approved_masks, pixel_size, use_imagej=False, images=self.images,
                output_dir=self.output_dir, pixel_size_map=pixel_size_map,
                masks_dir=self.masks_dir, soma_group_map=soma_group_map
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
            # Build O(1) lookup: (image_base, soma_id) -> flat_data
            mask_lookup = {}
            for flat_data in self.all_masks_flat:
                if flat_data['mask_data'].get('approved'):
                    key = (os.path.splitext(flat_data['image_name'])[0],
                           flat_data['mask_data'].get('soma_id'))
                    mask_lookup[key] = flat_data

            for i, result in enumerate(all_results):
                img_name = result.get('image_name', '')
                key = (img_name, result.get('soma_id'))
                flat_data = mask_lookup.get(key)
                if flat_data:
                    # Reload from disk if evicted
                    if flat_data['mask_data'].get('mask') is None:
                        self._reload_mask_from_disk(flat_data['mask_data'], flat_data['image_name'])
                    mask = flat_data['mask_data'].get('mask')
                    full_img_name = flat_data['image_name']
                    if mask is not None:
                        coloc_results = self.calculate_colocalization(mask, full_img_name)
                        result.update(coloc_results)
                        if coloc_results.get('coloc_status') == 'ok':
                            coloc_count += 1
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

        # Build O(1) lookup: image_basename -> img_data
        img_by_base = {}
        for full_name, img_data in self.images.items():
            img_by_base[os.path.splitext(full_name)[0]] = img_data

        # Add animal_id and treatment to each result
        for result in results:
            img_name_base = result['image_name']
            img_data = img_by_base.get(img_name_base)
            if img_data:
                result['animal_id'] = img_data.get('animal_id', '')
                result['treatment'] = img_data.get('treatment', '')
            else:
                result['animal_id'] = ''
                result['treatment'] = ''

        # Combined results file
        combined_path = os.path.join(self.output_dir, "combined_morphology_results.csv")

        # Collect all keys across all results (some may have coloc fields, others not)
        all_keys_set = set()
        for r in results:
            all_keys_set.update(r.keys())
        keys = list(all_keys_set)
        # Remove these to reorder them
        for key in ['soma_id', 'image_name', 'animal_id', 'treatment', 'soma_idx', 'soma_group']:
            if key in keys:
                keys.remove(key)

        # Separate colocalization keys from morphology keys for better organization
        coloc_keys = ['coloc_status', 'coloc_ch1', 'coloc_ch2',
                      'n_mask_pixels', 'n_ch1_signal', 'n_ch2_signal',
                      'ch1_mean_intensity', 'ch2_mean_intensity',
                      'n_coloc_pixels', 'ch1_coloc_percent', 'ch2_coloc_percent',
                      'pearson_r']
        morph_keys = [k for k in keys if k not in coloc_keys]
        coloc_present = [k for k in coloc_keys if k in keys]

        # Put them in the desired order: identifiers, morphology, colocalization
        # Only include soma_group column if any result has a non-empty group
        has_groups = any(r.get('soma_group', '') for r in results)
        id_keys = ['image_name', 'animal_id', 'treatment']
        if has_groups:
            id_keys.append('soma_group')
        id_keys.extend(['soma_id', 'soma_idx'])
        ordered_keys = id_keys + sorted(morph_keys) + coloc_present

        with open(combined_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)

        self.log(f"Combined results saved to: {combined_path}")

        # Export metadata CSV for ImageJ analysis matching
        metadata_path = os.path.join(self.masks_dir, "mask_metadata.csv")
        with open(metadata_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['mask_filename', 'soma_filename', 'image_name', 'soma_id', 'soma_idx',
                             'soma_x', 'soma_y', 'soma_area_um2', 'cell_area_um2',
                             'perimeter', 'eccentricity', 'roundness',
                             'avg_centroid_distance', 'polarity_index', 'principal_angle',
                             'major_axis_um', 'minor_axis_um', 'animal_id', 'treatment'])

            for result in results:
                img_name = result['image_name']
                soma_id = result['soma_id']
                soma_idx = result.get('soma_idx', 0)

                # Get soma position using O(1) lookup
                soma_x, soma_y = 0, 0
                img_data = img_by_base.get(img_name)
                if img_data and soma_idx < len(img_data['somas']):
                    soma_x, soma_y = img_data['somas'][soma_idx]

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
                    result.get('perimeter', 0),
                    result.get('eccentricity', 0),
                    result.get('roundness', 0),
                    result.get('avg_centroid_distance', 0),
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
                writer = csv.DictWriter(f, fieldnames=ordered_keys, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(img_results)

        self.log(f"Per-image results saved for {len(by_image)} images")

    # ----------------------------------------------------------------
    # 3D MORPHOLOGY CALCULATION
    # ----------------------------------------------------------------

    def _batch_calculate_morphology_3d(self):
        """Calculate 3D morphology metrics for approved masks."""
        try:
            voxel_xy = float(self.pixel_size_input.text())
            voxel_z = float(self.voxel_z_input.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Invalid voxel dimensions")
            return

        # Reload evicted masks
        reload_count = 0
        for flat in self.all_masks_flat:
            if flat['mask_data'].get('approved') and flat['mask_data'].get('mask') is None:
                if self._reload_3d_mask_from_disk(flat['mask_data'], flat['image_name']):
                    reload_count += 1
        if reload_count > 0:
            self.log(f"   Reloaded {reload_count} evicted masks from disk")

        approved = [f for f in self.all_masks_flat if f['mask_data'].get('approved')]
        if not approved:
            QMessageBox.warning(self, "Warning", "No approved masks to analyze")
            return

        self.log("=" * 50)
        self.log(f"Calculating 3D morphology for {len(approved)} masks...")
        self.log("=" * 50)

        self.progress_bar.setVisible(True)
        self.progress_status_label.setVisible(True)
        self.progress_bar.setValue(0)
        self.start_timer()
        self.batch_calculate_btn.setEnabled(False)

        # Use a worker thread
        from PyQt5.QtCore import QThread, pyqtSignal

        class _MorphThread3D(QThread):
            progress = pyqtSignal(int, str)
            finished = pyqtSignal(list)
            error_occurred = pyqtSignal(str)

            def __init__(self, approved_masks, vxy, vz):
                super().__init__()
                self.approved_masks = approved_masks
                self.vxy = vxy
                self.vz = vz

            def run(self):
                try:
                    calc = Morphology3DCalculator(self.vxy, self.vz)
                    results = []
                    total = len(self.approved_masks)
                    for i, flat in enumerate(self.approved_masks):
                        mask_data = flat['mask_data']
                        img_name = flat['image_name']
                        mask_3d = mask_data.get('mask')
                        if mask_3d is None:
                            continue
                        self.progress.emit(int((i + 1) / total * 100),
                                           f"Analyzing {i + 1}/{total}: {mask_data.get('soma_id', '')}")
                        try:
                            metrics = calc.calculate_all(mask_3d)
                        except Exception as e:
                            metrics = calc._empty_metrics()
                            metrics['error'] = str(e)
                        # Skeleton analysis
                        try:
                            skeleton, bp, ep, n_branches = skeletonize_3d_mask(mask_3d)
                            metrics['n_branches'] = n_branches
                            metrics['n_branch_points'] = len(bp)
                            metrics['n_endpoints'] = len(ep)
                            metrics['total_branch_length_um'] = round(
                                np.sum(skeleton) * ((self.vxy + self.vz) / 2), 4)
                        except Exception:
                            metrics['n_branches'] = 0
                            metrics['n_branch_points'] = 0
                            metrics['n_endpoints'] = 0
                            metrics['total_branch_length_um'] = 0
                        # Fractal analysis
                        try:
                            fd, lac, _, _ = fractal_dimension_3d(mask_3d, self.vxy, self.vz)
                            metrics['fractal_dimension_3d'] = fd
                            metrics['lacunarity_3d'] = lac
                        except Exception:
                            metrics['fractal_dimension_3d'] = 0
                            metrics['lacunarity_3d'] = 0
                        row = {
                            'image_name': os.path.splitext(img_name)[0],
                            'soma_id': mask_data.get('soma_id', ''),
                            'soma_idx': mask_data.get('soma_idx', 0),
                            'target_volume_um3': mask_data.get('volume_um3', 0),
                        }
                        row.update(metrics)
                        results.append(row)
                    self.finished.emit(results)
                except Exception as e:
                    self.error_occurred.emit(str(e))

        self._morph_thread_3d = _MorphThread3D(approved, voxel_xy, voxel_z)
        self._morph_thread_3d.progress.connect(self._on_morph_3d_progress)
        self._morph_thread_3d.finished.connect(self._on_morph_3d_finished)
        self._morph_thread_3d.error_occurred.connect(self._on_morph_3d_error)
        self._morph_thread_3d.start()

    def _on_morph_3d_progress(self, percentage, status):
        self.progress_bar.setValue(percentage)
        self.progress_status_label.setText(status)

    def _on_morph_3d_finished(self, all_results):
        self.stop_timer()
        self.progress_bar.setVisible(False)
        self.progress_status_label.setVisible(False)
        self.batch_calculate_btn.setEnabled(True)
        self.timer_label.setVisible(False)

        self._save_3d_results(all_results)

        for img_name, img_data in self.images.items():
            if img_data['selected'] and img_data['status'] == 'qa_complete':
                img_data['status'] = 'analyzed'
                self._update_file_list_item(img_name)

        self.log("=" * 50)
        self.log(f"3D morphology calculated for {len(all_results)} cells")
        self.log(f"Results saved to: {self.output_dir}")
        self.log("=" * 50)
        self._auto_save()
        QMessageBox.information(self, "Success",
                                f"3D morphology calculated for {len(all_results)} cells!\n\n"
                                f"Results saved to:\n{self.output_dir}")

    def _on_morph_3d_error(self, error_msg):
        self.stop_timer()
        self.progress_bar.setVisible(False)
        self.progress_status_label.setVisible(False)
        self.timer_label.setVisible(False)
        self.batch_calculate_btn.setEnabled(True)
        self.log(f"ERROR: {error_msg}")
        QMessageBox.critical(self, "Error", f"Morphology calculation failed:\n{error_msg}")

    def _save_3d_results(self, results):
        """Save 3D morphology results to CSV."""
        if not self.output_dir or not results:
            return

        for result in results:
            img_name_base = result['image_name']
            for full_name, img_data in self.images.items():
                if os.path.splitext(full_name)[0] == img_name_base:
                    result['animal_id'] = img_data.get('animal_id', '')
                    result['treatment'] = img_data.get('treatment', '')
                    break
            else:
                result['animal_id'] = ''
                result['treatment'] = ''

        combined_path = os.path.join(self.output_dir, "3d_morphology_results.csv")
        fieldnames = [
            'image_name', 'animal_id', 'treatment', 'soma_id', 'soma_idx',
            'target_volume_um3', 'volume_um3', 'surface_area_um2', 'sphericity',
            'elongation', 'cell_spread_3d_um', 'soma_volume_um3',
            'polarity_index_3d', 'principal_azimuth', 'principal_elevation',
            'major_axis_um', 'mid_axis_um', 'minor_axis_um',
            'n_branches', 'n_branch_points', 'n_endpoints',
            'total_branch_length_um', 'fractal_dimension_3d', 'lacunarity_3d',
        ]
        with open(combined_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)
        self.log(f"Results saved to: {combined_path}")

        # Per-image files
        by_image = {}
        for result in results:
            img = result['image_name']
            if img not in by_image:
                by_image[img] = []
            by_image[img].append(result)

        for img_name, img_results in by_image.items():
            path = os.path.join(self.output_dir, f"{img_name}_3d_morphology.csv")
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(img_results)
        self.log(f"Per-image results saved for {len(by_image)} Z-stacks")


def _generate_microglia_icon():
    """Generate a microglia-cell-themed icon programmatically."""
    size = 256
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(0, 0, 0, 0))  # transparent background

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    cx, cy = size // 2, size // 2

    # Draw branching processes (arms) radiating from center
    branch_pen = QPen(QColor(100, 180, 255), 5)
    branch_pen.setCapStyle(Qt.RoundCap)
    painter.setPen(branch_pen)
    import math as _m
    branches = [
        (0, 90), (45, 80), (90, 85), (135, 75),
        (180, 88), (225, 82), (270, 78), (315, 86),
        (22, 60), (67, 55), (112, 65), (157, 58),
        (202, 62), (247, 57), (292, 63), (337, 60),
    ]
    for angle_deg, length in branches:
        angle = _m.radians(angle_deg)
        x1 = cx + int(30 * _m.cos(angle))
        y1 = cy + int(30 * _m.sin(angle))
        x2 = cx + int(length * _m.cos(angle))
        y2 = cy + int(length * _m.sin(angle))
        painter.drawLine(x1, y1, x2, y2)
        # Small fork at tips
        for fork in (-25, 25):
            fa = angle + _m.radians(fork)
            fx = x2 + int(20 * _m.cos(fa))
            fy = y2 + int(20 * _m.sin(fa))
            painter.drawLine(x2, y2, fx, fy)

    # Draw soma (cell body) — filled circle in center
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(QColor(60, 140, 220)))
    painter.drawEllipse(cx - 28, cy - 28, 56, 56)
    # Nucleus highlight
    painter.setBrush(QBrush(QColor(180, 220, 255)))
    painter.drawEllipse(cx - 10, cy - 12, 18, 18)

    painter.end()
    return QIcon(pixmap)


def _get_app_icon():
    """Load MMPS app icon from common locations, or generate a default."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Check for icon files in order of preference
    icon_names = ['MMPS.icns', 'MMPS.ico', 'MMPS.png', 'MMPS.icn']
    search_dirs = [
        script_dir,
        os.path.join(script_dir, 'resources'),
        # macOS .app bundle: Contents/Resources
        os.path.join(script_dir, '..', 'Resources'),
    ]
    for d in search_dirs:
        for name in icon_names:
            path = os.path.join(d, name)
            if os.path.isfile(path):
                icon = QIcon(path)
                if not icon.isNull():
                    return icon
    # No file found — generate a microglia icon
    return _generate_microglia_icon()


def main():
    # Required for PyInstaller on macOS — prevents duplicate processes
    multiprocessing.freeze_support()

    # Tell macOS this is a regular GUI app (prevents duplicate dock icons)
    if sys.platform == 'darwin':
        try:
            from Foundation import NSBundle
            bundle = NSBundle.mainBundle()
            info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
            if info:
                info['LSUIElement'] = False
        except ImportError:
            pass

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Set application icon
    icon = _get_app_icon()
    if icon:
        app.setWindowIcon(icon)

    window = MicrogliaAnalysisGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
