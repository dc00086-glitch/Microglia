#!/usr/bin/env python3
"""
Scan a masks folder, compute all morphology metrics from the TIFF files,
and write a fresh CSV.  No existing CSV needed.

Metrics computed (matching MMPSv2.py):
    perimeter, mask_area, eccentricity, roundness, cell_spread,
    soma_area (from somas/ folder if available),
    polarity_index, principal_angle, major_axis_um, minor_axis_um

Usage:
    python compute_morphology_from_masks.py

    Then select:
      1. The masks folder containing *_mask.tif files
      2. (Optional) The somas folder containing *_soma.tif files
      3. Enter pixel size in µm/px
"""

import sys
import os
import csv
import re
import numpy as np
import tifffile
from skimage import measure

# ── filename pattern: {image_name}_{soma_id}_area{N}_mask.tif ──────────
MASK_RE = re.compile(r'^(.+?)_(soma_\d+_\d+)_area(\d+)_mask\.tif$')


def parse_mask_filename(filename):
    """Extract image_name, soma_id, area from a mask filename."""
    m = MASK_RE.match(filename)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return None, None, None


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
    params = {}

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


def get_soma_area(somas_dir, image_name, soma_id, pixel_size):
    """Try to find the soma outline TIFF and compute its area."""
    if not somas_dir:
        return None

    # Try common naming patterns
    candidates = [
        f"{image_name}_{soma_id}_soma.tif",
        f"{image_name}_{soma_id}.tif",
    ]
    for c in candidates:
        path = os.path.join(somas_dir, c)
        if os.path.exists(path):
            soma = tifffile.imread(path)
            soma = (soma > 0).astype(np.uint8)
            if np.any(soma):
                return np.sum(soma) * (pixel_size ** 2)
            return None

    # Fallback: search for any file with soma_id
    for f in os.listdir(somas_dir):
        if soma_id in f and f.endswith(".tif"):
            path = os.path.join(somas_dir, f)
            soma = tifffile.imread(path)
            soma = (soma > 0).astype(np.uint8)
            if np.any(soma):
                return np.sum(soma) * (pixel_size ** 2)
            return None

    return None


def main():
    # ── Pick folders ───────────────────────────────────────────────────
    try:
        from PyQt5.QtWidgets import QApplication, QFileDialog, QInputDialog
        app = QApplication(sys.argv)

        masks_dir = QFileDialog.getExistingDirectory(
            None, "Select masks folder (contains *_mask.tif files)",
            options=QFileDialog.DontUseNativeDialog
        )
        if not masks_dir:
            print("No masks folder selected – exiting.")
            return

        somas_dir = QFileDialog.getExistingDirectory(
            None, "Select somas folder (optional – Cancel to skip)",
            options=QFileDialog.DontUseNativeDialog
        )
        if not somas_dir:
            somas_dir = None

        pixel_size, ok = QInputDialog.getDouble(
            None, "Pixel Size", "Enter pixel size (µm/px):",
            0.3, 0.01, 10.0, 4
        )
        if not ok:
            print("Cancelled – exiting.")
            return

    except ImportError:
        if len(sys.argv) < 3:
            print("Usage: python compute_morphology_from_masks.py <masks_dir> <pixel_size> [somas_dir]")
            return
        masks_dir = sys.argv[1]
        pixel_size = float(sys.argv[2])
        somas_dir = sys.argv[3] if len(sys.argv) > 3 else None

    # ── Scan mask files ────────────────────────────────────────────────
    all_files = sorted(os.listdir(masks_dir))
    mask_files = [f for f in all_files if MASK_RE.match(f)]

    print(f"\nFound {len(all_files)} files in masks folder.")
    print(f"Matched {len(mask_files)} mask files (pattern: *_soma_X_Y_areaN_mask.tif)")
    if somas_dir:
        soma_count = len([f for f in os.listdir(somas_dir) if f.endswith(".tif")])
        print(f"Found {soma_count} soma files in somas folder.")
    print(f"Pixel size: {pixel_size} µm/px\n")

    if not mask_files:
        print("No mask files matched the pattern. Example files in folder:")
        for f in all_files[:10]:
            print(f"  {f}")
        return

    # ── Compute metrics for each mask ──────────────────────────────────
    results = []
    errors = 0

    for i, filename in enumerate(mask_files):
        image_name, soma_id, area = parse_mask_filename(filename)
        mask_path = os.path.join(masks_dir, filename)

        # Look up soma area from somas folder
        soma_area = get_soma_area(somas_dir, image_name, soma_id, pixel_size)

        try:
            metrics = compute_metrics(mask_path, pixel_size, soma_area)
        except Exception as e:
            print(f"  ERROR: {filename}: {e}")
            errors += 1
            continue

        if metrics is None:
            print(f"  SKIP (empty mask): {filename}")
            continue

        treatment = image_name.split('_')[0] if image_name else ''

        row = {
            'treatment': treatment,
            'image_name': image_name,
            'soma_id': soma_id,
            'area_um2': area,
        }
        row.update(metrics)
        results.append(row)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(mask_files)}...")

    # ── Write CSV ──────────────────────────────────────────────────────
    if not results:
        print("No results to write.")
        return

    output_path = os.path.join(os.path.dirname(masks_dir), "combined_morphology_results.csv")

    fieldnames = [
        'treatment', 'image_name', 'soma_id', 'area_um2',
        'mask_area', 'perimeter', 'roundness', 'eccentricity',
        'cell_spread', 'soma_area',
        'polarity_index', 'principal_angle',
        'major_axis_um', 'minor_axis_um',
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone!")
    print(f"  Masks processed: {len(results)}")
    print(f"  Errors:          {errors}")
    print(f"  Output CSV:      {output_path}")


if __name__ == "__main__":
    main()
