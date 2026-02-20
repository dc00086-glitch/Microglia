#!/usr/bin/env python3
"""
One-time script to recompute roundness and eccentricity for an existing
combined_morphology.csv using the mask TIFF files on disk.

Updated formulas (matching MMPSv2.py):
    axis_ratio   = minor_axis / major_axis
    eccentricity = sqrt(1 - axis_ratio^2)   (0 = circle, 1 = elongated)
    roundness    = (minor_axis / major_axis)^2  (0 = elongated, 1 = circle)

Usage:
    python recompute_roundness_eccentricity.py

    Then select:
      1. The combined_morphology CSV file
      2. The masks folder containing *_mask.tif files
"""

import sys
import os
import csv
import re
import numpy as np
import tifffile
from skimage import measure


def recompute_for_mask(mask_path):
    """Load a mask TIFF and return updated roundness + eccentricity."""
    mask = tifffile.imread(mask_path)
    mask = (mask > 0).astype(np.uint8)

    if not np.any(mask):
        return None, None

    props = measure.regionprops(mask.astype(int))
    if not props:
        return None, None

    p = props[0]
    major = p.major_axis_length
    minor = p.minor_axis_length

    if major > 0:
        axis_ratio = minor / major
        eccentricity = round(np.sqrt(1 - axis_ratio ** 2), 6)
        roundness = round(axis_ratio ** 2, 6)
    else:
        eccentricity = 0.0
        roundness = 0.0

    return roundness, eccentricity


def find_mask_file(masks_dir, image_name, soma_id, area_um2):
    """Try to locate the mask file matching this CSV row."""
    area_int = int(float(area_um2))

    # Try multiple naming conventions (MMPSv2 current + old MMPS patterns)
    candidates = [
        f"{image_name}_{soma_id}_area{area_int}_mask.tif",   # MMPSv2 approved masks
        f"{image_name}_{soma_id}_area{area_int}.tif",        # MMPSv2 generated (no _mask)
        f"{image_name}_{soma_id}_mask.tif",                  # old MMPS (no area in name)
        f"{image_name}_{soma_id}.tif",                       # old MMPS bare
    ]
    for c in candidates:
        path = os.path.join(masks_dir, c)
        if os.path.exists(path):
            return path

    # Broad fallback: any file matching image + soma_id + area
    for f in os.listdir(masks_dir):
        if f.startswith(image_name) and soma_id in f and f.endswith(".tif"):
            if f"area{area_int}" in f:
                return os.path.join(masks_dir, f)

    # Last resort: any file matching image + soma_id (ignore area)
    for f in os.listdir(masks_dir):
        if f.startswith(image_name) and soma_id in f and f.endswith(".tif"):
            return os.path.join(masks_dir, f)

    return None


def main():
    # ── Pick the CSV and masks folder ──────────────────────────────────
    try:
        from PyQt5.QtWidgets import QApplication, QFileDialog
        app = QApplication(sys.argv)

        csv_path, _ = QFileDialog.getOpenFileName(
            None, "Select combined_morphology CSV", "",
            "CSV Files (*.csv);;All Files (*)",
            options=QFileDialog.DontUseNativeDialog
        )
        if not csv_path:
            print("No CSV selected – exiting.")
            return

        masks_dir = QFileDialog.getExistingDirectory(
            None, "Select masks folder",
            options=QFileDialog.DontUseNativeDialog
        )
        if not masks_dir:
            print("No masks folder selected – exiting.")
            return
    except ImportError:
        # No Qt – fall back to command-line args
        if len(sys.argv) < 3:
            print("Usage: python recompute_roundness_eccentricity.py <csv_path> <masks_dir>")
            return
        csv_path = sys.argv[1]
        masks_dir = sys.argv[2]

    # ── Read the CSV ───────────────────────────────────────────────────
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if "roundness" not in fieldnames or "eccentricity" not in fieldnames:
        print("ERROR: CSV does not contain 'roundness' and/or 'eccentricity' columns.")
        return

    # Determine which column holds the area used in mask filenames
    area_col = None
    for candidate_col in ["area_um2", "mask_area"]:
        if candidate_col in fieldnames:
            area_col = candidate_col
            break
    if area_col is None:
        print("ERROR: CSV has no 'area_um2' or 'mask_area' column to match mask files.")
        return

    # ── List mask files in folder for diagnostics ────────────────────
    all_mask_files = sorted(f for f in os.listdir(masks_dir) if f.endswith(".tif"))
    print(f"\nFound {len(all_mask_files)} TIFF files in masks folder.")
    if all_mask_files:
        print(f"  First few: {all_mask_files[:5]}")
    print(f"CSV has {len(rows)} rows, matching on column '{area_col}'.\n")

    # ── Recompute ──────────────────────────────────────────────────────
    updated = 0
    skipped = 0
    errors = 0
    skipped_rows = []

    for row in rows:
        image_name = row.get("image_name", "")
        soma_id = row.get("soma_id", "")
        area = row.get(area_col, "0")

        mask_path = find_mask_file(masks_dir, image_name, soma_id, area)
        if mask_path is None:
            print(f"  SKIP (no file): {image_name} / {soma_id} / area={area}")
            skipped += 1
            skipped_rows.append(row)
            continue

        try:
            roundness, eccentricity = recompute_for_mask(mask_path)
        except Exception as e:
            print(f"  ERROR: {mask_path}: {e}")
            errors += 1
            skipped_rows.append(row)
            continue

        if roundness is None:
            print(f"  SKIP (empty mask): {mask_path}")
            skipped += 1
            skipped_rows.append(row)
            continue

        row["roundness"] = str(roundness)
        row["eccentricity"] = str(eccentricity)
        updated += 1

    # ── Report rows that still have bad values ─────────────────────────
    if skipped_rows:
        print(f"\n⚠️  {len(skipped_rows)} rows were NOT updated (no matching mask file).")
        print("These rows still have their OLD roundness/eccentricity values:")
        for r in skipped_rows:
            print(f"    {r.get('image_name','?')} / {r.get('soma_id','?')} / "
                  f"area={r.get(area_col,'?')}  "
                  f"roundness={r.get('roundness','?')}  eccentricity={r.get('eccentricity','?')}")
        print()

    # ── Write updated CSV ──────────────────────────────────────────────
    backup_path = csv_path.replace(".csv", "_old_backup.csv")
    os.rename(csv_path, backup_path)
    print(f"\nBackup saved to: {backup_path}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone!  Updated: {updated}  |  Skipped: {skipped}  |  Errors: {errors}")
    print(f"Updated CSV: {csv_path}")


if __name__ == "__main__":
    main()
