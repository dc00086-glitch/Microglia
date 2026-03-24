#!/usr/bin/env python3
"""
Ruffle Distribution Analysis
=============================
Standalone script to quantify the spatial distribution of membrane ruffles
(processes) relative to the soma in 3D microglia segmentation masks.

Ruffles = cell_mask minus soma_mask (i.e. the processes extending from the soma).

Outputs per-cell metrics:
  - ruffle_polarity_index   : PCA-based (0 = evenly distributed, 1 = all on one side)
  - ruffle_principal_azimuth: azimuthal angle of the dominant ruffle direction (0-360°)
  - ruffle_principal_elevation: elevation angle of the dominant ruffle direction (0-180°)
  - ruffle_centroid_offset_um: distance between soma centroid and ruffle centroid
  - ruffle_centroid_angle_azimuth: azimuth from soma center to ruffle center of mass
  - ruffle_centroid_angle_elevation: elevation from soma center to ruffle center of mass
  - ruffle_dispersion       : mean distance of ruffle voxels from soma centroid,
                              normalized by soma equivalent radius (low = clustered, high = spread)
  - ruffle_major_axis_um    : extent along the dominant PCA axis
  - ruffle_mid_axis_um      : extent along the second PCA axis
  - ruffle_minor_axis_um    : extent along the third PCA axis

Usage:
  # Easiest: load from an MMPS session file (reads pixel sizes, paths, etc.)
  python ruffle_distribution.py --session my_project.mmps3d_session

  # Single pair of masks
  python ruffle_distribution.py --cell-mask cell_mask3d.tif --soma-mask soma_mask3d.tif \
      --vxy 0.22 --vz 1.0

  # Batch mode: folder of MMPS-exported masks + somas
  python ruffle_distribution.py --masks-dir output/masks --somas-dir output/somas \
      --vxy 0.22 --vz 1.0 -o ruffle_distribution_results.csv

  # 2D mode (single-slice masks)
  python ruffle_distribution.py --cell-mask cell_mask.tif --soma-mask soma.tif \
      --vxy 0.22 --mode 2d
"""

import argparse
import csv
import json
import os
import re
import sys

import numpy as np
import tifffile


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def compute_ruffle_distribution(cell_mask, soma_mask, vxy, vz=1.0, mode="3d"):
    """Compute ruffle spatial distribution metrics.

    Parameters
    ----------
    cell_mask : ndarray
        Binary cell mask (2D or 3D).
    soma_mask : ndarray
        Binary soma mask, same shape as cell_mask.
    vxy : float
        Pixel size in XY (µm/pixel).
    vz : float
        Voxel depth in Z (µm/slice). Ignored for 2D.
    mode : str
        '2d' or '3d'.

    Returns
    -------
    dict  with metric name → value, or None if no ruffle voxels.
    """
    cell_bin = (cell_mask > 0).astype(np.uint8)
    soma_bin = (soma_mask > 0).astype(np.uint8)

    # Ruffle = cell minus soma
    ruffle_bin = cell_bin & (~soma_bin.astype(bool)).astype(np.uint8)
    ruffle_coords = np.argwhere(ruffle_bin > 0)  # (N, ndim)

    if len(ruffle_coords) < 3:
        return None

    soma_coords = np.argwhere(soma_bin > 0)
    if len(soma_coords) == 0:
        return None

    is_3d = (mode == "3d") and (cell_mask.ndim == 3)

    # Scale coordinates to physical units
    if is_3d:
        scale = np.array([vz, vxy, vxy])  # (Z, Y, X)
    else:
        scale = np.array([vxy, vxy])  # (row, col)
        # Flatten to 2D if needed
        if ruffle_coords.shape[1] == 3:
            ruffle_coords = ruffle_coords[:, 1:]
        if soma_coords.shape[1] == 3:
            soma_coords = soma_coords[:, 1:]

    ruffle_phys = ruffle_coords * scale
    soma_phys = soma_coords * scale

    soma_centroid = soma_phys.mean(axis=0)
    ruffle_centroid = ruffle_phys.mean(axis=0)

    # --- PCA on ruffle coordinates (centered on soma centroid) ---
    centered = ruffle_phys - soma_centroid
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh returns ascending order
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    major_val = eigenvalues[0]
    minor_val = eigenvalues[-1]
    major_vec = eigenvectors[:, 0]

    metrics = {}

    # Polarity index
    if major_val > 0:
        metrics["ruffle_polarity_index"] = round(1.0 - (minor_val / major_val), 4)
    else:
        metrics["ruffle_polarity_index"] = 0.0

    # Principal direction angles
    if is_3d:
        # Azimuth (XY plane, 0-360°) and elevation (from Z axis, 0-180°)
        dz, dy, dx = major_vec
        azimuth = np.degrees(np.arctan2(dy, dx)) % 360
        elevation = np.degrees(np.arccos(np.clip(dz / (np.linalg.norm(major_vec) + 1e-12), -1, 1)))
        metrics["ruffle_principal_azimuth"] = round(azimuth, 2)
        metrics["ruffle_principal_elevation"] = round(elevation, 2)
    else:
        angle = np.degrees(np.arctan2(major_vec[0], major_vec[1])) % 180
        metrics["ruffle_principal_angle"] = round(angle, 2)

    # Axis extents
    metrics["ruffle_major_axis_um"] = round(2 * np.sqrt(max(eigenvalues[0], 0)), 4)
    if is_3d:
        metrics["ruffle_mid_axis_um"] = round(2 * np.sqrt(max(eigenvalues[1], 0)), 4)
    metrics["ruffle_minor_axis_um"] = round(2 * np.sqrt(max(eigenvalues[-1], 0)), 4)

    # --- Centroid offset ---
    offset_vec = ruffle_centroid - soma_centroid
    offset_dist = np.linalg.norm(offset_vec)
    metrics["ruffle_centroid_offset_um"] = round(offset_dist, 4)

    # Direction from soma center to ruffle center
    if is_3d:
        dz, dy, dx = offset_vec
        c_az = np.degrees(np.arctan2(dy, dx)) % 360
        c_el = np.degrees(np.arccos(np.clip(dz / (offset_dist + 1e-12), -1, 1)))
        metrics["ruffle_centroid_angle_azimuth"] = round(c_az, 2)
        metrics["ruffle_centroid_angle_elevation"] = round(c_el, 2)
    else:
        c_ang = np.degrees(np.arctan2(offset_vec[0], offset_vec[1])) % 180
        metrics["ruffle_centroid_angle"] = round(c_ang, 2)

    # --- Dispersion (normalized by soma equivalent radius) ---
    distances = np.linalg.norm(ruffle_phys - soma_centroid, axis=1)
    mean_dist = distances.mean()
    # Soma equivalent radius
    soma_volume = len(soma_coords) * np.prod(scale)
    if is_3d:
        soma_eq_radius = (3 * soma_volume / (4 * np.pi)) ** (1 / 3)
    else:
        soma_eq_radius = np.sqrt(soma_volume / np.pi)
    if soma_eq_radius > 0:
        metrics["ruffle_dispersion"] = round(mean_dist / soma_eq_radius, 4)
    else:
        metrics["ruffle_dispersion"] = 0.0

    return metrics


# ---------------------------------------------------------------------------
# Mask filename parsing (compatible with MMPS export conventions)
# ---------------------------------------------------------------------------

MASK_RE_3D = re.compile(r'^(.+?)_(soma_\d+_\d+_\d+)_vol(\d+)_mask3d\.tif$')
MASK_RE_2D = re.compile(r'^(.+?)_(soma_\d+_\d+)_area(\d+)_mask\.tif$')


def parse_mask_filename(filename, mode="3d"):
    pat = MASK_RE_3D if mode == "3d" else MASK_RE_2D
    m = pat.match(filename)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return None, None, None


def find_soma_mask(somas_dir, image_name, soma_id):
    """Locate the matching soma mask file."""
    candidates = [
        f"{image_name}_{soma_id}_soma.tif",
        f"{image_name}_{soma_id}.tif",
    ]
    for c in candidates:
        path = os.path.join(somas_dir, c)
        if os.path.exists(path):
            return path
    # Fallback: search for soma_id substring
    if os.path.isdir(somas_dir):
        for f in os.listdir(somas_dir):
            if soma_id in f and f.endswith(".tif"):
                return os.path.join(somas_dir, f)
    return None


# ---------------------------------------------------------------------------
# Single-pair mode
# ---------------------------------------------------------------------------

def run_single(args):
    print(f"Loading cell mask: {args.cell_mask}")
    cell = tifffile.imread(args.cell_mask)
    print(f"Loading soma mask: {args.soma_mask}")
    soma = tifffile.imread(args.soma_mask)

    if cell.shape != soma.shape:
        # If 3D cell but 2D soma, broadcast soma across Z
        if cell.ndim == 3 and soma.ndim == 2:
            print("  Broadcasting 2D soma mask across Z slices.")
            soma = np.broadcast_to(soma[np.newaxis], cell.shape).copy()
        else:
            print(f"ERROR: Shape mismatch: cell {cell.shape} vs soma {soma.shape}", file=sys.stderr)
            sys.exit(1)

    mode = args.mode
    if mode == "auto":
        mode = "3d" if cell.ndim == 3 else "2d"

    vz = args.vz if args.vz is not None else 1.0
    metrics = compute_ruffle_distribution(cell, soma, args.vxy, vz, mode=mode)
    if metrics is None:
        print("No ruffle voxels found (cell minus soma is empty).")
        return

    print("\n--- Ruffle Distribution Metrics ---")
    for k, v in metrics.items():
        print(f"  {k:40s} = {v}")

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)
        print(f"\nResults saved to {args.output}")


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def run_batch(args):
    masks_dir = args.masks_dir
    somas_dir = args.somas_dir
    mode = args.mode

    mask_files = sorted(f for f in os.listdir(masks_dir) if f.endswith(".tif"))
    if not mask_files:
        print(f"No .tif files found in {masks_dir}")
        sys.exit(1)

    # Detect mode from filenames if auto
    if mode == "auto":
        if any(MASK_RE_3D.match(f) for f in mask_files):
            mode = "3d"
        else:
            mode = "2d"

    vz = args.vz if args.vz is not None else 1.0
    print(f"Mode: {mode.upper()}")
    print(f"Masks dir : {masks_dir}")
    print(f"Somas dir : {somas_dir}")
    print(f"Pixel size: vxy={args.vxy} µm, vz={vz} µm")
    print()

    all_results = []
    for fname in mask_files:
        image_name, soma_id, area_px = parse_mask_filename(fname, mode)
        if image_name is None:
            continue

        soma_path = find_soma_mask(somas_dir, image_name, soma_id)
        if soma_path is None:
            print(f"  SKIP {fname}: no matching soma mask found")
            continue

        cell = tifffile.imread(os.path.join(masks_dir, fname))
        soma = tifffile.imread(soma_path)

        if cell.shape != soma.shape:
            if cell.ndim == 3 and soma.ndim == 2:
                soma = np.broadcast_to(soma[np.newaxis], cell.shape).copy()
            else:
                print(f"  SKIP {fname}: shape mismatch cell {cell.shape} vs soma {soma.shape}")
                continue

        metrics = compute_ruffle_distribution(cell, soma, args.vxy, vz, mode=mode)
        if metrics is None:
            print(f"  SKIP {fname}: no ruffle voxels")
            continue

        if mode == "3d":
            area_um2 = area_px * (args.vxy ** 2) * vz
        else:
            area_um2 = area_px * (args.vxy ** 2)

        row = {"image_name": image_name, "soma_id": soma_id, "area_um2": area_um2}
        row.update(metrics)
        all_results.append(row)
        print(f"  OK   {fname}  polarity={metrics['ruffle_polarity_index']:.3f}  "
              f"offset={metrics['ruffle_centroid_offset_um']:.2f} µm  "
              f"dispersion={metrics['ruffle_dispersion']:.2f}")

    if not all_results:
        print("\nNo cells processed.")
        return

    # Write CSV
    out_path = args.output or os.path.join(os.path.dirname(masks_dir), "ruffle_distribution_results.csv")
    fieldnames = list(all_results[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n{len(all_results)} cells processed. Results saved to {out_path}")


# ---------------------------------------------------------------------------
# Session file mode
# ---------------------------------------------------------------------------

def resolve_session_path(session_dir, absolute_path, relative_path):
    """Resolve a path from session JSON, trying relative first for portability."""
    if relative_path:
        resolved = os.path.normpath(os.path.join(session_dir, relative_path))
        if os.path.exists(resolved):
            return resolved
    if absolute_path and os.path.exists(absolute_path):
        return absolute_path
    return None


def run_session(args):
    """Load an MMPS session file and batch-process all approved masks."""
    session_path = args.session
    if not os.path.isfile(session_path):
        print(f"ERROR: Session file not found: {session_path}", file=sys.stderr)
        sys.exit(1)

    with open(session_path, "r") as f:
        session = json.load(f)

    session_dir = os.path.dirname(os.path.abspath(session_path))

    # --- Resolve masks and somas directories ---
    masks_dir = resolve_session_path(
        session_dir,
        session.get("masks_dir"),
        session.get("masks_dir_rel"),
    )
    if not masks_dir or not os.path.isdir(masks_dir):
        # Fallback: look for masks/ inside output_dir
        output_dir = resolve_session_path(
            session_dir,
            session.get("output_dir"),
            session.get("output_dir_rel"),
        )
        if output_dir:
            masks_dir = os.path.join(output_dir, "masks")

    if not masks_dir or not os.path.isdir(masks_dir):
        print("ERROR: Could not locate masks directory from session file.", file=sys.stderr)
        sys.exit(1)

    somas_dir = os.path.join(os.path.dirname(masks_dir), "somas")
    if not os.path.isdir(somas_dir):
        print(f"ERROR: Somas directory not found at {somas_dir}", file=sys.stderr)
        sys.exit(1)

    # --- Read pixel sizes ---
    is_3d = session.get("mode_3d", False)
    mode = "3d" if is_3d else "2d"

    # Global pixel size (fallback)
    global_vxy = None
    ps_str = session.get("pixel_size", "")
    try:
        global_vxy = float(ps_str)
    except (ValueError, TypeError):
        pass

    global_vz = 1.0
    vz_str = session.get("voxel_size_z", "")
    try:
        global_vz = float(vz_str)
    except (ValueError, TypeError):
        pass

    # CLI overrides if provided
    if args.vxy is not None:
        global_vxy = args.vxy
    if args.vz is not None:
        global_vz = args.vz

    # --- Build per-image pixel size lookup ---
    # Session keys include file extension (e.g. "image.tif") but mask
    # filename parsing yields stems (e.g. "image"). Store both forms.
    image_pixel_sizes = {}
    for img_name, img_data in session.get("images", {}).items():
        ps = img_data.get("pixel_size")
        if ps is not None:
            try:
                val = float(ps)
                image_pixel_sizes[img_name] = val
                image_pixel_sizes[os.path.splitext(img_name)[0]] = val
            except (ValueError, TypeError):
                pass

    # --- Collect approved masks ---
    # Store with both the full session key and the stem so lookups from
    # parsed mask filenames (which lack extensions) can match.
    approved_masks = set()
    for img_name, img_data in session.get("images", {}).items():
        stem = os.path.splitext(img_name)[0]
        for mqa in img_data.get("mask_qa_state", []):
            if mqa.get("approved") is True:
                soma_id = mqa.get("soma_id", "")
                approved_masks.add((img_name, soma_id))
                approved_masks.add((stem, soma_id))

    print(f"Session : {session_path}")
    print(f"Mode    : {mode.upper()}")
    print(f"Masks   : {masks_dir}")
    print(f"Somas   : {somas_dir}")
    print(f"Default : vxy={global_vxy} µm, vz={global_vz} µm")
    if image_pixel_sizes:
        print(f"Per-image pixel sizes found for {len(image_pixel_sizes)} image(s)")
    if approved_masks:
        print(f"Approved masks: {len(approved_masks)}")
        # Show a sample to help diagnose matching issues
        sample = list(approved_masks)[:3]
        for img, sid in sample:
            print(f"  e.g. image={img!r}  soma_id={sid!r}")
    else:
        print("No approval filter found — processing all masks")
    print()

    mask_files = sorted(f for f in os.listdir(masks_dir) if f.endswith(".tif"))
    if not mask_files:
        print(f"No .tif files found in {masks_dir}")
        sys.exit(1)

    # Show sample parsed mask filenames
    sample_shown = 0
    for f_ in mask_files:
        n_, s_, _ = parse_mask_filename(f_, mode)
        if n_ is not None and sample_shown < 3:
            print(f"  Parsed: {f_!r} -> image={n_!r}  soma_id={s_!r}")
            sample_shown += 1
    unparsed = sum(1 for f_ in mask_files if parse_mask_filename(f_, mode)[0] is None)
    if unparsed:
        print(f"  ({unparsed} mask files did not match the {mode.upper()} filename pattern)")
    print()

    all_results = []
    skipped = 0
    for fname in mask_files:
        image_name, soma_id, area_px = parse_mask_filename(fname, mode)
        if image_name is None:
            continue

        # If we have approval info, only process approved masks
        if approved_masks and (image_name, soma_id) not in approved_masks:
            skipped += 1
            continue

        # Per-image pixel size or global fallback
        vxy = image_pixel_sizes.get(image_name, global_vxy)
        if vxy is None:
            print(f"  SKIP {fname}: no pixel size found (set --vxy or check session)")
            continue

        vz = global_vz

        soma_path = find_soma_mask(somas_dir, image_name, soma_id)
        if soma_path is None:
            print(f"  SKIP {fname}: no matching soma mask found")
            continue

        cell = tifffile.imread(os.path.join(masks_dir, fname))
        soma = tifffile.imread(soma_path)

        if cell.shape != soma.shape:
            if cell.ndim == 3 and soma.ndim == 2:
                soma = np.broadcast_to(soma[np.newaxis], cell.shape).copy()
            else:
                print(f"  SKIP {fname}: shape mismatch cell {cell.shape} vs soma {soma.shape}")
                continue

        metrics = compute_ruffle_distribution(cell, soma, vxy, vz, mode=mode)
        if metrics is None:
            print(f"  SKIP {fname}: no ruffle voxels")
            continue

        # Compute mask area in µm²
        if mode == "3d":
            area_um2 = area_px * (vxy ** 2) * vz
        else:
            area_um2 = area_px * (vxy ** 2)

        # Get animal_id and treatment from session if available
        images_dict = session.get("images", {})
        img_session = images_dict.get(image_name) or images_dict.get(image_name + ".tif") or images_dict.get(image_name + ".tiff")

        row = {
            "image_name": image_name,
            "soma_id": soma_id,
            "animal_id": img_session.get("animal_id", "") if img_session else "",
            "treatment": img_session.get("treatment", "") if img_session else "",
            "pixel_size_xy": vxy,
            "area_um2": area_um2,
        }
        row.update(metrics)
        all_results.append(row)
        print(f"  OK   {fname}  (vxy={vxy})  polarity={metrics['ruffle_polarity_index']:.3f}  "
              f"offset={metrics['ruffle_centroid_offset_um']:.2f} µm  "
              f"dispersion={metrics['ruffle_dispersion']:.2f}")

    if skipped:
        print(f"\n  ({skipped} unapproved masks skipped)")

    if not all_results:
        print("\nNo cells processed.")
        return

    # Write CSV
    out_path = args.output or os.path.join(os.path.dirname(masks_dir), "ruffle_distribution_results.csv")
    fieldnames = list(all_results[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n{len(all_results)} cells processed. Results saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Quantify ruffle (process) spatial distribution relative to the soma.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--vxy", type=float, default=None, help="XY pixel size in µm (read from session if omitted)")
    parser.add_argument("--vz", type=float, default=None, help="Z voxel depth in µm (read from session if omitted)")
    parser.add_argument("--mode", choices=["2d", "3d", "auto"], default="auto",
                        help="Analysis mode (default: auto-detect)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output CSV path")

    # Session mode
    parser.add_argument("--session", type=str,
                        help="Path to .mmps_session or .mmps3d_session file (easiest — reads all settings)")

    # Single-pair mode
    parser.add_argument("--cell-mask", type=str, help="Path to a single cell mask TIFF")
    parser.add_argument("--soma-mask", type=str, help="Path to the matching soma mask TIFF")

    # Batch mode
    parser.add_argument("--masks-dir", type=str, help="Folder of MMPS-exported cell masks")
    parser.add_argument("--somas-dir", type=str, help="Folder of MMPS-exported soma masks")

    args = parser.parse_args()

    if args.session:
        run_session(args)
    elif args.cell_mask and args.soma_mask:
        if args.vxy is None:
            parser.error("--vxy is required when using --cell-mask / --soma-mask")
        run_single(args)
    elif args.masks_dir and args.somas_dir:
        if args.vxy is None:
            parser.error("--vxy is required when using --masks-dir / --somas-dir")
        run_batch(args)
    else:
        # No arguments given — open a file dialog to pick a session file
        args.session = _pick_session_file()
        if args.session:
            run_session(args)
        else:
            parser.error("Provide --session, or --cell-mask + --soma-mask, or --masks-dir + --somas-dir")


def _pick_session_file():
    """Open a Qt file dialog and return the chosen path, or None."""
    try:
        from PyQt5.QtWidgets import QApplication, QFileDialog
    except ImportError:
        return None
    try:
        app = QApplication.instance() or QApplication(sys.argv)
        path, _ = QFileDialog.getOpenFileName(
            None,
            "Select MMPS Session File",
            "",
            "Session Files (*.mmps_session *.mmps3d_session);;JSON Files (*.json);;All Files (*)",
        )
        if path:
            return path
    except Exception as e:
        print(f"Could not open file dialog: {e}", file=sys.stderr)
    return None


if __name__ == "__main__":
    main()
