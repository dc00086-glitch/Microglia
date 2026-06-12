#!/usr/bin/env python3
"""Quick standalone test for the bulbous-ending (spheroid) detector.

Run it on a single mask TIFF or a whole folder of masks — no app build needed.
Prints the per-mask metrics and (optionally) saves an overlay PNG with each
detected bulb circled, so you can eyeball-calibrate the thresholds against
cells you'd score as dystrophic by eye.

Usage:
    # one mask
    python3 test_bulb_detection.py /path/to/cell_mask.tif --pixel-size 0.5

    # a folder of *_mask.tif files, save overlays into ./bulb_overlays/
    python3 test_bulb_detection.py /path/to/masks/ --pixel-size 0.5 --overlay

    # tune the thresholds
    python3 test_bulb_detection.py mask.tif --pixel-size 0.5 \
        --swelling-ratio 1.75 --min-bulb-diameter 1.5 --overlay
"""

import os
import sys
import argparse

import numpy as np
import tifffile
from scipy import ndimage
from skimage.morphology import skeletonize


def detect_bulbous_endings(mask, pixel_size, swelling_ratio=1.75,
                           min_bulb_diameter_um=1.5):
    """Identical logic to MMPSv2.12.py — kept standalone for easy testing."""
    result = {'num_bulbous_endings': 0, 'mean_bulb_diameter_um': 0.0,
              'beading_index': 0.0, 'bulb_coords': []}

    binary = (mask > 0)
    if not np.any(binary):
        return result

    skeleton = skeletonize(binary)
    if not np.any(skeleton):
        return result
    radius = ndimage.distance_transform_edt(binary)

    skel_u8 = skeleton.astype(np.uint8)
    neighbor_count = ndimage.convolve(
        skel_u8, np.ones((3, 3), dtype=np.uint8),
        mode='constant', cval=0) - skel_u8
    endpoints = skeleton & (neighbor_count == 1)

    ep_rows, ep_cols = np.nonzero(endpoints)
    n_endpoints = int(ep_rows.size)
    if n_endpoints == 0:
        return result

    median_radius = float(np.median(radius[skeleton]))
    if median_radius <= 0:
        return result

    min_bulb_radius_px = (min_bulb_diameter_um / pixel_size) / 2.0
    ep_radii = radius[ep_rows, ep_cols]
    is_bulb = (ep_radii >= swelling_ratio * median_radius) & \
              (ep_radii >= min_bulb_radius_px)

    n_bulbs = int(np.count_nonzero(is_bulb))
    result['num_bulbous_endings'] = n_bulbs
    result['beading_index'] = round(n_bulbs / n_endpoints, 4)
    if n_bulbs > 0:
        bulb_diam_um = 2.0 * ep_radii[is_bulb] * pixel_size
        result['mean_bulb_diameter_um'] = round(float(np.mean(bulb_diam_um)), 4)
        result['bulb_coords'] = list(zip(ep_rows[is_bulb].tolist(),
                                         ep_cols[is_bulb].tolist(),
                                         (2.0 * ep_radii[is_bulb]).tolist()))
    return result


def save_overlay(mask, res, out_path):
    """Save a PNG of the mask with each detected bulb circled in red."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mask > 0, cmap='gray')
    for r, c, diam_px in res['bulb_coords']:
        ax.add_patch(plt.Circle((c, r), max(diam_px, 6),
                                color='red', fill=False, linewidth=2))
        ax.plot(c, r, 'r+', markersize=8)
    ax.set_title(f"bulbs={res['num_bulbous_endings']}  "
                 f"beading_index={res['beading_index']}  "
                 f"mean_diam={res['mean_bulb_diameter_um']}um")
    ax.axis('off')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def run_one(path, args, overlay_dir=None):
    mask = tifffile.imread(path)
    res = detect_bulbous_endings(mask, args.pixel_size,
                                 args.swelling_ratio, args.min_bulb_diameter)
    name = os.path.basename(path)
    print(f"{name:40s}  bulbs={res['num_bulbous_endings']:3d}  "
          f"beading_index={res['beading_index']:.3f}  "
          f"mean_diam_um={res['mean_bulb_diameter_um']:.3f}")
    if overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)
        out = os.path.join(overlay_dir, os.path.splitext(name)[0] + "_bulbs.png")
        save_overlay(mask, res, out)
    return res


def main():
    ap = argparse.ArgumentParser(description="Test the bulbous-ending detector")
    ap.add_argument("path", help="mask TIFF, or a folder of *_mask.tif files")
    ap.add_argument("--pixel-size", type=float, required=True,
                    help="microns per pixel")
    ap.add_argument("--swelling-ratio", type=float, default=1.75)
    ap.add_argument("--min-bulb-diameter", type=float, default=1.5,
                    help="minimum bulb diameter in microns")
    ap.add_argument("--overlay", action="store_true",
                    help="save overlay PNGs marking detected bulbs")
    args = ap.parse_args()

    overlay_dir = "bulb_overlays" if args.overlay else None

    if os.path.isdir(args.path):
        masks = sorted(f for f in os.listdir(args.path)
                       if f.lower().endswith((".tif", ".tiff")))
        if not masks:
            print("No .tif files found in folder.")
            sys.exit(1)
        for f in masks:
            run_one(os.path.join(args.path, f), args, overlay_dir)
    else:
        run_one(args.path, args, overlay_dir)

    if overlay_dir:
        print(f"\nOverlays saved to ./{overlay_dir}/")


if __name__ == "__main__":
    main()
