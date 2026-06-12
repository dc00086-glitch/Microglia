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
                           min_bulb_diameter_um=1.5, soma_mask=None,
                           soma_area_um2=None, soma_margin=1.3,
                           soma_dilation_px=3, baseline_percentile=25,
                           min_branch_px=5):
    """Identical logic to MMPSv2.12.py — kept standalone for easy testing.

    Swellings are detected along each branch, relative to that branch's own
    baseline width. The soma is removed first (real ``soma_mask`` footprint, else
    a circular estimate). Returns ``soma_region`` / ``soma_center`` /
    ``exclusion_px`` so the overlay can show what was excluded.
    """
    result = {'num_bulbous_endings': 0, 'mean_bulb_diameter_um': 0.0,
              'beading_index': 0.0, 'bulb_coords': [],
              'soma_region': None, 'soma_center': None, 'exclusion_px': 0.0,
              'used_real_soma': False}

    binary = (mask > 0)
    if not np.any(binary):
        return result

    skeleton = skeletonize(binary)
    if not np.any(skeleton):
        return result
    radius = ndimage.distance_transform_edt(binary)

    use_real_soma = (soma_mask is not None
                     and np.shape(soma_mask) == binary.shape
                     and np.any(soma_mask))
    if use_real_soma:
        soma_region = (np.asarray(soma_mask) > 0)
        if soma_dilation_px and soma_dilation_px > 0:
            soma_region = ndimage.binary_dilation(
                soma_region, iterations=int(soma_dilation_px))
        result['soma_region'] = soma_region
        result['used_real_soma'] = True
    else:
        soma_center = np.unravel_index(int(np.argmax(radius)), radius.shape)
        if soma_area_um2 and soma_area_um2 > 0:
            soma_radius_px = np.sqrt(soma_area_um2 / np.pi) / pixel_size
        else:
            soma_radius_px = float(radius[soma_center])
        exclusion_px = soma_radius_px * soma_margin
        rr, cc = np.indices(binary.shape)
        soma_region = (np.hypot(rr - soma_center[0],
                                cc - soma_center[1]) <= exclusion_px)
        result['soma_center'] = (int(soma_center[0]), int(soma_center[1]))
        result['exclusion_px'] = float(exclusion_px)

    skeleton = skeleton & ~soma_region
    if not np.any(skeleton):
        return result

    struct = np.ones((3, 3), dtype=int)
    skel_u8 = skeleton.astype(np.uint8)
    neighbor_count = ndimage.convolve(
        skel_u8, np.ones((3, 3), dtype=np.uint8),
        mode='constant', cval=0) - skel_u8
    junctions = skeleton & (neighbor_count >= 3)
    branches = skeleton & ~junctions
    labeled, n_labels = ndimage.label(branches, structure=struct)
    if n_labels == 0:
        return result

    min_bulb_radius_px = (min_bulb_diameter_um / pixel_size) / 2.0
    swelling = np.zeros(binary.shape, dtype=bool)
    n_branches = 0
    for lab in range(1, n_labels + 1):
        idx = (labeled == lab)
        if int(np.count_nonzero(idx)) < min_branch_px:
            continue
        n_branches += 1
        branch_radii = radius[idx]
        baseline = float(np.percentile(branch_radii, baseline_percentile))
        if baseline <= 0:
            baseline = max(float(branch_radii.min()), 1.0)
        threshold = max(swelling_ratio * baseline, min_bulb_radius_px)
        swelling |= (idx & (radius >= threshold))

    if not np.any(swelling) or n_branches == 0:
        return result

    bulb_labeled, n_bulbs = ndimage.label(swelling, structure=struct)
    if n_bulbs == 0:
        return result

    coords = []
    diameters = []
    for b in range(1, n_bulbs + 1):
        bidx = (bulb_labeled == b)
        peak = np.unravel_index(int(np.argmax(np.where(bidx, radius, 0))),
                                radius.shape)
        diam_px = 2.0 * float(radius[peak])
        diameters.append(diam_px * pixel_size)
        coords.append((int(peak[0]), int(peak[1]), diam_px))

    result['num_bulbous_endings'] = int(n_bulbs)
    result['mean_bulb_diameter_um'] = round(float(np.mean(diameters)), 4)
    result['beading_index'] = round(n_bulbs / n_branches, 4)
    result['bulb_coords'] = coords
    return result


def find_soma_mask(mask_path):
    """Find the soma TIFF matching a cell mask and return it as a binary array.

    Mask:  <img>_<soma_id>_area<NNN>_mask.tif
    Soma:  <img>_<soma_id>_soma.tif  (in a sibling ``somas/`` folder or alongside)
    """
    import re
    name = os.path.basename(mask_path)
    m = re.match(r'^(.+?)_(soma_\d+_\d+)_area\d+_mask\.tif{1,2}$', name, re.I)
    if not m:
        return None
    img, soma_id = m.group(1), m.group(2)
    soma_name = f"{img}_{soma_id}_soma.tif"
    mask_dir = os.path.dirname(os.path.abspath(mask_path))
    for cand in [
        os.path.join(os.path.dirname(mask_dir), "somas", soma_name),  # ../somas/
        os.path.join(mask_dir, "somas", soma_name),                    # ./somas/
        os.path.join(mask_dir, soma_name),                             # alongside
    ]:
        if os.path.exists(cand):
            return (tifffile.imread(cand) > 0)
    return None


def save_overlay(mask, res, out_path):
    """Save a PNG of the mask with each detected bulb circled in red."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mask > 0, cmap='gray')
    # Show the excluded soma region: real outline (blue) or circular estimate.
    if res.get('soma_region') is not None:
        ax.contour(res['soma_region'], levels=[0.5],
                   colors='deepskyblue', linewidths=1.5)
    elif res.get('soma_center') and res.get('exclusion_px'):
        sr, sc = res['soma_center']
        ax.add_patch(plt.Circle((sc, sr), res['exclusion_px'],
                                color='deepskyblue', fill=False,
                                linewidth=1.5, linestyle='--'))
    for r, c, diam_px in res['bulb_coords']:
        ax.add_patch(plt.Circle((c, r), max(diam_px, 6),
                                color='red', fill=False, linewidth=2))
        ax.plot(c, r, 'r+', markersize=8)
    soma_tag = "real soma" if res.get('used_real_soma') else "est. soma"
    ax.set_title(f"bulbs={res['num_bulbous_endings']}  "
                 f"beading_index={res['beading_index']}  "
                 f"mean_diam={res['mean_bulb_diameter_um']}um  [{soma_tag}]")
    ax.axis('off')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def run_one(path, args, overlay_dir=None):
    mask = tifffile.imread(path)
    soma_mask = None if args.no_soma else find_soma_mask(path)
    res = detect_bulbous_endings(mask, args.pixel_size,
                                 args.swelling_ratio, args.min_bulb_diameter,
                                 soma_mask=soma_mask,
                                 soma_margin=args.soma_margin,
                                 soma_dilation_px=args.soma_dilation,
                                 baseline_percentile=args.baseline_percentile,
                                 min_branch_px=args.min_branch_px)
    name = os.path.basename(path)
    soma_tag = "real-soma" if res.get('used_real_soma') else "est-soma "
    print(f"{name:40s}  [{soma_tag}]  bulbs={res['num_bulbous_endings']:3d}  "
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
    ap.add_argument("--soma-margin", type=float, default=1.3,
                    help="circular soma exclusion factor (only when no soma mask found)")
    ap.add_argument("--soma-dilation", type=int, default=3,
                    help="pixels to dilate the real soma mask before excluding")
    ap.add_argument("--baseline-percentile", type=float, default=25,
                    help="percentile of each branch's radius used as its shaft width")
    ap.add_argument("--min-branch-px", type=int, default=5,
                    help="ignore skeleton branches shorter than this many pixels")
    ap.add_argument("--no-soma", action="store_true",
                    help="ignore soma TIFFs and use the circular estimate")
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
