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
    python3 test_bulb_detection.py mask.tif --pixel-size 0.104 \
        --min-bulb-diameter 1.4 --open-radius 4 --min-conn-len 10 --overlay
"""

import os
import sys
import argparse

import numpy as np
import tifffile
from scipy import ndimage
from skimage.morphology import skeletonize


def detect_bulbous_endings(mask, pixel_size, min_bulb_diameter_um=1.4,
                           soma_mask=None, soma_area_um2=None, soma_margin=1.3,
                           open_radius_px=None, soma_dilation_px=3,
                           min_tip_dist_factor=1.5, min_conn_len_px=10,
                           max_connections=1):
    """Identical logic to MMPSv2.12.py — kept standalone for easy testing.

    A bulb is a rounded terminal lobe: a blob (isolated by morphological opening,
    soma/body removed) above a size floor, beyond the soma-distance gate, with at
    most one substantial thin-process connection (junctions have more). Returns
    soma_region / bulb_coords for the overlay.
    """
    from skimage.morphology import opening, disk

    result = {'num_bulbous_endings': 0, 'mean_bulb_diameter_um': 0.0,
              'beading_index': 0.0, 'bulb_coords': [],
              'soma_region': None, 'used_real_soma': False}

    binary = (mask > 0)
    if not np.any(binary):
        return result

    radius = ndimage.distance_transform_edt(binary)
    skel = skeletonize(binary)
    struct = np.ones((3, 3), dtype=int)

    use_real_soma = (soma_mask is not None
                     and np.shape(soma_mask) == binary.shape
                     and np.any(soma_mask))
    if use_real_soma:
        soma_bin = (np.asarray(soma_mask) > 0)
        srows, scols = np.nonzero(soma_bin)
        soma_center = (float(srows.mean()), float(scols.mean()))
        soma_radius_px = np.sqrt(srows.size / np.pi)
        soma_region = soma_bin
        if soma_dilation_px and soma_dilation_px > 0:
            soma_region = ndimage.binary_dilation(soma_region,
                                                  iterations=int(soma_dilation_px))
        result['used_real_soma'] = True
    else:
        soma_center = np.unravel_index(int(np.argmax(radius)), radius.shape)
        if soma_area_um2 and soma_area_um2 > 0:
            soma_radius_px = np.sqrt(soma_area_um2 / np.pi) / pixel_size
        else:
            soma_radius_px = float(radius[soma_center])
        rr, cc = np.indices(binary.shape)
        soma_region = (np.hypot(rr - soma_center[0],
                                cc - soma_center[1]) <= soma_radius_px * soma_margin)
    result['soma_region'] = soma_region

    if open_radius_px is None:
        proc_dt = radius[skel & ~soma_region]
        thin_half = float(np.percentile(proc_dt, 25)) if proc_dt.size else 2.0
        ceiling = max(3, int((min_bulb_diameter_um / pixel_size) / 2.0) - 1)
        open_radius_px = int(min(max(round(thin_half + 2), 3), ceiling))

    opened = opening(binary, disk(int(open_radius_px)))
    labeled, n_labels = ndimage.label(opened, structure=struct)
    if n_labels == 0:
        return result

    sc = (int(round(soma_center[0])), int(round(soma_center[1])))
    soma_label = labeled[sc] if (0 <= sc[0] < labeled.shape[0]
                                 and 0 <= sc[1] < labeled.shape[1]) else 0
    if soma_label == 0:
        overlap = labeled[soma_region]; overlap = overlap[overlap > 0]
        if overlap.size:
            soma_label = int(np.bincount(overlap).argmax())

    proc_skel = skel & ~opened
    proc_labeled, _ = ndimage.label(proc_skel, structure=struct)
    proc_sizes = np.bincount(proc_labeled.ravel())

    min_bulb_radius_px = (min_bulb_diameter_um / pixel_size) / 2.0
    min_tip_dist_px = min_tip_dist_factor * soma_radius_px

    coords = []
    diameters = []
    for lab in range(1, n_labels + 1):
        if lab == soma_label:
            continue
        comp = (labeled == lab)
        blob_radius = float(radius[comp].max())
        if blob_radius < min_bulb_radius_px:
            continue
        ys, xs = np.nonzero(comp); cy, cx = ys.mean(), xs.mean()
        if np.hypot(cy - soma_center[0], cx - soma_center[1]) < min_tip_dist_px:
            continue
        ring = ndimage.binary_dilation(comp, iterations=2)
        touching = set(np.unique(proc_labeled[ring & proc_skel])) - {0}
        n_conn = sum(1 for t in touching if proc_sizes[t] >= min_conn_len_px)
        if n_conn > max_connections:
            continue
        peak = np.unravel_index(int(np.argmax(np.where(comp, radius, 0))),
                                radius.shape)
        coords.append((int(peak[0]), int(peak[1]), 2.0 * blob_radius))
        diameters.append(2.0 * blob_radius * pixel_size)

    if not coords:
        return result

    skel_nosoma = skel & ~soma_region
    su = skel_nosoma.astype(np.uint8)
    nbr = ndimage.convolve(su, np.ones((3, 3), dtype=np.uint8),
                           mode='constant', cval=0) - su
    n_tips = int(np.count_nonzero(skel_nosoma & (nbr == 1)))

    result['num_bulbous_endings'] = len(coords)
    result['mean_bulb_diameter_um'] = round(float(np.mean(diameters)), 4)
    result['beading_index'] = round(len(coords) / n_tips, 4) if n_tips else 0.0
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
                                 min_bulb_diameter_um=args.min_bulb_diameter,
                                 soma_mask=soma_mask,
                                 soma_margin=args.soma_margin,
                                 open_radius_px=args.open_radius,
                                 soma_dilation_px=args.soma_dilation,
                                 min_tip_dist_factor=args.min_tip_dist_factor,
                                 min_conn_len_px=args.min_conn_len)
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
    ap.add_argument("--min-bulb-diameter", type=float, default=1.4,
                    help="minimum bulb diameter in microns (size floor)")
    ap.add_argument("--open-radius", type=int, default=None,
                    help="opening disk radius in px; omit for auto (sized to this cell's process width)")
    ap.add_argument("--soma-margin", type=float, default=1.3,
                    help="circular soma exclusion factor (only when no soma mask found)")
    ap.add_argument("--soma-dilation", type=int, default=3,
                    help="pixels to dilate the real soma mask before excluding")
    ap.add_argument("--min-conn-len", type=int, default=10,
                    help="min length (px) of a process to count as a connection; a bulb has <=1")
    ap.add_argument("--min-tip-dist-factor", type=float, default=1.5,
                    help="bulb must be at least this many soma-radii from the soma center")
    ap.add_argument("--no-soma", action="store_true",
                    help="ignore soma TIFFs and use the circular estimate")
    ap.add_argument("--overlay", action="store_true",
                    help="save overlay PNGs marking detected bulbs")
    args = ap.parse_args()

    overlay_dir = "bulb_overlays" if args.overlay else None

    if os.path.isdir(args.path):
        # Skip hidden / macOS AppleDouble (._*) sidecar files, not just by name
        # but only keep real *_mask.tif files.
        masks = sorted(f for f in os.listdir(args.path)
                       if f.lower().endswith((".tif", ".tiff"))
                       and not f.startswith("."))
        if not masks:
            print("No .tif files found in folder.")
            sys.exit(1)
        for f in masks:
            try:
                run_one(os.path.join(args.path, f), args, overlay_dir)
            except Exception as e:
                print(f"  SKIP {f}: {e}")
    else:
        run_one(args.path, args, overlay_dir)

    if overlay_dir:
        print(f"\nOverlays saved to ./{overlay_dir}/")


if __name__ == "__main__":
    main()
