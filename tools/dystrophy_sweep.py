#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep the dystrophy fragment settings over an MMPS output folder.

The absolute fragment count depends on where the thresholds are set, so the
question that decides whether the metric is usable is not "what is the number"
but "does the RANKING of cells survive changing the settings". This runs the
detector over a grid of gap / search-radius / threshold values and reports the
Spearman rank correlation of every variant against the baseline.

Read it this way:
  rho > 0.9  the ranking is stable, the setting is not driving your result
  rho < 0.7  the setting IS driving your result; do not report it as biology

The detector itself is read straight out of MMPSv2.12.py, so this can never
drift from what the app computes.

Usage:
    python3 tools/dystrophy_sweep.py /path/to/mmps_output --pixel-size 0.1024
    python3 tools/dystrophy_sweep.py /path/to/mmps_output --pixel-size 0.1024 \\
        --processed-dir /path/to/processed --limit-images 5 --out sweep.csv

Expects, under the output folder:
    masks/   <image>_<soma_id>_area<N>_mask.tif
    somas/   <image>_<soma_id>_soma.tif
    <image>_processed.tif   (in the output folder, or --processed-dir)

Requirements: numpy, scipy, scikit-image, tifffile
"""

import argparse
import ast
import csv
import os
import re
import sys
from collections import defaultdict

import numpy as np
import tifffile
from scipy import ndimage
from scipy.stats import spearmanr
from skimage import measure

MASK_RE = re.compile(r'^(?P<image>.+)_(?P<soma>soma_\d+_\d+)_area(?P<area>\d+)_mask\.tif$')


def load_detector(mmps_path):
    """Exec just the dystrophy functions out of MMPSv2.12.py (no PyQt needed)."""
    tree = ast.parse(open(mmps_path).read())
    want_fn = {'_empty_fragment_params', '_avg_centroid_distance_um',
               '_soma_radius_um', '_fragment_search_radius_um',
               '_dystrophy_signal_threshold', '_detect_disconnected_fragments',
               # the fragment pass also recomputes beading on attached material
               '_detect_bulbous_endings', '_branch_extends_past'}
    want_const = {'DYSTROPHY_GAP_UM', 'DYSTROPHY_MIN_FRAGMENT_EXTENT_UM',
                  'DYSTROPHY_MAX_FRAGMENT_AREA_UM2', 'DYSTROPHY_MIN_SEARCH_RADIUS_UM',
                  'DYSTROPHY_SEARCH_RADIUS_SCALE', '_FRAGMENT_KEYS'}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want_fn:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in want_const for t in node.targets):
            nodes.append(node)
    missing = want_fn - {n.name for n in nodes if isinstance(n, ast.FunctionDef)}
    if missing:
        sys.exit(f"ERROR: {mmps_path} has no {', '.join(sorted(missing))}")
    ns = {'np': np, 'ndimage': ndimage, 'measure': measure}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<mmps>', 'exec'), ns)
    return ns


def discover(output_dir):
    """{image: {soma_id: (area_um2, mask_path)}} keeping each soma's largest area."""
    masks_dir = os.path.join(output_dir, 'masks')
    if not os.path.isdir(masks_dir):
        sys.exit(f"ERROR: no masks/ folder in {output_dir}")
    found = defaultdict(dict)
    for name in sorted(os.listdir(masks_dir)):
        m = MASK_RE.match(name)
        if not m:
            continue
        image, soma, area = m.group('image'), m.group('soma'), float(m.group('area'))
        prev = found[image].get(soma)
        if prev is None or area > prev[0]:
            found[image][soma] = (area, os.path.join(masks_dir, name))
    return found


def processed_path(output_dir, processed_dir, image):
    for d in [processed_dir, output_dir]:
        if not d:
            continue
        p = os.path.join(d, f"{image}_processed.tif")
        if os.path.exists(p):
            return p
    return None


def build_cells(ns, output_dir, image, somas, pixel_size, shape):
    somas_dir = os.path.join(output_dir, 'somas')
    cells = []
    for soma_id, (_area, mask_path) in sorted(somas.items()):
        mask = tifffile.imread(mask_path) > 0
        if mask.shape != shape:
            continue
        soma_path = os.path.join(somas_dir, f"{image}_{soma_id}_soma.tif")
        soma = None
        if os.path.exists(soma_path):
            soma = tifffile.imread(soma_path) > 0
            if soma.shape != shape:
                soma = None
        if soma is not None and soma.any():
            ys, xs = np.nonzero(soma)
            centroid = (ys.mean(), xs.mean())
        else:
            ys, xs = np.nonzero(mask)
            if ys.size == 0:
                continue
            centroid = (ys.mean(), xs.mean())
        cells.append({'soma_id': soma_id, 'centroid': centroid, 'mask': mask,
                      'soma_mask': soma,
                      'search_radius_um': ns['_fragment_search_radius_um'](
                          mask, soma, pixel_size)})
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('output_dir', help='MMPS output folder (contains masks/ and somas/)')
    ap.add_argument('--pixel-size', type=float, required=True, help='um per pixel')
    ap.add_argument('--processed-dir', default=None,
                    help='where <image>_processed.tif live, if not the output folder')
    ap.add_argument('--mmps', default=None, help='path to MMPSv2.12.py')
    ap.add_argument('--limit-images', type=int, default=0, help='0 = all')
    ap.add_argument('--out', default='dystrophy_sweep.csv')
    args = ap.parse_args()

    mmps = args.mmps or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', 'MMPSv2.12.py')
    ns = load_detector(os.path.normpath(mmps))
    detect = ns['_detect_disconnected_fragments']
    px = args.pixel_size

    # Baseline first; every other row is compared against it.
    base = {'gap_um': ns['DYSTROPHY_GAP_UM'],
            'search_radius_scale': ns['DYSTROPHY_SEARCH_RADIUS_SCALE'],
            'threshold_scale': 1.0}
    variants = [dict(base, label='baseline')]
    for gap in [1.0, 2.0, 2.5, 3.0]:
        variants.append(dict(base, gap_um=gap, label=f'gap={gap}'))
    for sc in [1.25, 1.5, 2.0, 3.0]:
        variants.append(dict(base, search_radius_scale=sc, label=f'radius x{sc}'))
    for ts in [0.8, 0.9, 1.1, 1.25]:
        variants.append(dict(base, threshold_scale=ts, label=f'threshold x{ts}'))

    images = discover(args.output_dir)
    names = sorted(images)
    if args.limit_images:
        names = names[:args.limit_images]
    if not names:
        sys.exit("ERROR: no MMPS mask files found")

    print(f"{len(names)} image(s), pixel size {px} um/px")
    per_variant = defaultdict(dict)   # label -> {(image, soma): params}
    skipped = []

    for i, image in enumerate(names, 1):
        ppath = processed_path(args.output_dir, args.processed_dir, image)
        if ppath is None:
            skipped.append(image)
            continue
        img = tifffile.imread(ppath)
        if img.ndim == 3:
            img = img.mean(axis=2)
        cells = build_cells(ns, args.output_dir, image, images[image], px, img.shape)
        if not cells:
            skipped.append(image)
            continue
        thr0 = ns['_dystrophy_signal_threshold'](img)
        print(f"  [{i}/{len(names)}] {image}: {len(cells)} cell(s), Otsu = {thr0:.1f}")
        for v in variants:
            res = detect(img, cells, px, gap_um=v['gap_um'],
                         search_radius_scale=v['search_radius_scale'],
                         threshold=thr0 * v['threshold_scale'])
            for soma_id, params in res.items():
                per_variant[v['label']][(image, soma_id)] = params

    if skipped:
        print(f"  skipped {len(skipped)} image(s) with no processed TIFF or no cells: "
              + ", ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else ""))
    if not per_variant:
        sys.exit("ERROR: nothing analysed - are the <image>_processed.tif files present?")

    keys = sorted(per_variant['baseline'])
    with open(args.out, 'w', newline='') as f:
        cols = ['setting', 'image_name', 'soma_id'] + list(ns['_FRAGMENT_KEYS']) \
               + ['frag_gap_um', 'frag_threshold']
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for v in variants:
            for k in keys:
                row = dict(per_variant[v['label']].get(k, {}))
                row.update(setting=v['label'], image_name=k[0], soma_id=k[1])
                w.writerow(row)
    print(f"\nPer-cell values written to {args.out}  ({len(keys)} cells x {len(variants)} settings)")

    def col(label, field):
        return np.array([per_variant[label].get(k, {}).get(field, 0) for k in keys], float)

    print("\nRank stability against the baseline (Spearman rho; >0.9 good, <0.7 unusable)")
    print(f"  {'setting':<18} {'rho(frag_index)':>16} {'rho(n_frag)':>12} "
          f"{'mean n_frag':>12} {'cells with >0':>14}")
    b_idx, b_n = col('baseline', 'fragmentation_index'), col('baseline', 'n_fragments')
    for v in variants:
        idx, n = col(v['label'], 'fragmentation_index'), col(v['label'], 'n_fragments')
        def rho(a, b):
            if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
                return float('nan')
            return spearmanr(a, b).correlation
        print(f"  {v['label']:<18} {rho(b_idx, idx):>16.3f} {rho(b_n, n):>12.3f} "
              f"{n.mean():>12.2f} {int((n > 0).sum()):>9}/{len(n)}")

    print("\nSearch radius is (avg_centroid_distance + soma_radius) x scale. If the "
          "\n  baseline row shows 'cells with >0' near zero, the disk is still inside "
          "\n  the arbor - raise DYSTROPHY_SEARCH_RADIUS_SCALE.")


if __name__ == '__main__':
    main()
