#!/usr/bin/env python3
"""Re-measure BBB leakage with a DIFFERENT tracer, reusing a finished run.

The situation this is for: the same fields were imaged again with another
tracer (BSA instead of dextran), so the vessels and the microglia are the same
objects and only the leak channel changed. Reviewing every vessel again would
be rework -- the segmentation already exists from the first run.

WHAT IS REUSED, AND WHAT IS NOT
    bbb_progress/<image>_vesselmask.tif   the REVIEWED vessel mask. Reused.
    Output/masks/*_mask.tif               the microglia masks. Reused.
    Output/somas/*_soma.tif               the soma outlines. Reused.
    bbb_overlays/*.png                    NOT usable. These are rendered
                                          composites -- a drawn outline in a
                                          lossy picture. Turning one back into
                                          a binary mask would invent the
                                          boundary, and every number here is
                                          measured against that boundary.

So this needs the FIRST run to have been made by a build that writes
bbb_progress/. If that folder is missing, the masks were never saved and the
vessels have to be reviewed again -- the script says so rather than guessing.

    python3 tools/bbb_rerun_tracer.py \
        --prior  "/Volumes/.../Dextran Output" \
        --images "/Volumes/.../BSA Images" \
        --tracer bsa --tracer-channel 2 \
        --pixel-size 0.0641 \
        --out    "/Volumes/.../BSA Output/simple_bsa_morphology.csv"

Writes a simple morphology CSV -- image_name, soma_id and the bbb_ columns for
the new tracer -- ready to tick as the morphology CSV in MMPS's ImageJ merge.
"""
import os
import re
import csv
import sys
import glob
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    import numpy as np
    import tifffile
    from scipy import ndimage
except ImportError as e:
    sys.exit(f"Missing a dependency ({e}). pip install numpy tifffile scipy")

# The measurements come from MMPS itself, so these numbers mean exactly what
# the app's own columns mean -- no second implementation to drift.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'mmps', os.path.join(ROOT, 'MMPSv2.12.py'))
mmps = importlib.util.module_from_spec(_spec)
sys.modules['mmps'] = mmps
_spec.loader.exec_module(mmps)

SOMA_RE = re.compile(r'^(?P<base>.+?)_(?P<sid>soma_\d+_\d+)_soma\.tiff?$')
MASK_RE = re.compile(r'^(?P<base>.+?)_(?P<sid>soma_\d+_\d+)_area(?P<a>\d+)_mask\.tiff?$')


def norm(name):
    n = str(name).strip()
    low = n.lower()
    for ext in ('.tiff', '.tif'):
        if low.endswith(ext):
            return n[:-len(ext)]
    return n


def load_plane(path, channel):
    """One channel of an image, however the file stores its axes."""
    arr = np.asarray(tifffile.imread(path))
    if arr.ndim == 2:
        return arr.astype(np.float64)
    if arr.ndim != 3:
        arr = arr.reshape(arr.shape[-3:])
    # channels are the smallest axis; pick it and index along it
    cax = int(np.argmin(arr.shape))
    if channel >= arr.shape[cax]:
        raise IndexError(f"channel {channel + 1} but the file has "
                         f"{arr.shape[cax]}")
    return np.moveaxis(arr, cax, 0)[channel].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prior', required=True,
                    help="the FINISHED run's Output folder -- the one holding "
                         "bbb_progress/, masks/ and somas/")
    ap.add_argument('--images', required=True,
                    help='folder of the new images carrying the new tracer')
    ap.add_argument('--tracer', required=True,
                    help='name for the new tracer, used as the column prefix '
                         '(e.g. bsa)')
    ap.add_argument('--tracer-channel', type=int, required=True,
                    help='1-based channel holding the new tracer')
    ap.add_argument('--pixel-size', type=float, required=True, help='µm/px')
    ap.add_argument('--extra-radius', type=float, default=0.0,
                    help='one more exposure ring in µm; microglia, 10, 20 and '
                         '30 µm are always measured')
    ap.add_argument('--out', required=True, help='CSV to write')
    a = ap.parse_args()

    prog = os.path.join(a.prior, 'bbb_progress')
    vmasks = {}
    for p in glob.glob(os.path.join(prog, '*_vesselmask.tif')):
        base = os.path.basename(p)[:-len('_vesselmask.tif')]
        vmasks[base] = p
    if not vmasks:
        sys.exit(
            f"No reviewed vessel masks in {prog}\n\n"
            f"That folder is written by MMPS as each image finishes. If the "
            f"first run predates it, only rendered PNGs were saved and the "
            f"vessel boundary cannot be recovered from those -- the vessels "
            f"have to be reviewed again on the new images.")

    somas = {}
    for p in glob.glob(os.path.join(a.prior, 'somas', '*_soma.tif*')):
        m = SOMA_RE.match(os.path.basename(p))
        if m:
            somas.setdefault(m.group('base'), {})[m.group('sid')] = p
    masks = {}
    for p in glob.glob(os.path.join(a.prior, 'masks', '*_mask.tif*')):
        m = MASK_RE.match(os.path.basename(p))
        if m:
            masks.setdefault(m.group('base'), {}).setdefault(
                m.group('sid'), []).append((int(m.group('a')), p))

    images = {}
    for p in glob.glob(os.path.join(a.images, '*.tif')) + \
             glob.glob(os.path.join(a.images, '*.tiff')) + \
             glob.glob(os.path.join(a.images, '*.TIF')):
        images[norm(os.path.basename(p))] = p

    ch = a.tracer_channel - 1
    radii = tuple(sorted(set(mmps._EXPOSURE_RADII_UM) |
                         ({float(a.extra_radius)} if a.extra_radius else set())))
    rows, skipped = [], []
    for base in sorted(vmasks):
        img_path = images.get(base)
        if img_path is None:
            skipped.append((base, 'no matching image in --images'))
            continue
        try:
            tracer = load_plane(img_path, ch)
        except Exception as e:
            skipped.append((base, f'could not read channel: {e}'))
            continue

        vessel = np.asarray(tifffile.imread(vmasks[base])) > 0
        if vessel.shape != tracer.shape:
            skipped.append((base, f'vessel mask {vessel.shape} does not match '
                                  f'image {tracer.shape} — are these the same '
                                  f'fields?'))
            continue

        dist_um = ndimage.distance_transform_edt(~vessel) * a.pixel_size
        tracers = {a.tracer: tracer}
        per_soma = masks.get(base, {})
        soma_files = somas.get(base, {})
        ids = sorted(set(per_soma) | set(soma_files))
        if not ids:
            skipped.append((base, 'no masks or somas for this image'))
            continue

        for sid in ids:
            cell = None
            sizes = per_soma.get(sid)
            if sizes:
                # the largest mask kept for this cell, as MMPS does
                _, path = max(sizes, key=lambda t: t[0])
                cell = np.asarray(tifffile.imread(path)) > 0
            soma = None
            sp = soma_files.get(sid)
            if sp:
                soma = np.asarray(tifffile.imread(sp)) > 0
            if cell is None:
                cell = soma
            if cell is None or cell.shape != vessel.shape:
                continue
            exp = mmps._microglia_leakage_exposure(
                cell, vessel, tracers, a.pixel_size, dist_um=dist_um,
                soma_mask=soma, halo_radii_um=radii)
            row = {'image_name': base, 'soma_id': sid}
            row.update(exp)
            rows.append(row)

    if not rows:
        sys.exit("Nothing measured. Check --images points at the new images "
                 "and their names match the first run's.")

    fields = ['image_name', 'soma_id']
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})

    print(f"{len(rows)} cells over {len(set(r['image_name'] for r in rows))} "
          f"images -> {a.out}")
    print(f"  columns: {', '.join(c for c in fields if c.startswith('bbb_'))}")
    if skipped:
        print(f"  {len(skipped)} image(s) skipped:")
        for b, why in skipped[:10]:
            print(f"    {b}: {why}")
        if len(skipped) > 10:
            print(f"    ...and {len(skipped) - 10} more")
    print("\nTick this as the morphology CSV in MMPS's ImageJ merge, or join "
          "it on image_name + soma_id.")


if __name__ == '__main__':
    main()
