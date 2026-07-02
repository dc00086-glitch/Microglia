#!/usr/bin/env python3
"""bbb_from_masks.py — standalone BBB metrics from masks you already generated.

Run this AFTER generating microglia masks in MMPS. For each raw multichannel
image it:
  * segments blood vessels from the CD31 channel (Otsu + close + despeckle),
  * quantifies tracer leakage per tracer (leakage index, perivascular rings),
  * for every mask, computes per-microglia exposure:
        dist_to_vessel_um       (measured from the SOMA, parsed from the filename)
        <tracer>_exposure_mean  (tracer the cell is bathed in, extravascular)
        cd31_exposure_mean      (vessel-marker signal in the cell)
        vessel_contact_fraction (fraction of the cell overlapping vessels)

Writes two CSVs: one per-microglia, one per-image (vessel/leakage summary).

USAGE:
    1. Edit the CONFIG block below (paths, pixel size, channel indices).
    2. python3 bbb_from_masks.py

The math mirrors MMPSv2.12.py exactly. Needs: numpy, scipy, scikit-image,
tifffile (the same packages MMPS uses).
"""

import os
import re
import glob
import csv
import numpy as np
from scipy import ndimage
import tifffile
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

# =========================== CONFIG — EDIT ME ============================
RAW_DIR    = "/Volumes/Expansion/.../Raw"      # folder of raw multichannel TIFFs
MASKS_DIR  = "/Volumes/Expansion/.../Output/masks"   # the *_mask.tif files
OUT_DIR    = "/Volumes/Expansion/.../Output"   # where the CSVs are written

PIXEL_SIZE_UM = 0.316          # microns per pixel (match what you used in MMPS)

# Channel numbers are 1-BASED, exactly as MMPS shows them (Channel 1, 2, 3...).
# (IBA1/microglia channel isn't needed here — BBB only uses CD31 + tracers.)
CD31_CHANNEL  = 3              # CD31 (blood vessels)
TRACERS = [                    # (name, channel number) — one per injected tracer
    ("dextran", 1),
]
SOMA_RADIUS_UM = 6.0           # disk radius around the soma for the distance basis
# ========================================================================


# ----------------------------------------------------------------------
# Image loading (robust to channel-axis order and Z-stacks)
# ----------------------------------------------------------------------
def load_multichannel(path):
    """Return an (H, W, C) float array. Handles (H,W,C), (C,H,W), and Z-stacks
    (max-projected). Channel axis is taken as the smallest axis <= 8."""
    arr = np.squeeze(np.asarray(tifffile.imread(path)))
    if arr.ndim == 2:
        return arr[:, :, None].astype(np.float64)
    if arr.ndim == 3:
        ax = int(np.argmin(arr.shape))
        if arr.shape[ax] <= 8:
            arr = np.moveaxis(arr, ax, -1)
        else:
            arr = arr.max(axis=0)[:, :, None]   # Z-stack, single channel
        return arr.astype(np.float64)
    # 4D+: find a channel axis (<=8), move last, max-project the rest
    chan_ax = next((i for i, s in enumerate(arr.shape) if s <= 8), None)
    if chan_ax is None:
        while arr.ndim > 2:
            arr = arr.max(axis=0)
        return arr[:, :, None].astype(np.float64)
    arr = np.moveaxis(arr, chan_ax, -1)
    while arr.ndim > 3:
        arr = arr.max(axis=0)
    return arr.astype(np.float64)


def read_mask(path):
    return (np.squeeze(np.asarray(tifffile.imread(path))) > 0).astype(np.uint8)


# ----------------------------------------------------------------------
# BBB math (copied from MMPSv2.12.py so results match the app)
# ----------------------------------------------------------------------
def _disk_struct(radius):
    r = int(radius)
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (yy * yy + xx * xx) <= r * r


def segment_vessels(cd31, ps, min_object_um2=5.0, close_radius_px=2):
    img = np.asarray(cd31, dtype=np.float64)
    empty = (np.zeros(img.shape, dtype=bool), {
        'vessel_area_fraction': 0.0, 'vessel_length_density_um_per_um2': 0.0,
        'vessel_branchpoint_density_per_mm2': 0.0, 'vessel_mean_diameter_um': 0.0,
        'vessel_threshold': 0.0})
    if img.max() <= 0 or not np.any(img > 0):
        return empty
    pos = img[img > 0]
    thr = float(threshold_otsu(pos)) if pos.min() < pos.max() else float(pos.min()) - 1.0
    vessels = img > thr
    if close_radius_px > 0:
        vessels = ndimage.binary_closing(vessels, structure=_disk_struct(close_radius_px))
    min_px = max(int(min_object_um2 / (ps ** 2)), 1)
    lab, n = ndimage.label(vessels)
    if n:
        sizes = np.bincount(lab.ravel())
        keep = np.where(sizes >= min_px)[0]
        keep = keep[keep != 0]
        vessels = np.isin(lab, keep)
    if not np.any(vessels):
        return empty
    area_um2 = vessels.size * (ps ** 2)
    skel = skeletonize(vessels)
    skel_len_um = int(skel.sum()) * ps
    dt = ndimage.distance_transform_edt(vessels)
    mean_diam_um = float(2 * dt[skel].mean() * ps) if skel.any() else 0.0
    su = skel.astype(np.uint8)
    nbr = ndimage.convolve(su, np.ones((3, 3), np.uint8), mode='constant', cval=0) - su
    n_branch = int(np.count_nonzero(skel & (nbr >= 3)))
    metrics = {
        'vessel_area_fraction': round(float(vessels.mean()), 4),
        'vessel_length_density_um_per_um2': round(skel_len_um / area_um2, 6),
        'vessel_branchpoint_density_per_mm2': round(n_branch / (area_um2 / 1e6), 2),
        'vessel_mean_diameter_um': round(mean_diam_um, 3),
        'vessel_threshold': round(thr, 2),
    }
    return vessels, metrics


def quantify_leakage(vessel_mask, tracer, ps, ring_edges_um=(0, 10, 20, 40)):
    t = np.asarray(tracer, dtype=np.float64)
    vessel_mask = vessel_mask > 0
    intra = t[vessel_mask]
    extra = t[~vessel_mask]
    intra_mean = float(intra.mean()) if intra.size else 0.0
    extra_mean = float(extra.mean()) if extra.size else 0.0
    leak_thr = float(np.median(intra)) if intra.size else 0.0
    extra_frac = float((extra > leak_thr).mean()) if extra.size else 0.0
    m = {
        'intravascular_mean': round(intra_mean, 3),
        'extravascular_mean': round(extra_mean, 3),
        'leakage_index': round(extra_mean / intra_mean, 4) if intra_mean > 0 else 0.0,
        'extravascular_area_fraction': round(extra_frac, 4),
    }
    dist_um = ndimage.distance_transform_edt(~vessel_mask) * ps
    for i in range(len(ring_edges_um) - 1):
        lo, hi = ring_edges_um[i], ring_edges_um[i + 1]
        ring = (dist_um > lo) & (dist_um <= hi)
        m['perivasc_%g_%gum_mean' % (lo, hi)] = (
            round(float(t[ring].mean()), 3) if np.any(ring) else 0.0)
    return m


def soma_disk(shape, cy, cx, ps, radius_um=SOMA_RADIUS_UM):
    h, w = shape[:2]
    cy = min(max(int(round(cy)), 0), h - 1)
    cx = min(max(int(round(cx)), 0), w - 1)
    r = max(1, int(round(radius_um / max(ps, 1e-6))))
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r)


def microglia_exposure(cell_mask, vessel_mask, tracers, ps, dist_um, soma_mask, cd31):
    m = {}
    vessel_mask = vessel_mask > 0
    cm = cell_mask > 0
    if not np.any(cm):
        return m
    dist_region = soma_mask if (soma_mask is not None and np.any(soma_mask)) else cm
    m['dist_to_vessel_um'] = round(float(dist_um[dist_region].min()), 3)
    outside = cm & ~vessel_mask
    reg = outside if np.any(outside) else cm
    for name, ch in tracers.items():
        m['%s_exposure_mean' % name] = round(float(ch[reg].mean()), 3)
    n_cell = int(cm.sum())
    m['vessel_contact_fraction'] = round(float((cm & vessel_mask).sum()) / n_cell, 4)
    m['cd31_exposure_mean'] = round(float(cd31[cm].mean()), 3)
    return m


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
# Mask filename: <img_base>_soma_<row>_<col>_area<N>_mask.tif
MASK_RE = re.compile(r'^(?P<base>.+)_(?P<sid>soma_(?P<r>\d+)_(?P<c>\d+))_area(?P<area>\d+)_mask\.tif+$', re.I)


def find_raw(base):
    for ext in ('.tif', '.tiff', '.TIF', '.TIFF'):
        p = os.path.join(RAW_DIR, base + ext)
        if os.path.exists(p):
            return p
    # case-insensitive fallback
    for p in glob.glob(os.path.join(RAW_DIR, '*')):
        if os.path.splitext(os.path.basename(p))[0].lower() == base.lower():
            return p
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # Convert 1-based channel numbers (as shown in MMPS) to 0-based array indices.
    cd31_idx = CD31_CHANNEL - 1
    tracer_specs = [(nm, ch - 1) for nm, ch in TRACERS]

    # Group masks by image base
    by_image = {}
    for mp in glob.glob(os.path.join(MASKS_DIR, '*_mask.tif*')):
        name = os.path.basename(mp)
        if name.startswith('._'):
            continue
        m = MASK_RE.match(name)
        if not m:
            continue
        by_image.setdefault(m.group('base'), []).append((mp, m))

    if not by_image:
        print(f"No *_mask.tif files matched in {MASKS_DIR}")
        return

    cell_rows, vessel_rows, misses = [], [], []
    for base, masks in sorted(by_image.items()):
        raw = find_raw(base)
        if raw is None:
            misses.append(base)
            continue
        color = load_multichannel(raw)
        nch = color.shape[2]
        if not (0 <= cd31_idx < nch):
            print(f"  ! {base}: CD31 channel {CD31_CHANNEL} out of range ({nch} ch) — skipped")
            continue
        cd31 = color[:, :, cd31_idx]
        vessel_mask, vmetrics = segment_vessels(cd31, PIXEL_SIZE_UM)
        dist_um = ndimage.distance_transform_edt(~(vessel_mask > 0)) * PIXEL_SIZE_UM

        tracers = {}
        vrow = {'image_name': base}
        vrow.update(vmetrics)
        for nm, ci in tracer_specs:
            if 0 <= ci < nch:
                tracers[nm] = color[:, :, ci]
                for k, v in quantify_leakage(vessel_mask, tracers[nm], PIXEL_SIZE_UM).items():
                    vrow['%s_%s' % (nm, k)] = v
        vessel_rows.append(vrow)

        for mp, mm in masks:
            mk = read_mask(mp)
            if mk.shape != vessel_mask.shape:
                print(f"  ! {os.path.basename(mp)}: shape {mk.shape} != image {vessel_mask.shape} — skipped")
                continue
            sm = soma_disk(vessel_mask.shape, int(mm.group('r')), int(mm.group('c')), PIXEL_SIZE_UM)
            exp = microglia_exposure(mk, vessel_mask, tracers, PIXEL_SIZE_UM, dist_um, sm, cd31)
            crow = {'image_name': base, 'soma_id': mm.group('sid')}
            crow.update(exp)
            cell_rows.append(crow)
        print(f"  {base}: {len(masks)} microglia, vessel area fraction {vmetrics['vessel_area_fraction']}")

    def write(path, rows):
        if not rows:
            return
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows -> {path}")

    write(os.path.join(OUT_DIR, 'bbb_microglia_from_masks.csv'), cell_rows)
    write(os.path.join(OUT_DIR, 'bbb_vessel_from_masks.csv'), vessel_rows)

    if misses:
        raw_stems = sorted({
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(RAW_DIR, '*'))
            if not os.path.basename(p).startswith('._')})
        print("\n" + "=" * 68)
        print(f"{len(misses)} image(s) had masks but NO matching raw file in RAW_DIR.")
        print(f"  masks want:  {', '.join(misses[:8])}{' ...' if len(misses) > 8 else ''}")
        print(f"  RAW_DIR   =  {RAW_DIR}")
        print(f"  it holds:    {', '.join(raw_stems[:8]) or '(nothing)'}"
              f"{' ...' if len(raw_stems) > 8 else ''}")
        if any('_processed' in s for s in raw_stems) or 'Processed' in RAW_DIR:
            print("\n  ⚠  RAW_DIR looks like your PROCESSED-images folder. Those are")
            print("     single-channel (_processed.tif) and have no CD31/tracer channels.")
            print("     Point RAW_DIR at the ORIGINAL MULTICHANNEL TIFFs you loaded into")
            print("     MMPS (the files with all 3 channels: dextran + IBA1 + CD31).")
        print("=" * 68)


if __name__ == '__main__':
    main()
