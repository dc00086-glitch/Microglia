#!/usr/bin/env python3
"""
add_bulbs_to_master.py
======================
Detect terminal ATP-sensor bulbs on MMPS-exported masks across timepoint groups
(1d / 3d / 7d / 28d) and merge the per-cell bulb metrics into your master sheet.

It does TWO things:
  1. Always writes a tidy results table  ->  bulb_results_all_groups.csv
     (one row per mask: timepoint, image_name, soma_id, area, mask_file,
      num_bulbous_endings, mean_bulb_diameter_um, beading_index)
  2. If you point it at a master sheet, writes a COPY of it with the three bulb
     columns added  ->  <master stem>_with_bulbs.(csv|xlsx)
     The original master file is never modified.

The detector is identical to MMPSv2.12 / test_bulb_detection.py (morphological
opening to isolate rounded terminal lobes, soma/body removed, size + distance
gates, one-connecting-branch terminal rule, and rejection of any lobe a branch
extends past).

------------------------------------------------------------------------------
USAGE
  1) Edit the CONFIG block below (paths + how your master sheet identifies cells).
  2) Install deps:
         python3 -m pip install numpy scipy scikit-image tifffile
         python3 -m pip install pandas openpyxl     # only needed for the merge
  3) Run:
         python3 add_bulbs_to_master.py
------------------------------------------------------------------------------
"""

import os
import re
import sys
import csv
import math

import numpy as np
import tifffile
from scipy import ndimage
from skimage.morphology import skeletonize, opening, disk

# ============================ CONFIG ========================================

# Parent folder that CONTAINS the timepoint folders (1d, 3d, 7d, 28d).
# Example: ".../TREM2 IBA1 Cortex CCI 63x"
BASE_DIR = "/Volumes/Expansion/CCI Young Rat NL1 Study Data/Raw Data/TREM2 IBA1 Cortex CCI 63x"

# Timepoint group folder names under BASE_DIR.
GROUPS = ["1d", "3d", "7d", "28d"]

# Where masks / somas live inside each group folder.
MASKS_SUBPATH = "Output/masks"
SOMAS_SUBPATH = "Output/somas"

# Pixel size in microns/pixel. None = auto-read from each TIFF's resolution tag
# (falls back to PIXEL_SIZE_FALLBACK if a file has no tag).
PIXEL_SIZE = None
PIXEL_SIZE_FALLBACK = 0.104

# ---- Master sheet merge (set MASTER_SHEET = None to skip and only get the CSV) ----
# This is pre-filled for the CORTEX run. For the INTERNAL CAPSULE sheet, point
# BASE_DIR at that region's mask folders and set MASTER_SHEET to:
#   ".../All Internal Capsule Morphology Data/Merged_All_Timepoints_IC_Morphology_Clean.csv"
MASTER_SHEET = "/Volumes/Expansion/CCI Young Rat NL1 Study Data/Raw Data/TREM2 IBA1 Cortex CCI 63x/Master Sheet Cortex_FIXED.csv"
MASTER_OUT = None            # output path; None = "<master stem>_with_bulbs.<ext>"

# Which master-sheet columns identify a cell. Matching is on image_name + soma_id
# (globally unique), with the .tif extension auto-stripped from image_name (the IC
# sheet stores it). Timepoint is NOT used for matching (the Day column is
# "Sham"/numeric, not the folder names). Leave None to auto-detect image/soma.
MASTER_IMAGE_COL = None      # auto-detects "image_name"
MASTER_SOMA_COL = None       # auto-detects "soma_id"
MASTER_GROUP_COL = None      # leave None (match on image+soma only)

# ---- Detector parameters (defaults match the validated MMPS detector) ----
MIN_BULB_DIAMETER_UM = 1.4
OPEN_RADIUS_PX = None        # None = auto (sized to each cell's process width)
SOMA_DILATION_PX = 3
MIN_TIP_DIST_FACTOR = 1.5
MIN_CONN_LEN_PX = 10
MAX_CONNECTIONS = 1
DISTAL_MIN_LEN_PX = 4
DISTAL_COS = -0.2

# ============================================================================

OUT_CSV = "bulb_results_all_groups.csv"
MASK_RE = re.compile(r'^(.+?)_(soma_\d+_\d+)_area(\d+)_mask\.tif{1,2}$', re.I)


# ---------------------------------------------------------------------------
# Detector (standalone — identical logic to MMPSv2.12 / test_bulb_detection.py)
# ---------------------------------------------------------------------------
def detect_bulbous_endings(mask, pixel_size, min_bulb_diameter_um=1.4,
                           soma_mask=None, soma_area_um2=None, soma_margin=1.3,
                           open_radius_px=None, soma_dilation_px=3,
                           min_tip_dist_factor=1.5, min_conn_len_px=10,
                           max_connections=1, distal_min_len_px=4,
                           distal_cos=-0.2):
    result = {'num_bulbous_endings': 0, 'mean_bulb_diameter_um': 0.0,
              'beading_index': 0.0, 'bulb_coords': []}

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
    else:
        soma_center = np.unravel_index(int(np.argmax(radius)), radius.shape)
        if soma_area_um2 and soma_area_um2 > 0:
            soma_radius_px = np.sqrt(soma_area_um2 / np.pi) / pixel_size
        else:
            soma_radius_px = float(radius[soma_center])
        rr, cc = np.indices(binary.shape)
        soma_region = (np.hypot(rr - soma_center[0],
                                cc - soma_center[1]) <= soma_radius_px * soma_margin)

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

    coords, diameters = [], []
    for lab in range(1, n_labels + 1):
        if lab == soma_label:
            continue
        comp = (labeled == lab)
        blob_radius = float(radius[comp].max())
        if blob_radius < min_bulb_radius_px:
            continue
        ys, xs = np.nonzero(comp)
        cy, cx = ys.mean(), xs.mean()
        if np.hypot(cy - soma_center[0], cx - soma_center[1]) < min_tip_dist_px:
            continue
        ring = ndimage.binary_dilation(comp, iterations=2)
        touching = set(np.unique(proc_labeled[ring & proc_skel])) - {0}
        n_conn = sum(1 for t in touching if proc_sizes[t] >= min_conn_len_px)
        if n_conn > max_connections:
            continue
        # Reject if a process extends PAST the bulb (far side from the soma).
        v_s = np.array([soma_center[0] - cy, soma_center[1] - cx], dtype=float)
        ns = np.linalg.norm(v_s) + 1e-9
        extends_past = False
        for t in touching:
            if proc_sizes[t] < distal_min_len_px:
                continue
            ty, tx = np.nonzero((proc_labeled == t) & ring)
            if ty.size == 0:
                continue
            v_t = np.array([ty.mean() - cy, tx.mean() - cx], dtype=float)
            if float(v_t @ v_s / (np.linalg.norm(v_t) * ns + 1e-9)) < distal_cos:
                extends_past = True
                break
        if extends_past:
            continue
        peak = np.unravel_index(int(np.argmax(np.where(comp, radius, 0))),
                                radius.shape)
        coords.append((int(peak[0]), int(peak[1])))
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_mask_filename(filename):
    """-> (image_name, soma_id, area) or (None, None, None)."""
    if filename.startswith("."):
        return None, None, None
    m = MASK_RE.match(filename)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return None, None, None


def read_pixel_size(path):
    """Read microns/pixel from a TIFF resolution tag; fall back to default."""
    if PIXEL_SIZE is not None:
        return PIXEL_SIZE
    try:
        with tifffile.TiffFile(path) as t:
            xr = t.pages[0].tags.get('XResolution')
            if xr and xr.value[0]:
                return xr.value[1] / xr.value[0]
    except Exception:
        pass
    return PIXEL_SIZE_FALLBACK


def find_soma(somas_dir, image_name, soma_id):
    if not somas_dir or not os.path.isdir(somas_dir):
        return None
    for cand in (f"{image_name}_{soma_id}_soma.tif",
                 f"{image_name}_{soma_id}.tif"):
        p = os.path.join(somas_dir, cand)
        if os.path.exists(p):
            return tifffile.imread(p) > 0
    for f in os.listdir(somas_dir):
        if soma_id in f and f.endswith(".tif") and not f.startswith("."):
            return tifffile.imread(os.path.join(somas_dir, f)) > 0
    return None


def process_group(group):
    """Run the detector on every mask in one timepoint group; return list of rows."""
    masks_dir = os.path.join(BASE_DIR, group, MASKS_SUBPATH)
    somas_dir = os.path.join(BASE_DIR, group, SOMAS_SUBPATH)
    rows = []
    if not os.path.isdir(masks_dir):
        print(f"  [{group}] masks folder not found: {masks_dir}")
        return rows
    files = sorted(f for f in os.listdir(masks_dir)
                   if f.lower().endswith((".tif", ".tiff")) and not f.startswith("."))
    print(f"  [{group}] {len(files)} mask files")
    for f in files:
        img, soma_id, area = parse_mask_filename(f)
        if img is None:
            continue
        path = os.path.join(masks_dir, f)
        try:
            mask = tifffile.imread(path)
            ps = read_pixel_size(path)
            soma = find_soma(somas_dir, img, soma_id)
            res = detect_bulbous_endings(
                mask, ps, min_bulb_diameter_um=MIN_BULB_DIAMETER_UM,
                soma_mask=soma, open_radius_px=OPEN_RADIUS_PX,
                soma_dilation_px=SOMA_DILATION_PX,
                min_tip_dist_factor=MIN_TIP_DIST_FACTOR,
                min_conn_len_px=MIN_CONN_LEN_PX, max_connections=MAX_CONNECTIONS,
                distal_min_len_px=DISTAL_MIN_LEN_PX, distal_cos=DISTAL_COS)
        except Exception as e:
            print(f"    SKIP {f}: {e}")
            continue
        rows.append({
            'timepoint': group,
            'image_name': img,
            'soma_id': soma_id,
            'target_area_um2': area,
            'mask_file': f,
            'num_bulbous_endings': res['num_bulbous_endings'],
            'mean_bulb_diameter_um': res['mean_bulb_diameter_um'],
            'beading_index': res['beading_index'],
        })
    return rows


BULB_COLS = ['num_bulbous_endings', 'mean_bulb_diameter_um', 'beading_index']


def merge_into_master(rows):
    """Write a copy of the master sheet with bulb columns added (non-destructive)."""
    try:
        import pandas as pd
    except ImportError:
        print("\npandas not installed -> skipping master merge "
              "(use the bulb_results CSV, or: pip install pandas openpyxl).")
        return

    ext = os.path.splitext(MASTER_SHEET)[1].lower()
    master = pd.read_excel(MASTER_SHEET) if ext in (".xlsx", ".xls") \
        else pd.read_csv(MASTER_SHEET)

    def find_col(explicit, candidates):
        if explicit:
            return explicit
        lower = {c.lower(): c for c in master.columns}
        for cand in candidates:
            if cand in lower:
                return lower[cand]
        return None

    img_col = find_col(MASTER_IMAGE_COL, ['image_name', 'image', 'imagename'])
    soma_col = find_col(MASTER_SOMA_COL, ['soma_id', 'soma', 'somaid'])
    # Only match on a timepoint column if explicitly configured — never auto-grab
    # 'Day' (it holds "Sham"/blank or numeric days, not the 1d/3d folder names).
    grp_col = find_col(MASTER_GROUP_COL, []) if MASTER_GROUP_COL else None
    if img_col is None or soma_col is None:
        print(f"\nCould not find image/soma columns in master "
              f"(have: {list(master.columns)}). Set MASTER_IMAGE_COL / "
              f"MASTER_SOMA_COL in CONFIG. Wrote bulb CSV only.")
        return

    # Build lookup keyed by (timepoint?, image_name, soma_id).
    def norm(v):
        return str(v).strip()

    def norm_img(v):
        """Normalize an image name: strip whitespace and a trailing .tif/.tiff
        (the IC sheet stores the full filename with extension; masks don't)."""
        s = str(v).strip()
        low = s.lower()
        if low.endswith('.tiff'):
            s = s[:-5]
        elif low.endswith('.tif'):
            s = s[:-4]
        return s

    lut = {}
    for r in rows:
        if grp_col is not None:
            key = (norm(r['timepoint']), norm_img(r['image_name']), norm(r['soma_id']))
        else:
            key = (norm_img(r['image_name']), norm(r['soma_id']))
        lut[key] = r

    matched = 0
    out_cols = {c: [] for c in BULB_COLS}
    for _, mr in master.iterrows():
        if grp_col is not None:
            key = (norm(mr[grp_col]), norm_img(mr[img_col]), norm(mr[soma_col]))
        else:
            key = (norm_img(mr[img_col]), norm(mr[soma_col]))
        hit = lut.get(key)
        if hit:
            matched += 1
        for c in BULB_COLS:
            out_cols[c].append(hit[c] if hit else np.nan)
    for c in BULB_COLS:
        master[c] = out_cols[c]

    out = MASTER_OUT
    if out is None:
        stem, e = os.path.splitext(MASTER_SHEET)
        out = f"{stem}_with_bulbs{e}"
    if out.lower().endswith((".xlsx", ".xls")):
        master.to_excel(out, index=False)
    else:
        master.to_csv(out, index=False)
    print(f"\nMerged bulb columns into {matched}/{len(master)} master rows "
          f"(matched on {'timepoint+' if grp_col else ''}{img_col}+{soma_col}).")
    print(f"Wrote: {out}   (original master untouched)")
    if matched == 0:
        print("  WARNING: 0 rows matched — check that the master's image/soma "
              "values match the mask filenames (and the timepoint column).")


def main():
    all_rows = []
    print(f"Scanning groups under: {BASE_DIR}")
    for g in GROUPS:
        all_rows.extend(process_group(g))
    if not all_rows:
        print("No masks processed — check BASE_DIR / GROUPS / MASKS_SUBPATH.")
        sys.exit(1)

    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    total_bulbs = sum(r['num_bulbous_endings'] for r in all_rows)
    print(f"\nWrote {len(all_rows)} rows ({total_bulbs} bulbs total) to {OUT_CSV}")
    by_grp = {}
    for r in all_rows:
        by_grp.setdefault(r['timepoint'], [0, 0])
        by_grp[r['timepoint']][0] += 1
        by_grp[r['timepoint']][1] += r['num_bulbous_endings']
    for g, (ncell, nb) in by_grp.items():
        print(f"   {g}: {ncell} cells, {nb} bulbs")

    if MASTER_SHEET:
        merge_into_master(all_rows)


if __name__ == "__main__":
    main()
