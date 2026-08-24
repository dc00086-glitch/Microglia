#!/usr/bin/env python3
"""train_mask_qa_model.py — learn mask QA from the decisions you already made.

WHAT THE USER ACTUALLY DECIDES
------------------------------
Mask QA looks like 16 independent accept/reject calls per cell, but it is not.
The grid QA screen approves one size and applies it: "double-click = accept this
size (smaller approved, larger rejected)". The saved labels confirm it — in
`MaskQAComplete.mmps_session`, 2075 of 2077 cells have a contiguous run of
approvals from the smallest target area upward. The decision is therefore a
single number per cell:

    the largest target area at which this cell's mask is still clean
    (0 if even the smallest one is not)

So that is what this trains: a per-cell CUTOFF, not 16 loose binaries. The
model still scores each (cell, target area) pair, but the scores are decoded
into a prefix, which is the only shape the labels ever take. That both matches
the UI and removes 16x of label noise the binary framing would have invented.

WHY THE MASK PIXELS ARE NOT A FEATURE
-------------------------------------
The obvious feature set is whole-object shape measured on the mask TIFF: area,
solidity, holes, components, and so on. It cannot be used, because MMPS DELETES
a mask's TIFF the moment it is rejected (`_delete_rejected_mask_tiff`). In the
28d session that leaves 14,552 approved masks on disk and 18,536 rejected ones
gone. Training whole-object features on what survives means training on the
positive class only; every negative would have to be dropped, and a classifier
fitted that way learns nothing except that masks exist.

The features here are computed instead from things that DO survive: the image,
the accepted soma outline, the positions of the other somas in the same image,
and the candidate target area. They describe the space a cell has to grow into
rather than the mask that resulted, which is also the thing the reviewer is
actually judging — a mask is rejected because the cell ran out of room and the
growth crossed into a neighbour, a vessel, or background.

That is not a workaround, it is the only leak-free option: at QA time the mask
pixels of an as-yet-unreviewed candidate are available, but in TRAINING data
they are available only for approved ones. Any feature computed from them would
be perfectly correlated with the label in training and useless in use.

To make whole-object features possible for FUTURE models, MMPS now writes
`mask_qa_features.csv` beside the masks at generation time, while every
candidate still exists. Pass it with `--features-csv` and its columns are
joined on and used as well. Nothing here requires it.

USAGE
    python3 train_mask_qa_model.py --session MaskQAComplete.mmps_session

  Multiple sessions, or a folder to scan for them, are fine:
    --session a.mmps_session b.mmps_session
    --session-root "/Volumes/Expansion/.../Raw Data"

  If the sessions were written on another machine, remap their stored paths:
    --root-map "/Volumes/Expansion=/mnt/expansion"

  Other options that matter:
    --image processed|raw    which image to measure (default processed)
    --channel N              1-based stain channel, for raw multi-channel images
    --features-csv PATH...   whole-object mask features captured by MMPS
    --limit 300              use only 300 cells, to check the wiring first
    --out mask_qa_model.joblib

The saved model is what MMPS loads to order the QA queue and pre-select a size.
"""

import os
import sys
import csv
import glob
import json
import argparse
import numpy as np

try:
    import tifffile
except ImportError:
    sys.exit("Missing tifffile.  pip install tifffile")
try:
    from scipy import ndimage as ndi
except ImportError:
    sys.exit("Missing scipy.  pip install scipy")
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    import joblib
except ImportError:
    sys.exit("Missing scikit-learn.  pip install scikit-learn joblib")


# ----------------------------------------------------------------------
# reading sessions
# ----------------------------------------------------------------------
def remap(path, mappings):
    """Apply --root-map OLD=NEW prefix rewrites to a path stored in a session.

    Sessions record absolute paths on the machine that wrote them, so a session
    copied off the acquisition drive points at a volume that is not mounted.
    """
    if not path:
        return path
    for old, new in mappings:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def find_sessions(paths, roots):
    """Collect .mmps_session files from explicit paths and/or scanned roots."""
    out = list(paths or [])
    for r in roots or []:
        out += sorted(glob.glob(os.path.join(r, '**', '*.mmps_session'),
                                recursive=True))
    seen, uniq = set(), []
    for p in out:
        rp = os.path.realpath(p)
        if rp not in seen and os.path.isfile(p):
            seen.add(rp)
            uniq.append(p)
    return uniq


def _cutoff_from_qa(entries):
    """-> (cutoff_um2, areas, labels) for one cell, or None if unusable.

    `cutoff` is the largest approved target area (0 if none approved). Cells
    whose approvals are NOT a contiguous run from the smallest area are dropped:
    they are the handful reviewed one mask at a time rather than in the grid, and
    a prefix model has no way to represent them. Dropping is honest; forcing them
    into a prefix would train on a label the user never gave.
    """
    by_area = {}
    for e in entries:
        a = e.get('area_um2', e.get('target_area_um2'))
        if a is None or e.get('approved') is None:
            continue
        # an auto-rejected duplicate is a rule, not a judgement -- see below
        if e.get('duplicate'):
            continue
        by_area[float(a)] = bool(e.get('approved'))
    if len(by_area) < 3:
        return None
    areas = sorted(by_area)
    labels = [by_area[a] for a in areas]
    n_true = sum(labels)
    if labels[:n_true] != [True] * n_true:      # not a prefix
        return None
    cutoff = areas[n_true - 1] if n_true else 0.0
    return cutoff, areas, labels


def read_session(path, mappings, dup_keys=frozenset()):
    """-> list of per-image dicts with everything needed to build features.

    `dup_keys` are (image, soma_id, area) triples MMPS auto-rejected as
    duplicates -- several target areas that produced pixel-identical masks once
    growth could not expand further. They are a rule, not a judgement: training
    on them teaches the model something it does not need and inflates the score
    with free correct answers. Sessions written before MMPS recorded the flag do
    not say which masks these were, so the flag comes from
    regen_masks_for_training.py's rebuild instead.
    """
    with open(path) as fh:
        sess = json.load(fh)
    px = sess.get('pixel_size')
    try:
        px = float(px)
    except (TypeError, ValueError):
        px = None
    images = []
    for img_name, img in (sess.get('images') or {}).items():
        qa = img.get('mask_qa_state') or []
        if not qa:
            continue
        outlines = {o.get('soma_id'): o for o in (img.get('soma_outlines') or [])}
        per_soma = {}
        for e in qa:
            area = e.get('area_um2', e.get('target_area_um2'))
            if area is not None and (img_name, e.get('soma_id'),
                                     float(area)) in dup_keys:
                continue
            per_soma.setdefault(e.get('soma_id'), []).append(e)
        somas = []
        for sid, entries in per_soma.items():
            got = _cutoff_from_qa(entries)
            if got is None:
                continue
            cutoff, areas, labels = got
            ol = outlines.get(sid)
            if ol is None:
                continue
            somas.append(dict(soma_id=sid, cutoff=cutoff, areas=areas,
                              labels=labels,
                              centroid=ol.get('centroid'),
                              polygon=ol.get('polygon_points'),
                              soma_area_um2=ol.get('soma_area_um2')))
        if not somas:
            continue
        # "Approve All Remaining" marks every unreviewed mask approved in one
        # click, which lands in the file looking exactly like a reviewer who
        # approved the largest size for each of those cells. Its fingerprint is
        # an unbroken run of them at the END of the image's queue -- the cells
        # nobody got to. Counted here so it can be reported rather than trained
        # on. (Checked on the 28d session: 2% of its saturated cells, so those
        # labels are real judgements.)
        a_max = max((max(s['areas']) for s in somas), default=0.0)
        tail = 0
        for s in reversed(somas):
            if s['cutoff'] >= a_max > 0:
                tail += 1
            else:
                break
        images.append(dict(
            saturated=sum(1 for s in somas if s['cutoff'] >= a_max > 0),
            bulk_tail=tail,
            session=os.path.basename(path),
            image_name=img_name,
            raw_path=remap(img.get('raw_path'), mappings),
            processed_path=remap(img.get('processed_path'), mappings),
            pixel_size=px,
            # every centroid in the image, including cells with no QA -- a
            # neighbour crowds this cell whether or not it was itself reviewed
            all_centroids=[c for c in (img.get('somas') or []) if c],
            somas=somas))
    return images


# ----------------------------------------------------------------------
# images
# ----------------------------------------------------------------------
def load_gray(path, channel=None):
    """Load an image as 2D float, keeping only the stain being measured.

    `channel` is 1-based. Leaving it None falls back to whichever channel carries
    the most total signal, which is only a guess -- on this study's images the
    brightest-channel guess picked a different channel on different images. Name
    the channel whenever the file is multi-channel; processed images are already
    single-channel and need nothing.
    """
    a = np.squeeze(np.asarray(tifffile.imread(path)))
    if a.ndim == 3:
        ax = int(np.argmin(a.shape))
        if a.shape[ax] <= 8:
            a = np.moveaxis(a, ax, -1)
            if channel is not None:
                if not 1 <= channel <= a.shape[2]:
                    raise ValueError(f"--channel {channel} but "
                                     f"{os.path.basename(path)} has "
                                     f"{a.shape[2]} channels")
                a = a[:, :, channel - 1]
            else:
                sums = [a[:, :, i].astype(np.float64).sum()
                        for i in range(a.shape[2])]
                a = a[:, :, int(np.argmax(sums))]
        else:
            a = a.max(axis=0)
    while a.ndim > 2:
        a = a.max(axis=0)
    return a.astype(np.float64)


def open_image(rec, prefer='processed', channel=None):
    """-> (gray, which) for one image record, or (None, reason)."""
    order = ([rec.get('processed_path'), rec.get('raw_path')]
             if prefer == 'processed' else
             [rec.get('raw_path'), rec.get('processed_path')])
    names = ['processed', 'raw'] if prefer == 'processed' else ['raw', 'processed']
    for p, nm in zip(order, names):
        if p and os.path.exists(p):
            try:
                # the processed image is already the single grey plane the mask
                # growth ran on, so it never needs a channel picked
                return load_gray(p, None if nm == 'processed' else channel), nm
            except Exception as e:
                return None, f"unreadable ({e})"
    return None, "no image file on disk"


# ----------------------------------------------------------------------
# features
#
# Everything below is COPIED VERBATIM into MMPSv2.12.py (prefixed `_mqa_`) so
# the app can score masks without importing this file. Keep the two identical;
# tools/test_mask_qa_parity.py fails if they drift. Pure numpy/scipy only, for
# the same reason -- cv2 in one copy and scipy in the other would agree on most
# inputs and disagree on the edges, which is the worst kind of drift.
# ----------------------------------------------------------------------
def polygon_mask(points, shape, offset=(0, 0)):
    """Rasterise an (row, col) polygon with an even-odd scanline test.

    Written out rather than delegated to cv2/skimage because this runs in two
    places and both must produce the SAME pixels; two libraries' fill rules
    differ by a pixel at the boundary, which is a percent of a small soma.
    """
    h, w = shape
    m = np.zeros((h, w), dtype=bool)
    if points is None or len(points) < 3:
        return m
    pts = np.asarray(points, dtype=np.float64)
    ys = pts[:, 0] - offset[0]
    xs = pts[:, 1] - offset[1]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    inside = np.zeros((h, w), dtype=bool)
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        y1, x1, y2, x2 = ys[i], xs[i], ys[j], xs[j]
        if y1 == y2:
            continue
        crosses = ((y1 > yy) != (y2 > yy))
        xint = x1 + (yy - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (xx < xint)
    return inside


def poly_area_perimeter(points):
    """Shoelace area and perimeter of a polygon, in pixel units."""
    p = np.asarray(points, dtype=np.float64)
    y, x = p[:, 0], p[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    per = float(np.sum(np.hypot(np.diff(np.append(y, y[0])),
                                np.diff(np.append(x, x[0])))))
    return float(area), per


def convex_area(points):
    """Convex-hull area of a polygon, for solidity. Falls back to its own area."""
    try:
        from scipy.spatial import ConvexHull
        p = np.asarray(points, dtype=np.float64)
        if len(p) < 3:
            return None
        return float(ConvexHull(p).volume)          # 2D "volume" is area
    except Exception:
        return None


MASK_QA_FEATURES = [
    'target_area_um2', 'r_eq_um', 'area_frac_of_max', 'area_over_soma',
    'soma_area_um2', 'soma_solidity', 'soma_circularity', 'soma_contrast',
    'dist_nn_um', 'nn_over_diameter', 'n_neighbours_in_disk',
    'n_neighbours_2r', 'border_headroom', 'touches_border',
    'own_frac_disk', 'own_frac_ann', 'fg_frac_disk', 'fg_frac_ann',
    'fg_area_over_target', 'own_fg_area_over_target',
    'int_disk_rel', 'int_ann_rel', 'int_drop',
    'n_fg_components_disk', 'largest_fg_frac_disk',
    'neighbours_within_25um', 'mean_dist_3_nearest_um',
]


def soma_feature_rows(gray, centroid, polygon, others, pixel_size_um, areas):
    """Features for one cell at every candidate target area -> (n_areas, n_feat).

    The reviewer is not judging the mask so much as the room the cell had: a
    mask is rejected when the growth ran past the cell into a neighbour, a
    vessel, or plain background. These measure that directly --

      * `own_frac_*`   what share of the footprint is closer to THIS soma than
                       to any other, i.e. how much of it is territory the cell
                       can defend. This is the single feature that states "it
                       bled into the neighbour".
      * `fg_frac_*`    how much of the footprint is above this cell's own
                       background, i.e. whether there is stain there at all.
      * `int_drop`     brightness at the growth front relative to the middle;
                       a front that has left the cell is dim.
      * `n_fg_components_disk` fragmentation, the other common reject reason.

    Everything is measured at the radius of a disk of the candidate area
    (`r_eq`), which is where the mask boundary sits for a compact cell and is
    close enough for a crowded one -- the point is to describe the neighbourhood
    at that scale, not to predict the exact mask.

    `others` are the OTHER soma centroids in the same image, (row, col) in image
    coordinates. `areas` are candidate target areas in um^2, ascending.
    """
    px = float(pixel_size_um)
    areas = [float(a) for a in areas]
    a_max = max(areas) if areas else 1.0
    r_max_px = np.sqrt(max(a_max, 1e-6) / np.pi) / px
    half = int(np.ceil(r_max_px * 1.6)) + 2

    H, W = gray.shape[:2]
    cy, cx = float(centroid[0]), float(centroid[1])
    y1, y2 = int(max(0, np.floor(cy) - half)), int(min(H, np.floor(cy) + half + 1))
    x1, x2 = int(max(0, np.floor(cx) - half)), int(min(W, np.floor(cx) + half + 1))
    p = gray[y1:y2, x1:x2].astype(np.float64)
    if p.size == 0:
        return np.zeros((len(areas), len(MASK_QA_FEATURES)), dtype=np.float32)

    ly, lx = cy - y1, cx - x1
    yy, xx = np.ogrid[:p.shape[0], :p.shape[1]]
    rad_um = np.hypot(yy - ly, xx - lx) * px

    # --- the cell itself -------------------------------------------------
    soma = polygon_mask(polygon, p.shape, offset=(y1, x1))
    if not soma.any():
        soma = rad_um <= max(px, 3.0 * px)
    if polygon is not None and len(polygon) >= 3:
        ar_px, per_px = poly_area_perimeter(polygon)
        soma_area_um2 = ar_px * px * px
        hull = convex_area(polygon)
        solidity = float(ar_px / hull) if hull else 1.0
        circ = float(4 * np.pi * ar_px / (per_px ** 2)) if per_px > 0 else 1.0
    else:
        soma_area_um2 = float(soma.sum()) * px * px
        solidity, circ = 1.0, 1.0

    # Background and this cell's own brightness. Both are LOCAL: staining and
    # exposure differ between images, and a threshold in raw units would mean
    # something different on each one.
    bg = float(np.percentile(p, 20))
    soma_level = float(np.median(p[soma])) if soma.any() else bg
    span = max(soma_level - bg, 1e-6)
    fg = p > (bg + 0.25 * span)
    hi = float(np.percentile(p, 99.5))
    contrast = float(span / max(hi - bg, 1e-6))

    # --- the neighbours --------------------------------------------------
    # `own` is this cell's Voronoi territory: pixels nearer to this soma than to
    # any other. Mask growth in MMPS is competitive, so this is close to the
    # boundary the growth itself will hit.
    own = np.ones(p.shape, dtype=bool)
    d_nn = np.inf
    nb, all_d = [], []
    for oc in (others or []):
        oy, ox = float(oc[0]), float(oc[1])
        d = np.hypot(oy - cy, ox - cx) * px
        if d < 1e-6:                                   # this same soma
            continue
        d_nn = min(d_nn, d)
        all_d.append(d)
        if d < r_max_px * px * 3.0:                    # only ones that can matter
            nb.append((oy - y1, ox - x1, d))
    for (ny, nx, _d) in nb:
        own &= (np.hypot(yy - ly, xx - lx) <= np.hypot(yy - ny, xx - nx))
    if not np.isfinite(d_nn):
        d_nn = float(max(H, W)) * px

    d_border_um = min(cy, cx, H - 1 - cy, W - 1 - cx) * px
    # Crowding, measured around THIS cell. An image-wide density would be one
    # number per image, and a forest handed 20 images' worth of those learns to
    # recognise the image and reproduce its habits -- which is the failure the
    # by-image split exists to expose, so do not hand it the feature.
    near = sorted(all_d)
    n_25 = float(sum(1 for d in near if d <= 25.0))
    d3 = float(np.mean(near[:3])) if near else float(max(H, W)) * px

    rows = []
    for a in areas:
        r_eq = float(np.sqrt(max(a, 1e-6) / np.pi))
        disk = rad_um <= r_eq
        ann = (rad_um > r_eq * 0.85) & (rad_um <= r_eq * 1.15)
        if not disk.any():
            disk = rad_um <= px
        if not ann.any():
            ann = disk

        fg_disk = fg & disk
        n_fg = int(fg_disk.sum())
        lab, ncomp = ndi.label(fg_disk)
        if ncomp:
            sizes = np.bincount(lab.ravel())[1:]
            largest = float(sizes.max()) / max(n_fg, 1)
        else:
            largest = 0.0
        int_disk = (float(p[disk].mean()) - bg) / span
        int_ann = (float(p[ann].mean()) - bg) / span

        rows.append([
            a,
            r_eq,
            a / a_max,
            a / max(soma_area_um2, 1e-6),
            soma_area_um2,
            solidity,
            circ,
            contrast,
            d_nn,
            d_nn / max(2.0 * r_eq, 1e-6),
            float(sum(1 for (_y, _x, d) in nb if d <= r_eq)),
            float(sum(1 for (_y, _x, d) in nb if d <= 2.0 * r_eq)),
            d_border_um / max(r_eq, 1e-6),
            float(d_border_um < r_eq),
            float(own[disk].mean()),
            float(own[ann].mean()),
            float(fg_disk.sum()) / max(float(disk.sum()), 1.0),
            float((fg & ann).sum()) / max(float(ann.sum()), 1.0),
            (n_fg * px * px) / max(a, 1e-6),
            (float((fg & own & disk).sum()) * px * px) / max(a, 1e-6),
            int_disk,
            int_ann,
            int_ann / (abs(int_disk) + 1e-6),
            float(ncomp),
            largest,
            n_25,
            d3,
        ])
    return np.asarray(rows, dtype=np.float32)


# ----------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------
def read_features_csv(paths):
    """Whole-object mask features captured by MMPS at generation time.

    Keyed (image, soma_id, target_area). Optional -- MMPS only started writing
    these after the model was scoped, so any session older than that has none,
    and the model must work without them.
    """
    table, cols, dups = {}, [], set()
    for path in paths or []:
        with open(path, newline='') as fh:
            rd = csv.DictReader(fh)
            # flag_* columns record decisions MMPS made by rule, not
            # measurements: flag_border_rejected predicts the label exactly and
            # would teach the model a rule it already has. They are carried in
            # the file so a person can exclude those masks; they are never fed
            # to the forest.
            keys = [c for c in (rd.fieldnames or [])
                    if c not in ('image', 'soma_id', 'target_area_um2')
                    and not c.startswith('flag_')]
            if not cols:
                cols = keys
            elif keys != cols:
                sys.exit(f"{path}: columns differ from the first features file")
            for row in rd:
                try:
                    k = (row['image'], row['soma_id'],
                         float(row['target_area_um2']))
                except (KeyError, ValueError):
                    continue
                table[k] = [float(row[c] or 0.0) for c in cols]
                if str(row.get('flag_duplicate', '')).strip() in ('1', 'True',
                                                                  'true'):
                    dups.add(k)
    return table, cols, dups


def build_dataset(records, prefer='processed', channel=None, limit=None,
                  feat_table=None, feat_cols=None, verbose=True):
    """-> X, y, groups, meta rows. One row per (cell, candidate target area)."""
    X, y, groups, meta = [], [], [], []
    used_cells = skipped_cells = 0
    n_feat_hits = 0
    for rec in records:
        gray, which = open_image(rec, prefer, channel)
        if gray is None:
            if verbose:
                print(f"  [skip] {rec['image_name']}: {which}")
            skipped_cells += len(rec['somas'])
            continue
        px = rec['pixel_size'] or 1.0
        centroids = rec['all_centroids'] or [s['centroid'] for s in rec['somas']]
        if verbose:
            print(f"  {rec['image_name'][:58]:58s} {which:9s} "
                  f"{len(rec['somas']):4d} cells")
        for s in rec['somas']:
            if limit is not None and used_cells >= limit:
                break
            if not s.get('centroid'):
                continue
            rows = soma_feature_rows(gray, s['centroid'], s.get('polygon'),
                                     centroids, px, s['areas'])
            for i, a in enumerate(s['areas']):
                extra = []
                if feat_cols:
                    hit = (feat_table or {}).get(
                        (rec['image_name'], s['soma_id'], float(a)))
                    if hit is None:
                        extra = [np.nan] * len(feat_cols)
                    else:
                        extra = hit
                        n_feat_hits += 1
                X.append(np.concatenate([rows[i], np.asarray(extra,
                                                             dtype=np.float32)])
                         if extra else rows[i])
                y.append(1 if s['labels'][i] else 0)
                groups.append(rec['image_name'])
                meta.append((rec['image_name'], s['soma_id'], float(a),
                             s['cutoff']))
            used_cells += 1
        if limit is not None and used_cells >= limit:
            break
    if not X:
        tried = next((r.get('processed_path') or r.get('raw_path')
                      for r in records if r.get('processed_path')
                      or r.get('raw_path')), None)
        hint = ""
        if tried:
            root = os.sep.join(tried.split(os.sep)[:3])
            hint = (f"\n\nThe sessions point at e.g.\n  {tried}\n"
                    f"If that volume is mounted elsewhere, add\n"
                    f"  --root-map \"{root}=/where/it/is/now\"")
        sys.exit("No usable rows: none of the images could be read." + hint)
    if feat_cols and verbose:
        print(f"  whole-object features joined on {n_feat_hits:,} of "
              f"{len(X):,} rows")
    if skipped_cells and verbose:
        print(f"  {skipped_cells:,} cells skipped for want of their image")
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int8),
            np.asarray(groups), meta)


# ----------------------------------------------------------------------
# decoding scores into the decision the user actually makes
# ----------------------------------------------------------------------
def decode_cutoff(areas, probs, cut=0.5):
    """-> (cutoff_um2, confidence). Longest run of approvals from the smallest.

    The labels only ever take this shape, so the prediction is made to take it
    too. Reading each probability independently would let the model propose
    "approve 50, reject 100, approve 150", which the QA screen cannot express
    and the user never chose.

    Confidence is how decisively the boundary sits where it does: how far the
    last approved and first rejected areas are from the cut. A cell whose
    probabilities step cleanly from 0.9 to 0.1 scores near 1; one that drifts
    across the cut scores near 0, and is exactly the cell worth looking at.
    """
    areas = list(areas)
    probs = list(probs)
    k = 0
    while k < len(probs) and probs[k] >= cut:
        k += 1
    cutoff = areas[k - 1] if k else 0.0
    below = probs[k - 1] if k else None            # last approved
    above = probs[k] if k < len(probs) else None   # first rejected
    parts = []
    if below is not None:
        parts.append(min(max((below - cut) / max(cut, 1e-6), 0.0), 1.0))
    if above is not None:
        parts.append(min(max((cut - above) / max(1.0 - cut, 1e-6), 0.0), 1.0))
    conf = float(np.mean(parts)) if parts else 0.0
    return float(cutoff), conf


def cells_from_rows(meta, prob, cut=0.5):
    """Group per-mask probabilities back into one prediction per cell."""
    order = {}
    for i, (img, sid, a, true_cut) in enumerate(meta):
        order.setdefault((img, sid), []).append((a, prob[i], true_cut))
    out = []
    for (img, sid), items in order.items():
        items.sort(key=lambda t: t[0])
        areas = [t[0] for t in items]
        probs = [t[1] for t in items]
        pred, conf = decode_cutoff(areas, probs, cut)
        out.append(dict(image=img, soma_id=sid, areas=areas,
                        true=items[0][2], pred=pred, conf=conf))
    return out


# ----------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------
def _auc(y, p):
    """ROC AUC without pulling in another sklearn import path."""
    y = np.asarray(y)
    if y.min() == y.max():
        return float('nan')
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def score_cells(cells, step):
    """-> dict of the numbers that decide whether this is worth using."""
    if not cells:
        return {}
    err = np.array([abs(c['pred'] - c['true']) for c in cells]) / max(step, 1e-6)
    return dict(n=len(cells),
                exact=float(np.mean(err < 0.5)),
                within1=float(np.mean(err < 1.5)),
                median_err=float(np.median(err)),
                mean_err=float(np.mean(err)))


def report_split(name, y, prob, cells, step):
    acc = float(np.mean((prob >= 0.5).astype(int) == y))
    pos, neg = y == 1, y == 0
    bal = float(0.5 * (np.mean(prob[pos] >= 0.5) + np.mean(prob[neg] < 0.5))) \
        if pos.any() and neg.any() else float('nan')
    s = score_cells(cells, step)
    print(f"  {name:12s} masks: acc {acc:.3f}  balanced {bal:.3f}  "
          f"AUC {_auc(y, prob):.3f}")
    print(f"  {'':12s} cells: exact cutoff {100 * s['exact']:.0f}%   "
          f"within one step {100 * s['within1']:.0f}%   "
          f"median error {s['median_err']:.1f} steps")
    return s


def baselines(cells, all_cutoffs, step):
    """What you get for free, so the model has something to beat.

    Two of them: always predict the commonest cutoff, and always predict the
    largest area (which is what "approve all" does). A model that does not
    clear both is not learning the cell, it is learning the study.
    """
    vals, counts = np.unique(all_cutoffs, return_counts=True)
    common = float(vals[int(np.argmax(counts))])
    out = {}
    for nm, v in (('commonest cutoff', common), ('always the largest',
                                                 float(max(all_cutoffs)))):
        err = np.array([abs(v - c['true']) for c in cells]) / max(step, 1e-6)
        out[nm] = (float(np.mean(err < 0.5)), float(np.mean(err < 1.5)), v)
    return out


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', nargs='*', default=[],
                    help='.mmps_session files holding QA decisions')
    ap.add_argument('--session-root', nargs='*', default=[],
                    help='folders to scan recursively for .mmps_session files')
    ap.add_argument('--root-map', nargs='*', default=[],
                    help='OLD=NEW prefix rewrites for paths stored in sessions')
    ap.add_argument('--image', choices=['processed', 'raw'], default='processed',
                    help='which image to measure (default processed: it is the '
                         'single grey plane the mask growth itself ran on)')
    ap.add_argument('--channel', type=int, default=None,
                    help='1-based stain channel, for raw multi-channel images. '
                         'Name it -- the brightest-channel guess picks '
                         'different channels on different images.')
    ap.add_argument('--features-csv', nargs='*', default=[],
                    help='mask_qa_features.csv files written by MMPS at mask '
                         'generation (optional whole-object features)')
    ap.add_argument('--drop-saturated', action='store_true',
                    help='drop cells whose cutoff is the largest candidate '
                         'area. Use when a session was finished with "Approve '
                         'All Remaining", which writes that label without '
                         'anyone having looked.')
    ap.add_argument('--limit', type=int, default=None, help='use at most N cells')
    ap.add_argument('--trees', type=int, default=400)
    ap.add_argument('--min-leaf', type=int, default=8)
    ap.add_argument('--folds', type=int, default=5,
                    help='cross-validation folds, split BY IMAGE')
    ap.add_argument('--cut', type=float, default=0.5,
                    help='probability above which a mask counts as approved')
    ap.add_argument('--out', default='mask_qa_model.joblib')
    a = ap.parse_args()

    mappings = []
    for m in a.root_map:
        if '=' not in m:
            sys.exit(f"--root-map wants OLD=NEW, got {m!r}")
        old, new = m.split('=', 1)
        mappings.append((old, new))

    sessions = find_sessions(a.session, a.session_root)
    if not sessions:
        sys.exit("No .mmps_session files given. Use --session or --session-root.")

    feat_table, feat_cols, dup_keys = read_features_csv(a.features_csv)
    if feat_cols:
        print(f"Whole-object features: {len(feat_cols)} columns, "
              f"{len(feat_table):,} masks, {len(dup_keys):,} of them "
              f"auto-rejected duplicates (excluded)\n")

    print(f"Sessions ({len(sessions)}):")
    records = []
    for sp in sessions:
        try:
            recs = read_session(sp, mappings, dup_keys)
        except Exception as e:
            print(f"  {os.path.basename(sp)}: unreadable ({e}) — skipped")
            continue
        cells = sum(len(r['somas']) for r in recs)
        print(f"  {os.path.basename(sp):40s} {len(recs):3d} images  "
              f"{cells:5d} reviewed cells")
        records += recs
    if not records:
        sys.exit("No reviewed cells found in those sessions.")

    # A cell whose approvals are not a contiguous run cannot be represented as a
    # cutoff. Say how many were dropped rather than let them vanish quietly.
    total_qa = kept = 0
    for sp in sessions:
        try:
            sess = json.load(open(sp))
        except Exception:
            continue
        for name, img in (sess.get('images') or {}).items():
            per = {}
            for e in (img.get('mask_qa_state') or []):
                ar = e.get('area_um2', e.get('target_area_um2'))
                if ar is not None and (name, e.get('soma_id'),
                                       float(ar)) in dup_keys:
                    continue
                per.setdefault(e.get('soma_id'), []).append(e)
            total_qa += len(per)
            kept += sum(1 for v in per.values() if _cutoff_from_qa(v))
    if total_qa:
        print(f"\n{kept:,} of {total_qa:,} reviewed cells are usable "
              f"({total_qa - kept} had non-contiguous approvals or no outline)")

    sat = sum(r.get('saturated', 0) for r in records)
    tail = sum(r.get('bulk_tail', 0) for r in records)
    n_cells = sum(len(r['somas']) for r in records)
    if sat:
        print(f"\n{sat:,} of {n_cells:,} cells ({100 * sat / n_cells:.0f}%) "
              f"approve every size.")
        if tail > 0.25 * sat:
            print(f"  {tail:,} of those sit in an unbroken run at the end of "
                  f"an image's queue -- the signature of \"Approve All "
                  f"Remaining\", not of review. Re-run with --drop-saturated.")
        else:
            print(f"  Only {tail:,} sit at the end of an image's queue, so "
                  f"these are judgements rather than a bulk approval.")
    if a.drop_saturated:
        for r in records:
            a_max = max((max(s['areas']) for s in r['somas']), default=0.0)
            r['somas'] = [s for s in r['somas'] if not (s['cutoff'] >= a_max > 0)]
        records = [r for r in records if r['somas']]
        kept_now = sum(len(r['somas']) for r in records)
        print(f"  --drop-saturated: {kept_now:,} cells left")
        if not records:
            sys.exit("Nothing left after --drop-saturated.")

    print("\nBuilding features…")
    X, y, groups, meta = build_dataset(records, a.image, a.channel, a.limit,
                                       feat_table, feat_cols)
    names = list(MASK_QA_FEATURES) + list(feat_cols or [])
    imgs = np.unique(groups)
    cutoffs = np.array([m[3] for m in meta])
    areas_all = sorted({m[2] for m in meta})
    step = float(np.median(np.diff(areas_all))) if len(areas_all) > 1 else 50.0
    print(f"\n{len(X):,} masks   {len(set((m[0], m[1]) for m in meta)):,} cells "
          f"  {len(imgs)} images   {len(names)} features")
    print(f"  approved {100 * y.mean():.0f}% of masks   "
          f"candidate areas {areas_all[0]:.0f}-{areas_all[-1]:.0f} µm² "
          f"in steps of {step:.0f}")
    if len(imgs) < 3:
        sys.exit(f"Only {len(imgs)} image(s). Splitting by image needs at "
                 f"least 3 -- with fewer, a held-out number would be a "
                 f"single image's luck.")

    # Split BY IMAGE, always. Masks from one image share illumination, staining
    # and cell density; splitting by mask lets the model memorise the image and
    # score well having learned nothing that transfers.
    n_folds = int(min(a.folds, len(imgs)))
    print(f"\nCross-validating over {n_folds} folds, split by image…")
    oof = np.zeros(len(X), dtype=np.float64)
    in_sample = np.zeros(len(X), dtype=np.float64)
    gkf = GroupKFold(n_splits=n_folds)
    for k, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        clf = RandomForestClassifier(
            n_estimators=a.trees, min_samples_leaf=a.min_leaf,
            class_weight='balanced_subsample', n_jobs=-1, random_state=0)
        Xtr = np.nan_to_num(X[tr], nan=0.0, posinf=0.0, neginf=0.0)
        Xte = np.nan_to_num(X[te], nan=0.0, posinf=0.0, neginf=0.0)
        clf.fit(Xtr, y[tr])
        oof[te] = clf.predict_proba(Xte)[:, 1]
        in_sample[tr] = np.maximum(in_sample[tr], clf.predict_proba(Xtr)[:, 1])
        held = np.unique(groups[te])
        print(f"  fold {k}: trained on {len(tr):,} masks, tested on "
              f"{len(te):,} from {len(held)} image(s)")

    print("\nHeld out (every cell scored by a model that never saw its image):")
    cells_oof = cells_from_rows(meta, oof, a.cut)
    s_oof = report_split('held-out', y, oof, cells_oof, step)
    cells_in = cells_from_rows(meta, in_sample, a.cut)
    print()
    s_in = report_split('training', y, in_sample, cells_in, step)

    gap = s_in['exact'] - s_oof['exact']
    print()
    if gap > 0.15:
        print(f"  Training beats held-out by {100 * gap:.0f} points -> it is "
              f"fitting each image, not the cell. More images will help more "
              f"than more features.")
    elif s_oof['exact'] < 0.35:
        print("  Training and held-out agree, and both are low -> the features "
              "or the labels are the limit, not transfer.")
    else:
        print("  Training and held-out agree -> what it learned transfers "
              "between images.")

    print("\nAgainst the free answers:")
    base = baselines(cells_oof, cutoffs, step)
    print(f"  {'rule':22s} {'exact':>7s} {'within 1':>9s}")
    for nm, (ex, w1, v) in base.items():
        print(f"  {nm + f' ({v:.0f} µm²)':22s} {100 * ex:6.0f}% {100 * w1:8.0f}%")
    print(f"  {'model':22s} {100 * s_oof['exact']:6.0f}% "
          f"{100 * s_oof['within1']:8.0f}%")

    print("\nWhat it is using:")
    full = RandomForestClassifier(
        n_estimators=a.trees, min_samples_leaf=a.min_leaf,
        class_weight='balanced_subsample', n_jobs=-1, random_state=0)
    full.fit(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), y)
    imp = full.feature_importances_
    for i in np.argsort(-imp)[:10]:
        print(f"  {names[i]:28s} {imp[i]:.3f}")

    # Confidence triage. The threshold has to be an ABSOLUTE value calibrated
    # here and carried in the model: ranking one batch and taking its top half
    # would auto-accept half of any batch, however badly that batch went.
    conf = np.array([c['conf'] for c in cells_oof])
    err = np.array([abs(c['pred'] - c['true']) for c in cells_oof]) / step
    conf_cal = {}
    if len(conf) >= 40:
        print("\nConfidence triage (cells sorted by how decisively the cutoff "
              "sits):")
        order = np.argsort(-conf)
        q = len(order) // 4
        for k in range(4):
            sl = order[k * q:(k + 1) * q] if k < 3 else order[3 * q:]
            print(f"  {['most', '2nd', '3rd', 'least'][k]:>5s} confident "
                  f"quarter: exact {100 * np.mean(err[sl] < 0.5):3.0f}%   "
                  f"within one step {100 * np.mean(err[sl] < 1.5):3.0f}%")
        for frac in (0.3, 0.5):
            thr = float(np.quantile(conf, 1.0 - frac))
            keep = conf >= thr
            if not keep.any():
                continue
            conf_cal[f'top{int(frac * 100)}'] = dict(
                threshold=thr,
                purity=float(np.mean(err[keep] < 0.5)),
                purity_within1=float(np.mean(err[keep] < 1.5)),
                covers=float(np.mean(keep)))
            print(f"  auto-accept the top {100 * frac:.0f}% "
                  f"(confidence >= {thr:.3f}): "
                  f"{100 * np.mean(err[keep] < 0.5):3.0f}% land on the exact "
                  f"size, {100 * np.mean(err[keep] < 1.5):3.0f}% within one step")
        # Does confidence order the work at all? If it does not, ordering the
        # queue by it is theatre and auto-accept should stay off. Only ask the
        # question when there is room for an answer: at 98% exact overall the
        # confident cells cannot be much better than the rest, and that says
        # nothing about the score.
        lo, hi = conf <= np.median(conf), conf > np.median(conf)
        lift = float(np.mean(err[hi] < 0.5) - np.mean(err[lo] < 0.5))
        headroom = float(np.mean(err < 0.5)) < 0.9
        conf_cal['ranks'] = bool(lift > 0.05 or not headroom)
        if headroom and lift <= 0.05:
            print("  Confidence does not separate right from wrong here "
                  "(+%.0f points). It is not worth ordering the queue by, and "
                  "nothing should be accepted on it." % (100 * lift))

    meta_out = dict(
        features=names,
        base_features=list(MASK_QA_FEATURES),
        extra_features=list(feat_cols or []),
        image=a.image, channel=a.channel, cut=a.cut,
        areas=areas_all, step=step,
        conf_cal=conf_cal,
        n_masks=int(len(X)), n_cells=int(len(cells_oof)), n_images=int(len(imgs)),
        heldout=dict(exact=s_oof['exact'], within1=s_oof['within1'],
                     median_err_steps=s_oof['median_err']),
        pixel_size_um=float(np.median([r['pixel_size'] or 0 for r in records])),
    )
    try:
        joblib.dump({'model': full, 'meta': meta_out}, a.out, compress=3)
    except OSError as e:
        sys.exit(f"\nCould not write {a.out}: {e}\n"
                 f"Free some disk space, or retrain with a larger --min-leaf "
                 f"to shrink the forest.")
    print(f"\nSaved model -> {a.out}  "
          f"({os.path.getsize(a.out) / 1e6:.0f} MB compressed)")

    print("\nHow to read this, for the QA workflow:")
    print("  Ordering the queue least-confident-first costs nothing and is")
    print("  worth doing at almost any accuracy -- it front-loads the cells")
    print("  that need a person and leaves the easy ones to the end.")
    print("  Pre-selecting a size is worth it when a wrong pick costs one")
    print("  click to correct, so roughly:")
    print("    exact above 60%    most cells are one confirm and done")
    print("    30 - 60%           still a real saving over choosing every size")
    print("    under 20%          leave it ordering the queue only")
    print("  Accepting cells UNREVIEWED needs the triage numbers, not the")
    print("  average: only the confident band, and only if its exact rate is")
    print("  high enough that you would not have caught those cells anyway.")


if __name__ == '__main__':
    main()
