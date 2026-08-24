#!/usr/bin/env python3
"""train_soma_model.py — learn soma outlining from the outlines you already accepted.

Every threshold-based detector we tried failed because the soma boundary in these
images is genuinely ambiguous: the soma is barely brighter than its own
processes, so no intensity cut separates them. Your accepted outlines, however,
DEFINE the boundary. This trains a model to reproduce that judgement.

Approach (Ilastik-style pixel classifier):
  * for every accepted soma, crop a patch around it in the matching image
  * compute multi-scale pixel features — smoothing, gradient magnitude, and
    Hessian eigenvalues (which distinguish BLOB-like from TUBE-like structure,
    i.e. soma vs process, without any hand-set threshold)
  * train a random forest: soma interior vs everything else
  * validate on IMAGES HELD OUT of training, reporting IoU against your outlines

Only needs numpy / scipy / scikit-image / scikit-learn / tifffile — no GPU.

USAGE
    python3 train_soma_model.py --root "/Volumes/Expansion/CCI Young Rat NL1 Study Data/Raw Data/TREM2 IBA1 Cortex CCI 63x"

  It walks each timepoint folder (1d, 3d, 7d, 28d), pairing
      <timepoint>/Output/somas/<image>_soma_<row>_<col>_soma.tif
  with
      <timepoint>/Image Directory/<image>.tif

  Start small to check it works, then run the lot:
    --limit 300              use only 300 somas
    --timepoints 1d 28d      only these folders
    --pixel-size 0.1046      µm/px (default 0.1046)
    --out soma_model.joblib  where to save the trained model

The saved model is what MMPS will load for auto-outlining.
"""

import os
import re
import sys
import glob
import argparse
import numpy as np

try:
    import tifffile
except ImportError:
    sys.exit("Missing tifffile.  pip install tifffile")
try:
    from scipy import ndimage
except ImportError:
    sys.exit("Missing scipy.  pip install scipy")
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupShuffleSplit
    import joblib
except ImportError:
    sys.exit("Missing scikit-learn.  pip install scikit-learn joblib")

MASK_RE = re.compile(r'^(?P<base>.+)_soma_(?P<r>\d+)_(?P<c>\d+)_soma\.tiff?$', re.I)
IMG_EXTS = ('.tif', '.tiff', '.TIF', '.TIFF')


# ----------------------------------------------------------------------
# data pairing
# ----------------------------------------------------------------------
def _norm(stem):
    """Collapse case and punctuation so 'YR 13-0 1d NL-1' == 'Yr_13-0_1d_Nl-1'."""
    return re.sub(r'[^a-z0-9]+', '_', stem.lower()).strip('_')


def _prefix_match(norm_index, key):
    """Last resort: an image whose normalised stem starts with the mask base
    (MMPS sometimes drops a trailing processing suffix). Only accept a unique
    hit -- an ambiguous match would silently train on the wrong pixels."""
    hits = [p for k, p in norm_index.items()
            if k.startswith(key) or key.startswith(k)]
    return hits[0] if len(hits) == 1 else None


def find_pairs(root, timepoints, limit=None):
    """Yield (mask_path, image_path, row, col, timepoint)."""
    pairs = []
    for tp in timepoints:
        somas_dir = os.path.join(root, tp, 'Output', 'somas')
        img_dir = os.path.join(root, tp, 'Image Directory')
        if not os.path.isdir(somas_dir):
            print(f"  [{tp}] no somas folder at {somas_dir} — skipped")
            continue
        if not os.path.isdir(img_dir):
            print(f"  [{tp}] no Image Directory at {img_dir} — skipped")
            continue
        # Index the image folder once. MMPS sanitises the image name when it
        # writes a mask (spaces and punctuation become underscores), so an exact
        # stem match often fails on real data -- index a normalised key too.
        index, norm_index = {}, {}
        for p in glob.glob(os.path.join(img_dir, '*')):
            bn = os.path.basename(p)
            if bn.startswith('._') or not bn.lower().endswith(('.tif', '.tiff')):
                continue
            stem = os.path.splitext(bn)[0]
            index.setdefault(stem.lower(), p)
            norm_index.setdefault(_norm(stem), p)
        found = missing = 0
        unmatched = []
        for mp in sorted(glob.glob(os.path.join(somas_dir, '*_soma.tif*'))):
            name = os.path.basename(mp)
            if name.startswith('._'):
                continue
            m = MASK_RE.match(name)
            if not m:
                continue
            base = m.group('base')
            ip = (index.get(base.lower())
                  or norm_index.get(_norm(base))
                  or _prefix_match(norm_index, _norm(base)))
            if ip is None:
                missing += 1
                if len(unmatched) < 3:
                    unmatched.append(base)
                continue
            pairs.append((mp, ip, int(m.group('r')), int(m.group('c')), tp))
            found += 1
        print(f"  [{tp}] {found} somas paired"
              + (f", {missing} had no matching image" if missing else ""))
        if missing and not found:
            print(f"        no image matched, e.g. {unmatched!r}")
            print(f"        Image Directory holds: "
                  f"{sorted(os.path.basename(p) for p in index.values())[:3]!r}")
    if limit:
        rng = np.random.RandomState(0)
        idx = rng.permutation(len(pairs))[:limit]
        pairs = [pairs[i] for i in sorted(idx)]
    return pairs


def describe_channels(path):
    """Report an image's channel layout so the right one can be named."""
    a = np.squeeze(np.asarray(tifffile.imread(path)))
    out = [f"    {os.path.basename(path)}", f"    shape {a.shape}  dtype {a.dtype}"]
    if a.ndim == 3 and a.shape[int(np.argmin(a.shape))] <= 8:
        b = np.moveaxis(a, int(np.argmin(a.shape)), -1)
        for i in range(b.shape[2]):
            ch = b[:, :, i].astype(np.float64)
            out.append(f"    channel {i + 1}: mean {ch.mean():8.2f}   "
                       f"max {ch.max():7.0f}   above-zero {100 * (ch > 0).mean():5.1f}%")
    else:
        out.append("    single channel")
    return "\n".join(out)


def load_gray(path, channel=None):
    """Load an image as 2D float, keeping only the stain we are training on.

    `channel` is 1-based. Leaving it None falls back to whichever channel carries
    the most total signal, which is only a guess -- if the microglia stain is not
    the brightest channel that silently trains the model on the wrong structure,
    so name the channel explicitly whenever the images are multi-channel.
    """
    a = np.squeeze(np.asarray(tifffile.imread(path)))
    if a.ndim == 3:
        # channel axis = the small one; otherwise treat as a stack
        ax = int(np.argmin(a.shape))
        if a.shape[ax] <= 8:
            a = np.moveaxis(a, ax, -1)
            if channel is not None:
                if not 1 <= channel <= a.shape[2]:
                    raise ValueError(f"--channel {channel} but this image has "
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


# ----------------------------------------------------------------------
# features
# ----------------------------------------------------------------------
def pixel_features(patch, scales=(1.0, 2.0, 4.0, 8.0), center=None):
    """Multi-scale per-pixel features -> (n_pixels, n_features).

    Hessian eigenvalues are the important ones: for a BLOB both eigenvalues are
    large and similar, for a TUBE (process) one is large and one near zero. That
    is the "prefer blobs, penalise branching" prior, learned rather than tuned.

    `center` is the (row, col) of the soma inside the patch -- the point the user
    clicked. Texture filters alone cannot tell the soma edge from a bright
    process 60 px away, so without this the classifier predicts the right AMOUNT
    of soma in the wrong PLACE. Two anchors are derived from it: how far a pixel
    is from the click, and how bright it is relative to this cell's own core
    (which also makes brightness comparable between images).
    """
    p = patch.astype(np.float64)
    lo, hi = np.percentile(p, 1), np.percentile(p, 99.5)
    p = (p - lo) / (hi - lo) if hi > lo else p * 0.0
    feats = [p]
    if center is not None:
        cy, cx = center
        yy, xx = np.ogrid[:p.shape[0], :p.shape[1]]
        rho = np.hypot(yy - cy, xx - cx) / max(p.shape) * 2.0
        rho = np.broadcast_to(rho, p.shape).astype(np.float64)
        # brightness of this cell's core, as a per-cell reference level
        core = p[max(0, int(cy) - 3):int(cy) + 4,
                 max(0, int(cx) - 3):int(cx) + 4]
        cval = float(np.median(core)) if core.size else 0.0
        feats += [rho, p - cval, p / (cval + 1e-3)]
    for s in scales:
        g = ndimage.gaussian_filter(p, s)
        feats.append(g)
        gy, gx = np.gradient(g)
        feats.append(np.hypot(gy, gx))                       # edge strength
        gyy = ndimage.gaussian_filter(p, s, order=[2, 0])
        gxx = ndimage.gaussian_filter(p, s, order=[0, 2])
        gxy = ndimage.gaussian_filter(p, s, order=[1, 1])
        tr = gxx + gyy
        det = gxx * gyy - gxy * gxy
        disc = np.sqrt(np.maximum((tr / 2.0) ** 2 - det, 0))
        l1, l2 = tr / 2.0 + disc, tr / 2.0 - disc            # Hessian eigenvalues
        feats += [l1, l2, np.abs(l1) - np.abs(l2)]           # blob vs tube
        feats.append(g - ndimage.gaussian_filter(p, s * 2))  # difference of gaussians
    return np.stack([f.ravel() for f in feats], axis=1).astype(np.float32)


FEATURE_SCALES = (1.0, 2.0, 4.0, 8.0)


def radial_contour(prob, center, cut, n_angles=180, smooth=9):
    """Turn a probability map into a smooth star-convex outline around `center`.

    A hand-drawn soma outline is a smooth closed contour; a pixel classifier
    emits ragged, speckled output. Thresholding it therefore loses several
    pixels of boundary accuracy for reasons that have nothing to do with how
    well the model located the soma. This casts a ray at each angle, takes the
    first crossing below `cut` as the boundary, median-filters the resulting
    radius profile circularly, and fills the polygon -- which enforces exactly
    the properties the accepted outlines have: one blob, no holes, no spurs,
    smooth edge.
    """
    cy, cx = float(center[0]), float(center[1])
    H, W = prob.shape
    R = int(max(H, W))
    rs = np.arange(0.0, R, 0.5)
    ang = np.linspace(0.0, 2 * np.pi, n_angles, endpoint=False)
    yy = cy + rs[None, :] * np.sin(ang)[:, None]
    xx = cx + rs[None, :] * np.cos(ang)[:, None]
    samp = ndimage.map_coordinates(prob, [yy.ravel(), xx.ravel()],
                                   order=1, mode='constant', cval=0.0)
    samp = samp.reshape(n_angles, len(rs))
    inside = samp >= cut
    # first radius at which the ray leaves the soma
    leaves = np.argmin(inside, axis=1)
    leaves[inside.all(axis=1)] = len(rs) - 1
    bnd = rs[leaves]
    if smooth > 1:
        bnd = ndimage.median_filter(bnd, size=smooth, mode='wrap')
    if not np.any(bnd > 0):
        return None
    gy, gx = np.ogrid[:H, :W]
    a = np.arctan2(gy - cy, gx - cx) % (2 * np.pi)
    idx = np.minimum((a / (2 * np.pi) * n_angles).astype(int), n_angles - 1)
    rad = np.hypot(gy - cy, gx - cx)
    return rad <= bnd[idx]


def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (y * y + x * x) <= r * r


def patch_around(img, r, c, half):
    h, w = img.shape[:2]
    y1, y2 = max(0, r - half), min(h, r + half)
    x1, x2 = max(0, c - half), min(w, c + half)
    return img[y1:y2, x1:x2], y1, x1


# ----------------------------------------------------------------------
# training set
# ----------------------------------------------------------------------
def build_dataset(pairs, half, per_soma=600, verbose_every=100, channel=None,
                  use_click=False):
    X, y, groups = [], [], []
    cache_path, cache_img = None, None
    for i, (mp, ip, r, c, tp) in enumerate(pairs):
        try:
            if ip != cache_path:
                cache_img = load_gray(ip, channel)
                cache_path = ip
            img = cache_img
            mask = np.squeeze(np.asarray(tifffile.imread(mp))) > 0
            if mask.shape != img.shape:
                continue
            ys, xs = np.nonzero(mask)
            if len(ys) < 20:
                continue
            # Centre the patch the same way scoring and MMPS will. Training on
            # the outline's centroid while predicting from the recorded click
            # makes every centre-anchored feature mean something different at
            # the two ends -- the clicks here sit up to ~80% of a soma radius
            # away from the centroid, which is far too big a shift to absorb.
            if use_click:
                cr, cc = int(r), int(c)
            else:
                cr, cc = int(ys.mean()), int(xs.mean())
            patch, y1, x1 = patch_around(img, cr, cc, half)
            mpatch = mask[y1:y1 + patch.shape[0], x1:x1 + patch.shape[1]]
            if patch.size == 0 or mpatch.sum() < 20:
                continue
            F = pixel_features(patch, FEATURE_SCALES, center=(cr - y1, cc - x1))
            lab = mpatch.ravel().astype(np.uint8)
            pos = np.flatnonzero(lab == 1)
            neg = np.flatnonzero(lab == 0)
            if len(pos) < 10 or len(neg) < 10:
                continue
            rng = np.random.RandomState(i)
            npick = min(per_soma // 2, len(pos), len(neg))
            # Hard negatives: pixels just OUTSIDE the accepted outline -- the
            # emerging processes. Sampling negatives uniformly fills the set with
            # trivial far-background and leaves the decision boundary untrained
            # exactly where it has to be sharp, so bias towards the rim.
            band = np.logical_and(
                ndimage.binary_dilation(mpatch, iterations=max(2, half // 6)),
                ~mpatch).ravel()
            near = np.flatnonzero(np.logical_and(lab == 0, band))
            far = np.flatnonzero(np.logical_and(lab == 0, ~band))
            n_near = min(int(npick * 0.65), len(near))
            n_far = min(npick - n_near, len(far))
            neg_sel = np.concatenate([
                rng.choice(near, n_near, replace=False),
                rng.choice(far, n_far, replace=False)])
            if len(neg_sel) < 10:
                neg_sel = rng.choice(neg, min(npick, len(neg)), replace=False)
            sel = np.concatenate([rng.choice(pos, npick, replace=False), neg_sel])
            X.append(F[sel])
            y.append(lab[sel])
            groups += [os.path.basename(ip)] * len(sel)     # group by IMAGE
        except Exception as e:
            print(f"    skipped {os.path.basename(mp)}: {e}")
            continue
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"    {i + 1}/{len(pairs)} somas processed")
    if not X:
        return None, None, None
    return np.vstack(X), np.concatenate(y), np.array(groups)


# ----------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------
def mask_from_prob(prob, center, prob_cut=0.5, open_r=0, mode='cc'):
    """Turn a probability map into a single soma mask.

    Split out from predict_mask because the probability map depends on neither
    the cut nor the mode: the sweep computes it once per soma and calls this for
    every combination, instead of re-running the forest 20 times per cell.
    """
    ly, lx = center
    if mode == 'radial':
        return radial_contour(prob, (ly, lx), prob_cut)
    binm = prob >= prob_cut
    # Sever thin structures BEFORE picking the component, so a process still
    # attached to the soma is dropped with it rather than dragged along. The
    # radius severs anything narrower than ~2*open_r while leaving the soma
    # (tens of px across) intact.
    if open_r > 0:
        binm = ndimage.binary_opening(binm, _disk(open_r))
    lab, n = ndimage.label(binm)
    if n == 0:
        return None
    ly = min(max(int(ly), 0), prob.shape[0] - 1)
    lx = min(max(int(lx), 0), prob.shape[1] - 1)
    cid = lab[ly, lx]
    if cid == 0:                                   # click just outside — nearest
        ys, xs = np.nonzero(lab)
        if not len(ys):
            return None
        i = int(np.argmin((ys - ly) ** 2 + (xs - lx) ** 2))
        cid = lab[ys[i], xs[i]]
    return ndimage.binary_fill_holes(lab == cid)


def predict_mask(clf, img, r, c, half, prob_cut=0.5, open_r=0, mode='cc'):
    """Outline one soma. This is the entry point MMPS will call."""
    patch, y1, x1 = patch_around(img, r, c, half)
    if patch.size == 0:
        return None, None, None
    F = pixel_features(patch, FEATURE_SCALES, center=(r - y1, c - x1))
    prob = clf.predict_proba(F)[:, 1].reshape(patch.shape)
    out = mask_from_prob(prob, (r - y1, c - x1), prob_cut, open_r, mode)
    return out, prob, (y1, x1)


def sweep_eval(clf, pairs, half, combos, use_click=False, verbose_every=200,
               channel=None):
    """Score every (mode, open_r, cut) combination in ONE pass over the somas.

    The forest runs once per soma; each combination then costs only a threshold
    and a little morphology. Scoring them separately re-ran the forest for every
    combination -- 20x the work for identical probability maps.

    use_click centres the patch on the filename's click coordinates instead of
    the ground-truth centroid, which is what MMPS actually has at outlining time.
    """
    acc = {k: [[], [], 0] for k in combos}
    cache_path, cache_img = None, None
    for n, (mp, ip, r, c, tp) in enumerate(pairs):
        try:
            if ip != cache_path:
                cache_img = load_gray(ip, channel)
                cache_path = ip
            img = cache_img
            truth_full = np.squeeze(np.asarray(tifffile.imread(mp))) > 0
            if truth_full.shape != img.shape:
                continue
            ys, xs = np.nonzero(truth_full)
            if len(ys) < 20:
                continue
            if use_click:
                cr, cc = int(r), int(c)
            else:
                cr, cc = int(ys.mean()), int(xs.mean())
            patch, y1, x1 = patch_around(img, cr, cc, half)
            if patch.size == 0:
                continue
            ctr = (cr - y1, cc - x1)
            F = pixel_features(patch, FEATURE_SCALES, center=ctr)
            prob = clf.predict_proba(F)[:, 1].reshape(patch.shape)
            truth = truth_full[y1:y1 + patch.shape[0], x1:x1 + patch.shape[1]]
            for key in combos:
                mode, orad, cut = key
                pred = mask_from_prob(prob, ctr, cut, orad, mode)
                if pred is None:
                    acc[key][2] += 1
                    continue
                inter = np.logical_and(pred, truth).sum()
                union = np.logical_or(pred, truth).sum()
                if union == 0:
                    acc[key][2] += 1
                    continue
                acc[key][0].append(inter / union)
                acc[key][1].append(pred.sum() / max(truth.sum(), 1))
        except Exception:
            for key in combos:
                acc[key][2] += 1
        if verbose_every and (n + 1) % verbose_every == 0:
            print(f"    {n + 1}/{len(pairs)} somas scored")
    return {k: (np.array(v[0]), np.array(v[1]), v[2]) for k, v in acc.items()}


def oracle_eval(clf, pairs, half, mode, open_r, use_click=False, channel=None):
    """Best IoU each cell could reach if its OWN threshold were chosen for it.

    A single global cut has to serve every cell. If cells disagree about where
    their boundary sits in probability terms, that one cut is wrong for most of
    them and the score understates what the probability map actually knows. This
    scores each cell at its own best cut, which is not achievable in practice --
    it needs the right answer to pick the cut -- but it bounds what per-cell
    calibration could buy. A large gap means the map locates the soma and the
    threshold is the problem; no gap means the map itself is the limit.
    """
    fine = np.arange(0.15, 0.91, 0.05)
    best_iou, best_cut, fixed = [], [], []
    cache_path, cache_img = None, None
    for mp, ip, r, c, tp in pairs:
        try:
            if ip != cache_path:
                cache_img = load_gray(ip, channel)
                cache_path = ip
            truth_full = np.squeeze(np.asarray(tifffile.imread(mp))) > 0
            if truth_full.shape != cache_img.shape:
                continue
            ys, xs = np.nonzero(truth_full)
            if len(ys) < 20:
                continue
            cr, cc = (int(r), int(c)) if use_click else (int(ys.mean()), int(xs.mean()))
            patch, y1, x1 = patch_around(cache_img, cr, cc, half)
            if patch.size == 0:
                continue
            ctr = (cr - y1, cc - x1)
            F = pixel_features(patch, FEATURE_SCALES, center=ctr)
            prob = clf.predict_proba(F)[:, 1].reshape(patch.shape)
            truth = truth_full[y1:y1 + patch.shape[0], x1:x1 + patch.shape[1]]
            per = []
            for cut in fine:
                pred = mask_from_prob(prob, ctr, float(cut), open_r, mode)
                if pred is None:
                    per.append(0.0)
                    continue
                u = np.logical_or(pred, truth).sum()
                per.append(np.logical_and(pred, truth).sum() / u if u else 0.0)
            per = np.array(per)
            best_iou.append(per.max())
            best_cut.append(float(fine[int(per.argmax())]))
        except Exception:
            continue
    return np.array(best_iou), np.array(best_cut)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='folder containing the 1d/3d/7d/28d subfolders')
    ap.add_argument('--timepoints', nargs='*', default=['1d', '3d', '7d', '28d'])
    ap.add_argument('--pixel-size', type=float, default=0.1046, help='µm/px')
    ap.add_argument('--soma-radius-um', type=float, default=8.0,
                    help='half-width of the analysis patch, in µm')
    ap.add_argument('--limit', type=int, default=None, help='use at most N somas')
    ap.add_argument('--per-soma', type=int, default=600, help='pixels sampled per soma')
    ap.add_argument('--trees', type=int, default=300)
    ap.add_argument('--min-leaf', type=int, nargs='*', default=[2, 20, 100],
                    help='leaf sizes to try; larger = smaller, more general '
                         'model. Several values trains one forest each and '
                         'reports accuracy against size')
    ap.add_argument('--scales', type=float, nargs='*', default=None,
                    help='filter sizes in px (default 1 2 4 8). Nothing in the '
                         'default set spans a whole soma (~40 px radius), so '
                         '"--scales 1 2 4 8 16 24" is worth testing')
    ap.add_argument('--channel', type=int, default=None,
                    help='1-based channel holding the microglia stain; without '
                         'it the brightest channel is guessed')
    ap.add_argument('--max-samples', type=int, default=None,
                    help='cap rows per tree; only needed if the fit runs out of '
                         'memory (real trees are far smaller than worst case)')
    ap.add_argument('--use-click', action='store_true',
                    help='centre patches on the recorded click instead of the '
                         'ground-truth centroid (matches how MMPS will run)')
    ap.add_argument('--out', default='soma_model.joblib')
    a = ap.parse_args()

    if a.scales:
        global FEATURE_SCALES
        FEATURE_SCALES = tuple(a.scales)
    print(f"feature scales: {FEATURE_SCALES}  "
          f"({1 + 3 + 6 * len(FEATURE_SCALES)} features per pixel)")
    half = max(16, int(round(a.soma_radius_um / a.pixel_size)))
    print(f"patch half-width: {half} px  ({a.soma_radius_um} µm at {a.pixel_size} µm/px)\n")

    print("Pairing accepted somas with images…")
    pairs = find_pairs(a.root, a.timepoints, a.limit)
    if not pairs:
        sys.exit("No soma/image pairs found — check --root and the folder names.")
    print(f"\n{len(pairs)} soma/image pairs\n")

    print("Channel layout of the first image:")
    print(describe_channels(pairs[0][1]))
    if a.channel:
        print(f"    -> training on channel {a.channel}\n")
    else:
        print("    -> no --channel given; guessing the brightest channel. If the "
              "microglia\n       stain is not the brightest, pass --channel N.")
        try:
            _a = np.squeeze(np.asarray(tifffile.imread(pairs[0][1])))
            if _a.ndim == 3 and _a.shape[int(np.argmin(_a.shape))] <= 8:
                _b = np.moveaxis(_a, int(np.argmin(_a.shape)), -1)
                _s = sorted((_b[:, :, i].astype(np.float64).sum(), i + 1)
                            for i in range(_b.shape[2]))[::-1]
                if len(_s) > 1 and _s[0][0] < 1.25 * _s[1][0]:
                    print(f"       WARNING channels {_s[0][1]} and {_s[1][1]} are "
                          f"within {100 * (_s[0][0] / _s[1][0] - 1):.0f}% of each "
                          f"other.\n       The guess can land on a different "
                          f"channel from one image to the next, which trains the\n"
                          f"       model on inconsistent stains. Pass --channel.")
        except Exception:
            pass
        print()

    # split by IMAGE so no image appears in both train and test
    imgs = np.array([p[1] for p in pairs])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    tr_idx, te_idx = next(gss.split(np.zeros(len(pairs)), groups=imgs))
    train_pairs = [pairs[i] for i in tr_idx]
    test_pairs = [pairs[i] for i in te_idx]
    print(f"train: {len(train_pairs)} somas from {len(set(imgs[tr_idx]))} images")
    print(f"test : {len(test_pairs)} somas from {len(set(imgs[te_idx]))} images "
          f"(held out entirely)\n")

    print("Extracting features…")
    X, y, _ = build_dataset(train_pairs, half, a.per_soma, channel=a.channel,
                            use_click=a.use_click)
    if X is None:
        sys.exit("No usable training data — do the soma masks match the image sizes?")
    print(f"  {X.shape[0]:,} pixels x {X.shape[1]} features\n")

    max_samples = (min(a.max_samples, X.shape[0]) if a.max_samples else None)
    open_radii = sorted(set([0, max(2, half // 16), max(3, half // 10)]))
    cuts = (0.35, 0.45, 0.5, 0.55, 0.65)
    combos = ([('cc', o, c) for o in open_radii for c in cuts]
              + [('radial', 0, c) for c in cuts])
    rng = np.random.RandomState(1)
    train_sub = [train_pairs[i] for i in
                 sorted(rng.permutation(len(train_pairs))[:len(test_pairs)])]

    # A leaf size of 2 on millions of rows lets each tree memorise individual
    # images: it drives the training score up, leaves the held-out score flat,
    # and produces a forest far too large to ship inside the app. Train at
    # several leaf sizes and report accuracy against size, so the trade is
    # visible instead of assumed.
    overall = None
    for leaf in a.min_leaf:
        print(f"Training random forest ({a.trees} trees, leaf {leaf}"
              + (f", {max_samples:,} rows/tree" if max_samples else "") + ")…")
        clf = RandomForestClassifier(n_estimators=a.trees, min_samples_leaf=leaf,
                                     max_samples=max_samples, bootstrap=True,
                                     n_jobs=-1, random_state=0,
                                     class_weight='balanced')
        clf.fit(X, y)
        # Measure the forest rather than predicting it -- node counts depend on
        # how separable the data turns out to be.
        nodes = sum(t.tree_.node_count for t in clf.estimators_)
        print(f"  {nodes:,} nodes, about {nodes * 80 / 1e6:.0f} MB uncompressed")

        res = sweep_eval(clf, test_pairs, half, combos, use_click=a.use_click,
                         channel=a.channel)
        best = None
        for key in combos:
            mode, orad, cut = key
            ious, ratios, fails = res[key]
            tag = 'radial   ' if mode == 'radial' else f'open {orad:>2}px'
            if len(ious) == 0:
                print(f"  {tag}  cut {cut}: no usable predictions")
                continue
            med = float(np.median(ious))
            print(f"  {tag}  cut {cut}:  median IoU {med:.3f}   "
                  f"median area ratio {np.median(ratios):.2f}x   "
                  f"IoU>0.7 {100 * np.mean(ious > 0.7):.0f}%   fails {fails}")
            if best is None or med > best[1]:
                best = (cut, med, orad, mode, float(np.mean(ious > 0.7)))
        if best is None:
            print("  no usable predictions at this leaf size\n")
            continue

        key = (best[3], best[2], best[0])
        tr = sweep_eval(clf, train_sub, half, [key], use_click=a.use_click,
                        verbose_every=0, channel=a.channel)[key][0]
        tr_med = float(np.median(tr)) if len(tr) else float('nan')
        print(f"  leaf {leaf}: held-out {best[1]:.3f}  train {tr_med:.3f}  "
              f"gap {tr_med - best[1]:+.3f}  IoU>0.7 {100 * best[4]:.0f}%  "
              f"{nodes * 80 / 1e6:.0f} MB\n")
        if overall is None or best[1] > overall['iou']:
            overall = dict(clf=clf, leaf=leaf, iou=best[1], train=tr_med,
                           cut=best[0], open_r=best[2], mode=best[3],
                           hit=best[4], nodes=nodes)

    if overall is None:
        sys.exit("Nothing evaluated successfully.")

    print(f"Best: leaf {overall['leaf']}, prob cut {overall['cut']}, "
          + ("radial contour" if overall['mode'] == 'radial'
             else f"opening {overall['open_r']}px")
          + f"  (median IoU {overall['iou']:.3f}, "
            f"IoU>0.7 {100 * overall['hit']:.0f}%)")

    gap = overall['train'] - overall['iou']
    print(f"\nDiagnostic  train-image median IoU {overall['train']:.3f}   "
          f"vs held-out {overall['iou']:.3f}")
    if overall['train'] < 0.65:
        print("  Both low -> the limit is the features or the outlines "
              "themselves, not generalisation.")
    elif gap > 0.10:
        print("  Fits training images but does not transfer -> the model is "
              "learning per-image appearance.")
        print("  More outlines will not help; the features have to be made "
              "invariant to how each image looks.")
    else:
        print("  Train and test agree -> the model is learning what it can; "
              "remaining error is boundary ambiguity.")

    print("\nPer-cell ceiling (each cell scored at its own best threshold)…")
    o_iou, o_cut = oracle_eval(overall['clf'], test_pairs, half,
                               overall['mode'], overall['open_r'],
                               use_click=a.use_click, channel=a.channel)
    if len(o_iou):
        print(f"  per-cell-best median IoU {np.median(o_iou):.3f}   "
              f"IoU>0.7 {100 * np.mean(o_iou > 0.7):.0f}%")
        print(f"  best threshold per cell: median {np.median(o_cut):.2f}, "
              f"10th-90th pct {np.percentile(o_cut, 10):.2f}-"
              f"{np.percentile(o_cut, 90):.2f}")
        lift = float(np.median(o_iou)) - overall['iou']
        if lift > 0.06:
            print(f"  +{lift:.3f} over the single global cut -> cells disagree "
                  f"about where their boundary sits.")
            print("  Calibrating a threshold per cell is a real lever worth "
                  "building.")
        else:
            print(f"  only +{lift:.3f} over the single global cut -> the "
                  f"probability map itself is the limit,")
            print("  not the threshold. Better thresholding will not rescue "
                  "this.")

    meta = dict(channel=a.channel, pixel_size_um=a.pixel_size,
                soma_radius_um=a.soma_radius_um, half=half,
                scales=FEATURE_SCALES, prob_cut=overall['cut'],
                open_r=overall['open_r'], mode=overall['mode'])
    try:
        joblib.dump({'model': overall['clf'], 'meta': meta}, a.out, compress=3)
    except OSError as e:
        sys.exit(f"\nCould not write {a.out}: {e}\n"
                 f"The forest holds {overall['nodes']:,} nodes. Free some disk "
                 f"space, or retrain with a larger --min-leaf to shrink it.")
    mb = os.path.getsize(a.out) / 1e6
    print(f"\nSaved model -> {a.out}  ({mb:.0f} MB compressed)")
    if mb > 200:
        print("  That is large to ship inside the MMPS app; a bigger --min-leaf "
              "trades a little accuracy for a much smaller file.")
    print("\nHow to read the result:")
    print("  median IoU > 0.80  excellent — automate with spot checks")
    print("  0.70 - 0.80        good — automate, review flagged low-confidence cells")
    print("  0.55 - 0.70        marginal — usable only with review of every cell")
    print("  < 0.55             do not automate; the outlines are too inconsistent")
    print("                     or the images lack the necessary signal")


if __name__ == '__main__':
    main()
