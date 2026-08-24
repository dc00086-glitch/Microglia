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


def load_gray(path):
    """Load an image as 2D float. Max-projects Z, takes the brightest channel."""
    a = np.squeeze(np.asarray(tifffile.imread(path)))
    if a.ndim == 3:
        # channel axis = the small one; otherwise treat as a stack
        ax = int(np.argmin(a.shape))
        if a.shape[ax] <= 8:
            a = np.moveaxis(a, ax, -1)
            # pick the channel with the most signal (the stained one)
            sums = [a[:, :, i].astype(np.float64).sum() for i in range(a.shape[2])]
            a = a[:, :, int(np.argmax(sums))]
        else:
            a = a.max(axis=0)
    while a.ndim > 2:
        a = a.max(axis=0)
    return a.astype(np.float64)


# ----------------------------------------------------------------------
# features
# ----------------------------------------------------------------------
def pixel_features(patch, scales=(1.0, 2.0, 4.0, 8.0)):
    """Multi-scale per-pixel features -> (n_pixels, n_features).

    Hessian eigenvalues are the important ones: for a BLOB both eigenvalues are
    large and similar, for a TUBE (process) one is large and one near zero. That
    is the "prefer blobs, penalise branching" prior, learned rather than tuned.
    """
    p = patch.astype(np.float64)
    lo, hi = np.percentile(p, 1), np.percentile(p, 99.5)
    p = (p - lo) / (hi - lo) if hi > lo else p * 0.0
    feats = [p]
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


def patch_around(img, r, c, half):
    h, w = img.shape[:2]
    y1, y2 = max(0, r - half), min(h, r + half)
    x1, x2 = max(0, c - half), min(w, c + half)
    return img[y1:y2, x1:x2], y1, x1


# ----------------------------------------------------------------------
# training set
# ----------------------------------------------------------------------
def build_dataset(pairs, half, per_soma=600, verbose_every=100):
    X, y, groups = [], [], []
    cache_path, cache_img = None, None
    for i, (mp, ip, r, c, tp) in enumerate(pairs):
        try:
            if ip != cache_path:
                cache_img = load_gray(ip)
                cache_path = ip
            img = cache_img
            mask = np.squeeze(np.asarray(tifffile.imread(mp))) > 0
            if mask.shape != img.shape:
                continue
            ys, xs = np.nonzero(mask)
            if len(ys) < 20:
                continue
            cr, cc = int(ys.mean()), int(xs.mean())      # true centroid of the outline
            patch, y1, x1 = patch_around(img, cr, cc, half)
            mpatch = mask[y1:y1 + patch.shape[0], x1:x1 + patch.shape[1]]
            if patch.size == 0 or mpatch.sum() < 20:
                continue
            F = pixel_features(patch, FEATURE_SCALES)
            lab = mpatch.ravel().astype(np.uint8)
            pos = np.flatnonzero(lab == 1)
            neg = np.flatnonzero(lab == 0)
            if len(pos) < 10 or len(neg) < 10:
                continue
            rng = np.random.RandomState(i)
            npick = min(per_soma // 2, len(pos), len(neg))
            sel = np.concatenate([rng.choice(pos, npick, replace=False),
                                  rng.choice(neg, npick, replace=False)])
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
def predict_mask(clf, img, r, c, half, prob_cut=0.5):
    patch, y1, x1 = patch_around(img, r, c, half)
    if patch.size == 0:
        return None, None, None
    F = pixel_features(patch, FEATURE_SCALES)
    prob = clf.predict_proba(F)[:, 1].reshape(patch.shape)
    binm = prob >= prob_cut
    lab, n = ndimage.label(binm)
    if n == 0:
        return None, prob, (y1, x1)
    ly, lx = r - y1, c - x1
    ly = min(max(ly, 0), patch.shape[0] - 1)
    lx = min(max(lx, 0), patch.shape[1] - 1)
    cid = lab[ly, lx]
    if cid == 0:                                   # click just outside — nearest
        ys, xs = np.nonzero(lab)
        if not len(ys):
            return None, prob, (y1, x1)
        i = int(np.argmin((ys - ly) ** 2 + (xs - lx) ** 2))
        cid = lab[ys[i], xs[i]]
    out = ndimage.binary_fill_holes(lab == cid)
    return out, prob, (y1, x1)


def evaluate(clf, pairs, half, prob_cut=0.5):
    ious, arearatios, fails = [], [], 0
    cache_path, cache_img = None, None
    for mp, ip, r, c, tp in pairs:
        try:
            if ip != cache_path:
                cache_img = load_gray(ip)
                cache_path = ip
            img = cache_img
            truth_full = np.squeeze(np.asarray(tifffile.imread(mp))) > 0
            if truth_full.shape != img.shape:
                continue
            ys, xs = np.nonzero(truth_full)
            if len(ys) < 20:
                continue
            cr, cc = int(ys.mean()), int(xs.mean())
            pred, prob, off = predict_mask(clf, img, cr, cc, half, prob_cut)
            if pred is None:
                fails += 1
                continue
            y1, x1 = off
            truth = truth_full[y1:y1 + pred.shape[0], x1:x1 + pred.shape[1]]
            inter = np.logical_and(pred, truth).sum()
            union = np.logical_or(pred, truth).sum()
            if union == 0:
                fails += 1
                continue
            ious.append(inter / union)
            arearatios.append(pred.sum() / max(truth.sum(), 1))
        except Exception:
            fails += 1
    return np.array(ious), np.array(arearatios), fails


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
    ap.add_argument('--trees', type=int, default=200)
    ap.add_argument('--out', default='soma_model.joblib')
    a = ap.parse_args()

    half = max(16, int(round(a.soma_radius_um / a.pixel_size)))
    print(f"patch half-width: {half} px  ({a.soma_radius_um} µm at {a.pixel_size} µm/px)\n")

    print("Pairing accepted somas with images…")
    pairs = find_pairs(a.root, a.timepoints, a.limit)
    if not pairs:
        sys.exit("No soma/image pairs found — check --root and the folder names.")
    print(f"\n{len(pairs)} soma/image pairs\n")

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
    X, y, _ = build_dataset(train_pairs, half, a.per_soma)
    if X is None:
        sys.exit("No usable training data — do the soma masks match the image sizes?")
    print(f"  {X.shape[0]:,} pixels x {X.shape[1]} features\n")

    print(f"Training random forest ({a.trees} trees)…")
    clf = RandomForestClassifier(n_estimators=a.trees, min_samples_leaf=4,
                                 n_jobs=-1, random_state=0, class_weight='balanced')
    clf.fit(X, y)
    print("  done\n")

    print("Evaluating on held-out images…")
    best = None
    for cut in (0.35, 0.45, 0.5, 0.55, 0.65):
        ious, ratios, fails = evaluate(clf, test_pairs, half, cut)
        if len(ious) == 0:
            print(f"  prob cut {cut}: no usable predictions")
            continue
        med = float(np.median(ious))
        print(f"  prob cut {cut}:  median IoU {med:.3f}   "
              f"median area ratio {np.median(ratios):.2f}x   "
              f"IoU>0.7 {100 * np.mean(ious > 0.7):.0f}%   fails {fails}")
        if best is None or med > best[1]:
            best = (cut, med)

    if best:
        print(f"\nBest probability cut: {best[0]}  (median IoU {best[1]:.3f})")
    meta = dict(pixel_size_um=a.pixel_size, soma_radius_um=a.soma_radius_um,
                half=half, scales=FEATURE_SCALES,
                prob_cut=(best[0] if best else 0.5))
    joblib.dump({'model': clf, 'meta': meta}, a.out)
    print(f"\nSaved model -> {a.out}")
    print("\nHow to read the result:")
    print("  median IoU > 0.80  excellent — automate with spot checks")
    print("  0.70 - 0.80        good — automate, review flagged low-confidence cells")
    print("  0.55 - 0.70        marginal — usable only with review of every cell")
    print("  < 0.55             do not automate; the outlines are too inconsistent")
    print("                     or the images lack the necessary signal")


if __name__ == '__main__':
    main()
