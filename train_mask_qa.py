#!/usr/bin/env python3
"""train_mask_qa.py — learn which microglia mask to accept, from your own QA.

Each soma is masked at several target areas and you keep at most one. This
learns that decision so MMPS can propose the LARGEST mask that should be
accepted, or say that none should be.

It is framed as accept/reject per mask rather than as predicting the size
directly. That uses every mask as an example instead of one per soma, and
"none acceptable" falls out for free: if no mask for a soma is predicted
acceptable, there is nothing to propose.

Features are whole-object, not per-pixel -- shape, how the mask sits against
the IBA1 signal, and whether it contains a nucleus. About 30 numbers per mask,
one row each.

DATA IT NEEDS
    <timepoint>/Output/masks/         masks you KEPT      (positive)
    <timepoint>/Output/rejected_masks/masks you REJECTED  (negative)
    <timepoint>/Output/somas/         the soma outlines
    <timepoint>/Image Directory/      the raw images

Rejected masks are only kept if "Advanced > Keep Rejected Masks" was on in
MMPS. Without them there is only one class and nothing can be learned.

USAGE
    python3 train_mask_qa.py --root "<study root>" --pixel-size 0.1046
"""

import os
import re
import sys
import csv
import glob
import hashlib
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
    import cv2
except ImportError:
    sys.exit("Missing opencv.  pip install opencv-python-headless")
try:
    from skimage.morphology import skeletonize
except ImportError:
    sys.exit("Missing scikit-image.  pip install scikit-image")
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import roc_auc_score
    import joblib
except ImportError:
    sys.exit("Missing scikit-learn.  pip install scikit-learn joblib")

MASK_RE = re.compile(
    r'^(?P<base>.+)_soma_(?P<r>\d+)_(?P<c>\d+)_area(?P<area>\d+)_mask\.tiff?$',
    re.I)
SOMA_RE = re.compile(r'^(?P<base>.+)_soma_(?P<r>\d+)_(?P<c>\d+)_soma\.tiff?$',
                     re.I)
CH_SUFFIX_RE = re.compile(r'_ch\d+$', re.I)


def _norm(stem):
    return re.sub(r'[^a-z0-9]+', '_', stem.lower()).strip('_')


# ----------------------------------------------------------------------
# gathering
# ----------------------------------------------------------------------
def index_images(img_dir):
    idx = {}
    for p in glob.glob(os.path.join(img_dir, '*')):
        bn = os.path.basename(p)
        if bn.startswith('._') or not bn.lower().endswith(('.tif', '.tiff')):
            continue
        stem = os.path.splitext(bn)[0]
        if CH_SUFFIX_RE.search(stem):
            continue
        idx.setdefault(_norm(stem), p)
        idx.setdefault(_norm(stem.replace('_processed', '')), p)
    return idx


def read_rejection_reasons(rej_dir):
    """filename -> reason, from the CSV MMPS writes alongside the masks."""
    out = {}
    path = os.path.join(rej_dir, 'rejected_masks.csv')
    if not os.path.exists(path):
        return out
    try:
        with open(path, newline='') as fh:
            for row in csv.DictReader(fh):
                if row.get('file'):
                    out[row['file']] = (row.get('reason') or 'user').strip()
    except Exception:
        pass
    return out


def gather(root, timepoints, image_subdir, drop_duplicates=True):
    """Collect every mask with its label, grouped by soma."""
    records = []
    n_dup = 0
    for tp in timepoints:
        out_dir = os.path.join(root, tp, 'Output')
        masks_dir = os.path.join(out_dir, 'masks')
        rej_dir = os.path.join(out_dir, 'rejected_masks')
        somas_dir = os.path.join(out_dir, 'somas')
        img_idx = index_images(os.path.join(root, tp, image_subdir))
        if not os.path.isdir(masks_dir) and not os.path.isdir(rej_dir):
            print(f"  [{tp}] no masks or rejected_masks folder — skipped")
            continue
        reasons = read_rejection_reasons(rej_dir)

        soma_paths = {}
        for sp in glob.glob(os.path.join(somas_dir, '*_soma.tif*')):
            m = SOMA_RE.match(os.path.basename(sp))
            if m:
                soma_paths[(m.group('base'), m.group('r'), m.group('c'))] = sp

        kept = rejected = 0
        for src_dir, label in ((masks_dir, 1), (rej_dir, 0)):
            if not os.path.isdir(src_dir):
                continue
            for mp in sorted(glob.glob(os.path.join(src_dir, '*_mask.tif*'))):
                fn = os.path.basename(mp)
                if fn.startswith('._'):
                    continue
                m = MASK_RE.match(fn)
                if not m:
                    continue
                # Duplicates are auto-rejected by a rule, not by judgement.
                # Training on them teaches a rule the model does not need and
                # pads the score with free correct answers.
                if label == 0 and drop_duplicates and \
                        reasons.get(fn, 'user') == 'duplicate':
                    n_dup += 1
                    continue
                base, r, c = m.group('base'), m.group('r'), m.group('c')
                ip = img_idx.get(_norm(base))
                if ip is None:
                    continue
                records.append(dict(
                    mask_path=mp, image_path=ip,
                    soma_path=soma_paths.get((base, r, c)),
                    base=base, row=int(r), col=int(c),
                    area=int(m.group('area')), label=label, tp=tp))
                kept += label
                rejected += (1 - label)
        print(f"  [{tp}] {kept} kept, {rejected} rejected")
    if n_dup:
        print(f"  ({n_dup} auto-rejected duplicates excluded)")
    # group by image so the per-image caches actually hit; without this each
    # mask reloads the image and recomputes its gradient
    records.sort(key=lambda r: (r['image_path'], r['base'], r['row'],
                                r['col'], r['area']))
    return records


# ----------------------------------------------------------------------
# features
# ----------------------------------------------------------------------
FEATURE_NAMES = [
    'target_area', 'area_px', 'area_um2', 'area_vs_target',
    'perimeter', 'circularity', 'solidity', 'extent', 'aspect_ratio',
    'eccentricity', 'n_components', 'n_holes', 'euler',
    'max_thickness', 'mean_thickness', 'skel_length', 'skel_endpoints',
    'skel_branchpoints', 'skel_per_area',
    'soma_frac', 'centroid_offset', 'touches_border',
    'sig_mean_in', 'sig_p90_in', 'sig_mean_ring', 'sig_contrast',
    'sig_frac_above_bg', 'boundary_gradient',
    'dapi_mean_in', 'dapi_frac_in', 'dapi_in_soma', 'dapi_dist',
    'nbr_dist', 'nbr_dist2', 'nbr_within_2r', 'frac_in_nbr_territory',
    'reach_vs_gap',
]


def mask_features(mask, soma_mask, sig, dapi, target_area, pixel_size,
                  grad=None, bg=None, dapi_thr=None, centre=None,
                  neighbours=None):
    """One mask -> one feature vector. Order must match FEATURE_NAMES.

    Everything is computed on a crop around the mask rather than the full
    frame. A mask occupies a few hundred pixels of a 1440x1920 image, so the
    distance transform, labelling and dilation were doing seventy times more
    work than they needed to, once per mask, over tens of thousands of masks.
    The crop keeps a margin wider than any neighbourhood used below, so every
    value is identical to the full-frame computation.
    """
    f = []
    m_full = mask > 0
    area = int(m_full.sum())
    if area < 5:
        return None
    px2 = pixel_size ** 2

    ys_f, xs_f = np.nonzero(m_full)
    H, W = m_full.shape
    MARGIN = 12                      # > the 6-px ring and the 3x3 skeleton pass
    cy0 = max(0, int(ys_f.min()) - MARGIN)
    cy1 = min(H, int(ys_f.max()) + 1 + MARGIN)
    cx0 = max(0, int(xs_f.min()) - MARGIN)
    cx1 = min(W, int(xs_f.max()) + 1 + MARGIN)
    m = m_full[cy0:cy1, cx0:cx1]
    sig_c = sig[cy0:cy1, cx0:cx1]
    dapi_c = dapi[cy0:cy1, cx0:cx1] if dapi is not None else None
    soma_c = (soma_mask[cy0:cy1, cx0:cx1]
              if soma_mask is not None else None)
    grad_c = grad[cy0:cy1, cx0:cx1] if grad is not None else None

    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = m[y0:y1, x0:x1]
    u8 = sub.astype(np.uint8)

    f += [float(target_area), float(area), area * px2,
          (area * px2) / max(float(target_area), 1e-6)]

    cnts, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea) if cnts else None
    per = cv2.arcLength(cnt, True) if cnt is not None else 0.0
    # circularity 4*pi*A/P^2: 1 for a disc, lower for ragged or elongated
    circ = (4 * np.pi * area / (per ** 2)) if per > 0 else 0.0
    hull_a = cv2.contourArea(cv2.convexHull(cnt)) if cnt is not None else 0.0
    solidity = area / hull_a if hull_a > 0 else 0.0
    extent = area / float(sub.size) if sub.size else 0.0
    h, w = sub.shape
    aspect = max(h, w) / max(min(h, w), 1)
    if cnt is not None and len(cnt) >= 5:
        (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
        a_, b_ = max(MA, ma) / 2, min(MA, ma) / 2
        ecc = float(np.sqrt(1 - (b_ * b_) / (a_ * a_))) if a_ > 0 else 0.0
    else:
        ecc = 0.0
    f += [float(per), float(circ), float(solidity), float(extent),
          float(aspect), float(ecc)]

    lab, ncomp = ndimage.label(m)
    filled = ndimage.binary_fill_holes(m)
    _, nholes = ndimage.label(filled & ~m)
    f += [float(ncomp), float(nholes), float(ncomp - nholes)]

    dt = ndimage.distance_transform_edt(m)
    f += [float(dt.max()), float(dt[m].mean())]

    sk = skeletonize(sub)
    nb = ndimage.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8),
                          mode='constant') - sk.astype(np.uint8)
    f += [float(sk.sum()), float(((nb == 1) & sk).sum()),
          float(((nb >= 3) & sk).sum()),
          float(sk.sum()) / max(area, 1)]

    # how much of the mask is soma, and whether it grew away from it
    if soma_c is not None and soma_c.any():
        s = soma_c > 0
        f.append(float(s.sum()) / max(area, 1))
        sy, sx = np.nonzero(s)
        f.append(float(np.hypot(ys.mean() - sy.mean(), xs.mean() - sx.mean())))
    else:
        f += [0.0, 0.0]
    f.append(1.0 if (int(ys_f.min()) == 0 or int(xs_f.min()) == 0
                     or int(ys_f.max()) >= H - 1
                     or int(xs_f.max()) >= W - 1) else 0.0)

    # IBA1 signal: is the mask sitting on the cell, and does it stop at an edge?
    ring = ndimage.binary_dilation(m, iterations=6) & ~m
    sin_ = sig_c[m]
    sring = sig_c[ring] if ring.any() else np.array([0.0])
    if bg is None:               # image-wide, so computed once per image
        bg = float(np.median(sig))
    f += [float(sin_.mean()), float(np.percentile(sin_, 90)),
          float(sring.mean()),
          float(sin_.mean()) / (float(sring.mean()) + 1e-6),
          float((sin_ > bg).mean())]
    # the gradient depends only on the image, so it is computed once per image
    if grad_c is None:
        gy, gx = np.gradient(ndimage.gaussian_filter(sig_c, 1.5))
        grad_c = np.hypot(gy, gx)
    edge = m & ~ndimage.binary_erosion(m)
    f.append(float(grad_c[edge].mean()) if edge.any() else 0.0)

    # DAPI: a real soma contains a nucleus
    if dapi_c is not None:
        d = dapi_c.astype(np.float64)
        thr = (dapi_thr if dapi_thr is not None
               else float(np.percentile(dapi, 99)) * 0.35)
        pos = d >= max(thr, 1.0)
        f += [float(d[m].mean()), float(pos[m].mean())]
        f.append(float(pos[soma_c > 0].mean())
                 if soma_c is not None and soma_c.any() else 0.0)
        if pos.any():
            dist = ndimage.distance_transform_edt(~pos)
            f.append(float(dist[m].min()))
        else:
            f.append(float(max(m.shape)))
    else:
        f += [0.0, 0.0, 0.0, 0.0]

    # Crowding. A mask is rejected when it runs into the cell next door, and
    # nothing above can see where the neighbours are -- every feature so far
    # describes the mask in isolation. These say how much room this cell had
    # and whether the mask used more than its share of it.
    cy_m, cx_m = (centre if centre is not None
                  else (float(ys_f.mean()), float(xs_f.mean())))
    nb = [n for n in (neighbours or [])
          if abs(n[0] - cy_m) > 1e-6 or abs(n[1] - cx_m) > 1e-6]
    far = float(max(H, W))
    if nb:
        d = sorted(float(np.hypot(n[0] - cy_m, n[1] - cx_m)) for n in nb)
        d1 = d[0]
        d2 = d[1] if len(d) > 1 else far
        reach = float(np.hypot(ys_f - cy_m, xs_f - cx_m).max())
        # how many neighbours sit within twice this mask's own reach
        n2r = float(sum(1 for x in d if x < 2 * max(reach, 1.0)))
        # pixels closer to some other soma than to this one: territory taken
        pts = np.stack([ys_f, xs_f], axis=1).astype(np.float64)
        own = np.hypot(pts[:, 0] - cy_m, pts[:, 1] - cx_m)
        best_other = np.full(own.shape, np.inf)
        for n in nb:
            best_other = np.minimum(
                best_other, np.hypot(pts[:, 0] - n[0], pts[:, 1] - n[1]))
        f += [d1, d2, n2r, float((best_other < own).mean()),
              reach / max(d1 / 2.0, 1.0)]
    else:
        f += [far, far, 0.0, 0.0, 0.0]

    return np.asarray(f, dtype=np.float32)


def image_stats(sig, dapi):
    """Everything about an image that does not vary between its masks.

    The gradient, the background level and the nuclear threshold are all
    image-wide, and computing them per mask meant scanning two million pixels
    tens of thousands of times.
    """
    gy, gx = np.gradient(ndimage.gaussian_filter(sig, 1.5))
    grad = np.hypot(gy, gx)
    bg = float(np.median(sig))
    dthr = (float(np.percentile(dapi, 99)) * 0.35) if dapi is not None else None
    return grad, bg, dthr


def load_channels(path, sig_ch, dapi_ch):
    a = np.squeeze(np.asarray(tifffile.imread(path)))
    if a.ndim == 2:
        return a.astype(np.float64), None
    ax = int(np.argmin(a.shape))
    if a.shape[ax] <= 8:
        a = np.moveaxis(a, ax, -1)
    else:
        return a.max(axis=0).astype(np.float64), None
    n = a.shape[2]
    # Clamping an out-of-range channel to the last one reads the wrong stain
    # and reports nothing, which is the failure that leaves no trace.
    if not 1 <= sig_ch <= n:
        raise ValueError(f"--signal-channel {sig_ch} but this image has "
                         f"{n} channels")
    sig = a[:, :, sig_ch - 1].astype(np.float64)
    dapi = None
    if dapi_ch:
        if not 1 <= dapi_ch <= n:
            raise ValueError(f"--dapi-channel {dapi_ch} but this image has "
                             f"{n} channels")
        dapi = a[:, :, dapi_ch - 1].astype(np.float64)
    # Every other channel is left unread: nothing downstream can see it.
    return sig, dapi


def build(records, sig_ch, dapi_ch, pixel_size, verbose_every=200):
    X, y, groups, keys = [], [], [], []
    skipped, skip_reasons = 0, []
    cache_img, cache_path = None, None
    cache_grad = cache_bg = cache_dthr = None
    cache_soma, cache_soma_path = None, None
    for i, rec in enumerate(records):
        try:
            if rec['image_path'] != cache_path:
                cache_img = load_channels(rec['image_path'], sig_ch, dapi_ch)
                cache_grad, cache_bg, cache_dthr = image_stats(*cache_img)
                cache_path = rec['image_path']
            sig, dapi = cache_img
            mask = np.squeeze(np.asarray(tifffile.imread(rec['mask_path'])))
            if mask.shape != sig.shape:
                continue
            sp = rec.get('soma_path')
            if sp and sp != cache_soma_path:
                cache_soma = np.squeeze(np.asarray(tifffile.imread(sp))) > 0
                cache_soma_path = sp
            soma = cache_soma if sp else None
            v = mask_features(mask, soma, sig, dapi, rec['area'], pixel_size,
                              grad=cache_grad, bg=cache_bg,
                              dapi_thr=cache_dthr,
                              centre=(rec['row'], rec['col']),
                              neighbours=per_image.get(rec['image_path'], []))
            if v is None:
                continue
            X.append(v)
            y.append(rec['label'])
            groups.append(rec['base'])
            keys.append((rec['base'], rec['row'], rec['col'], rec['area']))
        except Exception as e:
            # A file being written by MMPS at the same moment reads back short
            # or empty. One mask is not worth stopping for, but a lot of them
            # means the run is measuring a fraction of the data.
            skipped += 1
            if len(skip_reasons) < 5:
                skip_reasons.append(
                    f"{os.path.basename(rec['mask_path'])}: {e}")
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"    {i + 1}/{len(records)} masks processed"
                  + (f"  ({skipped} unreadable)" if skipped else ""))
    if skipped:
        print(f"\n  {skipped} of {len(records)} masks could not be read "
              f"({100 * skipped / max(len(records), 1):.1f}%)")
        for r in skip_reasons:
            print(f"    {r}")
        if skipped > 0.05 * len(records):
            print("  That is a lot. If MMPS was running at the same time it "
                  "may have been\n  writing these files — close it and rerun "
                  "rather than trusting this.")
    if not X:
        return None, None, None, None
    return (np.vstack(X), np.asarray(y), np.asarray(groups), keys)


# ----------------------------------------------------------------------
# the metric that matters
# ----------------------------------------------------------------------
def pick(areas, probs, cut, rule):
    """Choose the mask to accept from one soma's ladder of sizes.

    Acceptability is a BAND, not a prefix: the smallest masks are too small,
    the largest too big, and the acceptable ones sit between. So taking the
    largest mask over the cut hands the answer to any single false positive at
    the top of the ladder -- and choosing too large is the expensive error,
    because an oversized mask pulls in neighbouring processes and inflates
    every morphology metric computed from it.

    'largest'  the largest mask over the cut.
    'band'     the largest mask in the run of accepts containing the most
               confident mask, so an isolated accept above a rejection cannot
               drag the choice upward.
    """
    ok = [pr >= cut for pr in probs]
    if not any(ok):
        return None
    if rule == 'edge':
        # Acceptability ENDS somewhere on the ladder: sizes below the boundary
        # are fine, above it the mask has swallowed a neighbour. Rather than
        # asking each size independently whether it clears a threshold, find
        # where confidence collapses and take the size just before it. The
        # threshold then only decides whether anything is acceptable at all,
        # not which size -- so the choice does not move every time it is tuned.
        if len(probs) < 2:
            return areas[0]
        drops = [probs[i] - probs[i + 1] for i in range(len(probs) - 1)]
        j = max(range(len(drops)), key=lambda i: drops[i])
        return areas[j]
    if rule == 'largest':
        return max(a for a, o in zip(areas, ok) if o)
    best_i = max(range(len(probs)), key=lambda i: probs[i])
    if not ok[best_i]:
        return None
    hi = best_i
    while hi + 1 < len(ok) and ok[hi + 1]:
        hi += 1
    return areas[hi]


def size_choice_report(keys, y_true, p, cut, rule='largest', quiet=False,
                       over_w=3.0, miss_w=4.0):
    """Does the model pick the same mask you did?

    Per-mask accuracy is not the task. The task is choosing, per soma, the
    LARGEST acceptable mask -- or none. Score that directly.
    """
    somas = {}
    for k, t, pr in zip(keys, y_true, p):
        somas.setdefault(k[:3], []).append((k[3], t, pr))
    exact = within = none_ok = none_tot = 0
    over = under = 0
    total = 0
    step_pen = 0.0      # how far off, in size steps, direction-weighted
    n_pen = 0
    for _, rows in somas.items():
        rows.sort(key=lambda z: z[0])
        areas = [z[0] for z in rows]
        truth = [z[0] for z in rows if z[1] == 1]
        t_best = max(truth) if truth else None
        p_best = pick(areas, [z[2] for z in rows], cut, rule)
        total += 1
        if t_best is None:
            none_tot += 1
            if p_best is None:
                none_ok += 1
            else:
                # proposing a mask where every size was rejected puts a bad
                # mask into the data, like an oversize
                step_pen += over_w
                n_pen += 1
            continue
        if p_best is None:
            # proposing nothing where a size was acceptable: the cell is lost
            under += 1
            step_pen += miss_w
            n_pen += 1
            continue
        ti, pi = areas.index(t_best), areas.index(p_best)
        if p_best == t_best:
            exact += 1
            within += 1
        else:
            if abs(ti - pi) <= 1:
                within += 1
            if pi > ti:
                over += 1
            else:
                under += 1
        # Distance in size steps, so being one step out is not scored the same
        # as being eight. Overshooting is weighted more: an oversized mask
        # absorbs neighbouring processes, where an undersized one is visible.
        d = pi - ti
        step_pen += (over_w * d) if d > 0 else (-d)
        n_pen += 1
    # Two different questions, so two different denominators. Mixing them --
    # scoring the size choice across somas that have no correct size -- reads
    # far worse than the model is behaving.
    scoreable = total - none_tot
    d = max(scoreable, 1)
    if not quiet:
        print(f"  {scoreable} somas had an acceptable mask; on those:")
        print(f"    picked exactly your mask:    {100 * exact / d:5.1f}%")
        print(f"    within one size step:        {100 * within / d:5.1f}%")
        print(f"    chose too large:             {100 * over / d:5.1f}%")
        print(f"    chose too small:             {100 * under / d:5.1f}%")
        print(f"    mean size error:             {step_pen / max(n_pen, 1):5.2f} "
              f"weighted steps (0 = always exact)")
        if none_tot:
            print(f"  {none_tot} somas had none acceptable; on those:")
            print(f"    correctly proposed nothing:  "
                  f"{100 * none_ok / none_tot:5.1f}%")
    return dict(total=total, scoreable=scoreable, none_tot=none_tot,
                exact=exact / d, within=within / d, over=over / d,
                under=under / d, steps=step_pen / max(n_pen, 1),
                none_ok=(none_ok / none_tot if none_tot else 1.0))


def bootstrap_ci(keys, y_true, p, cut, rule, over_w, n_boot=2000, seed=0):
    """95% intervals by resampling IMAGES, not somas.

    Somas from one image share staining, background and crowding, so they are
    not independent observations. A binomial interval over somas treats 500 of
    them as 500 independent trials and comes out about half as wide as it
    should be. Resampling whole images keeps that correlation in the estimate.
    """
    by_img = {}
    for i, k in enumerate(keys):
        by_img.setdefault(k[0], []).append(i)
    imgs = sorted(by_img)
    rng = np.random.default_rng(seed)
    fields = ('exact', 'within', 'over', 'under')
    acc = {f: [] for f in fields}
    for _ in range(n_boot):
        pick = rng.integers(0, len(imgs), len(imgs))
        idx = []
        for j in pick:
            idx.extend(by_img[imgs[j]])
        # a resampled image appears more than once; make its somas distinct so
        # they are not collapsed into one group by the reporter
        kk, tt, pp = [], [], []
        for rep, j in enumerate(pick):
            for i in by_img[imgs[j]]:
                k = keys[i]
                kk.append((f"{k[0]}#{rep}",) + tuple(k[1:]))
                tt.append(y_true[i])
                pp.append(p[i])
        r = size_choice_report(kk, np.asarray(tt), np.asarray(pp), cut, rule,
                               quiet=True, over_w=over_w)
        for f in fields:
            acc[f].append(r[f])
    return {f: (float(np.percentile(acc[f], 2.5)),
                float(np.percentile(acc[f], 97.5))) for f in fields}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--timepoints', nargs='*',
                    default=['1d', '3d', '7d', '28d'])
    ap.add_argument('--image-subdir', default='Image Directory')
    ap.add_argument('--pixel-size', type=float, default=0.1046)
    ap.add_argument('--signal-channel', type=int, default=1,
                    help='microglia stain (red)')
    ap.add_argument('--dapi-channel', type=int, default=3,
                    help='nuclear stain (blue); 0 to disable')
    ap.add_argument('--boot', type=int, default=2000,
                    help='bootstrap draws for the confidence intervals; 0 to '
                         'skip')
    ap.add_argument('--cache', default=None,
                    help='save extracted features here, and reuse them next '
                         'time. Feature extraction is the slow part; the '
                         'decision rule can then be retried in seconds.')
    ap.add_argument('--max-masks', type=int, default=None,
                    help='use at most N masks, sampled evenly, for a quick check')
    ap.add_argument('--trees', type=int, default=400)
    ap.add_argument('--min-leaf', type=int, default=4)
    ap.add_argument('--oversize-cost', type=float, default=3.0,
                    help='how much worse choosing TOO LARGE is than choosing '
                         'too small, when picking the threshold. An oversized '
                         'mask pulls in neighbouring processes and inflates '
                         'every downstream metric, where an undersized one is '
                         'visible. 1.0 treats them equally.')
    ap.add_argument('--reject-weight', type=float, default=2.0,
                    help='how much more a wrongly ACCEPTED mask costs the '
                         'forest during training than a wrongly rejected one. '
                         '1.0 is a symmetric loss.')
    ap.add_argument('--keep-duplicates', action='store_true',
                    help='include auto-rejected duplicates (normally excluded: '
                         'they are a rule, not a judgement)')
    ap.add_argument('--out', default='mask_qa_model.joblib')
    a = ap.parse_args()

    try:
        print("script fingerprint:",
              hashlib.md5(open(__file__, 'rb').read()).hexdigest()[:8])
    except Exception:
        pass
    if a.dapi_channel and a.dapi_channel == a.signal_channel:
        sys.exit(f"--signal-channel and --dapi-channel are both "
                 f"{a.signal_channel}; they must be different stains.")
    used = {a.signal_channel: 'microglia (IBA1)'}
    if a.dapi_channel:
        used[a.dapi_channel] = 'nuclear (DAPI)'
    print("Channels read:")
    for c in sorted(used):
        print(f"  channel {c}: {used[c]}")
    ignored = [c for c in (1, 2, 3, 4) if c not in used]
    print(f"  ignored: {', '.join('channel %d' % c for c in ignored)} "
          f"— never loaded, so no feature can depend on them\n")

    print("Reading the mask folders. If MMPS is open on this study, close it "
          "first —\nit rewrites these files as it works and a half-written "
          "one cannot be read.\n")
    print("Gathering masks…")
    recs = gather(a.root, a.timepoints, a.image_subdir,
                  drop_duplicates=not a.keep_duplicates)
    if not recs:
        sys.exit("No masks found. Was 'Keep Rejected Masks' on during QA?")
    n_pos = sum(r['label'] for r in recs)
    n_neg = len(recs) - n_pos
    print(f"\n{len(recs)} masks: {n_pos} kept, {n_neg} rejected")
    if n_pos < 30 or n_neg < 30:
        sys.exit("Too few of one class to learn from — do more QA first.")

    if a.max_masks and len(recs) > a.max_masks:
        # Sample whole SOMAS. Taking every nth mask splits a soma's ladder of
        # sizes, and a soma whose accepted mask falls outside the sample then
        # looks like one where every size was rejected -- inflating "no
        # acceptable mask" and distorting the size-choice score.
        by_soma = {}
        for r in recs:
            by_soma.setdefault((r['base'], r['row'], r['col']), []).append(r)
        keys = sorted(by_soma)
        take, out = 0, []
        step = max(1, int(round(len(recs) / float(a.max_masks))))
        for k in keys[::1]:
            out.extend(by_soma[k])
            take += 1
            if len(out) >= a.max_masks:
                break
        recs = out
        print(f"  sampled down to {len(recs)} masks from {take} whole somas")

    X = None
    if a.cache and os.path.exists(a.cache):
        z = np.load(a.cache, allow_pickle=True)
        X, y, groups = z['X'], z['y'], z['groups']
        keys = [tuple(k) for k in z['keys']]
        if X.shape[1] != len(FEATURE_NAMES):
            print(f"\n{a.cache} holds {X.shape[1]} features but this version "
                  f"computes {len(FEATURE_NAMES)} — re-extracting.")
            X = None
        else:
            print(f"\nReusing features from {a.cache} "
                  f"({X.shape[0]:,} masks x {X.shape[1]} features)")
    if X is None:
        print("\nExtracting features…")
        X, y, groups, keys = build(recs, a.signal_channel, a.dapi_channel,
                                   a.pixel_size)
        if X is not None and a.cache:
            np.savez_compressed(a.cache, X=X, y=y, groups=groups,
                                keys=np.array(keys, dtype=object))
            print(f"  cached to {a.cache}")
    if X is None:
        sys.exit("No usable masks.")
    print(f"  {X.shape[0]:,} masks x {X.shape[1]} features")

    # split by IMAGE: masks from one image share staining and background, and
    # splitting by mask lets the forest memorise the image instead of the
    # decision
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    tr, te = next(gss.split(X, y, groups=groups))
    print(f"\ntrain: {len(tr)} masks from {len(set(groups[tr]))} images")
    print(f"test : {len(te)} masks from {len(set(groups[te]))} images "
          f"(held out entirely)\n")

    # Weight the classes so a wrongly accepted mask hurts more than a wrongly
    # rejected one. 'balanced' alone only corrects for class sizes and treats
    # the two mistakes as equally bad, which they are not here.
    n_pos_tr = int((y[tr] == 1).sum())
    n_neg_tr = int((y[tr] == 0).sum())
    w = {1: len(tr) / (2.0 * max(n_pos_tr, 1)),
         0: len(tr) / (2.0 * max(n_neg_tr, 1)) * a.reject_weight}
    print(f"class weights: keep {w[1]:.2f}, reject {w[0]:.2f} "
          f"(reject-weight {a.reject_weight})")
    clf = RandomForestClassifier(n_estimators=a.trees,
                                 min_samples_leaf=a.min_leaf,
                                 n_jobs=-1, random_state=0,
                                 class_weight=w)
    clf.fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:, 1]

    print("Per-mask accept/reject on held-out images")
    try:
        print(f"  ROC AUC {roc_auc_score(y[te], p):.3f}   "
              f"(0.5 = coin flip, 1.0 = perfect)")
    except ValueError:
        pass
    for cut in (0.3, 0.4, 0.5, 0.6, 0.7):
        pred = p >= cut
        tp_ = int((pred & (y[te] == 1)).sum())
        fp_ = int((pred & (y[te] == 0)).sum())
        fn_ = int((~pred & (y[te] == 1)).sum())
        prec = tp_ / max(tp_ + fp_, 1)
        rec = tp_ / max(tp_ + fn_, 1)
        print(f"  cut {cut}: precision {prec:.3f} recall {rec:.3f}")

    print("\nSize choice — the actual task")
    te_keys = [keys[i] for i in te]
    cuts = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    print("  exact/within1/too big/too small are of somas that HAVE an "
          "acceptable mask;")
    print("  'none ok' is of somas where you rejected every size.")
    print(f"\n  {'rule':>8} {'cut':>5} {'exact':>7} {'within1':>8} "
          f"{'too big':>8} {'too small':>10} {'none ok':>8} {'steps':>7}")
    best = None
    for rule in ('largest', 'band', 'edge'):
        for cut in cuts:
            r = size_choice_report(te_keys, y[te], p, cut, rule, quiet=True,
                                   over_w=a.oversize_cost)
            # Mean distance from your choice, in ladder steps, overshoots
            # weighted heavier. Counting only the DIRECTION of the error let a
            # threshold that is almost always far too small score best, since
            # never overshooting is trivially achieved by never reaching.
            cost = r['steps']
            mark = ''
            if best is None or cost < best['cost']:
                best = dict(rule=rule, cut=cut, cost=cost, **r)
                mark = '  <-'
            print(f"  {rule:>8} {cut:>5} {100 * r['exact']:6.1f}% "
                  f"{100 * r['within']:7.1f}% {100 * r['over']:7.1f}% "
                  f"{100 * r['under']:9.1f}% {100 * r['none_ok']:7.1f}% "
                  f"{cost:7.3f}{mark}")

    print(f"\nChosen: {best['rule']} rule at cut {best['cut']}")
    size_choice_report(te_keys, y[te], p, best['cut'], best['rule'],
                       over_w=a.oversize_cost)
    if a.boot:
        print(f"\n  95% confidence intervals ({a.boot} draws, resampling "
              f"images):")
        ci = bootstrap_ci(te_keys, y[te], p, best['cut'], best['rule'],
                          a.oversize_cost, n_boot=a.boot)
        for lbl, f in (('picked exactly your mask', 'exact'),
                       ('within one size step', 'within'),
                       ('chose too large', 'over'),
                       ('chose too small', 'under')):
            lo, hi = ci[f]
            print(f"    {lbl:26s} {100 * lo:5.1f}% – {100 * hi:5.1f}%")
    print(f"\n  (chosen by the smallest mean size error, counting a step too "
          f"large as {a.oversize_cost} steps too small)")

    imp = sorted(zip(FEATURE_NAMES, clf.feature_importances_),
                 key=lambda z: -z[1])[:10]
    print("\nMost useful features")
    for n, v in imp:
        print(f"  {v:.3f}  {n}")

    joblib.dump({'model': clf, 'meta': dict(
        features=FEATURE_NAMES, pixel_size_um=a.pixel_size,
        signal_channel=a.signal_channel, dapi_channel=a.dapi_channel,
        prob_cut=best['cut'], select_rule=best['rule'],
        oversize_cost=a.oversize_cost)}, a.out, compress=3)
    print(f"\nSaved -> {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")
    print("\nHow to read this:")
    print("  'picked exactly your mask' is the number that matters. Every")
    print("  soma you would have had to size by hand, it sizes for you.")
    print("  'within one size step' is the softer version -- close enough that")
    print("  a glance confirms it.")
    print("  'chose too large' is the dangerous direction: an oversized mask")
    print("  pulls in neighbouring processes and inflates every morphology")
    print("  metric downstream. Prefer a higher cut if that number is not small.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted — nothing saved", flush=True)
        os._exit(130)
