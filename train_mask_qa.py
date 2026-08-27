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
]


def mask_features(mask, soma_mask, sig, dapi, target_area, pixel_size):
    """One mask -> one feature vector. Order must match FEATURE_NAMES."""
    f = []
    m = mask > 0
    area = int(m.sum())
    if area < 5:
        return None
    px2 = pixel_size ** 2

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
    if soma_mask is not None and soma_mask.any():
        s = soma_mask > 0
        f.append(float(s.sum()) / max(area, 1))
        sy, sx = np.nonzero(s)
        f.append(float(np.hypot(ys.mean() - sy.mean(), xs.mean() - sx.mean())))
    else:
        f += [0.0, 0.0]
    f.append(1.0 if (y0 == 0 or x0 == 0 or y1 >= m.shape[0]
                     or x1 >= m.shape[1]) else 0.0)

    # IBA1 signal: is the mask sitting on the cell, and does it stop at an edge?
    ring = ndimage.binary_dilation(m, iterations=6) & ~m
    sin_ = sig[m]
    sring = sig[ring] if ring.any() else np.array([0.0])
    bg = float(np.median(sig))
    f += [float(sin_.mean()), float(np.percentile(sin_, 90)),
          float(sring.mean()),
          float(sin_.mean()) / (float(sring.mean()) + 1e-6),
          float((sin_ > bg).mean())]
    gy, gx = np.gradient(ndimage.gaussian_filter(sig, 1.5))
    gm = np.hypot(gy, gx)
    edge = m & ~ndimage.binary_erosion(m)
    f.append(float(gm[edge].mean()) if edge.any() else 0.0)

    # DAPI: a real soma contains a nucleus
    if dapi is not None:
        d = dapi.astype(np.float64)
        thr = float(np.percentile(d, 99)) * 0.35
        pos = d >= max(thr, 1.0)
        f += [float(d[m].mean()), float(pos[m].mean())]
        f.append(float(pos[soma_mask > 0].mean())
                 if soma_mask is not None and soma_mask.any() else 0.0)
        if pos.any():
            dist = ndimage.distance_transform_edt(~pos)
            f.append(float(dist[m].min()))
        else:
            f.append(float(max(m.shape)))
    else:
        f += [0.0, 0.0, 0.0, 0.0]

    return np.asarray(f, dtype=np.float32)


def load_channels(path, sig_ch, dapi_ch):
    a = np.squeeze(np.asarray(tifffile.imread(path)))
    if a.ndim == 2:
        return a.astype(np.float64), None
    ax = int(np.argmin(a.shape))
    if a.shape[ax] <= 8:
        a = np.moveaxis(a, ax, -1)
    else:
        return a.max(axis=0).astype(np.float64), None
    sig = a[:, :, min(sig_ch - 1, a.shape[2] - 1)].astype(np.float64)
    dapi = (a[:, :, dapi_ch - 1].astype(np.float64)
            if dapi_ch and dapi_ch <= a.shape[2] else None)
    return sig, dapi


def build(records, sig_ch, dapi_ch, pixel_size, verbose_every=200):
    X, y, groups, keys = [], [], [], []
    cache_img, cache_path = None, None
    cache_soma, cache_soma_path = None, None
    for i, rec in enumerate(records):
        try:
            if rec['image_path'] != cache_path:
                cache_img = load_channels(rec['image_path'], sig_ch, dapi_ch)
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
            v = mask_features(mask, soma, sig, dapi, rec['area'], pixel_size)
            if v is None:
                continue
            X.append(v)
            y.append(rec['label'])
            groups.append(rec['base'])
            keys.append((rec['base'], rec['row'], rec['col'], rec['area']))
        except Exception as e:
            print(f"    skipped {os.path.basename(rec['mask_path'])}: {e}")
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"    {i + 1}/{len(records)} masks processed")
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
    if rule == 'largest':
        return max(a for a, o in zip(areas, ok) if o)
    best_i = max(range(len(probs)), key=lambda i: probs[i])
    if not ok[best_i]:
        return None
    hi = best_i
    while hi + 1 < len(ok) and ok[hi + 1]:
        hi += 1
    return areas[hi]


def size_choice_report(keys, y_true, p, cut, rule='largest', quiet=False):
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
            continue
        if p_best is None:
            under += 1
            continue
        if p_best == t_best:
            exact += 1
            within += 1
        else:
            ti, pi = areas.index(t_best), areas.index(p_best)
            if abs(ti - pi) <= 1:
                within += 1
            if pi > ti:
                over += 1
            else:
                under += 1
    if not quiet:
        print(f"  somas scored: {total}   (of which {none_tot} had no "
              f"acceptable mask at all)")
        print(f"  picked exactly your mask:      "
              f"{100 * exact / max(total, 1):5.1f}%")
        print(f"  within one size step:          "
              f"{100 * within / max(total, 1):5.1f}%")
        print(f"  chose too large:               "
              f"{100 * over / max(total, 1):5.1f}%")
        print(f"  chose too small / nothing:     "
              f"{100 * under / max(total, 1):5.1f}%")
        if none_tot:
            print(f"  correctly said 'none':         "
                  f"{100 * none_ok / none_tot:5.1f}% of the {none_tot}")
    return dict(total=total, exact=exact / max(total, 1),
                within=within / max(total, 1), over=over / max(total, 1),
                under=under / max(total, 1))


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
    print(f"signal channel {a.signal_channel}, DAPI channel "
          f"{a.dapi_channel or 'none'}\n")

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

    print("\nExtracting features…")
    X, y, groups, keys = build(recs, a.signal_channel, a.dapi_channel,
                               a.pixel_size)
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
    print(f"\n  {'rule':>8} {'cut':>5} {'exact':>7} {'within1':>8} "
          f"{'too big':>8} {'too small':>10} {'cost':>7}")
    best = None
    for rule in ('largest', 'band'):
        for cut in cuts:
            r = size_choice_report(te_keys, y[te], p, cut, rule, quiet=True)
            # what we actually minimise: a miss costs 1, an oversize costs more
            cost = r['under'] + a.oversize_cost * r['over']
            mark = ''
            if best is None or cost < best['cost']:
                best = dict(rule=rule, cut=cut, cost=cost, **r)
                mark = '  <-'
            print(f"  {rule:>8} {cut:>5} {100 * r['exact']:6.1f}% "
                  f"{100 * r['within']:7.1f}% {100 * r['over']:7.1f}% "
                  f"{100 * r['under']:9.1f}% {cost:7.3f}{mark}")

    print(f"\nChosen: {best['rule']} rule at cut {best['cut']}")
    size_choice_report(te_keys, y[te], p, best['cut'], best['rule'])
    print(f"\n  (chosen by lowest cost, where choosing too large counts "
          f"{a.oversize_cost}x a miss)")

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
