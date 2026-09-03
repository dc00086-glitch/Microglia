#!/usr/bin/env python3
"""train_mask_ring.py — learn where to STOP growing a mask.

The per-mask classifier tells a good mask from a bad one well (AUC 0.887) but
locates the right SIZE only 41% of the time. Adjacent masks share about 85% of
their pixels, so their whole-mask features -- area, skeleton length,
circularity -- barely differ, and the model is asked to split two nearly
identical vectors.

That is not the decision being made. Going from 200 to 250 um2 adds a RING of
pixels, and the question is whether that new tissue belongs to this cell or the
one next door. It is a property of the increment, and the increment is where the
signal is: it changes sharply between the last good step and the first bad one,
where whole-mask features change smoothly.

So this scores each STEP up the ladder, then walks up while the steps are good
and stops at the first bad one.

    python3 train_mask_ring.py --root "<study root>" --timepoints . \\
        --image-subdir "Image Storage Directory" --pixel-size 0.104

Shares the gathering and image loading with train_mask_qa.py, which must sit
beside it.
"""
import os
import sys
import hashlib
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import tifffile
    from scipy import ndimage
    import cv2
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import roc_auc_score
    import joblib
except ImportError as e:
    sys.exit(f"Missing dependency: {e}")

import train_mask_qa as Q


RING_FEATURES = [
    'from_area', 'to_area', 'step_index', 'n_steps',
    'ring_px', 'ring_frac_of_new',
    'ring_mean', 'ring_p90', 'ring_frac_above_bg',
    'ring_vs_interior', 'ring_vs_beyond',
    'ring_components', 'ring_compactness',
    'reach_before', 'reach_growth',
    'ring_nbr_dist', 'ring_frac_nbr_territory',
    'ring_dapi_mean', 'ring_dapi_frac',
    'prev_solidity', 'prev_area_um2',
]


def ring_features(prev, new, sig, dapi, centre, neighbours, pixel_size,
                  bg, dapi_thr, from_area, to_area, step_index, n_steps):
    """Describe the pixels ADDED going from one size to the next."""
    ring = new & ~prev
    n = int(ring.sum())
    if n < 5:
        return None
    H, W = new.shape
    f = [float(from_area), float(to_area), float(step_index), float(n_steps),
         float(n), n / max(float(new.sum()), 1.0)]

    ys, xs = np.nonzero(ring)
    y0, y1 = max(0, ys.min() - 8), min(H, ys.max() + 9)
    x0, x1 = max(0, xs.min() - 8), min(W, xs.max() + 9)
    r_c = ring[y0:y1, x0:x1]
    s_c = sig[y0:y1, x0:x1]
    p_c = prev[y0:y1, x0:x1]

    rv = s_c[r_c]
    f += [float(rv.mean()), float(np.percentile(rv, 90)),
          float((rv > bg).mean())]

    # Is the new tissue as bright as what is already in the mask? A ring that
    # is much dimmer is background being swept up rather than more cell.
    iv = s_c[p_c] if p_c.any() else rv
    f.append(float(rv.mean()) / (float(iv.mean()) + 1e-6))
    # And does signal continue beyond it, or has the cell ended here?
    beyond = ndimage.binary_dilation(r_c, iterations=6) & ~r_c & ~p_c
    bv = s_c[beyond] if beyond.any() else np.array([bg])
    f.append(float(bv.mean()) / (float(rv.mean()) + 1e-6))

    # A legitimate step adds a connected collar. Reaching into a neighbour
    # arrives as several disconnected spidery pieces.
    lab, ncomp = ndimage.label(r_c)
    per = float(np.logical_xor(r_c, ndimage.binary_erosion(r_c)).sum())
    f += [float(ncomp), (per * per) / max(float(n), 1.0)]

    cy, cx = float(centre[0]), float(centre[1])
    pys, pxs = np.nonzero(prev)
    reach_prev = float(np.hypot(pys - cy, pxs - cx).max()) if len(pys) else 0.0
    reach_new = float(np.hypot(ys - cy, xs - cx).max())
    f += [reach_prev, reach_new / max(reach_prev, 1.0)]

    nb = [q for q in (neighbours or [])
          if abs(q[0] - cy) > 1e-6 or abs(q[1] - cx) > 1e-6]
    if nb:
        pts = np.stack([ys, xs], axis=1).astype(np.float64)
        own = np.hypot(pts[:, 0] - cy, pts[:, 1] - cx)
        other = np.full(own.shape, np.inf)
        for q in nb:
            other = np.minimum(other, np.hypot(pts[:, 0] - q[0],
                                               pts[:, 1] - q[1]))
        f += [float(other.min()), float((other < own).mean())]
    else:
        f += [float(max(H, W)), 0.0]

    if dapi is not None:
        d_c = dapi[y0:y1, x0:x1]
        dv = d_c[r_c]
        f += [float(dv.mean()),
              float((dv >= (dapi_thr if dapi_thr else 0)).mean())]
    else:
        f += [0.0, 0.0]

    # a little context about where we already are
    cnts, _ = cv2.findContours(p_c.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.contourArea(cv2.convexHull(c))
        f.append(float(prev.sum()) / hull if hull > 0 else 0.0)
    else:
        f.append(0.0)
    f.append(float(prev.sum()) * (pixel_size ** 2))
    return np.asarray(f, dtype=np.float32)


def build_steps(recs, sig_ch, dapi_ch, pixel_size, verbose_every=200):
    """One row per STEP up the ladder. Label 1 = that step was still accepted."""
    by_soma = {}
    for r in recs:
        by_soma.setdefault((r['image_path'], r['base'], r['row'], r['col']),
                           []).append(r)
    per_image = {}
    for r in recs:
        per_image.setdefault(r['image_path'], set()).add((r['row'], r['col']))
    per_image = {k: sorted(v) for k, v in per_image.items()}

    X, y, groups, keys = [], [], [], []
    cache_path = cache = grad = bg = dthr = None
    done = skipped = 0
    for (ip, base, rr, cc), items in sorted(by_soma.items()):
        items.sort(key=lambda r: r['area'])
        try:
            if ip != cache_path:
                cache = Q.load_channels(ip, sig_ch, dapi_ch)
                grad, bg, dthr = Q.image_stats(*cache)
                cache_path = ip
            sig, dapi = cache
            masks = []
            for r in items:
                m = np.squeeze(np.asarray(tifffile.imread(r['mask_path'])))
                if m.shape != sig.shape:
                    raise ValueError('mask/image size mismatch')
                masks.append(m > 0)
            sp = items[0].get('soma_path')
            base_mask = (np.squeeze(np.asarray(tifffile.imread(sp))) > 0
                         if sp and os.path.exists(sp) else None)
            if base_mask is None:
                base_mask = np.zeros_like(masks[0])
            nbrs = per_image.get(ip, [])
            prev = base_mask
            for i, (r, m) in enumerate(zip(items, masks)):
                v = ring_features(prev, m | prev, sig, dapi, (rr, cc), nbrs,
                                  pixel_size, bg, dthr,
                                  items[i - 1]['area'] if i else 0.0,
                                  r['area'], i, len(items))
                prev = m | prev
                if v is None:
                    continue
                X.append(v)
                y.append(r['label'])          # was this size still accepted?
                groups.append(base)
                keys.append((base, rr, cc, r['area']))
            done += 1
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"    skipped soma {base[:40]} {rr},{cc}: {e}")
        if verbose_every and done % verbose_every == 0:
            print(f"    {done}/{len(by_soma)} somas processed")
    if skipped:
        print(f"  {skipped} somas skipped")
    if not X:
        return None, None, None, None
    return np.vstack(X), np.asarray(y), np.asarray(groups), keys


def walk(areas, probs, cut):
    """Climb the ladder while each step is judged good; stop at the first bad."""
    chosen = None
    for a, p in zip(areas, probs):
        if p < cut:
            break
        chosen = a
    return chosen


def report(keys, y_true, p, cut, quiet=False, over_w=3.0, miss_w=4.0):
    somas = {}
    for k, t, pr in zip(keys, y_true, p):
        somas.setdefault(k[:3], []).append((k[3], t, pr))
    exact = within = over = under = 0
    none_ok = none_tot = total = 0
    pen = 0.0
    npen = 0
    for _, rows in somas.items():
        rows.sort(key=lambda z: z[0])
        areas = [z[0] for z in rows]
        truth = [z[0] for z in rows if z[1] == 1]
        t_best = max(truth) if truth else None
        p_best = walk(areas, [z[2] for z in rows], cut)
        total += 1
        if t_best is None:
            none_tot += 1
            if p_best is None:
                none_ok += 1
            else:
                pen += over_w
                npen += 1
            continue
        if p_best is None:
            under += 1
            pen += miss_w
            npen += 1
            continue
        ti, pi = areas.index(t_best), areas.index(p_best)
        if pi == ti:
            exact += 1
            within += 1
        else:
            if abs(ti - pi) <= 1:
                within += 1
            if pi > ti:
                over += 1
            else:
                under += 1
        d = pi - ti
        pen += (over_w * d) if d > 0 else (-d)
        npen += 1
    sc = max(total - none_tot, 1)
    if not quiet:
        print(f"  {total - none_tot} somas had an acceptable mask; on those:")
        print(f"    picked exactly your mask:    {100 * exact / sc:5.1f}%")
        print(f"    within one size step:        {100 * within / sc:5.1f}%")
        print(f"    chose too large:             {100 * over / sc:5.1f}%")
        print(f"    chose too small:             {100 * under / sc:5.1f}%")
        print(f"    mean size error:             {pen / max(npen, 1):5.2f} steps")
        if none_tot:
            print(f"  {none_tot} somas had none acceptable; on those:")
            print(f"    correctly proposed nothing:  "
                  f"{100 * none_ok / none_tot:5.1f}%")
    return dict(exact=exact / sc, within=within / sc, over=over / sc,
                under=under / sc, steps=pen / max(npen, 1),
                none_ok=(none_ok / none_tot if none_tot else 1.0))


def bootstrap_ci(keys, y_true, p, cut, over_w, n_boot=2000, seed=0):
    """95% intervals by resampling IMAGES. See train_mask_qa.bootstrap_ci."""
    by_img = {}
    for i, k in enumerate(keys):
        by_img.setdefault(k[0], []).append(i)
    imgs = sorted(by_img)
    rng = np.random.default_rng(seed)
    fields = ('exact', 'within', 'over', 'under')
    acc = {f: [] for f in fields}
    for _ in range(n_boot):
        pick = rng.integers(0, len(imgs), len(imgs))
        kk, tt, pp = [], [], []
        for rep, j in enumerate(pick):
            for i in by_img[imgs[j]]:
                k = keys[i]
                kk.append((f"{k[0]}#{rep}",) + tuple(k[1:]))
                tt.append(y_true[i])
                pp.append(p[i])
        r = report(kk, np.asarray(tt), np.asarray(pp), cut, quiet=True,
                   over_w=over_w)
        for f in fields:
            acc[f].append(r[f])
    return {f: (float(np.percentile(acc[f], 2.5)),
                float(np.percentile(acc[f], 97.5))) for f in fields}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--timepoints', nargs='*', default=['.'])
    ap.add_argument('--image-subdir', default='Image Storage Directory')
    ap.add_argument('--pixel-size', type=float, default=0.104)
    ap.add_argument('--signal-channel', type=int, default=1)
    ap.add_argument('--dapi-channel', type=int, default=3)
    ap.add_argument('--trees', type=int, default=400)
    ap.add_argument('--min-leaf', type=int, default=4)
    ap.add_argument('--oversize-cost', type=float, default=3.0)
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--cache', default=None)
    ap.add_argument('--out', default='mask_ring_model.joblib')
    a = ap.parse_args()

    try:
        print("script fingerprint:",
              hashlib.md5(open(__file__, 'rb').read()).hexdigest()[:8])
    except Exception:
        pass
    print(f"Channels read: {a.signal_channel} (IBA1), {a.dapi_channel} (DAPI)\n")

    print("Gathering masks…")
    recs = Q.gather(a.root, a.timepoints, a.image_subdir)
    if not recs:
        sys.exit("No masks found.")
    print(f"\n{len(recs)} masks")

    X = None
    if a.cache and os.path.exists(a.cache):
        z = np.load(a.cache, allow_pickle=True)
        X, y, groups = z['X'], z['y'], z['groups']
        keys = [tuple(k) for k in z['keys']]
        if X.shape[1] != len(RING_FEATURES):
            print(f"cache has {X.shape[1]} features, need {len(RING_FEATURES)}"
                  f" — re-extracting")
            X = None
        else:
            print(f"Reusing {a.cache} ({X.shape[0]:,} steps)")
    if X is None:
        print("\nDescribing each step up the ladder…")
        X, y, groups, keys = build_steps(recs, a.signal_channel,
                                         a.dapi_channel, a.pixel_size)
        if X is None:
            sys.exit("Nothing usable.")
        if a.cache:
            np.savez_compressed(a.cache, X=X, y=y, groups=groups,
                                keys=np.array(keys, dtype=object))
            print(f"  cached to {a.cache}")
    print(f"  {X.shape[0]:,} steps x {X.shape[1]} features")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    tr, te = next(gss.split(X, y, groups=groups))
    print(f"\ntrain: {len(tr)} steps from {len(set(groups[tr]))} images")
    print(f"test : {len(te)} steps from {len(set(groups[te]))} images\n")

    clf = RandomForestClassifier(n_estimators=a.trees,
                                 min_samples_leaf=a.min_leaf, n_jobs=-1,
                                 random_state=0, class_weight='balanced')
    clf.fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:, 1]
    try:
        print(f"Per-step AUC {roc_auc_score(y[te], p):.3f}")
    except ValueError:
        pass

    te_keys = [keys[i] for i in te]
    print(f"\n  {'cut':>5} {'exact':>7} {'within1':>8} {'too big':>8} "
          f"{'too small':>10} {'none ok':>8} {'steps':>7}")
    best = None
    for cut in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        r = report(te_keys, y[te], p, cut, quiet=True, over_w=a.oversize_cost)
        mark = ''
        if best is None or r['steps'] < best['steps']:
            best = dict(cut=cut, **r)
            mark = '  <-'
        print(f"  {cut:>5} {100 * r['exact']:6.1f}% {100 * r['within']:7.1f}% "
              f"{100 * r['over']:7.1f}% {100 * r['under']:9.1f}% "
              f"{100 * r['none_ok']:7.1f}% {r['steps']:7.3f}{mark}")

    print(f"\nChosen cut {best['cut']}")
    report(te_keys, y[te], p, best['cut'], over_w=a.oversize_cost)
    if a.boot:
        print(f"\n  95% confidence intervals ({a.boot} draws, resampling "
              f"images):")
        ci = bootstrap_ci(te_keys, y[te], p, best['cut'], a.oversize_cost,
                          n_boot=a.boot)
        for lbl, f in (('picked exactly your mask', 'exact'),
                       ('within one size step', 'within'),
                       ('chose too large', 'over'),
                       ('chose too small', 'under')):
            lo, hi = ci[f]
            print(f"    {lbl:26s} {100 * lo:5.1f}% – {100 * hi:5.1f}%")

    imp = sorted(zip(RING_FEATURES, clf.feature_importances_),
                 key=lambda z: -z[1])[:10]
    print("\nMost useful features")
    for n_, v in imp:
        print(f"  {v:.3f}  {n_}")

    joblib.dump({'model': clf, 'meta': dict(
        features=RING_FEATURES, pixel_size_um=a.pixel_size,
        signal_channel=a.signal_channel, dapi_channel=a.dapi_channel,
        prob_cut=best['cut'], mode='ring')}, a.out, compress=3)
    print(f"\nSaved -> {a.out}")
    print("\nCompare against the per-mask model: 40.6% exact, 72.3% within one.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
        os._exit(130)
