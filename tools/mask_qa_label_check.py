#!/usr/bin/env python3
"""What is in the QA labels, before any image is opened.

A .mmps_session carries every QA decision and every soma position, so two
questions can be answered from the session file alone, on a laptop, with the
acquisition drive nowhere in sight:

  1. What shape do the labels have? Are they a per-cell cutoff, and were they
     reviewed or bulk-approved?
  2. How much of the cutoff is CROWDING -- how much can be predicted from where
     the cells are, with no intensity at all?

The second is run by feeding the feature code a flat image, so every intensity
feature comes out constant and the forest can only use geometry. Whatever it
scores is a floor: it is what the model knows before it has looked at a single
pixel of stain. If that already beats "approve everything", crowding carries
the decision. If it does not, the intensity features are doing the work and
this cannot be judged without the images.

    python3 tools/mask_qa_label_check.py MaskQAComplete.mmps_session
"""
import os
import sys
import collections
import importlib.util

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_trainer():
    spec = importlib.util.spec_from_file_location(
        'trainer', os.path.join(ROOT, 'train_mask_qa_model.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip().splitlines()[-1].strip())
    t = load_trainer()
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold

    recs = []
    for path in sys.argv[1:]:
        recs += t.read_session(path, [])
    n_cells = sum(len(r['somas']) for r in recs)
    if not recs:
        sys.exit("No reviewed cells in those sessions.")
    print(f"{len(recs)} images, {n_cells} reviewed cells\n")

    # ---- 1. the shape of the labels ------------------------------------
    cut = collections.Counter()
    for r in recs:
        for s in r['somas']:
            cut[s['cutoff']] += 1
    a_max = max(max(s['areas']) for r in recs for s in r['somas'])
    print("cutoff (largest approved size), µm²:")
    for a in sorted(cut):
        bar = '#' * int(60 * cut[a] / max(cut.values()))
        print(f"  {int(a):4d}  {cut[a]:5d}  {bar}")
    sat = cut[a_max]
    tail = sum(r.get('bulk_tail', 0) for r in recs)
    print(f"\n  reject everything: {cut[0]:,} cells "
          f"({100 * cut[0] / n_cells:.0f}%)")
    print(f"  approve everything: {sat:,} cells "
          f"({100 * sat / n_cells:.0f}%), of which {tail:,} sit in a run at "
          f"the end of an image's queue")
    if tail > 0.25 * max(sat, 1):
        print("  -> looks like \"Approve All Remaining\" rather than review; "
              "train with --drop-saturated")

    # ---- 2. how far crowding alone gets --------------------------------
    print("\nCrowding only (flat image, no intensity), split by image…")
    H = W = 0
    for r in recs:
        for c in r['all_centroids']:
            H = max(H, int(c[0]) + 200)
            W = max(W, int(c[1]) + 200)
    flat = np.zeros((H, W), dtype=np.float64)
    X, y, g, meta = [], [], [], []
    for r in recs:
        px = r['pixel_size'] or 0.316
        cents = r['all_centroids'] or [s['centroid'] for s in r['somas']]
        for s in r['somas']:
            rows = t.soma_feature_rows(flat, s['centroid'], s['polygon'],
                                       cents, px, s['areas'])
            for i, a in enumerate(s['areas']):
                X.append(rows[i])
                y.append(1 if s['labels'][i] else 0)
                g.append(r['image_name'])
                meta.append((r['image_name'], s['soma_id'], float(a),
                             s['cutoff']))
    X = np.nan_to_num(np.asarray(X, dtype=np.float64))
    y, g = np.asarray(y), np.asarray(g)
    n_folds = min(5, len(set(g)))
    if n_folds < 3:
        sys.exit("Need at least 3 images to split by image.")
    oof = np.zeros(len(X))
    for tr, te in GroupKFold(n_splits=n_folds).split(X, y, g):
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=8,
                                     class_weight='balanced_subsample',
                                     n_jobs=-1, random_state=0)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]

    cells = t.cells_from_rows(meta, oof, 0.5)
    step = float(np.median(np.diff(sorted({m[2] for m in meta}))))
    s = t.score_cells(cells, step)
    print(f"  masks: acc {np.mean((oof >= .5).astype(int) == y):.3f}   "
          f"AUC {t._auc(y, oof):.3f}")
    print(f"  cells: exact {100 * s['exact']:.0f}%   "
          f"within one step {100 * s['within1']:.0f}%   "
          f"median error {s['median_err']:.1f} steps")
    for nm, (ex, w1, v) in t.baselines(
            cells, np.array([m[3] for m in meta]), step).items():
        print(f"  baseline {nm} ({v:.0f} µm²): exact {100 * ex:.0f}%   "
              f"within one {100 * w1:.0f}%")

    conf = np.array([c['conf'] for c in cells])
    err = np.array([abs(c['pred'] - c['true']) for c in cells]) / step
    o = np.argsort(-conf)
    q = len(o) // 4
    print("\n  confidence triage:")
    for k in range(4):
        sl = o[k * q:(k + 1) * q] if k < 3 else o[3 * q:]
        print(f"    {['most', '2nd', '3rd', 'least'][k]:>5s} quarter: "
              f"exact {100 * np.mean(err[sl] < 0.5):3.0f}%   "
              f"within one {100 * np.mean(err[sl] < 1.5):3.0f}%")

    best = max(v[0] for v in t.baselines(
        cells, np.array([m[3] for m in meta]), step).values())
    print()
    if s['exact'] > best + 0.05:
        print("  Crowding alone already beats the free answer -> where the "
              "cells sit carries a real part of the decision.")
    else:
        print("  Crowding alone does NOT beat the free answer. Either the "
              "intensity features carry the decision -- which needs the "
              "images -- or the cutoff is not predictable from the cell at "
              "all. Nothing here can tell those apart; run "
              "train_mask_qa_model.py against the drive to find out.")


if __name__ == '__main__':
    main()
