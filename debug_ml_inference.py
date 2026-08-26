#!/usr/bin/env python3
"""debug_ml_inference.py — why does confidence read 0.00 in the app?

Confidence is the overlap between the outline at a loose cut and at a strict
one, so exactly 0.00 means the strict cut returned nothing. Validation and the
app run the same model through slightly different code -- validation feeds the
patch straight in, the app crops a physical window and resamples it -- so this
runs BOTH on the same somas and prints what each produces.

    python3 debug_ml_inference.py --root "<study root>" --model soma_model.joblib

No rebuild needed. It imports the feature code out of train_soma_model.py, so
it is the same implementation both ends.
"""
import os
import re
import sys
import glob
import types
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib
import tifffile
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))


def load_trainer():
    """Exec train_soma_model.py's functions without running main()."""
    src = open(os.path.join(HERE, 'train_soma_model.py')).read()
    src = src.split("# ----------------------------------------------------------------------\ndef main()")[0]
    mod = types.ModuleType('t')
    mod.__dict__.update({'np': np, 'ndimage': ndimage, 'tifffile': tifffile,
                         'os': os, 're': re, 'glob': glob, 'sys': sys,
                         'joblib': joblib})
    exec(compile(src, 'train_soma_model.py', 'exec'), mod.__dict__)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--model', default='soma_model.joblib')
    ap.add_argument('--pixel-size', type=float, default=0.1046)
    ap.add_argument('--app-pixel-size', type=float, default=None,
                    help='pixel size MMPS is using, if different')
    ap.add_argument('--n', type=int, default=6)
    a = ap.parse_args()

    T = load_trainer()
    b = joblib.load(a.model)
    clf, m = b['model'], b.get('meta', {}) or {}
    scales = tuple(m.get('scales') or (1., 2., 4., 8.))
    half = int(m.get('half') or 76)
    cut = float(m.get('prob_cut') or 0.5)
    mode = m.get('mode') or 'cc'
    open_r = int(m.get('open_r') or 0)
    ch = m.get('channel')
    extra_ch = list(m.get('extra_channels') or [])
    train_px = float(m.get('pixel_size_um') or a.pixel_size)
    app_px = a.app_pixel_size or a.pixel_size

    print(f"model: trained_on={m.get('trained_on')} channel={ch} "
          f"extra={extra_ch} scales={len(scales)} cut={cut} mode={mode} "
          f"open_r={open_r}")
    print(f"       trained at {train_px} um/px; app running at {app_px} um/px")
    print(f"       half={half} -> app crops "
          f"{max(8, int(round(half * train_px / max(app_px, 1e-6))))} px "
          f"then resamples to {2 * half}\n")

    pairs = T.find_pairs(a.root, ['1d', '3d', '7d', '28d'], a.n,
                         m.get('image_subdir') or 'Image Directory')
    if not pairs:
        sys.exit("no pairs found")

    try:
        import cv2
    except ImportError:
        cv2 = None
        print("cv2 unavailable — only the validation path will run\n")

    print(f"{'soma':>4} {'path':>11} {'max':>6} {'@click':>7} "
          f"{'>=0.35':>7} {'>=0.65':>7} {'strict':>7} {'conf':>6}")
    for i, (mp, ip, r, c, tp) in enumerate(pairs[:a.n]):
        img = T.load_gray(ip, ch)
        ex_full = T.load_channels(ip, extra_ch)

        for label in ('validation', 'app'):
            if label == 'app' and cv2 is None:
                continue
            if label == 'validation':
                patch, y1, x1 = T.patch_around(img, int(r), int(c), half)
                ctr = (int(r) - y1, int(c) - x1)
                ex = [e[y1:y1 + patch.shape[0], x1:x1 + patch.shape[1]]
                      for e in ex_full]
                work = patch
            else:
                hi_px = max(8, int(round(half * train_px / max(app_px, 1e-6))))
                patch, y1, x1 = T.patch_around(img, int(r), int(c), hi_px)
                side = 2 * half
                work = cv2.resize(patch, (side, side),
                                  interpolation=cv2.INTER_LINEAR)
                ctr = ((int(r) - y1) * side / patch.shape[0],
                       (int(c) - x1) * side / patch.shape[1])
                ex = [cv2.resize(e[y1:y1 + patch.shape[0],
                                   x1:x1 + patch.shape[1]],
                                 (side, side), interpolation=cv2.INTER_LINEAR)
                      for e in ex_full]
            F = T.pixel_features(work, scales, center=ctr,
                                 extra=(ex if ex else None))
            prob = clf.predict_proba(F)[:, 1].reshape(work.shape)
            cy = min(max(int(ctr[0]), 0), prob.shape[0] - 1)
            cx = min(max(int(ctr[1]), 0), prob.shape[1] - 1)
            lo_m = T.mask_from_prob(prob, ctr, 0.35, open_r, mode)
            hi_m = T.mask_from_prob(prob, ctr, 0.65, open_r, mode)
            conf = 0.0
            if lo_m is not None and hi_m is not None:
                u = np.logical_or(lo_m, hi_m).sum()
                conf = float(np.logical_and(lo_m, hi_m).sum()) / u if u else 0.0
            print(f"{i:>4} {label:>11} {prob.max():6.3f} {prob[cy, cx]:7.3f} "
                  f"{100 * (prob >= 0.35).mean():6.1f}% "
                  f"{100 * (prob >= 0.65).mean():6.1f}% "
                  f"{'None' if hi_m is None else int(hi_m.sum()):>7} "
                  f"{conf:6.3f}")
        print()


if __name__ == '__main__':
    main()
