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


def _conf_at(T, clf, img, ex_full, r, c, scales, half, mode, open_r, app_px,
             train_px):
    """Confidence for one soma at a given assumed pixel size and click."""
    import cv2
    hi_px = max(8, int(round(half * train_px / max(app_px, 1e-6))))
    patch, y1, x1 = T.patch_around(img, int(r), int(c), hi_px)
    if patch.size == 0 or min(patch.shape[:2]) < 8:
        return None
    side = 2 * half
    work = cv2.resize(patch, (side, side), interpolation=cv2.INTER_LINEAR)
    ctr = ((int(r) - y1) * side / patch.shape[0],
           (int(c) - x1) * side / patch.shape[1])
    ex = [cv2.resize(e[y1:y1 + patch.shape[0], x1:x1 + patch.shape[1]],
                     (side, side), interpolation=cv2.INTER_LINEAR)
          for e in ex_full]
    F = T.pixel_features(work, scales, center=ctr, extra=(ex if ex else None))
    prob = clf.predict_proba(F)[:, 1].reshape(work.shape)
    lo = T.mask_from_prob(prob, ctr, 0.35, open_r, mode)
    hi = T.mask_from_prob(prob, ctr, 0.65, open_r, mode)
    if lo is None or hi is None:
        return 0.0
    u = np.logical_or(lo, hi).sum()
    return float(np.logical_and(lo, hi).sum()) / u if u else 0.0


def run_sweep(T, clf, pairs, scales, half, cut, mode, open_r, ch, extra_ch,
              train_px):
    """Which wrong assumption would drive confidence to zero?

    Everything measurable outside the app checks out, so the fault is in an
    argument the app supplies. There are two: the pixel size, which sets how
    large a window is cropped, and the click, which anchors every centre-based
    feature. Vary each and watch where confidence collapses.
    """
    print("\nPIXEL SIZE the app believes it has (trained at "
          f"{train_px} um/px)")
    print(f"  {'px':>7} {'crop':>6}   median confidence over "
          f"{len(pairs)} somas")
    for app_px in (0.052, 0.1046, 0.104, 0.15, 0.2, 0.25, 0.316, 0.5):
        vals = []
        for mp, ip, r, c, tp in pairs:
            img = T.load_gray(ip, ch)
            ex = T.load_channels(ip, extra_ch)
            v = _conf_at(T, clf, img, ex, r, c, scales, half, mode, open_r,
                         app_px, train_px)
            if v is not None:
                vals.append(v)
        crop = max(8, int(round(half * train_px / max(app_px, 1e-6))))
        med = float(np.median(vals)) if vals else float('nan')
        flag = '   <-- collapses' if med < 0.05 else ''
        print(f"  {app_px:7.4f} {crop:5d}px   {med:.3f}{flag}")

    print("\nCLICK OFFSET from the recorded soma position")
    print(f"  {'offset':>8}   median confidence")
    for off in (0, 5, 10, 20, 30, 45, 60, 80):
        vals = []
        for mp, ip, r, c, tp in pairs:
            img = T.load_gray(ip, ch)
            ex = T.load_channels(ip, extra_ch)
            v = _conf_at(T, clf, img, ex, int(r) + off, int(c), scales, half,
                         mode, open_r, train_px, train_px)
            if v is not None:
                vals.append(v)
        med = float(np.median(vals)) if vals else float('nan')
        flag = '   <-- collapses' if med < 0.05 else ''
        print(f"  {off:6d}px   {med:.3f}{flag}")
    print()


def load_mmps_loader():
    """Pull load_tiff_image out of MMPSv2.12.py without importing the GUI."""
    src = open(os.path.join(HERE, 'MMPSv2.12.py')).read()
    m = re.search(r'^def load_tiff_image\(.*?(?=\n\ndef |\n\nclass )',
                  src, re.S | re.M)
    if not m:
        sys.exit("could not find load_tiff_image in MMPSv2.12.py")
    mod = types.ModuleType('mm')
    mod.__dict__.update({'np': np, 'os': os, 'sys': sys})
    exec(compile(m.group(0), 'MMPSv2.12.py', 'exec'), mod.__dict__)
    return mod.load_tiff_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--model', default='soma_model.joblib')
    ap.add_argument('--pixel-size', type=float, default=0.1046)
    ap.add_argument('--app-pixel-size', type=float, default=None,
                    help='pixel size MMPS is using, if different')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--sweep', action='store_true',
                    help='vary the pixel size and the click position to find '
                         'which one collapses confidence to zero')
    ap.add_argument('--mmps-loader', action='store_true',
                    help="read the image through MMPS's own load_tiff_image "
                         "instead of tifffile.imread, which is the remaining "
                         "difference between this script and the running app")
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

    if a.sweep:
        run_sweep(T, clf, pairs[:a.n], scales, half, cut, mode, open_r, ch,
                  extra_ch, train_px)
        return

    print(f"{'soma':>4} {'path':>11} {'max':>6} {'@click':>7} "
          f"{'>=0.35':>7} {'>=0.65':>7} {'strict':>7} {'conf':>6}")
    mmps_load = load_mmps_loader() if a.mmps_loader else None
    if mmps_load:
        print("reading through MMPS's load_tiff_image\n")

    for i, (mp, ip, r, c, tp) in enumerate(pairs[:a.n]):
        if mmps_load:
            raw = np.squeeze(np.asarray(mmps_load(ip)))
            if raw.ndim == 3:
                ax = int(np.argmin(raw.shape))
                if raw.shape[ax] <= 8:
                    raw = np.moveaxis(raw, ax, -1)
            img = raw[:, :, (ch or 1) - 1].astype(np.float64)
            ex_full = [raw[:, :, k - 1].astype(np.float64) for k in extra_ch]
            if i == 0:
                print(f"  MMPS loader gave {raw.shape} {raw.dtype}; "
                      f"channel means "
                      + ", ".join(f"{raw[:, :, k].mean():.2f}"
                                  for k in range(raw.shape[2])))
                t = np.squeeze(tifffile.imread(ip))
                print(f"  tifffile gave    {t.shape} {t.dtype}; "
                      f"channel means "
                      + ", ".join(f"{t[:, :, k].mean():.2f}"
                                  for k in range(t.shape[2])) + "\n")
        else:
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
