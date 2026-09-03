#!/usr/bin/env python3
"""Fail if MMPS's copy of the mask-QA feature code drifts from the trainer's.

The forest is fitted on features from train_mask_qa.py. MMPS holds a copy so
the app stays one file. If they diverge the model keeps choosing mask sizes --
worse ones -- with nothing raising, which is the hardest failure to notice.

    python3 tools/test_maskqa_parity.py
"""
import os
import re
import sys
import types

import numpy as np
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED = ['_mq_image_stats', '_mq_mask_features', '_mq_pick',
            '_mq_soma_confidence']


def load(path, names, prefix=''):
    src = open(path).read()
    mod = types.ModuleType('probe')
    mod.__dict__.update({'np': np, 'ndimage': ndimage, 'ndi': ndimage})
    import cv2
    from skimage.morphology import skeletonize
    mod.__dict__.update({'cv2': cv2, 'skeletonize': skeletonize})
    out = []
    for n in names:
        m = re.search(
            rf'^def {prefix}{n}\(.*?(?=\n\ndef |\n\nclass |\n\n# ---|\n\nFEATURE|\n\n_MQ)',
            src, re.S | re.M)
        if not m:
            sys.exit(f"could not find {prefix}{n} in {os.path.basename(path)}")
        out.append(m.group(0))
    code = "\n\n".join(out)
    if prefix:
        for n in names:
            code = code.replace(f'{prefix}{n}(', f'{n}(')
    exec(compile(code, path, 'exec'), mod.__dict__)
    return mod


def check_structure():
    src = open(os.path.join(ROOT, 'MMPSv2.12.py')).read()
    missing = [n for n in REQUIRED if f'def {n}(' not in src]
    if '_MQ_FEATURE_NAMES' not in src:
        missing.append('_MQ_FEATURE_NAMES')
    if missing:
        sys.exit("FAIL: MMPSv2.12.py is missing part of the mask-QA path: "
                 + ", ".join(missing))
    print(f"PASS: all {len(REQUIRED) + 1} mask-QA symbols present")


def main():
    check_structure()
    fns = ['image_stats', 'mask_features', 'pick', 'soma_confidence']
    tr = load(os.path.join(ROOT, 'train_mask_qa.py'), fns)
    ap = load(os.path.join(ROOT, 'MMPSv2.12.py'), fns, prefix='_mq_')

    rng = np.random.default_rng(11)
    fails = 0
    for trial in range(12):
        H = W = int(rng.integers(140, 260))
        yy, xx = np.ogrid[:H, :W]
        cy, cx = H // 2, W // 2
        soma = ((yy - cy) ** 2 + (xx - cx) ** 2) < 12 ** 2
        r = int(rng.integers(25, 55))
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2) < r ** 2
        sig = mask * 180 + rng.normal(30, 9, (H, W))
        dapi = soma * 200 + rng.normal(20, 7, (H, W))

        g1, b1, d1 = tr.image_stats(sig, dapi)
        g2, b2, d2 = ap.image_stats(sig, dapi)
        if not (np.allclose(g1, g2) and b1 == b2 and d1 == d2):
            print(f"  trial {trial}: IMAGE STATS DIFFER")
            fails += 1
            continue

        nbrs = [(cy, cx), (cy, cx + int(rng.integers(60, 160)))]
        a = tr.mask_features(mask, soma, sig, dapi, 200, 0.104, g1, b1, d1,
                             centre=(cy, cx), neighbours=nbrs)
        b = ap.mask_features(mask, soma, sig, dapi, 200, 0.104, g2, b2, d2,
                             centre=(cy, cx), neighbours=nbrs)
        if a is None or b is None:
            if (a is None) != (b is None):
                print(f"  trial {trial}: one returned None")
                fails += 1
            continue
        if a.shape != b.shape or not np.array_equal(a, b):
            bad = np.flatnonzero(a != b) if a.shape == b.shape else []
            print(f"  trial {trial}: FEATURES DIFFER at indices {list(bad)[:6]}")
            fails += 1

        areas = [50 * (i + 1) for i in range(8)]
        probs = list(rng.random(8))
        for rule in ('largest', 'band', 'edge'):
            for cut in (0.4, 0.6):
                if tr.pick(areas, probs, cut, rule) != ap.pick(areas, probs,
                                                               cut, rule):
                    print(f"  trial {trial} {rule}@{cut}: PICK DIFFERS")
                    fails += 1
        ch = tr.pick(areas, probs, 0.5, 'band')
        if tr.soma_confidence(areas, probs, ch) != ap.soma_confidence(
                areas, probs, ch):
            print(f"  trial {trial}: CONFIDENCE DIFFERS")
            fails += 1

    if fails:
        sys.exit(f"\nFAIL: {fails} mismatch(es). MMPS's mask-QA code has "
                 f"drifted from train_mask_qa.py — the model will choose worse "
                 f"sizes until they match again.")
    print("PASS: MMPS and train_mask_qa.py produce identical features, "
          "picks and confidence (12 random cells, 3 rules, 2 cuts).")


if __name__ == '__main__':
    main()
