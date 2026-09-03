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
    # A class has been deleted from this file before by an edit whose pattern
    # ran past its end, and the only symptom was a button that stopped doing
    # anything. Name every piece of the path explicitly.
    for n in ('_MaskQAModel',):
        if f'class {n}:' not in src:
            missing.append(f'class {n}')
    for n in ('get_mask_qa_model', 'mask_qa_model_paths', 'score_soma',
              '_run_ml_mask_qa', '_mq_apply_choice', '_ml_mask_qa_dialog',
              '_mq_channels_for', '_write_mq_decisions'):
        if f'def {n}(' not in src:
            missing.append(n)
    if missing:
        sys.exit("FAIL: MMPSv2.12.py is missing part of the mask-QA path: "
                 + ", ".join(missing))
    print(f"PASS: all {len(REQUIRED) + 10} mask-QA symbols present")


def load_model_class():
    """The app's _MaskQAModel, wired to its own copies of the feature code."""
    src = open(os.path.join(ROOT, 'MMPSv2.12.py')).read()
    parts = []
    m = re.search(r'^_MQ_FEATURE_NAMES = \[.*?^\]', src, re.S | re.M)
    parts.append(m.group(0))
    for n in REQUIRED:
        m = re.search(
            rf'^def {n}\(.*?(?=\n\ndef |\n\nclass |\n\n# ---)', src,
            re.S | re.M)
        parts.append(m.group(0))
    m = re.search(r'^class _MaskQAModel:.*?(?=\n\n_MQ_MODEL = )', src,
                  re.S | re.M)
    if not m:
        sys.exit("FAIL: could not read class _MaskQAModel out of MMPS")
    parts.append(m.group(0))

    mod = types.ModuleType('probe_cls')
    import cv2
    from skimage.morphology import skeletonize
    mod.__dict__.update({'np': np, 'ndi': ndimage, 'ndimage': ndimage,
                         'cv2': cv2, 'skeletonize': skeletonize})
    exec(compile("\n\n".join(parts), 'MMPSv2.12.py', 'exec'), mod.__dict__)
    return mod


class _StubForest:
    """Returns a prepared probability per row, in the order it is handed them.

    The point is to check that score_soma keeps the ladder in the order the
    rule expects. A model that scored the rungs correctly but handed them to
    the rule reversed would choose the smallest mask every time and nothing
    would raise.
    """

    def __init__(self, probs):
        self.probs = probs

    def predict_proba(self, X):
        p = np.asarray(self.probs[:len(X)], dtype=float)
        return np.c_[1 - p, p]


def check_score_soma(tr, mod, rng):
    """score_soma must reach the same size the trainer's rule would."""
    import joblib
    fails = 0
    for trial in range(6):
        H = W = 220
        yy, xx = np.ogrid[:H, :W]
        cy, cx = H // 2, W // 2
        soma = ((yy - cy) ** 2 + (xx - cx) ** 2) < 12 ** 2
        ladder = []
        for r in (18, 24, 30, 36, 42, 48):
            ladder.append((r * 10,
                           ((yy - cy) ** 2 + (xx - cx) ** 2) < r ** 2))
        sig = ladder[-1][1] * 180 + rng.normal(30, 9, (H, W))
        dapi = soma * 200 + rng.normal(20, 7, (H, W))
        probs = list(rng.random(len(ladder)))

        bundle = {'model': _StubForest(probs),
                  'meta': dict(features=mod._MQ_FEATURE_NAMES,
                               pixel_size_um=0.104, signal_channel=1,
                               dapi_channel=3, prob_cut=0.5,
                               select_rule='band')}
        path = os.path.join(ROOT, 'tools', '_parity_stub.joblib')
        joblib.dump(bundle, path)
        try:
            mq = mod._MaskQAModel(path)
            got, conf, per = mq.score_soma(
                ladder, sig, dapi, soma, 0.104, (cy, cx),
                [(cy, cx), (cy, cx + 90)])
        finally:
            os.remove(path)

        areas = [a for a, _ in ladder]
        want = tr.pick(areas, probs, 0.5, 'band')
        want_conf = tr.soma_confidence(areas, probs, want)
        if got != want:
            print(f"  trial {trial}: score_soma chose {got}, rule says {want}")
            fails += 1
        if abs(conf - want_conf) > 1e-9:
            print(f"  trial {trial}: confidence {conf} vs {want_conf}")
            fails += 1
        if [a for a, _ in per] != areas:
            print(f"  trial {trial}: ladder order changed: "
                  f"{[a for a, _ in per]}")
            fails += 1
    return fails


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

    fails += check_score_soma(tr, load_model_class(), rng)

    if fails:
        sys.exit(f"\nFAIL: {fails} mismatch(es). MMPS's mask-QA code has "
                 f"drifted from train_mask_qa.py — the model will choose worse "
                 f"sizes until they match again.")
    print("PASS: MMPS and train_mask_qa.py produce identical features, "
          "picks and confidence (12 random cells, 3 rules, 2 cuts).")
    print("PASS: _MaskQAModel.score_soma reaches the trainer's own choice "
          "and confidence on 6 synthetic ladders.")


if __name__ == '__main__':
    main()
