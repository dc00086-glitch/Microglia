#!/usr/bin/env python3
"""Fail if MMPS's copy of the mask-QA feature code drifts from the trainer's.

The forest is fitted on features produced by train_mask_qa_model.py. MMPS holds
a copy so the app stays a single file. If the two ever diverge the model keeps
proposing sizes -- worse ones -- with nothing raising, which is the hardest kind
of failure to notice.

Checks both ways it can go wrong: the source text (caught early, names the
line) and the numbers (caught even if the text matches but the surrounding
module gives a name a different meaning).

    python3 tools/test_mask_qa_parity.py
"""
import os
import re
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINER = os.path.join(ROOT, 'train_mask_qa_model.py')
APP = os.path.join(ROOT, 'MMPSv2.12.py')
FUNCS = ['polygon_mask', 'poly_area_perimeter', 'convex_area',
         'soma_feature_rows', 'decode_cutoff']
PREFIX = '_mqa_'


def grab(src, name, prefix=''):
    m = re.search(rf'^def {prefix}{name}\(.*?'
                  rf'(?=\n\ndef |\n\nclass |\n\n# ---|\n\n# ===|\nMASK_QA_)',
                  src, re.S | re.M)
    if not m:
        sys.exit(f"could not find {prefix}{name}")
    return m.group(0).rstrip() + '\n'


def grab_const(src, name):
    m = re.search(rf'^{name} = \[.*?^\]', src, re.S | re.M)
    if not m:
        sys.exit(f"could not find {name}")
    return m.group(0)


def build_module(src, names, prefix=''):
    """Exec just the named functions, without importing the whole file."""
    mod = types.ModuleType('probe')
    from scipy import ndimage
    mod.__dict__.update(np=np, ndimage=ndimage, ndi=ndimage)
    code = "\n\n".join(grab(src, n, prefix) for n in names)
    if prefix:
        code = code.replace(prefix, '')
    exec(compile(code, '<probe>', 'exec'), mod.__dict__)
    return mod


def main():
    tsrc, asrc = open(TRAINER).read(), open(APP).read()
    fails = 0

    # --- the feature list has to be the same list, in the same order ---
    if grab_const(tsrc, 'MASK_QA_FEATURES') != grab_const(asrc, 'MASK_QA_FEATURES'):
        print("FAIL MASK_QA_FEATURES differs between trainer and app")
        fails += 1

    # --- source text ---
    for n in FUNCS:
        a = grab(tsrc, n)
        b = grab(asrc, n, PREFIX).replace(PREFIX, '')
        if a != b:
            print(f"FAIL {n}: source text differs")
            for i, (la, lb) in enumerate(zip(a.split('\n'), b.split('\n'))):
                if la != lb:
                    print(f"     first difference at line {i + 1}")
                    print(f"       trainer: {la}")
                    print(f"       app:     {lb}")
                    break
            fails += 1

    # --- the numbers ---
    trainer = build_module(tsrc, FUNCS)
    app = build_module(asrc, FUNCS, PREFIX)
    names = eval(grab_const(tsrc, 'MASK_QA_FEATURES').split('=', 1)[1].strip())

    rng = np.random.default_rng(11)
    for trial in range(12):
        h = int(rng.integers(120, 260))
        gray = rng.random((h, h)) * rng.integers(200, 4000)
        cy, cx = float(rng.integers(30, h - 30)), float(rng.integers(30, h - 30))
        th = np.linspace(0, 2 * np.pi, int(rng.integers(6, 20)), endpoint=False)
        rr = rng.uniform(4, 10, len(th))
        poly = [[cy + r * np.sin(t), cx + r * np.cos(t)] for t, r in zip(th, rr)]
        others = [[float(rng.integers(0, h)), float(rng.integers(0, h))]
                  for _ in range(int(rng.integers(0, 12)))]
        px = float(rng.uniform(0.08, 0.5))
        areas = [50.0 * (i + 1) for i in range(int(rng.integers(4, 17)))]

        A = trainer.soma_feature_rows(gray, (cy, cx), poly, others, px, areas)
        B = app.soma_feature_rows(gray, (cy, cx), poly, others, px, areas)
        if A.shape != B.shape:
            print(f"FAIL trial {trial}: shape {A.shape} vs {B.shape}")
            fails += 1
            continue
        if A.shape[1] != len(names):
            print(f"FAIL trial {trial}: {A.shape[1]} columns but "
                  f"{len(names)} feature names")
            fails += 1
            continue
        if not np.allclose(A, B, rtol=0, atol=0, equal_nan=True):
            bad = np.argwhere(A != B)
            col = names[bad[0][1]]
            print(f"FAIL trial {trial}: features differ, first at '{col}'")
            fails += 1

        probs = list(rng.random(len(areas)))
        if trainer.decode_cutoff(areas, probs) != app.decode_cutoff(areas, probs):
            print(f"FAIL trial {trial}: decode_cutoff differs")
            fails += 1

    if fails:
        sys.exit(f"\n{fails} parity failure(s) — the app and the trainer no "
                 f"longer compute the same features. Re-copy the block marked "
                 f"COPIED VERBATIM in MMPSv2.12.py.")
    print(f"parity OK — {len(FUNCS)} functions, {len(names)} features identical")


if __name__ == '__main__':
    main()
