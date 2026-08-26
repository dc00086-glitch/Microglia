#!/usr/bin/env python3
"""Fail if MMPS's copy of the ML feature code drifts from the trainer's.

The forest is fitted on features produced by train_soma_model.py. MMPS holds a
copy of that code so the app stays a single file. If the two ever diverge the
model keeps returning outlines -- worse ones -- with nothing raising, which is
the hardest kind of failure to notice. This compares them numerically.

    python3 tools/test_ml_parity.py
"""
import os
import re
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUNCS = ['pixel_features', 'radial_contour', 'mask_from_prob', 'otsu']


def load_module(path, names, prefix=''):
    """Exec just the named functions out of a file, without importing it."""
    src = open(path).read()
    mod = types.ModuleType('probe')
    mod.__dict__['np'] = np
    from scipy import ndimage
    mod.__dict__['ndimage'] = ndimage
    mod.__dict__['ndi'] = ndimage
    out = []
    for n in names:
        pat = rf'^def {prefix}{n}\(' if n != 'otsu' else rf'^def {prefix}_?otsu\('
        m = re.search(pat + r'.*?(?=\n\ndef |\n\nclass |\n\n# ---)',
                      src, re.S | re.M)
        if not m:
            sys.exit(f"could not find {prefix}{n} in {os.path.basename(path)}")
        out.append(m.group(0))
    code = "\n\n".join(out)
    if prefix:
        for n in names + ['disk', 'patch_around']:
            code = code.replace(f'{prefix}{n}(', f'{n}(')
        code = code.replace('def disk(', 'def _disk(').replace(' disk(', ' _disk(')
        code = code.replace('def otsu(', 'def _otsu(').replace(' otsu(', ' _otsu(')
    exec(compile(code, path, 'exec'), mod.__dict__)
    return mod


REQUIRED = [
    '_MLSomaOutliner', 'get_ml_outliner', 'ml_model_paths', '_ml_note',
    'drain_ml_messages', '_ml_pixel_features', '_ml_mask_from_prob',
    '_ml_radial_contour', '_ml_otsu', '_ml_disk', '_ml_patch_around',
]
REQUIRED_METHODS = ['__init__', 'accept_threshold', 'describe',
                    '_probability_map', 'outline']
REQUIRED_GLOBALS = ['_ML_OUTLINER', '_ML_OUTLINER_PATH', '_ML_LOAD_MESSAGES']


def check_structure():
    """Every piece of the ML path is still defined.

    A regex edit once matched from one function to the next `def` and, because
    the class in between starts with `class`, deleted _MLSomaOutliner and the
    module globals with it. Nothing failed to import -- the loader just caught
    the NameError and reported no model -- so it shipped. Byte-comparing
    features cannot catch a whole symbol going missing; this can.
    """
    import ast
    src = open(os.path.join(ROOT, 'MMPSv2.12.py')).read()
    tree = ast.parse(src)
    names = {n.name for n in ast.walk(tree)
             if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    missing = [n for n in REQUIRED if n not in names]
    cls = next((n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == '_MLSomaOutliner'),
               None)
    if cls is not None:
        have = {m.name for m in cls.body if isinstance(m, ast.FunctionDef)}
        missing += [f'_MLSomaOutliner.{m}' for m in REQUIRED_METHODS
                    if m not in have]
    missing += [g for g in REQUIRED_GLOBALS
                if not re.search(rf'^{g} = ', src, re.M)]
    if missing:
        sys.exit("FAIL: MMPSv2.12.py is missing part of the ML path: "
                 + ", ".join(missing))
    print(f"PASS: all {len(REQUIRED) + len(REQUIRED_METHODS) + len(REQUIRED_GLOBALS)}"
          f" ML symbols present in MMPSv2.12.py")


def main():
    check_structure()
    trainer = load_module(os.path.join(ROOT, 'train_soma_model.py'),
                          FUNCS + ['_disk', 'patch_around'])
    app = load_module(os.path.join(ROOT, 'MMPSv2.12.py'),
                      FUNCS + ['disk', 'patch_around'], prefix='_ml_')

    rng = np.random.default_rng(7)
    fails = 0
    for trial in range(25):
        h = int(rng.integers(40, 160))
        patch = rng.random((h, h)) * rng.integers(1, 4096)
        ctr = (float(rng.integers(5, h - 5)), float(rng.integers(5, h - 5)))
        scales = (1.0, 2.0, 4.0, 8.0) if trial % 2 else (1.0, 2.0, 4.0, 8.0, 16.0, 24.0)

        extra = ([rng.random((h, h)) * 500] if trial % 3 == 0 else
                 [rng.random((h, h)) * 500, rng.random((h, h)) * 90]
                 if trial % 3 == 1 else None)
        a = trainer.pixel_features(patch, scales, center=ctr, extra=extra)
        b = app.pixel_features(patch, scales, center=ctr, extra=extra)
        if a.shape != b.shape or not np.array_equal(a, b):
            print(f"  trial {trial}: FEATURES DIFFER  {a.shape} vs {b.shape}")
            fails += 1
            continue

        prob = rng.random((h, h))
        for mode in ('cc', 'radial', 'radial_h2', 'radial_h4', 'radial_h6'):
            for cut in (0.35, 0.5, 0.65):
                ma = trainer.mask_from_prob(prob, ctr, cut, 3, mode)
                mb = app.mask_from_prob(prob, ctr, cut, 3, mode)
                if (ma is None) != (mb is None):
                    print(f"  trial {trial} {mode}@{cut}: one returned None")
                    fails += 1
                elif ma is not None and not np.array_equal(ma, mb):
                    print(f"  trial {trial} {mode}@{cut}: MASKS DIFFER")
                    fails += 1

    if fails:
        sys.exit(f"\nFAIL: {fails} mismatch(es). MMPS's ML feature code has "
                 f"drifted from train_soma_model.py — the trained model will "
                 f"misbehave silently until they match again.")
    print("PASS: MMPS and train_soma_model.py produce identical features "
          "and masks (25 random patches, both scale sets, all 5 mask modes, "
          "with and without extra channels).")


if __name__ == '__main__':
    main()
