#!/usr/bin/env python3
"""End-to-end check of the mask-QA model, on data that can live in the repo.

The real training data is a study on an external drive, so nothing here can be
checked against it automatically. This builds a small synthetic study instead --
cells scattered in an image, with a cutoff rule that a person would recognise
(a cell may grow until its footprint reaches half way to its nearest
neighbour) -- and runs the whole path on it:

    session + images -> features -> trained model -> MMPS's own inference

It proves the wiring, not the science. A high score here says the pipeline
carries a learnable rule from one end to the other; it says nothing about how
well the rule people actually apply can be learned.

    python3 tools/test_mask_qa_model.py
"""
import os
import re
import json
import shutil
import subprocess
import sys
import tempfile
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PX = 0.316
AREAS = [50.0 * i for i in range(1, 17)]


def make_study(out, n_images=6, seed=3):
    """Write images and a .mmps_session whose cutoffs follow one clear rule."""
    rng = np.random.default_rng(seed)
    import tifffile
    os.makedirs(out, exist_ok=True)
    images = {}
    for gi in range(n_images):
        H = W = 360
        img = rng.normal(300, 25, (H, W))
        cents = [(float(rng.uniform(40, H - 40)), float(rng.uniform(40, W - 40)))
                 for _ in range(int(rng.integers(16, 26)))]
        yy, xx = np.mgrid[0:H, 0:W]
        for (cy, cx) in cents:
            r = rng.uniform(4.5, 7.0)
            img += 2600 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))
            for _ in range(int(rng.integers(3, 7))):
                th, L = rng.uniform(0, 2 * np.pi), rng.uniform(12, 30)
                for t in np.linspace(0, L, 40):
                    py, pxx = cy + t * np.sin(th), cx + t * np.cos(th)
                    img += 900 * np.exp(-((yy - py) ** 2 + (xx - pxx) ** 2) / 5.12)
        name = f"synth_{gi}.tif"
        tifffile.imwrite(os.path.join(out, name), img.astype(np.uint16))

        qa, outlines = [], []
        for si, (cy, cx) in enumerate(cents):
            d_nn = min([np.hypot(cy - oy, cx - ox) * PX for (oy, ox) in cents
                        if (oy, ox) != (cy, cx)] or [50.0])
            cutoff = 0.0
            for a in AREAS:
                if np.sqrt(a / np.pi) <= d_nn / 2.0:
                    cutoff = a
            sid = f"soma_{int(cy)}_{int(cx)}"
            qa += [{'soma_id': sid, 'area_um2': a, 'approved': a <= cutoff}
                   for a in AREAS]
            th = np.linspace(0, 2 * np.pi, 14, endpoint=False)
            rr = 6.0 + rng.normal(0, 0.4, 14)
            outlines.append({
                'soma_idx': si, 'soma_id': sid, 'centroid': [cy, cx],
                'soma_area_um2': float(np.pi * 36.0 * PX * PX),
                'polygon_points': [[float(cy + r * np.sin(t)),
                                    float(cx + r * np.cos(t))]
                                   for t, r in zip(th, rr)]})
        images[name] = dict(
            raw_path=os.path.join(out, name),
            processed_path=os.path.join(out, name), extra_channel_paths={},
            status='qa_complete', selected=True,
            somas=[list(c) for c in cents],
            soma_ids=[o['soma_id'] for o in outlines],
            soma_outlines=outlines, mask_qa_state=qa, mask_files=[])
    sess = os.path.join(out, 'synth.mmps_session')
    with open(sess, 'w') as fh:
        json.dump(dict(version=2, pixel_size=str(PX), images=images), fh)
    return sess


def load_app_block():
    """Exec MMPS's mask-QA block on its own, without importing the whole app.

    Importing MMPSv2.12.py would need PyQt and a display; the block under test
    needs neither.
    """
    src = open(os.path.join(ROOT, 'MMPSv2.12.py')).read()
    m = re.search(r'^MASK_QA_FEATURES = \[.*?(?=\n\ndef auto_outline_soma_blob)',
                  src, re.S | re.M)
    if not m:
        sys.exit("could not find the mask-QA block in MMPSv2.12.py")
    mod = types.ModuleType('mmps_mask_qa')
    from scipy import ndimage
    mod.__dict__.update(np=np, ndimage=ndimage, ndi=ndimage, os=os, sys=sys)
    exec(compile(m.group(0), 'MMPSv2.12.py', 'exec'), mod.__dict__)
    return mod


def main():
    tmp = tempfile.mkdtemp(prefix='mask_qa_test_')
    try:
        print("building a synthetic study…")
        sess = make_study(os.path.join(tmp, 'study'))
        model_path = os.path.join(tmp, 'mask_qa_model.joblib')

        print("training…")
        run = subprocess.run(
            [sys.executable, os.path.join(ROOT, 'train_mask_qa_model.py'),
             '--session', sess, '--folds', '3', '--trees', '120',
             '--out', model_path],
            capture_output=True, text=True)
        if run.returncode != 0:
            print(run.stdout[-3000:])
            print(run.stderr[-3000:])
            sys.exit("trainer failed")
        out = run.stdout
        print("\n".join(l for l in out.splitlines()
                        if 'cells:' in l or 'masks:' in l or 'model  ' in l))

        m = re.search(r'held-out.*?cells: exact cutoff\s+(\d+)%', out, re.S)
        if not m:
            print(out[-3000:])
            sys.exit("could not read the held-out score out of the trainer")
        exact = int(m.group(1))
        fails = []
        if exact < 70:
            fails.append(f"held-out exact cutoff only {exact}% on a rule that "
                         f"is one of the features — the pipeline is losing it")

        # --- MMPS's own inference must agree with the trainer's ------------
        print("\nchecking MMPS's inference against the trainer…")
        sys.path.insert(0, ROOT)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'trainer', os.path.join(ROOT, 'train_mask_qa_model.py'))
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)
        app = load_app_block()

        ml = app.get_ml_mask_qa(model_path)
        if ml is None:
            sys.exit("MMPS could not load the model it was just given")
        if ml.unusable:
            sys.exit(f"MMPS refused the model: {ml.unusable}")

        import joblib
        bundle = joblib.load(model_path)
        clf = bundle['model']
        recs = trainer.read_session(sess, [])
        checked = disagreed = 0
        for rec in recs[:2]:
            gray, _which = trainer.open_image(rec, 'processed', None)
            cents = rec['all_centroids']
            for s in rec['somas']:
                rows = trainer.soma_feature_rows(gray, s['centroid'],
                                                 s['polygon'], cents,
                                                 rec['pixel_size'], s['areas'])
                probs = clf.predict_proba(np.nan_to_num(rows))[:, 1]
                want = trainer.decode_cutoff(s['areas'], list(probs))
                got = ml.suggest(gray, s['centroid'], s['polygon'], cents,
                                 rec['pixel_size'], s['areas'])
                checked += 1
                if got is None or abs(got['cutoff'] - want[0]) > 1e-6 \
                        or abs(got['conf'] - want[1]) > 1e-9:
                    disagreed += 1
        if disagreed:
            fails.append(f"MMPS and the trainer disagree on {disagreed} of "
                         f"{checked} cells")
        else:
            print(f"  {checked} cells: same size, same confidence")

        # --- the whole-object capture path has to run at all ---------------
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[30:50, 30:50] = 1
        gray = np.random.default_rng(0).random((80, 80)) * 1000
        vals = app._mqa_object_features(mask, gray, mask, PX, 200.0, (40, 40))
        if len(vals) != len(app.MASK_QA_OBJECT_FEATURES):
            fails.append(f"object features: {len(vals)} values for "
                         f"{len(app.MASK_QA_OBJECT_FEATURES)} names")
        elif not all(np.isfinite(v) for v in vals):
            fails.append("object features: produced a non-finite value")
        else:
            print(f"  whole-object capture: "
                  f"{len(vals)} finite features on a test mask")

        if fails:
            sys.exit("\nFAIL\n  " + "\n  ".join(fails))
        print("\nPASS: session -> features -> model -> MMPS inference, "
              "end to end.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
