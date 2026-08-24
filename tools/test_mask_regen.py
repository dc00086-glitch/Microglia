#!/usr/bin/env python3
"""Can the masks QA deleted be rebuilt, and are the rebuilt ones the real ones?

Simulates the whole situation end to end on synthetic data:

  generate every candidate mask  ->  pick a cutoff per cell  ->  delete the
  rejected TIFFs, exactly as QA does  ->  rebuild from the session alone  ->
  check the rebuild against the survivors  ->  write features for ALL
  candidates  ->  train on them, negatives included

The session it writes is deliberately a version-2 one with the smoothing and
constraint settings left out, because that is what the real session on the
drive looks like -- so the settings search has to find them back.

    python3 tools/test_mask_regen.py
"""
import os
import sys
import json
import shutil
import subprocess
import tempfile
import importlib.util

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PX = 1.0
AREAS = list(range(50, 801, 50))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_study(root, regen, mm, n_images=3, seed=5):
    """Images, outlines, every candidate mask, a QA decision for each."""
    import tifffile
    rng = np.random.default_rng(seed)
    out_dir = os.path.join(root, 'Output')
    masks_dir = os.path.join(out_dir, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    images = {}
    n_deleted = n_kept = 0

    for gi in range(n_images):
        H = W = 320
        img = rng.normal(300, 15, (H, W))
        yy, xx = np.mgrid[0:H, 0:W]
        cents = []
        for _ in range(6):
            cents.append((float(rng.uniform(50, H - 50)),
                          float(rng.uniform(50, W - 50))))
        for (cy, cx) in cents:
            img += 3000 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 7.0 ** 2))
            for _ in range(6):
                th = rng.uniform(0, 2 * np.pi)
                for t in np.linspace(0, 40, 70):
                    img += 1500 * np.exp(
                        -((yy - cy - t * np.sin(th)) ** 2
                          + (xx - cx - t * np.cos(th)) ** 2) / 8.0)
        img = np.clip(img, 0, 65535).astype(np.uint16)
        name = f"synth_{gi}.tif"
        proc = os.path.join(out_dir, f"synth_{gi}_processed.tif")
        tifffile.imwrite(proc, img)

        outlines = []
        for i, (cy, cx) in enumerate(cents):
            th = np.linspace(0, 2 * np.pi, 14, endpoint=False)
            pts = [[float(cy + 7 * np.sin(t)), float(cx + 7 * np.cos(t))]
                   for t in th]
            outlines.append(dict(
                soma_idx=i, soma_id=f"soma_{int(cy)}_{int(cx)}",
                centroid=[cy, cx],
                soma_area_um2=float(np.pi * 49 * PX * PX),
                polygon_points=pts))

        # every candidate, with MMPS's own code
        mm.CFG = regen.Settings()
        gen_outlines = [dict(o, outline=mm._polygon_to_mask(
            o['polygon_points'], img.shape)) for o in outlines]
        masks = mm._create_competitive_masks(
            img.astype(np.float64), gen_outlines, AREAS, PX, name)

        # a decision per cell, then QA's deletions
        qa, kept_files = [], []
        cutoffs = {}
        for i, o in enumerate(outlines):
            d_nn = min([np.hypot(o['centroid'][0] - p['centroid'][0],
                                 o['centroid'][1] - p['centroid'][1]) * PX
                        for p in outlines if p is not o] or [500.0])
            cutoffs[o['soma_id']] = max(
                [a for a in AREAS if np.sqrt(a / np.pi) <= d_nn / 2.0] or [0])
        for md in masks:
            sid = md['soma_id']
            area = float(md.get('target_area_um2', 0))
            approved = area <= cutoffs[sid] and not md.get('duplicate')
            qa.append({'soma_id': sid, 'area_um2': int(area),
                       'approved': bool(approved)})
            if md.get('mask') is None:
                continue
            fn = mm._mask_tif_name(os.path.splitext(name)[0], sid, area)
            if approved:      # survives QA
                tifffile.imwrite(os.path.join(masks_dir, fn),
                                 (np.asarray(md['mask']) > 0).astype(np.uint8) * 255)
                kept_files.append(fn)
                n_kept += 1
            else:             # QA deleted it
                n_deleted += 1

        images[name] = dict(
            raw_path=proc, processed_path=proc, extra_channel_paths={},
            status='qa_complete', selected=True,
            somas=[list(c) for c in cents],
            soma_ids=[o['soma_id'] for o in outlines],
            soma_outlines=outlines, mask_qa_state=qa, mask_files=kept_files)

    # version 2: no smoothing or constraint settings, like the real session
    sess = os.path.join(root, 'synth.mmps_session')
    with open(sess, 'w') as fh:
        json.dump(dict(version=2, pixel_size=str(PX), output_dir=out_dir,
                       masks_dir=masks_dir, mask_min_area=50,
                       mask_max_area=800, mask_step_size=50,
                       use_min_intensity=True, min_intensity_percent=5,
                       rolling_ball_radius=50,
                       mask_segmentation_method='competitive',
                       images=images), fh)
    return sess, masks_dir, n_kept, n_deleted


def main():
    tmp = tempfile.mkdtemp(prefix='mask_regen_test_')
    fails = []
    try:
        regen = load('regen', os.path.join(ROOT, 'regen_masks_for_training.py'))
        mm = regen.load_mmps()
        print("building a study and running a QA pass over it…")
        sess, masks_dir, kept, deleted = build_study(tmp, regen, mm)
        print(f"  {kept} masks approved and on disk, {deleted} rejected and "
              f"deleted\n")
        if deleted == 0:
            sys.exit("the simulated QA deleted nothing — test is not testing "
                     "anything")

        print("rebuilding from the session alone…")
        run = subprocess.run(
            [sys.executable, os.path.join(ROOT, 'regen_masks_for_training.py'),
             '--session', sess, '--settings-search', '--verify-only'],
            capture_output=True, text=True)
        tail = [l for l in run.stdout.splitlines()
                if 'identical' in l or 'best:' in l]
        print("  " + "\n  ".join(tail[-4:]))
        if run.returncode != 0:
            print(run.stdout[-2000:], run.stderr[-2000:])
            fails.append("verify pass failed outright")
        elif 'pixel-identical (100.0%)' not in run.stdout:
            fails.append("the rebuild is not pixel-identical to the survivors")

        print("\nwriting features for every candidate…")
        csv_path = os.path.join(tmp, 'mask_qa_features.csv')
        run = subprocess.run(
            [sys.executable, os.path.join(ROOT, 'regen_masks_for_training.py'),
             '--session', sess, '--settings-search', '--out', csv_path],
            capture_output=True, text=True)
        if run.returncode != 0 or not os.path.exists(csv_path):
            print(run.stdout[-2000:], run.stderr[-2000:])
            sys.exit("feature write failed")
        import csv as _csv
        with open(csv_path) as fh:
            rows = list(_csv.DictReader(fh))
        sess_json = json.load(open(sess))
        labels = {}
        for nm, im in sess_json['images'].items():
            for e in im['mask_qa_state']:
                labels[(nm, e['soma_id'], float(e['area_um2']))] = e['approved']
        neg = sum(1 for r in rows
                  if labels.get((r['image'], r['soma_id'],
                                 float(r['target_area_um2']))) is False)
        print(f"  {len(rows)} rows, {neg} of them masks QA had deleted")
        if neg < 0.5 * deleted:
            fails.append(f"only {neg} of {deleted} deleted masks came back")

        print("\ntraining with the whole-object features…")
        run = subprocess.run(
            [sys.executable, os.path.join(ROOT, 'train_mask_qa_model.py'),
             '--session', sess, '--features-csv', csv_path, '--folds', '3',
             '--trees', '80', '--out', os.path.join(tmp, 'm.joblib')],
            capture_output=True, text=True)
        if run.returncode != 0:
            print(run.stdout[-2500:], run.stderr[-2500:])
            fails.append("training on the recovered features failed")
        else:
            for l in run.stdout.splitlines():
                if 'whole-object features joined' in l or 'features' in l[:40] \
                        or 'cells:' in l:
                    print("  " + l.strip())
            if 'whole-object features joined on' not in run.stdout:
                fails.append("the trainer did not join the recovered features")

        # --- and MMPS has to be able to RUN a model trained that way --------
        print("\nchecking MMPS can run a whole-object model…")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from test_mask_qa_model import load_app_block
        app = load_app_block()
        model_path = os.path.join(tmp, 'm.joblib')
        ml = app.get_ml_mask_qa(model_path) if os.path.exists(model_path) else None
        if ml is None:
            fails.append("MMPS could not load the whole-object model")
        elif ml.unusable:
            fails.append(f"MMPS refused the whole-object model: {ml.unusable}")
        elif not ml.needs_masks:
            fails.append("MMPS did not notice the model needs the masks")
        else:
            trainer = load('trainer',
                           os.path.join(ROOT, 'train_mask_qa_model.py'))
            recs = trainer.read_session(sess, [])
            rec = recs[0]
            gray, _ = trainer.open_image(rec, 'processed', None)
            obj_by_cell = {}
            for r in rows:
                if r['image'] != rec['image_name']:
                    continue
                vals = [float(r[c]) for c in app.MASK_QA_OBJECT_FEATURES]
                obj_by_cell.setdefault(r['soma_id'], {})[
                    float(r['target_area_um2'])] = vals
            n_ok = n_none = 0
            for c in rec['somas']:
                got = ml.suggest(gray, c['centroid'], c['polygon'],
                                 rec['all_centroids'], rec['pixel_size'],
                                 c['areas'],
                                 object_features=obj_by_cell.get(c['soma_id']))
                if got is None:
                    n_none += 1
                else:
                    n_ok += 1
            print(f"  scored {n_ok} cells with measured masks "
                  f"({n_none} skipped for want of one)")
            if n_ok == 0:
                fails.append("MMPS scored no cells with the whole-object model")
            # a whole-object model must REFUSE to guess when the masks are gone
            blind = ml.suggest(gray, rec['somas'][0]['centroid'],
                               rec['somas'][0]['polygon'],
                               rec['all_centroids'], rec['pixel_size'],
                               rec['somas'][0]['areas'])
            if blind is not None:
                fails.append("MMPS scored a cell without measuring its masks — "
                             "it would propose a size from a made-up vector")
            else:
                print("  and refuses to score a cell whose masks it cannot "
                      "measure")

        if fails:
            sys.exit("\nFAIL\n  " + "\n  ".join(fails))
        print("\nPASS: masks deleted by QA rebuilt exactly, and trained on.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
