#!/usr/bin/env python3
"""A BBB run must survive being interrupted, and pick up where it stopped.

Vessel review is the expensive part of BBB analysis and it was the one thing
never persisted. Every row was buffered in memory and written after the LAST
image, so a run that ended early -- a drive ejecting, a bad file, the review
window closed -- wrote nothing at all and every review in it was lost.

Worse, finishing the job later ERASED the first part, because the morphology
merge blanks the BBB columns of any row the run did not cover:

    for c in bbb_cols:
        r[c] = hit.get(c, '') if hit else ''

So this pins the checkpoint cycle: one file per finished image, written the
moment that image is done; a later run skips those images; and the CSVs are
written from every checkpoint, not only from what the current run reviewed.

    python3 tools/test_bbb_resume.py
"""
import os
import sys
import json
import shutil
import tempfile
import warnings

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
except ImportError as e:
    print(f"SKIP: needs numpy ({e})")
    sys.exit(0)

import importlib.util
spec = importlib.util.spec_from_file_location(
    'mmps', os.path.join(ROOT, 'MMPSv2.12.py'))
mmps = importlib.util.module_from_spec(spec)
sys.modules['mmps'] = mmps
spec.loader.exec_module(mmps)


def main():
    d = tempfile.mkdtemp()
    fails = []

    mask = np.zeros((40, 50), np.uint8)
    mask[10:20, 10:40] = 1

    # --- a finished image is saved the moment it is done -------------------
    ok = mmps._bbb_save_checkpoint(
        d, 'img1',
        {'image_name': 'img1', 'vessel_area_frac': 0.15},
        [{'image_name': 'img1', 'soma_id': 's1', 'bbb_dist_to_vessel_um': 3.2},
         {'image_name': 'img1', 'soma_id': 's2', 'bbb_dist_to_vessel_um': 9.9}],
        {'use_tubeness': True, 'thr_scale': 1.4, 'target_area_pct': None},
        vessel_mask=mask)
    if not ok:
        fails.append("saving a checkpoint reported failure")
    if not os.path.exists(os.path.join(d, 'bbb_progress', 'img1.json')):
        fails.append("no checkpoint file was written")

    # --- and read back whole ------------------------------------------------
    got = mmps._bbb_load_checkpoints(d)
    if set(got) != {'img1'}:
        fails.append(f"loading found {sorted(got)}, expected ['img1']")
    else:
        p = got['img1']
        if p['vessel_row'].get('vessel_area_frac') != 0.15:
            fails.append("the vessel row did not survive the round trip")
        if len(p.get('cell_rows') or []) != 2:
            fails.append("the per-cell rows did not survive the round trip")
        if abs(p['settings'].get('thr_scale', 0) - 1.4) > 1e-9:
            fails.append("the review settings did not survive; resuming could "
                         "not reproduce this image's segmentation")

    # --- the REVIEWED mask survives, which is the expensive decision --------
    back = mmps._bbb_load_checkpoint_mask(d, got['img1'])
    if back is None:
        fails.append("the reviewed vessel mask was not saved; a resume would "
                     "have to ask about this image again")
    elif not np.array_equal(np.asarray(back) > 0, mask > 0):
        fails.append("the reviewed vessel mask came back different")

    # --- a half-written checkpoint must not read as a finished image --------
    part = os.path.join(d, 'bbb_progress', 'img_crash.json.part')
    with open(part, 'w') as f:
        f.write('{"image": "img_crash", "vessel_ro')
    if 'img_crash' in mmps._bbb_load_checkpoints(d):
        fails.append("a half-written checkpoint counted as a finished image")

    # ...and neither must a corrupt finished one
    with open(os.path.join(d, 'bbb_progress', 'img_bad.json'), 'w') as f:
        f.write('{not json at all')
    loaded = mmps._bbb_load_checkpoints(d)
    if 'img_bad' in loaded:
        fails.append("a corrupt checkpoint counted as a finished image")
    if 'img1' not in loaded:
        fails.append("one corrupt checkpoint hid the good ones")

    # --- a checkpoint with no vessel row is not 'done' ----------------------
    with open(os.path.join(d, 'bbb_progress', 'img_empty.json'), 'w') as f:
        json.dump({'image': 'img_empty', 'vessel_row': None,
                   'cell_rows': []}, f)
    if 'img_empty' in mmps._bbb_load_checkpoints(d):
        fails.append("an image with no vessel row counted as finished; it "
                     "would be skipped forever")

    # --- resume arithmetic: what a second run would still have to do --------
    for b in ('img2', 'img3'):
        mmps._bbb_save_checkpoint(
            d, b, {'image_name': b}, [{'image_name': b, 'soma_id': 'x'}], {})
    selected = ['img1.tif', 'img2.tif', 'img3.tif', 'img4.tif', 'img5.tif']
    done = mmps._bbb_load_checkpoints(d)
    todo = [n for n in selected if os.path.splitext(n)[0] not in done]
    if todo != ['img4.tif', 'img5.tif']:
        fails.append(f"a resume would review {todo}, expected the two that "
                     f"have no checkpoint")

    # --- and the CSVs cover everything, not just this run -------------------
    # What the run itself produced, plus every checkpoint it did not re-review.
    vessel_rows = [{'image_name': 'img4'}, {'image_name': 'img5'}]
    cell_rows = [{'image_name': 'img4', 'soma_id': 'x'}]
    run_images = {r['image_name'] for r in vessel_rows}
    for base, payload in sorted(done.items()):
        if base in run_images:
            continue
        vessel_rows.append(payload['vessel_row'])
        cell_rows.extend(payload.get('cell_rows') or [])
    names = sorted(r['image_name'] for r in vessel_rows)
    if names != ['img1', 'img2', 'img3', 'img4', 'img5']:
        fails.append(f"the CSV would cover {names}, not all five images")
    if len(cell_rows) != 1 + 2 + 1 + 1:
        fails.append(f"the CSV would hold {len(cell_rows)} cell rows, "
                     f"expected 5 across all images")

    # --- re-reviewing an image replaces it, never duplicates it -------------
    mmps._bbb_save_checkpoint(d, 'img1', {'image_name': 'img1',
                                          'vessel_area_frac': 0.99},
                              [{'image_name': 'img1', 'soma_id': 's1'}], {})
    again = mmps._bbb_load_checkpoints(d)
    if len([b for b in again if b == 'img1']) != 1:
        fails.append("re-reviewing an image left two checkpoints for it")
    if again['img1']['vessel_row'].get('vessel_area_frac') != 0.99:
        fails.append("re-reviewing an image did not replace its saved result")

    shutil.rmtree(d, ignore_errors=True)
    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("OK: BBB saves every finished image, resumes on the rest, and the "
          "CSVs cover all runs")


if __name__ == '__main__':
    main()
