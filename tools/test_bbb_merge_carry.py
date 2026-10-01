#!/usr/bin/env python3
"""The BBB columns must survive the ImageJ merge.

BBB writes its per-cell columns into combined_morphology_results.csv, or into
bbb_microglia_exposure.csv when that master did not exist yet. The ImageJ merge
reads ONE morphology CSV:

    morphology_path = simple_path or .../combined_morphology_results.csv

so picking a simple-characteristics export in the merge dialog produced a
merged sheet with no BBB columns at all -- nothing errored, they were simply
not in the file it read, and the numbers looked lost.

    python3 tools/test_bbb_merge_carry.py
"""
import os
import csv
import sys
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

BBB_COLS = ['bbb_dist_to_vessel_um', 'bbb_red_dextran_exposure_microglia']


def write(path, fields, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def simple_sheet():
    fields = ['image_name', 'soma_id', 'area_um2', 'circularity']
    rows = [
        {'image_name': 'i1.tif', 'soma_id': 'soma_1_1', 'area_um2': '300',
         'circularity': '0.7'},
        {'image_name': 'i2.tif', 'soma_id': 'soma_2_2', 'area_um2': '300',
         'circularity': '0.4'},
        {'image_name': 'i3.tif', 'soma_id': 'soma_3_3', 'area_um2': '300',
         'circularity': '0.9'},          # never covered by a BBB run
    ]
    return fields, rows


def main():
    fails = []
    d = tempfile.mkdtemp()

    master = os.path.join(d, 'combined_morphology_results.csv')
    write(master, ['image_name', 'soma_id', 'area_um2'] + BBB_COLS, [
        {'image_name': 'i1.tif', 'soma_id': 'soma_1_1', 'area_um2': '300',
         BBB_COLS[0]: '4.25', BBB_COLS[1]: '881'},
        {'image_name': 'i2.tif', 'soma_id': 'soma_2_2', 'area_um2': '300',
         BBB_COLS[0]: '12.0', BBB_COLS[1]: '90'},
    ])

    # --- a simple export picks the BBB columns up ---------------------------
    simple = os.path.join(d, 'simple_characteristics.csv')
    fields, rows = simple_sheet()
    write(simple, fields, rows)
    added = mmps._attach_bbb_columns(rows, fields, d, simple)
    if sorted(added) != sorted(BBB_COLS):
        fails.append(f"merging a simple morphology CSV carried {added} across; "
                     f"the merged sheet would have no BBB columns")
    else:
        if rows[0].get(BBB_COLS[0]) != '4.25':
            fails.append("a BBB value landed on the wrong cell, or not at all")
        if rows[1].get(BBB_COLS[1]) != '90':
            fails.append("the second cell's BBB value did not come across")
        if rows[2].get(BBB_COLS[0], 'MISSING') != '':
            fails.append("a cell no BBB run covered got a value instead of a "
                         "blank")

    # --- and they must survive the WRITE, not just reach the rows ----------
    # The merged sheet is written with extrasaction='ignore', so a column that
    # reaches the rows but never reaches fieldnames is dropped in silence --
    # which is this bug all over again, one step later. Reproduce the app's
    # own fieldname assembly and check a value comes out the other side.
    fields_w, rows_w = simple_sheet()
    added_w = mmps._attach_bbb_columns(rows_w, fields_w, d, simple)
    morph_fieldnames = fields_w + [c for c in added_w if c not in fields_w]
    all_keys = morph_fieldnames + ['Sholl_max', 'Skel_branches']
    seen, ordered = set(), []
    for k in all_keys:
        if k not in seen:
            seen.add(k)
            ordered.append(k)
    out = os.path.join(d, 'merged.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ordered, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows_w)
    with open(out, newline='') as f:
        back = list(csv.DictReader(f))
    missing = [c for c in BBB_COLS if c not in (back[0] if back else {})]
    if missing:
        fails.append(f"{missing} reached the rows but not the merged file's "
                     f"header, so extrasaction='ignore' dropped them silently")
    elif back[0].get(BBB_COLS[0]) != '4.25':
        fails.append("the BBB column is in the merged header but its value did "
                     "not survive the write")

    # --- .tif / .tiff / bare names must match, as elsewhere -----------------
    fields2, rows2 = simple_sheet()
    rows2[0]['image_name'] = 'i1'          # no extension
    rows2[1]['image_name'] = 'i2.tiff'     # the other extension
    mmps._attach_bbb_columns(rows2, fields2, d, simple)
    if rows2[0].get(BBB_COLS[0]) != '4.25' or rows2[1].get(BBB_COLS[0]) != '12.0':
        fails.append("image names differing only by extension did not match; "
                     "every row would come back blank")

    # --- the standalone BBB file is used when there is no master ------------
    d2 = tempfile.mkdtemp()
    write(os.path.join(d2, 'bbb_microglia_exposure.csv'),
          ['image_name', 'soma_id'] + BBB_COLS,
          [{'image_name': 'i1.tif', 'soma_id': 'soma_1_1',
            BBB_COLS[0]: '7.5', BBB_COLS[1]: '500'}])
    fields3, rows3 = simple_sheet()
    if not mmps._attach_bbb_columns(rows3, fields3, d2, simple):
        fails.append("BBB results in bbb_microglia_exposure.csv were ignored; "
                     "that is where they go when the master does not exist yet")
    elif rows3[0].get(BBB_COLS[0]) != '7.5':
        fails.append("the standalone BBB file's values did not come across")

    # --- two tracers: a sheet with one must still pick up the other --------
    # With a second tracer a sheet can carry bbb_bsa_* and still be missing
    # bbb_dextran_*. Bailing on "has any bbb_ column" would leave the second
    # run's numbers out and look exactly like the bug this guards against.
    d5 = tempfile.mkdtemp()
    write(os.path.join(d5, 'combined_morphology_results.csv'),
          ['image_name', 'soma_id', 'bbb_dist_to_vessel_um',
           'bbb_dextran_exposure_microglia'],
          [{'image_name': 'i1.tif', 'soma_id': 'soma_1_1',
            'bbb_dist_to_vessel_um': '4.25',
            'bbb_dextran_exposure_microglia': '881'}])
    bsa_fields = ['image_name', 'soma_id', 'bbb_bsa_exposure_microglia']
    bsa_rows = [{'image_name': 'i1.tif', 'soma_id': 'soma_1_1',
                 'bbb_bsa_exposure_microglia': '404'}]
    got5 = mmps._attach_bbb_columns(bsa_rows, bsa_fields, d5,
                                    os.path.join(d5, 'simple_bsa.csv'))
    if 'bbb_dextran_exposure_microglia' not in got5:
        fails.append("a sheet carrying bbb_bsa_* did not pick up the dextran "
                     "columns; one tracer's numbers would be missing from the "
                     "merge")
    if 'bbb_bsa_exposure_microglia' in got5:
        fails.append("a column the sheet already has was added again")
    if bsa_rows[0].get('bbb_bsa_exposure_microglia') != '404':
        fails.append("the sheet's own tracer value was overwritten by the "
                     "carry")
    if bsa_rows[0].get('bbb_dextran_exposure_microglia') != '881':
        fails.append("the other tracer's value did not come across")
    shutil.rmtree(d5, ignore_errors=True)

    # --- tracer NAMES must not matter, only the bbb_ prefix ----------------
    # Tracers are named by the user in the BBB panel, so nothing may key on a
    # particular name. Two tracers with names this code has never seen must
    # both come across, with every ring and the vessel mean.
    d6 = tempfile.mkdtemp()
    odd = ['bbb_FITC_albumin_488_exposure_microglia',
           'bbb_FITC_albumin_488_exposure_10um',
           'bbb_FITC_albumin_488_exposure_20um',
           'bbb_FITC_albumin_488_exposure_30um',
           'bbb_FITC_albumin_488_exposure_45um',   # a custom extra radius
           'bbb_FITC_albumin_488_vessel_mean',
           'bbb_cadaverine_555_exposure_microglia',
           'bbb_cadaverine_555_vessel_mean',
           'bbb_dist_to_vessel_um']
    write(os.path.join(d6, 'combined_morphology_results.csv'),
          ['image_name', 'soma_id'] + odd,
          [dict({'image_name': 'i1.tif', 'soma_id': 'soma_1_1'},
                **{c: str(i + 1) for i, c in enumerate(odd)})])
    f6, r6 = simple_sheet()
    got6 = mmps._attach_bbb_columns(r6, f6, d6,
                                    os.path.join(d6, 'simple.csv'))
    missing6 = [c for c in odd if c not in got6]
    if missing6:
        fails.append(f"these tracer columns were not carried: {missing6} — "
                     f"something is keying on the tracer's name rather than "
                     f"the bbb_ prefix")
    if r6[0].get('bbb_cadaverine_555_vessel_mean') != '8':
        fails.append("an unfamiliar tracer's value did not land on its cell")

    # the same must hold for adopting a prior run's CSV
    import csv as _c6
    with open(os.path.join(d6, 'bbb_vessel_leakage.csv'), 'w', newline='') as f:
        w = _c6.DictWriter(f, fieldnames=['image_name', 'vessel_area_fraction'])
        w.writeheader()
        w.writerow({'image_name': 'i1.tif', 'vessel_area_fraction': '0.02'})
    ad6 = mmps._bbb_adopt_prior_csv(d6, {})
    if 'i1' not in ad6:
        fails.append("adopting a prior run did not find the image")
    else:
        carried = ad6['i1']['cell_rows']
        if not carried:
            fails.append("adoption carried no cell rows for an unfamiliar "
                         "tracer")
        else:
            miss = [c for c in odd if c not in carried[0]]
            if miss:
                fails.append(f"adoption dropped {miss}; it is keying on "
                             f"tracer names rather than the bbb_ prefix")
    shutil.rmtree(d6, ignore_errors=True)

    # --- a sheet that already has them is left alone ------------------------
    if mmps._attach_bbb_columns(
            [{'image_name': 'i1.tif', 'soma_id': 'soma_1_1'}],
            ['image_name', 'soma_id', BBB_COLS[0]], d, master):
        fails.append("a sheet that already carries the BBB columns had them "
                     "attached a second time")

    # --- nothing to carry is not an error -----------------------------------
    empty = tempfile.mkdtemp()
    fields4, rows4 = simple_sheet()
    if mmps._attach_bbb_columns(rows4, fields4, empty, simple):
        fails.append("BBB columns were invented with no BBB results present")
    if any(BBB_COLS[0] in r for r in rows4):
        fails.append("an empty carry still wrote BBB keys into the rows")

    # --- a sheet with no soma_id cannot be joined, and must not be mangled ---
    rows5 = [{'image_name': 'i1.tif', 'area_um2': '300'}]
    if mmps._attach_bbb_columns(rows5, ['image_name', 'area_um2'], d, simple):
        fails.append("rows with no soma_id were joined anyway; there is no key "
                     "to match a cell on")

    for p in (d, d2, empty):
        shutil.rmtree(p, ignore_errors=True)

    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("OK: the BBB columns survive an ImageJ merge that reads another "
          "morphology CSV")


if __name__ == '__main__':
    main()
