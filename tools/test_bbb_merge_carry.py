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
