#!/usr/bin/env python3
"""Joining a second tracer must not overwrite the first's vessel geometry.

Two BBB runs over the same cells with different leak channels. The
tracer-specific columns merge cleanly, but the VESSEL GEOMETRY columns carry
the same names in both files -- bbb_dist_to_vessel_um and friends. Letting
them overwrite replaces one run's geometry with the other's, and where a mask
was RECONSTRUCTED rather than reviewed those are not the same number.

    python3 tools/test_merge_tracer_results.py
"""
import os
import csv
import sys
import shutil
import tempfile
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, 'tools', 'merge_tracer_results.py')


def write(path, fields, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def main():
    fails = []
    d = tempfile.mkdtemp()

    base = os.path.join(d, 'combined_morphology_results.csv')
    write(base, ['image_name', 'soma_id', 'area_um2',
                 'bbb_dist_to_vessel_um', 'bbb_dextran_exposure_microglia'], [
        {'image_name': 'f1.tif', 'soma_id': 's1', 'area_um2': '300',
         'bbb_dist_to_vessel_um': '4.5',
         'bbb_dextran_exposure_microglia': '900'},
        {'image_name': 'f2.tif', 'soma_id': 's2', 'area_um2': '310',
         'bbb_dist_to_vessel_um': '20.0',
         'bbb_dextran_exposure_microglia': '10'},
        {'image_name': 'f3.tif', 'soma_id': 's3', 'area_um2': '320',
         'bbb_dist_to_vessel_um': '8.0',
         'bbb_dextran_exposure_microglia': '50'},   # no BSA row for this one
    ])

    add = os.path.join(d, 'simple_bsa_morphology.csv')
    write(add, ['image_name', 'soma_id', 'bbb_vessel_mask_source',
                'bbb_dist_to_vessel_um', 'bbb_bsa_exposure_microglia'], [
        # same mask -> identical geometry
        {'image_name': 'f1.tif', 'soma_id': 's1',
         'bbb_vessel_mask_source': 'reviewed',
         'bbb_dist_to_vessel_um': '4.5', 'bbb_bsa_exposure_microglia': '700'},
        # reconstructed mask -> geometry drifts
        {'image_name': 'f2', 'soma_id': 's2',          # bare name must match
         'bbb_vessel_mask_source': 'reconstructed',
         'bbb_dist_to_vessel_um': '21.3', 'bbb_bsa_exposure_microglia': '5'},
        # a cell the base sheet does not have
        {'image_name': 'f9.tif', 'soma_id': 's9',
         'bbb_vessel_mask_source': 'reviewed',
         'bbb_dist_to_vessel_um': '1.0', 'bbb_bsa_exposure_microglia': '1'},
    ])

    out = os.path.join(d, 'merged.csv')
    p = subprocess.run([sys.executable, TOOL, '--base', base, '--add', add,
                        '--suffix', 'bsa', '--out', out],
                       capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout + p.stderr)
        fails.append(f"the merge exited {p.returncode}")

    rows = read(out)
    if len(rows) != 3:
        fails.append(f"merged sheet has {len(rows)} rows, expected the base's 3")
    by = {r['soma_id']: r for r in rows}

    # --- the base's geometry must be untouched -----------------------------
    if by['s1'].get('bbb_dist_to_vessel_um') != '4.5' or \
            by['s2'].get('bbb_dist_to_vessel_um') != '20.0':
        fails.append("the second run overwrote the first run's vessel geometry")
    # --- and the second run's kept beside it -------------------------------
    if by['s2'].get('bbb_dist_to_vessel_um__bsa') != '21.3':
        fails.append("the second run's geometry was dropped instead of kept "
                     "under a suffix; the disagreement becomes invisible")
    # --- tracer columns merge cleanly --------------------------------------
    if by['s1'].get('bbb_bsa_exposure_microglia') != '700':
        fails.append("the new tracer's value did not land on its cell")
    if by['s1'].get('bbb_dextran_exposure_microglia') != '900':
        fails.append("the first tracer's value was altered")
    # --- a bare image name still matches -----------------------------------
    if not by['s2'].get('bbb_bsa_exposure_microglia'):
        fails.append("an image named without .tif did not match; every row "
                     "would come back blank")
    # --- an unmatched base row is blank, not wrong -------------------------
    if by['s3'].get('bbb_bsa_exposure_microglia', 'X') != '':
        fails.append("a cell with no second-tracer row got a value")
    if by['s3'].get('bbb_dextran_exposure_microglia') != '50':
        fails.append("an unmatched row lost its own data")
    # --- a row only in --add must not appear -------------------------------
    if any(r['soma_id'] == 's9' for r in rows):
        fails.append("a cell absent from --base was added; --base decides "
                     "which cells exist")

    # --- the disagreement has to be REPORTED, not just survivable ----------
    if 'identical' not in p.stdout:
        fails.append("the collision report did not say how far the two runs "
                     "agree")
    if '1.3' not in p.stdout:
        fails.append("the largest geometry difference was not reported, so a "
                     "run measuring different vessels would pass unnoticed")

    # --- no key columns is a refusal, not a silent empty merge -------------
    nokey = os.path.join(d, 'nokey.csv')
    write(nokey, ['cell', 'value'], [{'cell': 'a', 'value': '1'}])
    p2 = subprocess.run([sys.executable, TOOL, '--base', nokey, '--add', add,
                         '--out', os.path.join(d, 'x.csv')],
                        capture_output=True, text=True)
    if p2.returncode == 0:
        fails.append("a sheet with no image_name/soma_id was merged anyway; "
                     "there is no key to join cells on")

    # --- inputs untouched ---------------------------------------------------
    if read(base)[0].get('bbb_dist_to_vessel_um') != '4.5':
        fails.append("the merge modified --base in place")

    shutil.rmtree(d, ignore_errors=True)
    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("OK: a second tracer joins on without overwriting the first run's "
          "vessel geometry, and the disagreement is reported")


if __name__ == '__main__':
    main()
