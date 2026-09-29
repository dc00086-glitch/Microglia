#!/usr/bin/env python3
"""Two BBB runs over different images must combine into one complete sheet.

MMPS folds the per-cell BBB columns into combined_morphology_results.csv and
BLANKS them for every row the run did not cover:

    for c in bbb_cols:
        r[c] = hit.get(c, '') if hit else ''

So a second run over the remaining images erases the first run's -- which is
exactly what happens when a run is interrupted part way and finished later.
This checks the recovery actually recovers.

    python3 tools/test_merge_bbb_runs.py
"""
import os
import csv
import sys
import shutil
import tempfile
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, 'tools', 'merge_bbb_runs.py')

BBB = ['bbb_dist_to_vessel_um', 'bbb_microglia_exposure_mean']


def write(path, fields, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def main():
    d = tempfile.mkdtemp()
    fails = []
    fields = ['image_name', 'soma_id', 'area_um2'] + BBB

    # 135 cells over 135 images. Run 1 covered the first 50, run 2 the rest --
    # and each blanked the other's rows, the way MMPS does.
    def master(covered):
        rows = []
        for i in range(135):
            r = {'image_name': f'img{i}.tif', 'soma_id': f's{i}',
                 'area_um2': str(100 + i)}
            for c in BBB:
                r[c] = f'{i}.{c[-3:]}' if i in covered else ''
            rows.append(r)
        return rows

    first, second = set(range(50)), set(range(50, 135))
    write(os.path.join(d, 'run1.csv'), fields, master(first))
    write(os.path.join(d, 'run2.csv'), fields, master(second))

    # ...and a vessel CSV each, over their own images.
    vfields = ['image_name', 'vessel_area_frac']
    for name, covered in (('v1.csv', first), ('v2.csv', second)):
        write(os.path.join(d, name), vfields,
              [{'image_name': f'img{i}.tif', 'vessel_area_frac': f'0.{i:03d}'}
               for i in sorted(covered)])

    out = os.path.join(d, 'merged')
    p = subprocess.run(
        [sys.executable, TOOL, '--master', os.path.join(d, 'run2.csv'),
         '--also', os.path.join(d, 'run1.csv'),
         '--vessels', os.path.join(d, 'v2.csv'), os.path.join(d, 'v1.csv'),
         '--out', out], capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout + p.stderr)
        fails.append(f"the merge exited {p.returncode}")

    merged = read(os.path.join(out, 'combined_morphology_results.csv'))
    if len(merged) != 135:
        fails.append(f"merged sheet has {len(merged)} rows, expected 135")

    missing = [r['soma_id'] for r in merged
               if any(not r.get(c, '').strip() for c in BBB)]
    if missing:
        fails.append(f"{len(missing)} row(s) still blank after merging, "
                     f"e.g. {missing[:5]} — the two runs did not combine")

    # the values must be the RIGHT ones, not just present
    wrong = []
    for i, r in enumerate(merged):
        for c in BBB:
            want = f'{i}.{c[-3:]}'
            if r.get(c, '').strip() != want:
                wrong.append(f"{r['soma_id']} {c}: {r.get(c)!r} != {want!r}")
    if wrong:
        fails.append(f"{len(wrong)} value(s) landed on the wrong row, "
                     f"e.g. {wrong[:3]}")

    # identity columns must survive untouched
    if any(r['area_um2'] != str(100 + i) for i, r in enumerate(merged)):
        fails.append("a non-BBB column was altered by the merge")

    vessels = read(os.path.join(out, 'bbb_vessel_leakage.csv'))
    if len(vessels) != 135:
        fails.append(f"vessel sheet has {len(vessels)} images, expected 135")
    if len({v['image_name'] for v in vessels}) != len(vessels):
        fails.append("the vessel sheet has duplicate images")

    # --- a disagreement must be reported, not silently resolved ------------
    clash = master(first)
    for r in clash[:1]:
        r[BBB[0]] = 'DIFFERENT'
    write(os.path.join(d, 'clash.csv'), fields, clash)
    out2 = os.path.join(d, 'merged2')
    p2 = subprocess.run(
        [sys.executable, TOOL, '--master', os.path.join(d, 'run1.csv'),
         '--also', os.path.join(d, 'clash.csv'), '--out', out2],
        capture_output=True, text=True)
    if 'disagreed' not in p2.stdout:
        fails.append("two runs disagreeing about a cell was not reported")
    if read(os.path.join(out2, 'combined_morphology_results.csv'))[0][BBB[0]] \
            == 'DIFFERENT':
        fails.append("a disagreement overwrote the master's value instead of "
                     "keeping it")

    # --- the originals must be untouched -----------------------------------
    want0 = f'0.{BBB[0][-3:]}'
    if read(os.path.join(d, 'run1.csv'))[0][BBB[0]] != want0:
        fails.append("the merge modified an input file")

    shutil.rmtree(d, ignore_errors=True)
    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("OK: two partial BBB runs combine into one complete sheet, with "
           "disagreements reported")


if __name__ == '__main__':
    main()
