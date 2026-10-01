#!/usr/bin/env python3
"""Join a second tracer's BBB results onto the first tracer's sheet.

Two runs over the same cells with different leak channels. Most columns are
tracer-specific and merge cleanly -- bbb_dextran_* beside bbb_bsa_*. But the
VESSEL GEOMETRY columns carry the same names in both files:

    bbb_dist_to_vessel_um, bbb_juxtavascular,
    bbb_vessel_contact_fraction, bbb_vessel_contact_area_um2

Letting those overwrite would quietly replace one run's geometry with the
other's, and for the images whose vessel mask was RECONSTRUCTED rather than
reviewed the two are not necessarily the same number. So a colliding column is
kept under a suffix instead, and how far the two disagree is reported.

That report is the point as much as the merge is: where the same vessel mask
was used, the geometry must agree exactly. A disagreement says the two runs did
not measure against the same vessels -- which is expected for the
reconstructed images and a red flag for any other.

    python3 tools/merge_tracer_results.py \
        --base "/Volumes/.../Dextran Output/combined_morphology_results.csv" \
        --add  "/Volumes/.../BSA Output/simple_bsa_morphology.csv" \
        --suffix bsa \
        --out  "/Volumes/.../Merged/dextran_plus_bsa.csv"

Nothing is written in place; --base and --add are read only.
"""
import os
import csv
import sys
import argparse

IDENTITY = ('image_name', 'soma_id', 'animal_id', 'treatment', 'region',
            'timepoint')


def norm(name):
    n = str(name).strip()
    low = n.lower()
    for ext in ('.tiff', '.tif'):
        if low.endswith(ext):
            return n[:-len(ext)]
    return n


def key_of(row):
    return (norm(row.get('image_name', '')),
            str(row.get('soma_id', '')).strip())


def read(path):
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def num(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True,
                    help='the sheet to join onto, keeping its rows and column '
                         'order (e.g. the dextran combined_morphology_results.csv)')
    ap.add_argument('--add', required=True,
                    help="the second tracer's CSV")
    ap.add_argument('--suffix', default='run2',
                    help='suffix for columns that exist in BOTH files, so '
                         'neither is lost (e.g. bsa -> bbb_dist_to_vessel_um__bsa)')
    ap.add_argument('--out', required=True, help='CSV to write')
    a = ap.parse_args()

    for p in (a.base, a.add):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")

    bf, brows = read(a.base)
    af, arows = read(a.add)
    for name, f in ((a.base, bf), (a.add, af)):
        if 'image_name' not in f or 'soma_id' not in f:
            sys.exit(f"{name} has no image_name/soma_id; there is no key to "
                     f"join cells on.")

    add_by_key = {}
    dup = 0
    for r in arows:
        k = key_of(r)
        if k in add_by_key:
            dup += 1
        add_by_key[k] = r

    incoming = [c for c in af if c not in IDENTITY]
    collisions = [c for c in incoming if c in bf]
    fresh = [c for c in incoming if c not in bf]
    renamed = {c: f"{c}__{a.suffix}" for c in collisions}

    out_fields = list(bf) + fresh + [renamed[c] for c in collisions]
    matched = 0
    # For the collision report: how far apart the two runs are, per column.
    diffs = {c: [] for c in collisions}
    for row in brows:
        src = add_by_key.get(key_of(row))
        if src is None:
            for c in fresh:
                row.setdefault(c, '')
            for c in collisions:
                row[renamed[c]] = ''
            continue
        matched += 1
        for c in fresh:
            row[c] = src.get(c, '')
        for c in collisions:
            row[renamed[c]] = src.get(c, '')
            x, y = num(row.get(c)), num(src.get(c))
            if x is not None and y is not None:
                diffs[c].append(abs(x - y))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction='ignore')
        w.writeheader()
        for row in brows:
            w.writerow({k: row.get(k, '') for k in out_fields})

    print(f"{a.out}")
    print(f"  {len(brows)} rows in --base, {matched} matched a row in --add")
    if matched < len(brows):
        print(f"  {len(brows) - matched} row(s) had no match and were left "
              f"blank in the new columns")
    extra = len(add_by_key) - matched
    if extra > 0:
        print(f"  {extra} row(s) in --add matched nothing in --base and are "
              f"NOT in the output; --base decides which cells exist")
    if dup:
        print(f"  {dup} duplicate key(s) in --add; the last won")
    print(f"  {len(fresh)} new column(s) added")
    if collisions:
        print(f"  {len(collisions)} column(s) exist in both and were kept "
              f"separately as __{a.suffix}:")
        for c in collisions:
            d = diffs[c]
            if not d:
                print(f"      {c}: nothing numeric to compare")
            else:
                worst = max(d)
                same = sum(1 for v in d if v == 0)
                print(f"      {c}: {same}/{len(d)} identical, "
                      f"largest difference {worst:g}")
        print("  where the same vessel mask was used these must be identical; "
              "a difference means the two runs measured against different "
              "vessels (expected for reconstructed masks, not for reviewed "
              "ones)")


if __name__ == '__main__':
    main()
