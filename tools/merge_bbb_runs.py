#!/usr/bin/env python3
"""Combine BBB results from two or more runs that each covered some images.

Why this is needed: MMPS folds the per-cell BBB columns into
combined_morphology_results.csv keyed on image_name + soma_id, and a row the
run did NOT cover gets those columns blanked --

    for c in bbb_cols:
        r[c] = hit.get(c, '') if hit else ''

-- so a second run over the remaining images erases the first run's. Same for
bbb_vessel_leakage.csv, which is opened 'w' and replaced outright.

Neither is a problem as long as the earlier file is copied aside first. This
puts them back together: for every BBB column, a row takes the value from
whichever run actually filled it.

    python3 tools/merge_bbb_runs.py \
        --master  run2/combined_morphology_results.csv \
        --also    run1_backup/combined_morphology_results.csv \
        --vessels run2/bbb_vessel_leakage.csv run1_backup/bbb_vessel_leakage.csv \
        --out     merged/

The FIRST --master is the layout that is kept (its rows, its column order).
Each --also supplies values for rows it filled and the master left blank.
Nothing is overwritten: a cell already filled in the master stays as it is,
and any disagreement is reported rather than silently resolved.
"""
import os
import csv
import sys
import argparse

# Written by MMPS itself, not by a BBB run -- never treated as BBB data.
IDENTITY = ('image_name', 'soma_id', 'animal_id', 'treatment', 'region',
            'timepoint')


def read_csv(path):
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def norm_img(s):
    s = str(s).strip()
    low = s.lower()
    if low.endswith('.tiff'):
        return s[:-5]
    if low.endswith('.tif'):
        return s[:-4]
    return s


def key_of(row):
    return (norm_img(row.get('image_name', '')),
            str(row.get('soma_id', '')).strip())


def blank(v):
    return v is None or str(v).strip() == ''


def merge_master(master_path, also_paths, out_path):
    fields, rows = read_csv(master_path)
    if 'image_name' not in fields or 'soma_id' not in fields:
        sys.exit(f"{master_path} has no image_name/soma_id; cannot match rows.")

    # BBB columns are the bbb_-prefixed ones plus any <tracer>_ column a run
    # added. Anything the master already had and no run filled is left alone.
    bbb_cols = [c for c in fields if c not in IDENTITY]

    by_key = {}
    for r in rows:
        by_key.setdefault(key_of(r), []).append(r)

    filled = conflicts = unmatched = 0
    conflict_examples = []
    for path in also_paths:
        f2, r2 = read_csv(path)
        for src in r2:
            targets = by_key.get(key_of(src))
            if not targets:
                unmatched += 1
                continue
            for dst in targets:
                for c in f2:
                    if c in IDENTITY or blank(src.get(c)):
                        continue
                    if c not in dst:
                        dst[c] = src[c]
                        if c not in bbb_cols:
                            bbb_cols.append(c)
                        filled += 1
                    elif blank(dst.get(c)):
                        dst[c] = src[c]
                        filled += 1
                    elif str(dst[c]).strip() != str(src[c]).strip():
                        conflicts += 1
                        if len(conflict_examples) < 5:
                            conflict_examples.append(
                                f"{key_of(src)} {c}: kept {dst[c]!r}, "
                                f"{os.path.basename(path)} had {src[c]!r}")

    out_fields = list(fields)
    for c in bbb_cols:
        if c not in out_fields:
            out_fields.append(c)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in out_fields})

    still_blank = sum(
        1 for r in rows
        if all(blank(r.get(c)) for c in bbb_cols if c.startswith('bbb_')))
    print(f"  {out_path}")
    print(f"      {len(rows)} rows, {filled} cells filled from {len(also_paths)} "
          f"other run(s), {unmatched} incoming row(s) matched nothing")
    if still_blank:
        print(f"      {still_blank} row(s) still have NO bbb_ value — those "
              f"cells were never covered by any run")
    if conflicts:
        print(f"      {conflicts} cell(s) disagreed between runs; the master's "
              f"value was kept:")
        for e in conflict_examples:
            print(f"        {e}")


def merge_vessels(paths, out_path):
    seen, order, fields = {}, [], []
    for path in paths:
        f, rows = read_csv(path)
        for c in f:
            if c not in fields:
                fields.append(c)
        for r in rows:
            k = norm_img(r.get('image_name', ''))
            if k in seen:
                continue          # first file wins, like the master does
            seen[k] = r
            order.append(k)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for k in order:
            w.writerow({c: seen[k].get(c, '') for c in fields})
    print(f"  {out_path}")
    print(f"      {len(order)} images from {len(paths)} run(s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--master', required=True,
                    help='the combined_morphology_results.csv whose rows and '
                         'column order are kept')
    ap.add_argument('--also', nargs='*', default=[],
                    help='other runs\' combined_morphology_results.csv, read '
                         'for the rows they filled')
    ap.add_argument('--vessels', nargs='*', default=[],
                    help='bbb_vessel_leakage.csv from every run; earlier files '
                         'win on a duplicate image')
    ap.add_argument('--out', required=True, help='folder to write into')
    a = ap.parse_args()

    for p in [a.master] + list(a.also) + list(a.vessels):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")
    os.makedirs(a.out, exist_ok=True)

    print("Merging BBB runs:")
    merge_master(a.master, a.also,
                 os.path.join(a.out, 'combined_morphology_results.csv'))
    if a.vessels:
        merge_vessels(a.vessels,
                      os.path.join(a.out, 'bbb_vessel_leakage.csv'))
    print("\nNothing was overwritten in place; the originals are untouched.")


if __name__ == '__main__':
    main()
