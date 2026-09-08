#!/usr/bin/env python3
"""Fail if ImageJ result CSVs stop merging into the master sheet.

The merge joins each ImageJ row to a morphology row on (image, soma, TARGET
area) -- the size that was asked for, which the mask filename is built from.

The trap is mask_area_um2. In the skeleton scripts it is the area the mask
MEASURED, which is never exactly the target: a 150 um2 mask comes out at 147.
Using it as the join key produced zero matches for every skeleton file, and
nothing raised -- the merged sheet was written with the skeleton columns
present and empty. In the fractal script the very same column name holds the
TARGET area instead, so no rule about the name alone is safe.

There are four CSV shapes in circulation and they do not agree on which
identifier columns exist:

  cluster skeleton   image_name, soma_id, target_area_um2  (all stated)
  standalone skel    cell_name + mask_file only, mask_area_um2 = MEASURED
  standalone fractal cell_name, image_name, soma_id, mask_area_um2 = TARGET
  sholl              Mask Name, Image Name, Soma ID, Mask Area (um2) = TARGET

    python3 tools/test_csv_merge.py
"""
import csv
import os
import re
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, 'MMPSv2.12.py')

IMG = 'Yr_13-0_1d_Nl-1__Slice_2'
SOMAS = ['soma_452_379', 'soma_733_714']
AREAS = [150, 200, 250]
MEASURED = {a: round(a * 0.982, 2) for a in AREAS}   # never equals the target


class _Log:
    def __init__(self):
        self.lines = []

    def log(self, m):
        self.lines.append(m)


def load_merge_code():
    """The loader and the lookup builder, taken out of MMPS verbatim."""
    src = open(TARGET).read()
    m = re.search(r'^    def _load_imagej_csv\(self.*?(?=\n    def )',
                  src, re.S | re.M)
    if not m:
        sys.exit("FAIL: could not find _load_imagej_csv in MMPS")
    loader = textwrap.dedent(m.group(0))
    loader = loader.replace('def _load_imagej_csv(self,',
                            'def load_imagej_csv(_self,')
    loader = loader.replace('self.log', '_self.log')

    m = re.search(r'^            def _build_ij_lookup\(ij_data\):'
                  r'.*?(?=\n            sholl_lookup)', src, re.S | re.M)
    if not m:
        sys.exit("FAIL: could not find _build_ij_lookup in MMPS")
    builder = textwrap.dedent(m.group(0))

    ns = {'csv': csv, 'os': os, 're': re}
    exec(compile(loader, TARGET, 'exec'), ns)
    exec(compile(builder, TARGET, 'exec'), ns)
    return ns['load_imagej_csv'], ns['_build_ij_lookup']


def write_csv(path, cols, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in cols})


def cell_name(mask_file):
    return re.sub(r'_area\d+_mask\.tif$', '', mask_file)


def make_files(d):
    """One CSV of each shape, all describing the same six cells."""
    made = {}
    base = [(s, a) for s in SOMAS for a in AREAS]

    # cluster-template skeleton: states every identifier
    rows = [dict(image_name=IMG, soma_id=s, target_area_um2=a, soma_idx=0,
                 animal_id='', treatment='',
                 cell_name=cell_name(f"{IMG}_{s}_area{a}_mask.tif"),
                 mask_file=f"{IMG}_{s}_area{a}_mask.tif",
                 skeleton_file=f"{IMG}_{s}_area{a}_skel.tif",
                 mask_area_um2=MEASURED[a], num_branches=11, num_junctions=5,
                 total_skeleton_length_um=88.5) for s, a in base]
    p = os.path.join(d, 'cluster_skeleton.csv')
    write_csv(p, ['image_name', 'animal_id', 'treatment', 'soma_id',
                  'soma_idx', 'target_area_um2', 'cell_name', 'mask_file',
                  'skeleton_file', 'mask_area_um2', 'num_branches',
                  'num_junctions', 'total_skeleton_length_um'], rows)
    made['cluster skeleton'] = ('skeleton', p)

    # standalone SkeletonAnalysisImageJ.py: no target column at all, and
    # mask_area_um2 is the MEASURED area
    rows = [dict(cell_name=cell_name(f"{IMG}_{s}_area{a}_mask.tif"),
                 mask_file=f"{IMG}_{s}_area{a}_mask.tif",
                 skeleton_file=f"{IMG}_{s}_area{a}_skel.tif",
                 pixel_size_um=0.104, upscale_factor=1,
                 mask_area_um2=MEASURED[a], num_branches=11, num_junctions=5,
                 total_skeleton_length_um=88.5) for s, a in base]
    p = os.path.join(d, 'standalone_skeleton.csv')
    write_csv(p, ['cell_name', 'mask_file', 'skeleton_file', 'pixel_size_um',
                  'upscale_factor', 'mask_area_um2', 'num_branches',
                  'num_junctions', 'total_skeleton_length_um'], rows)
    made['standalone skeleton'] = ('skeleton', p)

    # standalone FractalAnalysis_ImageJ.py: mask_area_um2 is the TARGET
    rows = [dict(cell_name=cell_name(f"{IMG}_{s}_area{a}_mask.tif"),
                 image_name=IMG, soma_id=s, mask_area_um2=a,
                 mask_file=f"{IMG}_{s}_area{a}_mask.tif",
                 fractal_dimension=1.42, hull_area_um2=310.0)
            for s, a in base]
    p = os.path.join(d, 'standalone_fractal.csv')
    write_csv(p, ['cell_name', 'image_name', 'soma_id', 'mask_area_um2',
                  'mask_file', 'fractal_dimension', 'hull_area_um2'], rows)
    made['standalone fractal'] = ('fractal', p)

    # Sholl: full mask filename as the name, target area under its own header
    rows = [{'Mask Name': f"{IMG}_{s}_area{a}_mask.tif", 'Image Name': IMG,
             'Soma ID': s, 'Mask Area (um2)': a, 'Centroid X (px)': 10,
             'Centroid Y (px)': 20, 'Start Radius (um)': 5,
             'Primary Branches': 4, 'Sum of Intersections': 33}
            for s, a in base]
    p = os.path.join(d, 'sholl.csv')
    write_csv(p, ['Mask Name', 'Image Name', 'Soma ID', 'Mask Area (um2)',
                  'Centroid X (px)', 'Centroid Y (px)', 'Start Radius (um)',
                  'Primary Branches', 'Sum of Intersections'], rows)
    made['sholl'] = ('sholl', p)
    return made


def main():
    import tempfile
    load_imagej_csv, build_lookup = load_merge_code()
    morph = [{'image_name': IMG, 'soma_id': s, 'target_area_um2': str(a)}
             for s in SOMAS for a in AREAS]

    fails = 0
    with tempfile.TemporaryDirectory() as d:
        for label, (kind, path) in make_files(d).items():
            loaded = load_imagej_csv(_Log(), path, kind)
            lookup = build_lookup(loaded)
            matched, wrong = 0, 0
            for row in morph:
                want = int(float(row['target_area_um2']))
                key = (row['image_name'], row['soma_id'], want)
                data = lookup.get(key) or lookup.get(
                    (row['image_name'], row['soma_id'], None))
                if data:
                    matched += 1
                    # An area-blind match would hand every size the same row.
                    if lookup.get(key) is None:
                        wrong += 1
            note = f"  {label:22s} {matched}/{len(morph)} matched"
            if matched != len(morph):
                print(note + "   <-- FAIL")
                fails += 1
            elif wrong:
                print(note + f"   <-- FAIL: {wrong} matched ignoring area, so "
                             f"every size gets the same row")
                fails += 1
            else:
                print(note)

    if fails:
        sys.exit(f"\nFAIL: {fails} CSV shape(s) do not merge. Rows are joined "
                 f"on the TARGET area; check that mask_area_um2 is not being "
                 f"used as the key, since in the skeleton files it is the "
                 f"measured area and never equals the target.")
    print("PASS: all four ImageJ CSV shapes merge onto the morphology sheet, "
          "each size matched to its own row.")


if __name__ == '__main__':
    main()
