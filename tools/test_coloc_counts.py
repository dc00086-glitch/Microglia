#!/usr/bin/env python3
"""Fail if colocalization counts cells by mask instead of by cell.

Morphology results hold one row per APPROVED MASK, and QA approves every size
at or below the one chosen -- so a cell sized at 400 contributes six rows and a
cell sized at 150 contributes one.

Two things went wrong with that:

  counts   Counting rows called each cell once per size it kept. Totals came
           out several times too high, and percent_coloc was weighted towards
           whichever cells accepted the most sizes -- so the number moved when
           mask sizing changed, with no change in the biology.

  source   The per-mask colocalisation lookup was keyed by (image, soma) with
           no size. The dict kept whichever mask came last, and the flat list
           runs largest first, so every row of a cell reported the SMALLEST
           approved mask's pixel counts and correlation while claiming to
           describe its own size.

    python3 tools/test_coloc_counts.py
"""
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, 'MMPSv2.12.py')

# Two cells. One accepted up to 400 (6 sizes), one only 150 (1 size).
# Counting rows says 7 cells, 6 of them coloc -> 85.7%.
# Counting cells says 2, 1 of them coloc -> 50.0%.
ROWS = (
    [{'image_name': 'img', 'soma_id': 'soma_1_1', 'target_area_um2': a,
      'soma_group': 'coloc', 'coloc_status': 'ok', 'n_coloc_pixels': 10}
     for a in (400, 350, 300, 250, 200, 150)]
    + [{'image_name': 'img', 'soma_id': 'soma_2_2', 'target_area_um2': 150,
        'soma_group': 'single_channel', 'coloc_status': 'ok',
        'n_coloc_pixels': 0}]
)


def run_summary(tmpdir):
    """Run MMPS's real _save_coloc_summary over synthetic rows."""
    import csv
    import textwrap
    import types

    src = open(TARGET).read()
    m = re.search(r'^    def _save_coloc_summary\(self, results\):.*?'
                  r'(?=\n    def )', src, re.S | re.M)
    if not m:
        sys.exit("FAIL: could not find _save_coloc_summary in MMPS — it has "
                 "been renamed or removed.")
    code = textwrap.dedent(m.group(0)).replace(
        'def _save_coloc_summary(self, results):',
        'def save_coloc_summary(self, results):')
    ns = {'os': os, 'csv': csv}
    exec(compile(code, TARGET, 'exec'), ns)

    app = types.SimpleNamespace(
        coloc_channel_1=0, coloc_channel_2=1,
        channel_names={0: 'Iba1', 1: 'TREM2'},
        output_dir=tmpdir, log=lambda *a, **k: None)
    ns['save_coloc_summary'](app, ROWS)
    path = os.path.join(tmpdir, 'colocalization_summary.csv')
    if not os.path.exists(path):
        sys.exit("FAIL: _save_coloc_summary wrote no file.")
    with open(path) as f:
        return list(csv.DictReader(f))


def check_counts():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        rows = run_summary(d)
    if len(rows) != 1:
        print(f"  expected one image row, got {len(rows)}")
        return False
    r = rows[0]
    total = int(r['total_cells'])
    coloc = int(r['TREM2+Iba1'])
    single = int(r['Iba1_only'])
    pct = float(r['percent_coloc'])
    ok = True
    if total != 2:
        print(f"  total_cells {total}, expected 2 — counting masks, not cells")
        ok = False
    if (coloc, single) != (1, 1):
        print(f"  coloc/single {coloc}/{single}, expected 1/1")
        ok = False
    if abs(pct - 50.0) > 1e-9:
        print(f"  percent_coloc {pct}, expected 50.0 — weighted by how many "
              f"sizes each cell kept")
        ok = False
    if ok:
        print(f"  counts            {total} cells, {coloc} coloc, "
              f"{single} single, {pct}% — one row per cell")
    return ok


def check_lookup_key():
    """The per-mask lookup must include the area, or sizes collapse."""
    src = open(TARGET).read()
    m = re.search(r'            mask_lookup = \{\}\n.*?\n(?=\s*if flat_data:)',
                  src, re.S)
    if not m:
        sys.exit("FAIL: could not find the colocalization mask lookup in MMPS.")
    block = m.group(0)
    # Both the key built and the key looked up must carry an area.
    writes = re.search(r'mask_lookup\[\(([^\]]*?)\)\]\s*=', block, re.S)
    reads = re.search(r'mask_lookup\.get\(\(([^;]*?)\)\)', block, re.S)
    if not writes or not reads:
        print("  lookup            could not read the key expressions")
        return False
    ok = True
    for label, mm in (('write', writes), ('read', reads)):
        parts = mm.group(1).count(',')
        if parts < 2:
            print(f"  lookup {label} key has {parts + 1} parts, needs 3 "
                  f"(image, soma, area) — every size of a cell collapses onto "
                  f"one mask")
            ok = False
    if 'area' not in writes.group(1) or 'area' not in reads.group(1):
        print("  lookup key does not mention an area")
        ok = False
    if ok:
        print("  mask lookup       keyed by (image, soma, area) — each row "
              "reads its own mask")
    return ok


def main():
    ast.parse(open(TARGET).read(), TARGET)
    ok = check_counts()
    ok = check_lookup_key() and ok
    if not ok:
        sys.exit("\nFAIL: colocalization counts are per-mask, not per-cell.")
    print("PASS: colocalization counts one row per cell and reads each mask's "
          "own signal.")


if __name__ == '__main__':
    main()
