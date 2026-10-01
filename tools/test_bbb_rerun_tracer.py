#!/usr/bin/env python3
"""Re-measuring with a second tracer must reuse the first run's vessel masks.

The same fields imaged again with another tracer: vessels and microglia are the
same objects, only the leak channel changed. Reviewing every vessel again would
be rework, so the finished run's REVIEWED masks are reused.

What must hold: the numbers come from MMPS's own function (so a column means
what the app's column means), a cell's value tracks the tracer it actually
sits in, and the script refuses rather than guesses when the vessel masks were
never saved -- the overlay PNGs are rendered composites, and recovering a
boundary from one would invent the thing every number is measured against.

    python3 tools/test_bbb_rerun_tracer.py
"""
import os
import csv
import sys
import shutil
import tempfile
import subprocess
import warnings

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, 'tools', 'bbb_rerun_tracer.py')

try:
    import numpy as np
    import tifffile
except ImportError as e:
    print(f"SKIP: needs numpy and tifffile ({e})")
    sys.exit(0)

H, W = 80, 100
PX = 0.5


def build(d, with_progress=True):
    """A finished dextran run, plus the same fields reshot with BSA."""
    prior = os.path.join(d, 'Dextran Output')
    imgs = os.path.join(d, 'BSA Images')
    for sub in ('bbb_progress', 'masks', 'somas', 'bbb_overlays'):
        os.makedirs(os.path.join(prior, sub), exist_ok=True)
    os.makedirs(imgs, exist_ok=True)

    vessel = np.zeros((H, W), np.uint8)
    vessel[38:42, :] = 1                      # one horizontal vessel

    for base in ('fieldA', 'fieldB'):
        if with_progress:
            tifffile.imwrite(
                os.path.join(prior, 'bbb_progress', base + '_vesselmask.tif'),
                vessel)
        # a rendered overlay, which must NOT be mistaken for a mask
        tifffile.imwrite(os.path.join(prior, 'bbb_overlays', base + '_bbb.png')
                         .replace('.png', '.tif'),
                         np.zeros((H, W, 3), np.uint8))

        # two cells: one hugging the vessel, one far from it
        for sid, r0 in (('soma_40_20', 34), ('soma_70_60', 66)):
            soma = np.zeros((H, W), np.uint8)
            soma[r0:r0 + 6, 20:26] = 1
            tifffile.imwrite(
                os.path.join(prior, 'somas', f'{base}_{sid}_soma.tif'), soma)
            for area in (200, 400):
                m = np.zeros((H, W), np.uint8)
                grow = 2 if area == 200 else 5
                m[r0 - grow:r0 + 6 + grow, 20 - grow:26 + grow] = 1
                tifffile.imwrite(
                    os.path.join(prior, 'masks',
                                 f'{base}_{sid}_area{area}_mask.tif'), m)

        # the new image: ch0 CD31, ch1 IBA1, ch2 BSA. BSA sits on the vessel
        # and fades away from it, so the near cell must read far higher.
        bsa = np.zeros((H, W), np.float32)
        rows = np.abs(np.arange(H) - 40)[:, None]
        bsa += np.clip(1000 - rows * 40, 0, None)
        stack = np.dstack([np.zeros((H, W), np.float32),
                           np.zeros((H, W), np.float32), bsa]).astype(np.uint16)
        tifffile.imwrite(os.path.join(imgs, base + '.tif'), stack)
    return prior, imgs


def main():
    fails = []
    d = tempfile.mkdtemp()
    prior, imgs = build(d)
    out = os.path.join(d, 'simple_bsa.csv')

    p = subprocess.run(
        [sys.executable, TOOL, '--prior', prior, '--images', imgs,
         '--tracer', 'bsa', '--tracer-channel', '3',
         '--pixel-size', str(PX), '--out', out],
        capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout + p.stderr)
        fails.append(f"the re-run exited {p.returncode}")

    rows = []
    if os.path.exists(out):
        with open(out, newline='') as f:
            rows = list(csv.DictReader(f))
    if len(rows) != 4:
        fails.append(f"wrote {len(rows)} cell rows, expected 4 (2 images x 2 "
                     f"cells)")

    if rows:
        cols = set(rows[0])
        for need in ('image_name', 'soma_id', 'bbb_dist_to_vessel_um',
                     'bbb_bsa_exposure_microglia', 'bbb_bsa_exposure_10um',
                     'bbb_bsa_exposure_20um', 'bbb_bsa_exposure_30um'):
            if need not in cols:
                fails.append(f"the CSV has no {need} column")
        if any(c.startswith('bbb_dextran') for c in cols):
            fails.append("a column from the FIRST tracer leaked into the "
                         "re-run; these are new measurements")

        near = [r for r in rows if r['soma_id'] == 'soma_40_20']
        far = [r for r in rows if r['soma_id'] == 'soma_70_60']

        def val(r, k):
            try:
                return float(r[k])
            except (ValueError, KeyError, TypeError):
                return None

        nd = [val(r, 'bbb_dist_to_vessel_um') for r in near]
        fd = [val(r, 'bbb_dist_to_vessel_um') for r in far]
        if any(v is None for v in nd + fd):
            fails.append("a distance came out blank; the vessel mask was not "
                         "used")
        elif not all(n < f for n in nd for f in fd):
            fails.append(f"the cell beside the vessel is not closer to it "
                         f"({nd} vs {fd}) — the reused mask is not being "
                         f"measured against")

        ne = [val(r, 'bbb_bsa_exposure_microglia') for r in near]
        fe = [val(r, 'bbb_bsa_exposure_microglia') for r in far]
        if any(v is None for v in ne + fe):
            fails.append("an exposure came out blank")
        elif not all(n > f for n in ne for f in fe):
            fails.append(f"the cell in the leak is not more exposed than the "
                         f"one away from it ({ne} vs {fe}) — the new tracer "
                         f"channel is not being read")

    # --- no saved vessel masks must REFUSE, not guess from the PNGs ---------
    d2 = tempfile.mkdtemp()
    prior2, imgs2 = build(d2, with_progress=False)
    p2 = subprocess.run(
        [sys.executable, TOOL, '--prior', prior2, '--images', imgs2,
         '--tracer', 'bsa', '--tracer-channel', '3',
         '--pixel-size', str(PX), '--out', os.path.join(d2, 'x.csv')],
        capture_output=True, text=True)
    if p2.returncode == 0:
        fails.append("with no saved vessel masks the script still produced "
                     "numbers; the overlay PNGs are pictures and the boundary "
                     "would be invented")
    if 'reviewed again' not in (p2.stdout + p2.stderr):
        fails.append("refusing did not say what to do instead")
    if os.path.exists(os.path.join(d2, 'x.csv')):
        fails.append("a CSV was written even though it refused")

    # --- a mismatched field must be skipped, not measured -------------------
    d3 = tempfile.mkdtemp()
    prior3, imgs3 = build(d3)
    tifffile.imwrite(os.path.join(imgs3, 'fieldB.tif'),
                     np.zeros((H + 10, W, 3), np.uint16))   # wrong shape
    out3 = os.path.join(d3, 'o.csv')
    p3 = subprocess.run(
        [sys.executable, TOOL, '--prior', prior3, '--images', imgs3,
         '--tracer', 'bsa', '--tracer-channel', '3',
         '--pixel-size', str(PX), '--out', out3],
        capture_output=True, text=True)
    with open(out3, newline='') as f:
        rows3 = list(csv.DictReader(f))
    if any(r['image_name'] == 'fieldB' for r in rows3):
        fails.append("an image whose shape does not match its vessel mask was "
                     "measured anyway")
    if 'does not match' not in p3.stdout:
        fails.append("the shape mismatch was not reported")

    for p_ in (d, d2, d3):
        shutil.rmtree(p_, ignore_errors=True)

    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("OK: a second tracer is measured against the first run's reviewed "
          "vessel masks, and refused when they were never saved")


if __name__ == '__main__':
    main()
