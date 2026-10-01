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

    # --- overlays are drawn on the NEW tracer, same boundaries -------------
    # The point of re-rendering is a figure that differs from the first run's
    # ONLY in the tracer: same vessel outline, same cell outlines, same
    # display range across images. Anything else and the two cannot be shown
    # side by side as a comparison.
    d7 = tempfile.mkdtemp()
    prior7, imgs7 = build(d7)
    ov = os.path.join(d7, 'bsa_overlays')
    out7 = os.path.join(d7, 'o.csv')
    p7 = subprocess.run(
        [sys.executable, TOOL, '--prior', prior7, '--images', imgs7,
         '--tracer', 'bsa', '--tracer-channel', '3', '--overlays', ov,
         '--pixel-size', str(PX), '--out', out7],
        capture_output=True, text=True)
    if p7.returncode != 0:
        print(p7.stdout + p7.stderr)
        fails.append(f"--overlays exited {p7.returncode}")
    made = sorted(os.listdir(ov)) if os.path.isdir(ov) else []
    if len(made) != 2:
        fails.append(f"wrote {len(made)} overlay(s), expected one per image")
    for f_ in made:
        if os.path.getsize(os.path.join(ov, f_)) < 5000:
            fails.append(f"{f_} is too small to be a rendered figure")
    if 'display range' not in p7.stdout:
        fails.append("no shared display range was measured; per-image "
                     "autoscaling would make a faint image look as bright as "
                     "a leaking one")

    # --- the outline colours are the user's to set -------------------------
    import importlib.util as _iu
    _sp = _iu.spec_from_file_location('rerun', TOOL)
    _rr = _iu.module_from_spec(_sp); _sp.loader.exec_module(_rr)
    if _rr.parse_colour('green', (9, 9, 9)) != (0.0, 1.0, 0.0):
        fails.append("'green' did not resolve to green")
    if _rr.parse_colour('0,0,255', (9, 9, 9)) != (0.0, 0.0, 1.0):
        fails.append("an 0-255 triple was not scaled to 0-1")
    if _rr.parse_colour('0.5,0.5,0.5', (9, 9, 9)) != (0.5, 0.5, 0.5):
        fails.append("an 0-1 triple was rescaled when it should not be")
    if _rr.parse_colour('not a colour', (1, 2, 3)) != (1, 2, 3):
        fails.append("an unparseable colour did not fall back")
    if _rr.parse_colour('', (1, 2, 3)) != (1, 2, 3):
        fails.append("an empty colour did not fall back")
    # vessel green and microglia blue by default, as asked
    import argparse as _ap
    src = open(TOOL).read()
    if "'--vessel-colour', default='green'" not in src:
        fails.append("vessels do not default to green")
    if "'--cell-colour', default='blue'" not in src:
        fails.append("microglia do not default to blue")

    # the first run's overlays must be untouched -- both are wanted
    orig = os.path.join(prior7, 'bbb_overlays')
    if len(os.listdir(orig)) != 2:
        fails.append("the first run's overlays were disturbed; the whole "
                     "point is to have both")
    if os.path.abspath(ov) == os.path.abspath(orig):
        fails.append("the new overlays were written over the old ones")

    # without --overlays nothing is rendered
    d8 = tempfile.mkdtemp()
    prior8, imgs8 = build(d8)
    subprocess.run(
        [sys.executable, TOOL, '--prior', prior8, '--images', imgs8,
         '--tracer', 'bsa', '--tracer-channel', '3',
         '--pixel-size', str(PX), '--out', os.path.join(d8, 'o.csv')],
        capture_output=True, text=True)
    if os.path.isdir(os.path.join(d8, 'bsa_overlays')):
        fails.append("overlays were written without being asked for")

    shutil.rmtree(d7, ignore_errors=True)
    shutil.rmtree(d8, ignore_errors=True)

    # --- image-level leakage joins onto an existing vessel CSV -------------
    # The perivascular rings live at IMAGE level, not per cell, so they belong
    # in bbb_vessel_leakage.csv beside the first tracer's.
    d9 = tempfile.mkdtemp()
    prior9, imgs9 = build(d9)
    vcsv = os.path.join(d9, 'bbb_vessel_leakage.csv')
    import csv as _c9
    with open(vcsv, 'w', newline='') as f:
        w = _c9.DictWriter(f, fieldnames=['image_name', 'vessel_area_fraction',
                                          'dextran_leakage_index'])
        w.writeheader()
        w.writerow({'image_name': 'fieldA.tif', 'vessel_area_fraction': '0.05',
                    'dextran_leakage_index': '0.40'})
        w.writerow({'image_name': 'fieldB', 'vessel_area_fraction': '0.06',
                    'dextran_leakage_index': '0.33'})
        w.writerow({'image_name': 'never_run.tif',
                    'vessel_area_fraction': '0.01',
                    'dextran_leakage_index': '0.10'})
    p9 = subprocess.run(
        [sys.executable, TOOL, '--prior', prior9, '--images', imgs9,
         '--tracer', 'bsa', '--tracer-channel', '3', '--vessel-csv', vcsv,
         '--pixel-size', str(PX), '--out', os.path.join(d9, 'o.csv')],
        capture_output=True, text=True)
    if p9.returncode != 0:
        print(p9.stdout + p9.stderr)
        fails.append(f"--vessel-csv exited {p9.returncode}")

    with open(vcsv, newline='') as f:
        back = list(_c9.DictReader(f))
    cols = set(back[0]) if back else set()
    rings = sorted(c for c in cols if 'perivasc' in c)
    if not rings:
        fails.append("no perivascular columns were added to the vessel CSV")
    if not any(c.startswith('bsa_') for c in rings):
        fails.append(f"the rings are not prefixed with the tracer: {rings}")
    for need in ('bsa_leakage_index', 'bsa_intravascular_mean',
                 'bsa_extravascular_mean'):
        if need not in cols:
            fails.append(f"{need} was not added")

    by9 = {r['image_name']: r for r in back}
    if len(back) != 3:
        fails.append(f"the vessel CSV has {len(back)} rows, expected its own 3")
    if by9.get('fieldA.tif', {}).get('dextran_leakage_index') != '0.40':
        fails.append("an existing column was altered")
    if not str(by9.get('fieldB', {}).get(rings[0] if rings else '', '')).strip():
        fails.append("a bare image name did not match, so its rings are blank")
    if str(by9.get('never_run.tif', {}).get(rings[0] if rings else '', 'X')).strip():
        fails.append("an image this run never measured got a ring value "
                     "instead of a blank")
    if not os.path.exists(vcsv + '.bak'):
        fails.append("the original vessel CSV was rewritten with no backup")

    # running again must not duplicate the columns
    before_cols = len(cols)
    subprocess.run(
        [sys.executable, TOOL, '--prior', prior9, '--images', imgs9,
         '--tracer', 'bsa', '--tracer-channel', '3', '--vessel-csv', vcsv,
         '--pixel-size', str(PX), '--out', os.path.join(d9, 'o2.csv')],
        capture_output=True, text=True)
    with open(vcsv, newline='') as f:
        again = list(_c9.DictReader(f))
    if again and len(set(again[0])) != before_cols:
        fails.append("running twice changed the column count; the columns are "
                     "being duplicated or dropped")
    shutil.rmtree(d9, ignore_errors=True)

    # --- images with no saved mask can be REBUILT from the recorded area ---
    # bbb_vessel_leakage.csv records vessel_area_fraction for every image the
    # first run measured, and _segment_vessels can segment TO an area
    # fraction. That reproduces a stated rule rather than decoding a figure.
    d4 = tempfile.mkdtemp()
    prior4, imgs4 = build(d4)
    os.remove(os.path.join(prior4, 'bbb_progress', 'fieldB_vesselmask.tif'))
    # give the CD31 channel real vessel signal so a re-segmentation has
    # something to find, and record the area the first run measured
    for base in ('fieldA', 'fieldB'):
        p_img = os.path.join(imgs4, base + '.tif')
        stack = np.asarray(tifffile.imread(p_img)).astype(np.uint16)
        cd = np.zeros((H, W), np.uint16)
        cd[38:42, :] = 3000
        stack[:, :, 0] = cd
        tifffile.imwrite(p_img, stack)
    with open(os.path.join(prior4, 'bbb_vessel_leakage.csv'), 'w',
              newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image_name',
                                          'vessel_area_fraction'])
        w.writeheader()
        for base in ('fieldA', 'fieldB'):
            w.writerow({'image_name': base + '.tif',
                        'vessel_area_fraction': f'{4.0 / H:.4f}'})

    out4 = os.path.join(d4, 'o.csv')
    p4 = subprocess.run(
        [sys.executable, TOOL, '--prior', prior4, '--images', imgs4,
         '--tracer', 'bsa', '--tracer-channel', '3', '--reconstruct',
         '--cd31-channel', '1', '--pixel-size', str(PX), '--out', out4],
        capture_output=True, text=True)
    if p4.returncode != 0:
        print(p4.stdout + p4.stderr)
        fails.append(f"--reconstruct exited {p4.returncode}")
    with open(out4, newline='') as f:
        rows4 = list(csv.DictReader(f))
    srcs = {r['image_name']: r.get('bbb_vessel_mask_source') for r in rows4}
    if srcs.get('fieldA') != 'reviewed':
        fails.append(f"fieldA has a saved mask but is marked "
                     f"{srcs.get('fieldA')!r}")
    if srcs.get('fieldB') != 'reconstructed':
        fails.append(f"fieldB had no saved mask; it is marked "
                     f"{srcs.get('fieldB')!r}, so a rebuilt mask could not be "
                     f"told from a reviewed one")
    if not any(r['image_name'] == 'fieldB' for r in rows4):
        fails.append("the image with no saved mask was not measured at all "
                     "despite --reconstruct")
    else:
        fb = [r for r in rows4 if r['image_name'] == 'fieldB']
        if all(str(r.get('bbb_dist_to_vessel_um', '')).strip() == ''
               for r in fb):
            fails.append("a reconstructed image produced only blanks; the "
                         "rebuilt mask found no vessel")

    # --- CD31 is re-segmented from the ORIGINAL images when told ----------
    # The recorded area fraction came from the first run's pixels, so those
    # are the pixels it describes. Applying that target to a differently
    # exposed second acquisition can land anywhere -- including on nothing.
    d10 = tempfile.mkdtemp()
    prior10, imgs10 = build(d10)
    os.remove(os.path.join(prior10, 'bbb_progress', 'fieldB_vesselmask.tif'))
    orig = os.path.join(d10, 'orig')
    os.makedirs(orig, exist_ok=True)
    for base in ('fieldA', 'fieldB'):
        cd = np.zeros((H, W), np.uint16); cd[38:42, :] = 3000
        tifffile.imwrite(os.path.join(orig, base + '.tif'),
                         np.dstack([cd, cd, cd]).astype(np.uint16))
        p_img = os.path.join(imgs10, base + '.tif')
        st = np.asarray(tifffile.imread(p_img)).astype(np.uint16)
        st[:, :, 0] = 5          # the NEW image's ch1 is flat: nothing to find
        tifffile.imwrite(p_img, st)
    import csv as _c10
    with open(os.path.join(prior10, 'bbb_vessel_leakage.csv'), 'w',
              newline='') as f:
        w = _c10.DictWriter(f, fieldnames=['image_name',
                                           'vessel_area_fraction'])
        w.writeheader()
        for base in ('fieldA', 'fieldB'):
            w.writerow({'image_name': base + '.tif',
                        'vessel_area_fraction': f'{4.0 / H:.4f}'})

    out10 = os.path.join(d10, 'from_orig.csv')
    p10 = subprocess.run(
        [sys.executable, TOOL, '--prior', prior10, '--images', imgs10,
         '--tracer', 'bsa', '--tracer-channel', '3', '--reconstruct',
         '--cd31-channel', '1', '--reconstruct-from', orig,
         '--pixel-size', str(PX), '--out', out10],
        capture_output=True, text=True)
    if 'reconstructing CD31 from' not in p10.stdout:
        fails.append("--reconstruct-from was ignored")
    r10 = []
    if os.path.exists(out10):
        with open(out10, newline='') as f:
            r10 = list(_c10.DictReader(f))
    if not any(x['image_name'] == 'fieldB' for x in r10):
        fails.append("segmenting the ORIGINAL image still produced nothing for "
                     "the image with no saved mask")

    # ...and without it, the flat new channel must be CAUGHT, not measured
    out11 = os.path.join(d10, 'from_new.csv')
    p11 = subprocess.run(
        [sys.executable, TOOL, '--prior', prior10, '--images', imgs10,
         '--tracer', 'bsa', '--tracer-channel', '3', '--reconstruct',
         '--cd31-channel', '1', '--pixel-size', str(PX), '--out', out11],
        capture_output=True, text=True)
    r11 = []
    if os.path.exists(out11):
        with open(out11, newline='') as f:
            r11 = list(_c10.DictReader(f))
    if any(x['image_name'] == 'fieldB' for x in r11):
        fails.append("a reconstruction that found far less vessel than "
                     "recorded was measured anyway; every distance and ring "
                     "would be taken from an empty mask")
    shutil.rmtree(d10, ignore_errors=True)

    # --reconstruct without --cd31-channel must refuse, not guess a plane
    p5 = subprocess.run(
        [sys.executable, TOOL, '--prior', prior4, '--images', imgs4,
         '--tracer', 'bsa', '--tracer-channel', '3', '--reconstruct',
         '--pixel-size', str(PX), '--out', os.path.join(d4, 'x.csv')],
        capture_output=True, text=True)
    if p5.returncode == 0:
        fails.append("--reconstruct ran without being told which plane is "
                     "CD31")

    # without --reconstruct the missing image is simply absent, not invented
    out6 = os.path.join(d4, 'o6.csv')
    subprocess.run(
        [sys.executable, TOOL, '--prior', prior4, '--images', imgs4,
         '--tracer', 'bsa', '--tracer-channel', '3',
         '--pixel-size', str(PX), '--out', out6],
        capture_output=True, text=True)
    with open(out6, newline='') as f:
        rows6 = list(csv.DictReader(f))
    if any(r['image_name'] == 'fieldB' for r in rows6):
        fails.append("an image with no saved mask was measured even without "
                     "--reconstruct")

    shutil.rmtree(d4, ignore_errors=True)

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
          "masks, rebuilt to the recorded area where none was saved, and "
          "refused when there is neither")


if __name__ == '__main__':
    main()
