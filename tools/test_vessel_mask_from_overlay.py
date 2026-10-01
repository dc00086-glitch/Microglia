#!/usr/bin/env python3
"""Recovering a vessel mask from an overlay PNG must be measured, not assumed.

A figure is normally not a mask. The BBB overlay is the exception: the vessel
is a translucent FILL rather than a hairline contour, and the heatmap under it
is inferno, whose ramp contains no green at any point -- so a green pixel in
that panel can only be fill.

It is still a rendered, resampled copy, so the only question that matters is
how close the recovery lands. This renders overlays with MMPS's own function
from masks it knows, recovers them back, and checks the IoU is high enough to
measure against -- and that the tool REFUSES when it is not.

    python3 tools/test_vessel_mask_from_overlay.py
"""
import os
import sys
import shutil
import tempfile
import subprocess
import warnings

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, 'tools', 'vessel_mask_from_overlay.py')

try:
    import numpy as np
    import tifffile
except ImportError as e:
    print(f"SKIP: needs numpy and tifffile ({e})")
    sys.exit(0)
try:
    import PIL  # noqa: F401
    import matplotlib  # noqa: F401
except ImportError as e:
    print(f"SKIP: needs Pillow and matplotlib ({e})")
    sys.exit(0)

import importlib.util
spec = importlib.util.spec_from_file_location(
    'mmps', os.path.join(ROOT, 'MMPSv2.12.py'))
mmps = importlib.util.module_from_spec(spec)
sys.modules['mmps'] = mmps
spec.loader.exec_module(mmps)

H, W = 220, 260
GREEN = (0.0, 1.0, 0.0)


def vessels(seed=0):
    """A branching vessel tree, not a blob — recovery is hardest at thin parts."""
    m = np.zeros((H, W), bool)
    m[100:112, 20:240] = True              # trunk
    m[40:100, 90:99] = True                # branch up
    m[112:190, 160:167] = True             # branch down
    m[60:66, 99:200] = True                # thin branch
    return m


def tracer(seed=0, bright=False):
    """A dim field, or one with a bright leak band over the vessel.

    The distinction is the whole result: inferno's top end runs yellow-white
    where green and red converge, so a vessel sitting in a BRIGHT leak is the
    hard case -- and the one that matters, since a leaking field is what the
    measurement is about.
    """
    rng = np.random.RandomState(seed)
    t = rng.rand(H, W) * 60
    if bright:
        yy, _ = np.ogrid[:H, :W]
        t = t + 700 * np.exp(-(((yy - 106) / 30.0) ** 2))
    return t


def run_case(bright, min_iou, tmp):
    """Render overlays from a known mask, recover them, return (rc, stdout)."""
    global BRIGHT
    BRIGHT = bright
    ov = os.path.join(tmp, 'bbb_overlays')
    prog = os.path.join(tmp, 'prior', 'bbb_progress')
    imgs = os.path.join(tmp, 'images')
    for p_ in (ov, prog, imgs):
        os.makedirs(p_, exist_ok=True)
    vm = vessels()
    cells = [np.zeros((H, W), bool) for _ in range(2)]
    cells[0][120:140, 40:60] = True
    cells[1][70:88, 180:198] = True
    for i, base in enumerate(['f1', 'f2', 'f3', 'f4']):
        t = tracer(i, bright=bright)
        tifffile.imwrite(os.path.join(imgs, base + '.tif'),
                         np.dstack([t, t, t]).astype(np.uint16))
        mmps._save_bbb_overlay(os.path.join(ov, base + '_bbb.png'),
                               vm, {'dextran': t}, cell_masks=cells,
                               vessel_colour=GREEN,
                               cell_colour=(0.25, .45, 1), source_label=base)
        if base != 'f4':
            tifffile.imwrite(os.path.join(prog, base + '_vesselmask.tif'),
                             vm.astype(np.uint8))
    out = os.path.join(tmp, 'recovered')
    p_ = subprocess.run(
        [sys.executable, TOOL, '--overlays', ov,
         '--prior', os.path.join(tmp, 'prior'), '--images', imgs,
         '--colour', 'green', '--min-iou', str(min_iou), '--out', out],
        capture_output=True, text=True)
    return p_, out, vm


BRIGHT = False


def main():
    fails = []

    # --- a dim field: the fill must be recovered closely ------------------
    d1 = tempfile.mkdtemp()
    p1, out1, vm = run_case(bright=False, min_iou=0.90, tmp=d1)
    print(p1.stdout.strip())
    if p1.returncode != 0:
        print(p1.stderr.strip())
        fails.append("recovery failed on a dim field, where it should work")
    if 'Scored against 3 known mask' not in p1.stdout:
        fails.append("it did not score against the images that have a saved "
                     "mask; the accuracy would be unmeasured")
    med1 = None
    for line in p1.stdout.splitlines():
        if 'median' in line and 'IoU' in line:
            med1 = float(line.split('median')[1].split()[0])
    if med1 is None:
        fails.append("no median IoU was reported")
    elif med1 < 0.95:
        fails.append(f"dim field median IoU {med1:.3f}; the translucent fill "
                     f"should come back almost exactly here")

    made = sorted(os.listdir(out1)) if os.path.isdir(out1) else []
    if 'f4_vesselmask.tif' not in made:
        fails.append("the one image with no saved mask got no recovery, which "
                     "is the only one that needed one")
    for b in ('f1', 'f2', 'f3'):
        if f'{b}_vesselmask.tif' in made:
            fails.append(f"{b} has a real saved mask but a recovery was "
                         f"written over it")
    if 'f4_vesselmask.tif' in made:
        got = np.asarray(tifffile.imread(
            os.path.join(out1, 'f4_vesselmask.tif'))) > 0
        u = int((got | vm).sum())
        iou = int((got & vm).sum()) / u if u else 0
        if iou < 0.95:
            fails.append(f"the written mask scores {iou:.3f} against truth")
        from scipy import ndimage as _nd
        if _nd.label(got)[1] > 3:
            fails.append(f"the recovered mask is in {_nd.label(got)[1]} "
                         f"pieces; a vessel tree is one or two")

    # --- a BRIGHT leaking field: it degrades, and must SAY SO -------------
    # Inferno runs yellow-white at the top, where green and red converge, so a
    # vessel inside a bright leak is the hard case. It is also the case the
    # measurement exists for, which is why the gate has to hold here.
    d2 = tempfile.mkdtemp()
    p2, out2, _ = run_case(bright=True, min_iou=0.95, tmp=d2)
    med2 = None
    for line in p2.stdout.splitlines():
        if 'median' in line and 'IoU' in line:
            med2 = float(line.split('median')[1].split()[0])
    print(f"  bright field median IoU: {med2}")
    if med2 is None:
        fails.append("no IoU was reported on the bright field")
    elif med2 >= med1:
        fails.append(f"the bright field scored {med2:.3f} against the dim "
                     f"field's {med1:.3f}; the known failure mode did not "
                     f"appear, so this test is not exercising it")
    if p2.returncode == 0:
        fails.append("a recovery below --min-iou still exited cleanly")
    if os.path.isdir(out2) and os.listdir(out2):
        fails.append("masks were written despite scoring below --min-iou; the "
                     "score has to gate the output or the measurement is "
                     "decoration")

    # --- a warm fill cannot be separated from a warm ramp -----------------
    p3 = subprocess.run(
        [sys.executable, TOOL, '--overlays',
         os.path.join(d1, 'bbb_overlays'),
         '--prior', os.path.join(d1, 'prior'),
         '--images', os.path.join(d1, 'images'),
         '--colour', 'orange', '--out', os.path.join(d1, 'warm')],
        capture_output=True, text=True)
    if p3.returncode == 0:
        fails.append("a warm fill colour was accepted; inferno is a warm ramp "
                     "so that cannot be told apart from the heatmap")

    # --- no scoring set at all must refuse ---------------------------------
    d3 = tempfile.mkdtemp()
    p4, _o, _v = run_case(bright=False, min_iou=0.90, tmp=d3)
    import shutil as _sh
    _sh.rmtree(os.path.join(d3, 'prior', 'bbb_progress'))
    p5 = subprocess.run(
        [sys.executable, TOOL, '--overlays', os.path.join(d3, 'bbb_overlays'),
         '--prior', os.path.join(d3, 'prior'),
         '--images', os.path.join(d3, 'images'),
         '--colour', 'green', '--out', os.path.join(d3, 'z')],
        capture_output=True, text=True)
    if p5.returncode == 0:
        fails.append("with nothing to score against it still wrote masks; an "
                     "unmeasured recovery is the thing this exists to prevent")

    for p_ in (d1, d2, d3):
        shutil.rmtree(p_, ignore_errors=True)
    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("OK: the vessel fill is recovered and scored against the real masks, "
          "and withheld whenever it does not match")


if __name__ == '__main__':
    main()
