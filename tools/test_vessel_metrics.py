#!/usr/bin/env python3
"""Check the vessel morphometrics against geometry whose answer is known.

Three things this was written for, all measured on synthetic vessels:

  * Skeleton length counted PIXELS, so a vessel at 45 degrees measured
    1/sqrt(2) = 0.69 of its true length. Vessels run at every angle, so length
    density was systematically short, worst at the angles a dense bed has most
    of. A diagonal step is sqrt(2) now.

  * Branch points counted junction PIXELS. A junction spans several adjacent
    skeleton pixels that all have three or more neighbours, so one Y measured
    as three branch points. Connected clusters count once now.

  * Nothing in the leakage row is defined when segmentation finds no vessel --
    no lumen to average, no barrier to be intact, "outside the vessels" is the
    whole frame. leakage_index came back 0.0, which reads as a perfectly
    intact barrier rather than "this could not be measured". Same for a
    perivascular ring the frame is too small to hold.

    python3 tools/test_vessel_metrics.py
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
    from scipy import ndimage
except ImportError as e:
    print(f"SKIP: needs numpy and scipy ({e})")
    sys.exit(0)

import importlib.util
spec = importlib.util.spec_from_file_location(
    'mmps', os.path.join(ROOT, 'MMPSv2.12.py'))
mmps = importlib.util.module_from_spec(spec)
sys.modules['mmps'] = mmps
spec.loader.exec_module(mmps)

PS = 1.0            # 1 um per pixel, so a pixel count IS a micron count
H = W = 200
AREA_UM2 = H * W * PS ** 2


def met(mask):
    return mmps._segment_vessels(None, PS, vessel_mask=mask)[1]


def main():
    fails = []

    # --- a straight band: area, length and diameter are all exact ----------
    v = np.zeros((H, W), bool)
    v[95:105, :] = True
    mv = met(v)
    if abs(mv['vessel_area_fraction'] - (10 * W) / float(H * W)) > 1e-4:
        fails.append(f"area fraction {mv['vessel_area_fraction']}")
    # End-to-end the number also carries skeletonize's own behaviour: the
    # skeleton of a border-touching rectangle tapers at the corners, so it is a
    # few percent shorter than the frame. Allow for that here and test the
    # length rule itself directly, below.
    got_len = mv['vessel_length_density_um_per_um2'] * AREA_UM2
    if not (0.92 * W <= got_len <= W):
        fails.append(f"a straight {W} um vessel measured {got_len:.1f} um long")

    # --- the length rule itself, on skeletons with no ambiguity -----------
    line = np.zeros((H, W), bool)
    line[100, 10:110] = True                      # 100 px, 99 steps
    got = mmps._skeleton_length_um(line, PS)
    if abs(got - 99.0) > 1e-6:
        fails.append(f"a horizontal 100-pixel skeleton measured {got} um, "
                     f"expected 99 (one step shorter than the pixel count)")
    diag = np.zeros((H, W), bool)
    for i in range(100):
        diag[20 + i, 20 + i] = True               # 100 px, 99 diagonal steps
    got = mmps._skeleton_length_um(diag, PS)
    want = 99.0 * float(np.sqrt(2.0))
    if abs(got - want) > 1e-6:
        fails.append(f"a diagonal 100-pixel skeleton measured {got:.3f} um, "
                     f"expected {want:.3f} — a diagonal step is sqrt(2), not 1")
    if mmps._skeleton_length_um(np.zeros((H, W), bool), PS) != 0.0:
        fails.append("an empty skeleton has non-zero length")
    if abs(mv['vessel_mean_diameter_um'] - 10.0) > 1e-6:
        fails.append(f"a 10 px band measured {mv['vessel_mean_diameter_um']} um wide")

    # --- a 45-degree vessel must not measure 1/sqrt(2) of its length ------
    d = np.zeros((H, W), bool)
    yy, xx = np.mgrid[:H, :W]
    d[np.abs(yy - xx) <= 4] = True
    true_len = float(np.hypot(H, W)) * PS
    got = met(d)['vessel_length_density_um_per_um2'] * AREA_UM2
    ratio = got / true_len
    if ratio < 0.93:
        fails.append(f"a 45-degree vessel measured {ratio:.3f} of its true "
                     f"length ({got:.1f} vs {true_len:.1f} um) — diagonal "
                     f"steps are being counted as one pixel, not sqrt(2)")

    # --- one Y junction is one branch point -------------------------------
    y = np.zeros((H, W), bool)
    y[100:104, 20:100] = True
    for i, c in enumerate(range(100, 180)):
        y[100 - i // 2 - 1:100 - i // 2 + 3, c] = True
        y[102 + i // 2:106 + i // 2, c] = True
    n = met(y)['vessel_branchpoint_density_per_mm2'] * (AREA_UM2 / 1e6)
    if abs(n - 1.0) > 0.5:
        fails.append(f"a single Y junction counted as {n:.0f} branch points — "
                     f"junction pixels are being counted instead of junctions")

    # --- rings start OUTSIDE the wall -------------------------------------
    t = np.zeros((H, W))
    t[v] = 1000.0
    dist = ndimage.distance_transform_edt(~v) * PS
    t[(dist > 0) & (dist <= 10)] = 1.0
    t[(dist > 10) & (dist <= 20)] = 2.0
    t[(dist > 20) & (dist <= 40)] = 3.0
    q = mmps._quantify_leakage(v, t, PS)
    for key, want in (('perivasc_0_10um_mean', 1.0),
                      ('perivasc_10_20um_mean', 2.0),
                      ('perivasc_20_40um_mean', 3.0)):
        if not isinstance(q[key], float) or abs(q[key] - want) > 1e-6:
            fails.append(f"{key} = {q[key]!r}, expected {want} — the ring is "
                         f"picking up lumen pixels")

    # --- the leakage index itself -----------------------------------------
    t2 = np.where(v, 100.0, 25.0)
    q2 = mmps._quantify_leakage(v, t2, PS)
    if abs(q2['leakage_index'] - 0.25) > 1e-6:
        fails.append(f"leakage_index {q2['leakage_index']}, expected 0.25")

    # --- nothing is invented when there is no vessel ----------------------
    none_found = mmps._quantify_leakage(np.zeros((H, W), bool),
                                        np.full((H, W), 50.0), PS)
    for key in ('intravascular_mean', 'leakage_index',
                'extravascular_area_fraction', 'perivasc_0_10um_mean'):
        if none_found[key] != '':
            fails.append(f"with no vessel found, {key} = {none_found[key]!r}; "
                         f"it is undefined and must be blank (0.0 for the "
                         f"leakage index reads as an intact barrier)")
    if met(np.zeros((H, W), bool))['vessel_mean_diameter_um'] != '':
        fails.append("with no vessel found, vessel_mean_diameter_um is not blank")

    # --- a ring the frame cannot hold -------------------------------------
    small = np.zeros((30, 30), bool)
    small[14:16, :] = True
    tiny = mmps._quantify_leakage(small, np.full((30, 30), 7.0), PS)
    if tiny['perivasc_20_40um_mean'] != '':
        fails.append(f"a 20-40 um ring that does not fit in the frame reported "
                     f"{tiny['perivasc_20_40um_mean']!r} instead of blank")
    if tiny['perivasc_0_10um_mean'] != 7.0:
        fails.append(f"the ring that DOES fit reported "
                     f"{tiny['perivasc_0_10um_mean']!r}, expected 7.0")

    # bbb_from_masks.py advertises identical math; hold it to that.
    alone_path = os.path.join(ROOT, 'bbb_from_masks.py')
    if os.path.exists(alone_path):
        import re
        import types
        src = open(alone_path).read()
        alone = types.ModuleType('alone')
        alone.__dict__.update({'np': np, 'ndimage': ndimage})
        for name in ('skeleton_length_um', 'count_branch_points',
                     'quantify_leakage'):
            mm = re.search(rf'^def {name}\(.*?(?=\n\ndef |\n\nclass |\Z)',
                           src, re.S | re.M)
            if not mm:
                fails.append(f"bbb_from_masks.py has no {name}")
                continue
            exec(compile(mm.group(0), alone_path, 'exec'), alone.__dict__)
        if hasattr(alone, 'skeleton_length_um'):
            if abs(alone.skeleton_length_um(diag, PS)
                   - mmps._skeleton_length_um(diag, PS)) > 1e-9:
                fails.append("bbb_from_masks.py disagrees on skeleton length")
            if alone.count_branch_points(y) != mmps._count_branch_points(y):
                fails.append("bbb_from_masks.py disagrees on branch points")
            a = alone.quantify_leakage(np.zeros((H, W), bool),
                                       np.full((H, W), 50.0), PS)
            if a.get('leakage_index') != '':
                fails.append("bbb_from_masks.py still invents a leakage index "
                             "when no vessel was found")

    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("OK: vessel length, junctions, rings and the leakage row all "
          "check out, in both files")


if __name__ == '__main__':
    main()
