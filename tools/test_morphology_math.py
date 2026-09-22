#!/usr/bin/env python3
"""Check the shape descriptors against an ellipse whose axes are known exactly.

The failure this was written for: major_axis_um and minor_axis_um came from
`2 * sqrt(eigenvalue)`, which is the SEMI-axis. Every value was exactly half
its own column name, and half what ImageJ or skimage report for the same cell
-- while eccentricity and roundness in the SAME row were built on skimage's
full-length axes, so one row carried two different definitions of "major axis"
and nothing flagged it.

    python3 tools/test_morphology_math.py
"""
import os
import sys
import warnings

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
    from skimage import measure
    from PyQt5.QtWidgets import QApplication
except ImportError as e:
    print(f"SKIP: needs numpy, scikit-image and PyQt5 ({e})")
    sys.exit(0)

import importlib.util
spec = importlib.util.spec_from_file_location(
    'mmps', os.path.join(ROOT, 'MMPSv2.12.py'))
mmps = importlib.util.module_from_spec(spec)
sys.modules['mmps'] = mmps
spec.loader.exec_module(mmps)

H = W = 400
PS = 1.0


def ellipse(a, b):
    yy, xx = np.ogrid[:H, :W]
    return (((xx - 200) / a) ** 2 + ((yy - 200) / b) ** 2 <= 1).astype(np.uint8)


def main():
    QApplication([])
    fails = []
    calc = mmps.MorphologyCalculator(
        (np.random.rand(H, W) * 255).astype(np.uint8), PS)

    for a, b in ((120.0, 40.0), (90.0, 90.0), (150.0, 20.0)):
        mask = ellipse(a, b)
        p = calc.calculate_all_parameters(mask, (200.0, 200.0), None)
        props = measure.regionprops(mask.astype(int))[0]

        # Axis lengths are FULL axes, i.e. 2a and 2b, same as skimage.
        for key, want, ski in (
                ('major_axis_um', 2 * a, props.axis_major_length),
                ('minor_axis_um', 2 * b, props.axis_minor_length)):
            got = p[key]
            if abs(got - want) > 0.02 * want:
                fails.append(f"ellipse a={a:.0f} b={b:.0f}: {key} = {got:.2f}, "
                             f"expected about {want:.0f} — a value near "
                             f"{want / 2:.0f} means the SEMI-axis is being "
                             f"reported under a full-axis name")
            elif abs(got - ski) > 0.05:
                fails.append(f"ellipse a={a:.0f} b={b:.0f}: {key} = {got:.2f} "
                             f"but skimage says {ski:.2f}")

        # These two are built on skimage's full axes and must agree with it.
        if abs(p['eccentricity'] - props.eccentricity) > 1e-3:
            fails.append(f"ellipse a={a:.0f} b={b:.0f}: eccentricity "
                         f"{p['eccentricity']:.4f} vs skimage "
                         f"{props.eccentricity:.4f}")
        want_round = (min(a, b) / max(a, b)) ** 2
        if abs(p['roundness'] - want_round) > 0.01:
            fails.append(f"ellipse a={a:.0f} b={b:.0f}: roundness "
                         f"{p['roundness']:.4f}, expected {want_round:.4f}")

        # Area, and the polarity index at the extremes.
        want_area = np.pi * a * b * PS ** 2
        if abs(p['mask_area'] - want_area) > 0.02 * want_area:
            fails.append(f"ellipse a={a:.0f} b={b:.0f}: mask_area "
                         f"{p['mask_area']:.0f}, expected {want_area:.0f}")
        if a == b and p['polarity_index'] > 0.02:
            fails.append(f"a circle reported polarity_index "
                         f"{p['polarity_index']} — should be ~0")

    # A horizontal ellipse points along 0 degrees.
    p = calc.calculate_all_parameters(ellipse(120.0, 40.0), (200.0, 200.0), None)
    ang = p['principal_angle'] % 180
    if min(ang, 180 - ang) > 2.0:
        fails.append(f"a horizontal ellipse reported principal_angle {ang}")

    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("OK: axes, eccentricity, roundness, area and polarity all check out")


if __name__ == '__main__':
    main()
