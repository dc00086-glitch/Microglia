#!/usr/bin/env python3
"""Fail if a few saturated pixels can wipe out the vessel mask.

vessel_area_fraction is the fraction of the frame inside the binary CD31 mask,
and that mask is an Otsu threshold on the RAW channel (or on its Sato tubeness
response). Otsu splits an image into two classes by intensity, so a handful of
pixels at the sensor ceiling -- a hot pixel, dust, a bright edge artefact --
become a class of their own and the threshold lands above every real vessel.

Measured before the fix, on a synthetic frame with three vessels:

    clean                 threshold  654   area fraction 0.052
    one saturated pixel   threshold 2102   area fraction 0.000

Three pixels cleared that threshold, the minimum-object filter removed them as
too small, and the image reported NO VESSELS -- with no error. Every number
downstream still came out: area fraction zero, every cell infinitely far from a
vessel, leakage index zero. All of it looks like a finding.

    python3 tools/test_vessel_robustness.py
"""
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, 'MMPSv2.12.py')
PS = 0.104
TOL = 0.25          # area fraction must stay within 25% of the clean value


def load():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    spec = importlib.util.spec_from_file_location("mmps", TARGET)
    m = importlib.util.module_from_spec(spec)
    sys.modules['mmps'] = m
    sys.argv = ['mmps']
    spec.loader.exec_module(m)
    return m


def frame(seed=7, h=400, w=400):
    """A raw-like CD31 field: camera offset, noise, three tube-like vessels."""
    rng = np.random.default_rng(seed)
    img = rng.normal(300, 25, (h, w))
    yy, xx = np.mgrid[:h, :w]
    for (y0, x0, ang) in ((80, 40, 0.3), (220, 20, -0.15), (330, 60, 0.5)):
        d = np.abs((yy - y0) * np.cos(ang) - (xx - x0) * np.sin(ang))
        img += 900 * np.exp(-(d ** 2) / (2 * 3.0 ** 2))
    return np.clip(img, 0, None)


def main():
    m = load()
    base = frame()
    fails = 0
    for tube in (True, False):
        _, clean = m._segment_vessels(base.copy(), PS, use_tubeness=tube)
        ref = clean['vessel_area_fraction']
        if ref <= 0:
            print(f"  tubeness={tube}: the clean frame itself found no "
                  f"vessels — the test image is wrong, not the code")
            fails += 1
            continue
        print(f"  tubeness={str(tube):5s} clean area fraction {ref:.4f}")
        for n_hot in (1, 3, 10, 50):
            img = base.copy()
            for i in range(n_hot):
                img[5 + (i % 20), 5 + (i // 20)] = 65535.0
            _, got = m._segment_vessels(img, PS, use_tubeness=tube)
            af = got['vessel_area_fraction']
            drift = abs(af - ref) / ref
            if af <= 0:
                print(f"    {n_hot:3d} saturated px -> area fraction 0.0000 "
                      f"— the mask was wiped out")
                fails += 1
            elif drift > TOL:
                print(f"    {n_hot:3d} saturated px -> {af:.4f} "
                      f"({100 * drift:.0f}% off the clean value)")
                fails += 1
            else:
                print(f"    {n_hot:3d} saturated px -> {af:.4f}  ok")

    # And the run must say so when a mask really is empty, rather than
    # reporting zeros as though they were measurements.
    src = open(TARGET).read()
    if '_bbb_no_vessels' not in src or 'NO VESSELS' not in src:
        print("  an empty vessel mask is not reported to the user — its zeros "
              "are indistinguishable from a real measurement")
        fails += 1
    else:
        print("  empty masks are reported rather than written out as zeros")

    if fails:
        sys.exit(f"\nFAIL: {fails} case(s). A few saturated pixels must not "
                 f"move the vessel threshold past the vessels.")
    print("PASS: vessel area fraction survives saturated pixels, and an empty "
          "mask is reported.")


if __name__ == '__main__':
    main()
