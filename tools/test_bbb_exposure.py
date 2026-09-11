#!/usr/bin/env python3
"""Fail if BBB tracer numbers ever count signal that is still inside a vessel.

Every leakage measure rests on one rule: tracer inside the lumen is blood, not
leak. A cell sitting on a bright vessel must not read as bathed in tracer just
because the vessel falls inside the region being averaged.

The failure this was written for is the quiet one. Per-cell exposure averaged
extravascular pixels, but when the region had NONE -- swallowed whole by the
vessel mask, which happens readily when CD31 over-segments -- it fell back to
averaging the lot, every pixel of it intravascular. The column then reported
pure blood signal under the name "exposure", numerically indistinguishable
from a genuinely tracer-soaked cell.

Also pinned: exposure samples the footprint grown 10 um outward, not the bare
footprint, and MMPS and bbb_from_masks.py still agree.

    python3 tools/test_bbb_exposure.py
"""
import os
import re
import sys
import types

import numpy as np
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, 'MMPSv2.12.py')
STANDALONE = os.path.join(ROOT, 'bbb_from_masks.py')

INTRA = 1000.0      # tracer inside the lumen
EXTRA = 10.0        # tracer out in the parenchyma
H = W = 80
PS = 1.0            # 1 um per pixel, so 10 um of halo is 10 px
HALO_UM = 10.0


def load(path, names):
    """Exec just the BBB math out of a file — importing MMPS builds a Qt app."""
    src = open(path).read()
    mod = types.ModuleType('probe')
    mod.__dict__.update({'np': np, 'ndimage': ndimage,
                         '_JUXTAVASCULAR_MAX_UM': 10.0,
                         '_EXPOSURE_HALO_UM': HALO_UM,
                         'EXPOSURE_HALO_UM': HALO_UM})
    for name in names:
        m = re.search(rf'^def {name}\(.*?(?=\n\n#|\n\ndef |\n\nclass |\Z)',
                      src, re.S | re.M)
        if not m:
            sys.exit(f"could not find {name} in {os.path.basename(path)}")
        exec(compile(m.group(0), path, 'exec'), mod.__dict__)
    return mod


def box(r0, r1, c0, c1):
    m = np.zeros((H, W), bool)
    m[r0:r1, c0:c1] = True
    return m


def vessel_stripe(c0, c1):
    v = np.zeros((H, W), bool)
    v[:, c0:c1] = True
    return v


def tracer_for(vessel):
    """Bright in the lumen, dim everywhere else. Any lumen pixel that sneaks
    into the average drags the mean far above EXTRA."""
    t = np.full((H, W), EXTRA)
    t[vessel] = INTRA
    return t


def expose(mod, cell, vessel, tracer, halo=HALO_UM):
    """Call either file's exposure function through one signature."""
    if hasattr(mod, '_microglia_leakage_exposure'):
        return mod._microglia_leakage_exposure(
            cell, vessel, {'dex': tracer}, PS, soma_mask=cell,
            halo_um=halo)
    dist = ndimage.distance_transform_edt(~(vessel > 0)) * PS
    return mod.microglia_exposure(cell, vessel, {'dex': tracer}, PS, dist,
                                  cell, halo_um=halo)


def check_no_lumen(mod, label, fails):
    vessel = vessel_stripe(38, 43)          # 5 px wide, down the middle
    tracer = tracer_for(vessel)
    for name, cell in (
            ("clear of the vessel",      box(20, 26, 8, 20)),
            ("straddling the wall",      box(20, 26, 35, 46)),
            ("mostly inside the vessel", box(20, 26, 37, 44)),
            ("entirely inside the vessel", box(20, 26, 39, 42))):
        got = expose(mod, cell, vessel, tracer).get('microglia_dex_exposure_mean')
        if not isinstance(got, float) or abs(got - EXTRA) > 1e-6:
            fails.append(f"{label}: a cell {name} reported exposure {got!r}, "
                         f"expected {EXTRA} — lumen signal ({INTRA:.0f}) is "
                         f"reaching the per-cell mean")


def check_undefined(mod, label, fails):
    """No extravascular pixel anywhere in the region => no value, not the
    lumen mean."""
    vessel = vessel_stripe(0, W)            # the whole frame is vessel
    tracer = tracer_for(vessel)
    out = expose(mod, box(30, 36, 30, 40), vessel, tracer)
    got = out.get('microglia_dex_exposure_mean')
    if got != '':
        fails.append(f"{label}: with every pixel intravascular, exposure came "
                     f"back as {got!r} — there is no extravascular value to "
                     f"report and {INTRA:.0f} is the blood signal itself")
    if out.get('exposure_region_um2') not in (0, 0.0, None) and got == '':
        fails.append(f"{label}: blank exposure but exposure_region_um2 = "
                     f"{out.get('exposure_region_um2')}")


def check_halo(mod, label, fails):
    """The region must actually reach past the footprint, by about 10 um."""
    vessel = np.zeros((H, W), bool)         # no vessels: isolate the halo
    cell = box(30, 40, 30, 40)
    # Dim under the cell, bright in the ring around it. A footprint-only mean
    # cannot see the ring; a 10 um halo must.
    tracer = np.zeros((H, W))
    tracer[cell] = 10.0
    ring = box(20, 50, 20, 50) & ~cell
    tracer[ring] = 100.0

    out = expose(mod, cell, vessel, tracer)
    halo_mean = out.get('microglia_dex_exposure_mean')
    cell_mean = out.get('microglia_dex_exposure_mean_cell_only')
    if not isinstance(cell_mean, float) or abs(cell_mean - 10.0) > 1e-6:
        fails.append(f"{label}: cell-only mean {cell_mean!r}, expected 10.0")
    if not isinstance(halo_mean, float) or halo_mean <= 10.0:
        fails.append(f"{label}: halo mean {halo_mean!r} did not rise above the "
                     f"footprint's own 10.0 — the 10 um halo is not being "
                     f"sampled")

    # Region size: the cell grown by 10 um. PS is 1 um/px here, so the pixel
    # count and the um2 figure are the same number.
    n = out.get('exposure_region_um2')
    want = float((ndimage.distance_transform_edt(~cell) <= HALO_UM / PS).sum())
    if n is None or abs(n - want) > 0.02 * want:
        fails.append(f"{label}: exposure_region_um2 {n}, expected about {want} "
                     f"for a {HALO_UM:g} um halo")

    # And a zero halo must reproduce the footprint-only number exactly.
    bare = expose(mod, cell, vessel, tracer, halo=0.0).get('microglia_dex_exposure_mean')
    if bare != cell_mean:
        fails.append(f"{label}: halo_um=0 gave {bare!r} but the cell-only "
                     f"column says {cell_mean!r}; they must agree")


def check_rings(mod, fails):
    """Image-level: rings start OUTSIDE the wall, and the split is clean."""
    vessel = vessel_stripe(38, 43)
    tracer = tracer_for(vessel)
    m = mod._quantify_leakage(vessel, tracer, PS, ring_edges_um=(0, 5, 10))
    for key in ('perivasc_0_5um_mean', 'perivasc_5_10um_mean'):
        got = m.get(key)
        if got is None:
            fails.append(f"ring {key} missing")
        elif abs(got - EXTRA) > 1e-6:
            fails.append(f"ring {key} = {got}, expected {EXTRA} — the first "
                         f"ring is picking up lumen pixels")
    if abs(m['intravascular_mean'] - INTRA) > 1e-6:
        fails.append(f"intravascular_mean = {m['intravascular_mean']}")
    if abs(m['extravascular_mean'] - EXTRA) > 1e-6:
        fails.append(f"extravascular_mean = {m['extravascular_mean']} — the "
                     f"image-level split counts lumen as parenchyma")


def main():
    fails = []
    mmps = load(TARGET, ['_grown_by_um', '_quantify_leakage',
                         '_microglia_leakage_exposure'])
    check_no_lumen(mmps, 'MMPS', fails)
    check_undefined(mmps, 'MMPS', fails)
    check_halo(mmps, 'MMPS', fails)
    check_rings(mmps, fails)

    # bbb_from_masks.py advertises identical math; hold it to that.
    if os.path.exists(STANDALONE):
        alone = load(STANDALONE, ['grown_by_um', 'microglia_exposure'])
        check_no_lumen(alone, 'bbb_from_masks.py', fails)
        check_undefined(alone, 'bbb_from_masks.py', fails)
        check_halo(alone, 'bbb_from_masks.py', fails)

    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("OK: exposure samples the mask plus 10 um and never counts lumen")


if __name__ == '__main__':
    main()
