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

Also pinned: exposure is measured at the microglia mask and at 10, 20 and 30
um out from it every run, a radius chosen in the dialog adds exactly one more
column, the per-cell columns are all prefixed bbb_ and are ONLY those, and
MMPS and bbb_from_masks.py still agree column for column.

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
RADII = (0.0, 10.0, 20.0, 30.0)
HALO_UM = 10.0


def load(path, names):
    """Exec just the BBB math out of a file — importing MMPS builds a Qt app."""
    src = open(path).read()
    mod = types.ModuleType('probe')
    mod.__dict__.update({'np': np, 'ndimage': ndimage,
                         '_JUXTAVASCULAR_MAX_UM': 10.0,
                         '_EXPOSURE_RADII_UM': RADII,
                         'EXPOSURE_RADII_UM': RADII,
                         '_JUXTAVASCULAR_MAX_UM': 10.0,
                         'JUXTAVASCULAR_MAX_UM': 10.0})
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


def expose(mod, cell, vessel, tracer, radii=RADII):
    """Call either file's exposure function through one signature."""
    if hasattr(mod, '_microglia_leakage_exposure'):
        return mod._microglia_leakage_exposure(
            cell, vessel, {'dex': tracer}, PS, soma_mask=cell,
            halo_radii_um=radii)
    dist = ndimage.distance_transform_edt(~(vessel > 0)) * PS
    return mod.microglia_exposure(cell, vessel, {'dex': tracer}, PS, dist,
                                  cell, halo_radii_um=radii)


def exposures(out):
    """Just the tracer-exposure columns, by radius tag."""
    return {k.split('_exposure_')[1]: v for k in out
            for _ in (0,) if '_exposure_' in k
            for v in (out[k],)}


def check_no_lumen(mod, label, fails):
    """No radius, on any overlap, may average a lumen pixel."""
    vessel = vessel_stripe(38, 43)          # 5 px wide, down the middle
    tracer = tracer_for(vessel)
    for name, cell, blank_ok in (
            ("clear of the vessel",        box(20, 26, 8, 20),  ()),
            ("straddling the wall",        box(20, 26, 35, 46), ()),
            ("mostly inside the vessel",   box(20, 26, 37, 44), ()),
            # A cell wholly inside the lumen has no extravascular pixel in its
            # own footprint, so the microglia-only column is legitimately
            # blank. Every wider radius reaches past the wall and must read
            # the parenchyma -- never the lumen it is sitting in.
            ("entirely inside the vessel", box(20, 26, 39, 42), ('microglia',))):
        out = expose(mod, cell, vessel, tracer)
        for tag, got in exposures(out).items():
            if got == '' and tag in blank_ok:
                continue
            if not isinstance(got, float) or abs(got - EXTRA) > 1e-6:
                fails.append(f"{label}: a cell {name}, radius {tag}, reported "
                             f"exposure {got!r}, expected {EXTRA} — lumen "
                             f"signal ({INTRA:.0f}) is reaching the mean")


def check_undefined(mod, label, fails):
    """No extravascular pixel anywhere => no exposure value, at any radius.
    The in-vessel mean, by contrast, is exactly what IS defined there."""
    vessel = vessel_stripe(0, W)            # the whole frame is vessel
    tracer = tracer_for(vessel)
    out = expose(mod, box(30, 36, 30, 40), vessel, tracer)
    for tag, got in exposures(out).items():
        if got != '':
            fails.append(f"{label}: with every pixel intravascular, exposure "
                         f"at radius {tag} came back as {got!r} — there is no "
                         f"extravascular value and {INTRA:.0f} is the blood "
                         f"signal itself")
    vm = out.get('bbb_dex_vessel_mean')
    if not isinstance(vm, float) or abs(vm - INTRA) > 1e-6:
        fails.append(f"{label}: bbb_dex_vessel_mean {vm!r}, expected {INTRA} — "
                     f"the in-vessel mean should read the lumen")


def check_halo(mod, label, fails):
    """Each radius must reach further than the last, and only those columns."""
    vessel = np.zeros((H, W), bool)         # no vessels: isolate the radii
    cell = box(35, 45, 35, 45)
    # Value falls off with distance from the cell, so a wider radius must
    # average lower. A radius that is not actually sampled cannot do that.
    d = ndimage.distance_transform_edt(~cell) * PS
    tracer = np.select([d <= 0, d <= 10, d <= 20, d <= 30],
                       [100.0, 80.0, 60.0, 40.0], default=20.0)

    out = expose(mod, cell, vessel, tracer)
    got = exposures(out)
    want_tags = ['microglia', '10um', '20um', '30um']
    if sorted(got) != sorted(want_tags):
        fails.append(f"{label}: exposure radii are {sorted(got)}, expected "
                     f"{sorted(want_tags)}")
        return
    vals = [got[t] for t in want_tags]
    if not all(isinstance(v, float) for v in vals):
        fails.append(f"{label}: non-numeric exposure among {vals}")
        return
    if not all(a > b for a, b in zip(vals, vals[1:])):
        fails.append(f"{label}: exposures {dict(zip(want_tags, vals))} do not "
                     f"fall with radius — a wider ring is not being sampled")
    if abs(vals[0] - 100.0) > 1e-6:
        fails.append(f"{label}: the microglia-only value is {vals[0]}, expected "
                     f"100.0 (the footprint itself)")

    # Exactly one extra column when the dialog names a radius, and none when
    # it names one already covered.
    extra = expose(mod, cell, vessel, tracer, radii=list(RADII) + [50.0])
    if sorted(exposures(extra)) != sorted(want_tags + ['50um']):
        fails.append(f"{label}: an extra 50 um radius gave "
                     f"{sorted(exposures(extra))}")
    same = expose(mod, cell, vessel, tracer, radii=list(RADII))
    if sorted(exposures(same)) != sorted(want_tags):
        fails.append(f"{label}: default radii gave {sorted(exposures(same))}")

    # And the per-cell column set is EXACTLY this, nothing else.
    allowed = {'bbb_dist_to_vessel_um', 'bbb_juxtavascular',
               'bbb_vessel_contact_fraction', 'bbb_vessel_contact_area_um2',
               'bbb_dex_vessel_mean'} | {
        'bbb_dex_exposure_%s' % t for t in want_tags}
    got_cols = set(out)
    if got_cols - allowed:
        fails.append(f"{label}: unexpected per-cell column(s) "
                     f"{sorted(got_cols - allowed)}")
    if allowed - got_cols:
        fails.append(f"{label}: missing per-cell column(s) "
                     f"{sorted(allowed - got_cols)}")
    if any(not c.startswith('bbb_') for c in got_cols):
        fails.append(f"{label}: column(s) not prefixed bbb_: "
                     f"{sorted(c for c in got_cols if not c.startswith('bbb_'))}")


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
    mmps = load(TARGET, ['_exposure_radius_tag', '_exposure_regions',
                         '_quantify_leakage', '_microglia_leakage_exposure'])
    check_no_lumen(mmps, 'MMPS', fails)
    check_undefined(mmps, 'MMPS', fails)
    check_halo(mmps, 'MMPS', fails)
    check_rings(mmps, fails)

    # bbb_from_masks.py advertises identical math; hold it to that.
    if os.path.exists(STANDALONE):
        alone = load(STANDALONE, ['exposure_radius_tag', 'exposure_regions',
                                  'microglia_exposure'])
        check_no_lumen(alone, 'bbb_from_masks.py', fails)
        check_undefined(alone, 'bbb_from_masks.py', fails)
        check_halo(alone, 'bbb_from_masks.py', fails)

    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("OK: exposure at microglia/10/20/30 um, bbb_-prefixed, no lumen")


if __name__ == '__main__':
    main()
