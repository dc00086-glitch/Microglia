#!/usr/bin/env python3
"""Check that region growing can follow a process across an out-of-focus break.

A microglia process that dips out of the focal plane for a few pixels drops
below the intensity floor, which stops the brightest-first growth dead. Every
pixel past the break is lost: the cell reads as truncated, or as beaded and
dystrophic when it is neither.

Filling holes in the finished mask (the existing "Smooth masks" option) cannot
fix that, and this test pins the distinction: hole-filling only closes gaps
that are ALREADY enclosed by mask, and the growth never reached the far side of
a break, so there is nothing there to enclose.

What must also hold, or the option is worse than the problem: a bridge is only
taken when real signal is found beyond the gap, it may not exceed the span
asked for, and it may not cross into a neighbouring cell's territory.

    python3 tools/test_mask_bridging.py
"""
import os
import sys
import warnings

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
except ImportError as e:
    print(f"SKIP: needs numpy ({e})")
    sys.exit(0)

import importlib.util
spec = importlib.util.spec_from_file_location(
    'mmps', os.path.join(ROOT, 'MMPSv2.12.py'))
mmps = importlib.util.module_from_spec(spec)
sys.modules['mmps'] = mmps
spec.loader.exec_module(mmps)

H = W = 120
FLOOR = 100.0
BREAK = slice(80, 83)        # a 3 px out-of-focus run
BEYOND = slice(83, 115)      # the part of the process past it


def scene(with_signal_beyond=True):
    """A soma with one long process, broken by 3 dim pixels."""
    roi = np.full((H, W), 10.0)
    soma = np.zeros((H, W), np.uint8)
    soma[58:63, 58:63] = 1
    roi[58:63, 58:63] = 900.0
    roi[60, 63:80] = 600.0
    roi[60, BREAK] = 12.0                 # above background, below the floor
    if with_signal_beyond:
        roi[60, BEYOND] = 600.0
    return roi, soma


def grow(roi, soma, bridge_px, territory=None, label=0):
    order, _ = mmps._priority_region_grow(
        roi, 60, 60, soma, FLOOR, None, territory, label, None, 5000,
        bridge_px)
    mask = np.zeros((H, W), np.uint8)
    for r, c in order:
        mask[r, c] = 1
    return mask


def main():
    fails = []
    roi, soma = scene()

    # --- off, and too short, must stop at the break ------------------------
    for bridge in (0, 1, 2):
        mask = grow(roi, soma, bridge)
        if int(mask[60, BEYOND].sum()) != 0:
            fails.append(f"bridge_px={bridge} crossed a 3 px break it should "
                         f"not reach")

    # --- long enough must recover the whole rest of the process ------------
    for bridge in (3, 5, 8):
        mask = grow(roi, soma, bridge)
        got = int(mask[60, BEYOND].sum())
        if got != 32:
            fails.append(f"bridge_px={bridge} recovered {got}/32 pixels past a "
                         f"3 px break")
        if int(mask[60, BREAK].sum()) != 3:
            fails.append(f"bridge_px={bridge} left the break itself unfilled; "
                         f"the mask would be two disconnected pieces")

    # --- the mask must be ONE connected piece at every prefix length --------
    # Masks are prefixes of the growth order, so a bridge committed out of turn
    # would produce a mask with a floating fragment.
    order, _ = mmps._priority_region_grow(
        roi, 60, 60, soma, FLOOR, None, None, 0, None, 5000, 5)
    from scipy import ndimage
    for n in range(4, len(order) + 1, 7):
        m = np.zeros((H, W), np.uint8)
        for r, c in order[:n]:
            m[r, c] = 1
        if ndimage.label(m)[1] > 1:
            fails.append(f"the mask is in {ndimage.label(m)[1]} pieces at "
                         f"prefix length {n} — a gap was committed before the "
                         f"pixel that justified it")
            break

    # --- two bridges that cross must not double-count the shared pixel ------
    # A horizontal probe and a vertical one can cross, and then the pixel where
    # their lines meet sits in BOTH spans. Committing it twice would put a
    # duplicate in the growth order, and since masks are prefixes of that list,
    # every target area would be reached a pixel early -- each mask comes out
    # smaller than the area asked for, silently.
    roi3 = np.full((H, W), 10.0)
    soma3 = np.zeros((H, W), np.uint8)
    soma3[58:63, 58:63] = 1
    roi3[58:63, 58:63] = 900.0
    roi3[60, 63:78] = 600.0        # arm A runs right from the soma
    roi3[60, 78:81] = 12.0         # ...and breaks
    roi3[60, 81:90] = 600.0        # ...then carries on
    roi3[50:58, 60] = 600.0        # arm B goes up,
    roi3[50, 60:81] = 600.0        # ...across,
    roi3[51:60, 80] = 600.0        # ...and back down, straight into arm A's
    roi3[60:63, 80] = 12.0         # break, so both gaps contain (60, 80)
    roi3[63:70, 80] = 600.0

    order3, _ = mmps._priority_region_grow(
        roi3, 60, 60, soma3, FLOOR, None, None, 0, None, 5000, 4)
    if len(order3) != len(set(order3)):
        dupes = sorted({p for p in order3 if order3.count(p) > 1})
        fails.append(f"crossing bridges put duplicates in the growth order "
                     f"({dupes}); masks are prefixes of it, so every mask "
                     f"would come out short of its target area")
    m3 = np.zeros((H, W), np.uint8)
    for r, c in order3:
        m3[r, c] = 1
    if int(m3[60, 81:90].sum()) != 9:
        fails.append(f"the horizontal arm past the crossing recovered "
                     f"{int(m3[60, 81:90].sum())}/9 pixels")
    if int(m3[63:70, 80].sum()) != 7:
        fails.append(f"the vertical arm past the crossing recovered "
                     f"{int(m3[63:70, 80].sum())}/7 pixels")
    for n in range(4, len(order3) + 1, 5):
        m = np.zeros((H, W), np.uint8)
        for r, c in order3[:n]:
            m[r, c] = 1
        if ndimage.label(m)[1] > 1:
            fails.append(f"crossing bridges left the mask in pieces at prefix "
                         f"length {n}")
            break

    # --- a process that simply ends must not be extended -------------------
    roi2, soma2 = scene(with_signal_beyond=False)
    mask = grow(roi2, soma2, 8)
    reach = int(np.max(np.nonzero(mask[60])[0]))
    if reach > 79:
        fails.append(f"with nothing beyond the gap the mask still reached "
                     f"x={reach}; a bridge must need signal on the far side")

    # --- a neighbour's territory is a hard stop ----------------------------
    terr = np.zeros((H, W), np.int32)
    terr[:, :82] = 1
    terr[:, 82:] = 2
    mask = grow(roi, soma, 8, territory=terr, label=1)
    reach = int(np.max(np.nonzero(mask[60])[0]))
    if reach >= 82:
        fails.append(f"the bridge crossed into another cell's territory "
                     f"(reached x={reach}, the boundary is at 82)")

    # --- hole filling is NOT a substitute, which is why this exists --------
    unbridged = grow(roi, soma, 0)
    smoothed = mmps._smooth_mask(unbridged.copy(), 10)
    if int(np.asarray(smoothed)[60, BEYOND].sum()) != 0:
        fails.append("_smooth_mask recovered pixels past the break; if that "
                     "ever becomes true this feature is redundant")

    # --- the toggle gates it -----------------------------------------------
    from PyQt5.QtWidgets import QApplication
    app = QApplication([])
    gui = mmps.MicrogliaAnalysisGUI()
    assert app is not None
    gui.mask_bridge_gaps = False
    gui.mask_bridge_px = 5
    if gui._bridge_px() != 0:
        fails.append("bridging is not off when the checkbox is off")
    gui.mask_bridge_gaps = True
    if gui._bridge_px() != 5:
        fails.append(f"bridging span is {gui._bridge_px()}, expected 5")

    # --- and it reaches the real grower -------------------------------------
    args = ((60.0, 60.0), [200.0], 1.0, 0, 's1', (H, W), roi, (0, H, 0, W),
            25.0, soma, None, 0, False, 0, True, 5, 'img.tif', 0, False, 4,
            'percent', 900.0)
    off = mmps._grow_masks_for_soma(args + (0,))[0]['mask']
    on = mmps._grow_masks_for_soma(args + (3,))[0]['mask']
    if int(np.asarray(off)[60, BEYOND].sum()) != 0:
        fails.append("_grow_masks_for_soma crossed the break with bridging off")
    if int(np.asarray(on)[60, BEYOND].sum()) != 32:
        fails.append("the bridge setting does not reach _grow_masks_for_soma")

    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("OK: growth follows a process across a short break, and only then")


if __name__ == '__main__':
    main()
