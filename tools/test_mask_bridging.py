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
    order, _, _ = mmps._priority_region_grow(
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
    order, _, _ = mmps._priority_region_grow(
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

    order3, _, _ = mmps._priority_region_grow(
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

    # --- competitive growth must bridge too, without swapping pixels -------
    # This is the mode the study data was grown in, so "the option exists" is
    # worth nothing if it stops at the other two. Competition makes it
    # stricter: the dark valley between two cells is what places the boundary,
    # so a bridge must never pull a pixel out of a neighbouring cell.
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    g = mmps.MicrogliaAnalysisGUI()
    g.use_min_intensity = True
    g.min_intensity_percent = 10
    g.local_intensity_window = 0
    g.use_circular_constraint = False
    g.mask_smooth_enabled = False

    # Two cells facing each other, each with a broken process. The break in
    # cell A's process sits 2 px from cell B's own signal.
    comp = np.full((H, W), 10.0)
    comp[28:33, 28:33] = 900.0                 # cell A soma
    comp[30, 33:44] = 600.0                    # A's process, running right
    comp[30, 44:47] = 12.0                     # ...broken
    comp[30, 47:58] = 600.0                    # ...and continuing
    comp[28:33, 78:83] = 900.0                 # cell B soma, well clear
    comp[30, 70:78] = 600.0                    # B's process, running left
    oa = np.zeros((H, W), np.uint8); oa[28:33, 28:33] = 1
    ob = np.zeros((H, W), np.uint8); ob[28:33, 78:83] = 1
    somas = [
        {'soma_idx': 0, 'soma_id': 'A', 'centroid': (30.0, 30.0),
         'soma_area_um2': 25.0, 'outline': oa},
        {'soma_idx': 1, 'soma_id': 'B', 'centroid': (30.0, 80.0),
         'soma_area_um2': 25.0, 'outline': ob},
    ]

    def comp_masks(bridge):
        g.mask_bridge_gaps = bool(bridge)
        g.mask_bridge_px = max(1, bridge)
        out = g._create_competitive_masks(
            comp.astype(np.uint16), somas, [400], 1.0, 'c.tif')
        by = {}
        for m in out:
            by[m['soma_id']] = np.asarray(m['mask'])
        return by

    off = comp_masks(0)
    on = comp_masks(4)

    if int(off['A'][30, 47:58].sum()) != 0:
        fails.append("competitive growth crossed the break with bridging off")
    got = int(on['A'][30, 47:58].sum())
    if got != 11:
        fails.append(f"competitive growth recovered {got}/11 pixels past the "
                     f"break — bridging does not reach this mode, which is the "
                     f"one the study data uses")

    # No pixel may end up in two cells at once, bridged or not.
    for tag, masks in (('off', off), ('on', on)):
        overlap = int((masks['A'] & masks['B']).sum())
        if overlap:
            fails.append(f"bridging {tag}: {overlap} pixels belong to both "
                         f"cells at once")

    # Each cell's mask must still be a single connected piece.
    for tag, masks in (('off', off), ('on', on)):
        for sid, m in masks.items():
            n = ndimage.label(m)[1]
            if n > 1:
                fails.append(f"bridging {tag}: cell {sid}'s mask is in {n} "
                             f"pieces")

    # A bridge must not reach into the neighbour's own process.
    if int(on['A'][30, 70:78].sum()):
        fails.append("cell A's bridge took pixels from cell B's process")

    # --- "Redo Masks (This Image)" must expose it, per image ----------------
    # Focus drift is a property of one slide, not of the batch. The redo dialog
    # is the place to fix it: turn bridging on for the image that drifted,
    # leave every other image grown exactly as it was. The settings it uses are
    # globals borrowed for one image, so they must also be handed back.
    from PyQt5.QtWidgets import QCheckBox, QSpinBox, QDialog
    from PyQt5.QtCore import QTimer

    proc = np.zeros((H, W), np.uint16)
    proc[:] = 10
    proc[58:63, 58:63] = 900
    proc[60, 63:80] = 600
    proc[60, BREAK] = 12
    proc[60, BEYOND] = 600
    outline = np.zeros((H, W), np.uint8)
    outline[58:63, 58:63] = 1

    gui.images['a.tif'] = {
        'raw_path': '/x/a.tif', 'processed': proc, 'processed_path': None,
        'somas': [(60.0, 60.0)], 'soma_ids': ['a_s1'], 'soma_groups': [],
        'soma_outlines': [{'soma_id': 'a_s1', 'soma_idx': 0,
                           'polygon_points': [(58, 58), (62, 58), (62, 62),
                                              (58, 62)],
                           'outline': outline, 'centroid': (60.0, 60.0),
                           'soma_area_um2': 25.0}],
        'masks': [], 'status': 'outlined', 'selected': True, 'animal_id': '',
        'treatment': '', 'region': '', 'timepoint': '',
        'rolling_ball_radius': 50, 'pixel_size': 1.0}
    gui.current_image_name = 'a.tif'
    gui.mask_min_area = 300
    gui.mask_max_area = 400
    gui.mask_step_size = 100
    gui.min_intensity_percent = 10
    gui.use_circular_constraint = False

    # The batch setting stays OFF throughout: what the dialog turns on must
    # reach this image and nothing else.
    gui.mask_bridge_gaps = False
    gui.mask_bridge_px = 3
    seen = {'checkbox': False, 'slider': False, 'typed': False, 'max': None}

    def reaper():
        dlg = QApplication.activeModalWidget()
        if dlg is None:
            return
        boxes = [b for b in dlg.findChildren(QCheckBox)
                 if 'across small breaks' in b.text()]
        if isinstance(dlg, QDialog) and boxes:
            seen['checkbox'] = True
            boxes[0].setChecked(True)
            for sb in dlg.findChildren(QSpinBox):
                if sb.objectName() == 'bridge_span':
                    seen['typed'] = True
                    seen['max'] = sb.maximum()
                    sb.setValue(6)
                    seen['slider'] = True
            dlg.accept()
        else:
            dlg.reject() if isinstance(dlg, QDialog) else dlg.close()

    timer = QTimer()
    timer.timeout.connect(reaper)
    timer.start(30)
    gui.regenerate_masks_current_image()
    timer.stop()

    # The span must be typed, not dragged: breaks are measured off the image,
    # and a slider capped at 15 px silently refuses anything wider (under 5 µm
    # at 0.316 µm/px). It must also reach a value a slider never could.
    if seen['max'] is not None and seen['max'] < 100:
        fails.append(f"the break-span control tops out at {seen['max']} px; a "
                     f"break wider than that cannot be entered at all")
    if not seen['typed']:
        fails.append("the break span is not a spin box, so it cannot be typed")

    if not seen['checkbox']:
        fails.append("the Redo Masks dialog has no gap-bridging checkbox; "
                     "breaks can only be addressed by changing the batch "
                     "setting and regenerating everything")
    if not seen['slider']:
        fails.append("the Redo Masks dialog offers no break-span slider")

    regen = gui.images['a.tif']['masks']
    if not regen:
        fails.append("Redo Masks produced no masks at all")
    else:
        biggest = max(regen, key=lambda m: int(np.asarray(m['mask']).sum()))
        got = int(np.asarray(biggest['mask'])[60, BEYOND].sum())
        if got == 0:
            fails.append("the redo dialog's bridging setting never reached "
                         "the grower — the mask still stops at the break")

    if gui.mask_bridge_gaps is not False or gui.mask_bridge_px != 3:
        fails.append(f"the per-image redo leaked its bridging setting into "
                     f"the batch settings (gaps={gui.mask_bridge_gaps}, "
                     f"px={gui.mask_bridge_px}); the next whole-batch "
                     f"generation would silently bridge too")

    # ...and it must be handed back even when the redo cannot run.
    gui.images['a.tif']['processed'] = None
    gui.images['a.tif']['processed_path'] = '/nonexistent/a.tif'
    timer.start(30)
    gui.regenerate_masks_current_image()
    timer.stop()
    if gui.mask_bridge_gaps is not False or gui.mask_bridge_px != 3:
        fails.append("a redo that failed part way left the batch bridging "
                     "setting overwritten")

    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("OK: growth follows a process across a short break, and only then")


if __name__ == '__main__':
    main()
