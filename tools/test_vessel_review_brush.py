#!/usr/bin/env python3
"""The vessel review opens ready to draw.

Marking vessels is what that window is for, but the draw mode started on Off,
so a correction began by clicking on the image, noticing nothing happened, and
then finding the combo. The brush is the default now, at a size wide enough to
trace a vessel in one pass rather than colouring it in.

    python3 tools/test_vessel_review_brush.py
"""
import os
import sys
import warnings

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
    from PyQt5.QtWidgets import QApplication
except ImportError as e:
    print(f"SKIP: needs PyQt5 and numpy ({e})")
    sys.exit(0)

import importlib.util
spec = importlib.util.spec_from_file_location(
    'mmps', os.path.join(ROOT, 'MMPSv2.12.py'))
mmps = importlib.util.module_from_spec(spec)
sys.modules['mmps'] = mmps
spec.loader.exec_module(mmps)


def main():
    app = QApplication([])
    fails = []
    assert app is not None

    cd31 = np.zeros((120, 140), np.uint16)
    cd31[55:65, 20:120] = 800          # one vessel
    dlg = mmps.VesselReviewDialog(None, cd31, 0.316, 'img1')

    if dlg.draw_combo.currentData() != 'mark':
        fails.append(f"the draw mode starts on "
                     f"{dlg.draw_combo.currentData()!r}; clicking the image "
                     f"would do nothing until the combo is found")
    if dlg.brush_spin.value() != 30:
        fails.append(f"the brush starts at {dlg.brush_spin.value()} px, "
                     f"expected 30")
    if not (dlg.brush_spin.minimum() <= 30 <= dlg.brush_spin.maximum()):
        fails.append("30 px is outside the brush range, so the default cannot "
                     "hold")

    # --- and a click actually marks, with no further setup -----------------
    before = int(dlg.marks.sum())
    dlg.paint_at(60, 70)
    after = int(dlg.marks.sum())
    if after <= before:
        fails.append("painting straight after opening marked nothing")
    else:
        # a 30 px brush is a disc of radius 30, so it must be much wider than
        # the 12 px one it replaced -- otherwise the size did not take effect
        if after < 500:
            fails.append(f"one stroke marked {after} px; a 30 px brush should "
                         f"cover far more, so the size is not being used")
    if int(dlg.erase_marks.sum()):
        fails.append("painting in 'mark' mode also wrote to the erase layer")

    # --- Off must still be reachable and still mean off --------------------
    dlg.draw_combo.setCurrentIndex(max(0, dlg.draw_combo.findData('off')))
    held = int(dlg.marks.sum())
    dlg.paint_at(20, 20)
    if int(dlg.marks.sum()) != held:
        fails.append("'Off' still painted; the mode is being ignored")

    # --- erase writes its own layer, and the two give way to each other ----
    # Erasing over an include mark is meant to clear it ("the other layer gives
    # way, so re-marking undoes an erase and vice versa"), so the test checks
    # that at a DIFFERENT spot the include layer is left alone, and separately
    # that an overlapping erase does take the include mark away.
    dlg.draw_combo.setCurrentIndex(max(0, dlg.draw_combo.findData('erase')))
    dlg.paint_at(10, 130)                      # nowhere near the marked vessel
    if int(dlg.erase_marks.sum()) == 0:
        fails.append("'Erase' marked nothing")
    if int(dlg.marks.sum()) != held:
        fails.append("erasing somewhere else still changed the include layer")

    dlg.paint_at(60, 70)                       # straight over the include mark
    if int(dlg.marks.sum()) >= held:
        fails.append("erasing over an include mark did not clear it, so the "
                     "two layers cannot undo each other")

    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("OK: vessel review opens on the brush at 30 px, and the other modes "
          "still behave")


if __name__ == '__main__':
    main()
