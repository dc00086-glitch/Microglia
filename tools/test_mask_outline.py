#!/usr/bin/env python3
"""The single mask view must draw a contour, the way the QA grid does.

The grid has always drawn the mask with findContours + drawContours, so it
reads as a cell. The single view filled every mask pixel instead, which draws
the interior exactly as grown -- speckled, full of one-pixel holes where growth
skipped a dim pixel -- and at full resolution that reads as a ragged blob. Same
mask, two answers, and the noisier one is the one you review on.

Fill still has a job: you cannot paint what you cannot see. So it stays
reachable, painting switches to it automatically, and leaving paint puts the
view back.

    python3 tools/test_mask_outline.py
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
    from PyQt5.QtGui import QPixmap, QPainter, QImage
except ImportError as e:
    print(f"SKIP: needs PyQt5 and numpy ({e})")
    sys.exit(0)

import importlib.util
spec = importlib.util.spec_from_file_location(
    'mmps', os.path.join(ROOT, 'MMPSv2.12.py'))
mmps = importlib.util.module_from_spec(spec)
sys.modules['mmps'] = mmps
spec.loader.exec_module(mmps)

N = 120


def speckled_mask():
    """A blob whose interior is full of holes — what growth really produces."""
    m = np.zeros((N, N), np.uint8)
    m[30:90, 30:90] = 1
    rng = np.random.RandomState(0)
    holes = rng.rand(N, N) < 0.25
    m[holes] = 0
    m[55:65, 55:65] = 1          # a solid core so it is not all holes
    return m


CANVAS = 700          # the label centres the pixmap, so leave room for the
                      # pan offset -- a tight canvas clips the whole overlay


def painted(label, mask):
    """Render the overlay to an image and return the green pixel count."""
    img = QImage(CANVAS, CANVAS, QImage.Format_RGB32)
    img.fill(0)
    label.pix_source = QPixmap(N, N)
    label.scaled_pixmap = QPixmap(N, N)
    label.mask_overlay = mask
    p = QPainter(img)
    label._draw_mask_overlay(p)
    p.end()
    arr = np.frombuffer(img.bits().asstring(CANVAS * CANVAS * 4),
                        np.uint8).reshape(CANVAS, CANVAS, 4)
    return int(((arr[:, :, 1] > 100) & (arr[:, :, 2] < 100)).sum())


def main():
    app = QApplication([])
    fails = []
    assert app is not None

    gui = mmps.MicrogliaAnalysisGUI()
    label = gui.mask_label
    mask = speckled_mask()
    interior = int(mask.sum())

    # --- outline is the default, and it is not a fill ----------------------
    if not label.mask_outline_only:
        fails.append("the single view still defaults to the filled overlay")
    if not gui.mask_outline_check.isChecked():
        fails.append("the Outline toggle does not start on")

    outline_px = painted(label, mask)
    label.mask_outline_only = False
    fill_px = painted(label, mask)

    if outline_px == 0:
        fails.append("outline mode drew nothing at all")
    if fill_px == 0:
        fails.append("fill mode drew nothing at all")
    # A contour of a 60x60 blob is a few hundred pixels; its interior is
    # thousands. If the outline is not far smaller, it is not an outline.
    if outline_px >= fill_px * 0.5:
        fails.append(f"outline painted {outline_px} px against the fill's "
                     f"{fill_px} — that is not a contour, the speckled "
                     f"interior is still being drawn")
    print(f"  interior {interior} px   fill drew {fill_px}   "
          f"outline drew {outline_px}")

    # --- the toggle reaches every label ------------------------------------
    gui.mask_outline_check.setChecked(False)
    for lbl in (gui.original_label, gui.preview_label, gui.processed_label,
                gui.mask_label):
        if lbl.mask_outline_only:
            fails.append("unchecking Outline did not reach every image label")
            break
    if gui.opacity_slider.isEnabled() is False:
        fails.append("the opacity slider stays disabled in fill mode, where "
                     "it is the only thing that does anything")
    gui.mask_outline_check.setChecked(True)
    if gui.opacity_slider.isEnabled():
        fails.append("the opacity slider is live in outline mode, where it "
                     "changes nothing")

    # --- painting needs the fill, and gives the view back ------------------
    gui.paint_fill_btn.setChecked(True)
    if gui.mask_outline_check.isChecked():
        fails.append("turning on Paint Fill left the outline up — you cannot "
                     "see the pixels you are painting")
    gui.paint_fill_btn.setChecked(False)
    if not gui.mask_outline_check.isChecked():
        fails.append("leaving paint mode stranded the view in fill instead of "
                     "restoring the outline")

    # ...including when the user hands off between the two paint tools
    gui.paint_fill_btn.setChecked(True)
    gui.paint_erase_btn.setChecked(True)      # switches off paint_fill
    if gui.mask_outline_check.isChecked():
        fails.append("handing off from paint to erase put the outline back "
                     "mid-edit")
    gui.paint_erase_btn.setChecked(False)
    if not gui.mask_outline_check.isChecked():
        fails.append("the outline never came back after the paint/erase "
                     "hand-off")

    # --- an empty mask must not crash or draw ------------------------------
    if painted(label, np.zeros((N, N), np.uint8)) != 0:
        fails.append("an empty mask still painted something")

    # --- and the MASK itself is the clean one, not just its picture --------
    # This is what makes area and skeleton length describe the shape being
    # reviewed. _smooth_mask alone cannot do it: where growth took roughly
    # every other pixel the gaps join up and reach the outside, so not one of
    # them counts as an enclosed hole at any gap size. Filling the outer
    # contour cannot do it either -- the contour traces every notch, so
    # filling it reproduces the lace exactly.
    lacy = np.zeros((N, N), np.uint8)
    lacy[30:90, 30:90] = 1
    rng2 = np.random.RandomState(0)
    lacy[rng2.rand(N, N) < 0.10] = 0
    lacy[55:65, 65:75] = 1
    from scipy import ndimage as _nd
    if _nd.label(lacy)[1] != 1:
        fails.append("the test mask is in pieces; growth produces a connected "
                     "mask and this would not be testing the real case")

    raw_holes = int((lacy[30:90, 30:90] == 0).sum())
    sm = np.asarray(mmps._smooth_mask(lacy.copy(), 50))
    sm_holes = int((sm[30:90, 30:90] == 0).sum())
    sol = np.asarray(mmps._solidify_mask(lacy.copy(), 2))
    sol_holes = int((sol[30:90, 30:90] == 0).sum())
    print(f"  holes in the blob: {raw_holes} as grown, {sm_holes} after "
          f"smoothing at gap size 50, {sol_holes} after solidify r=2")

    if sm_holes < 15:
        fails.append("smoothing alone already cleaned the mask, so this test "
                     "does not show what solidify is for")
    if sol_holes >= sm_holes * 0.6:
        fails.append(f"solidify left {sol_holes} holes against smoothing's "
                     f"{sm_holes} — it is not closing the ragged edge")

    # It must only ever ADD pixels: an area can never come out smaller.
    if not np.all(sol[lacy > 0]):
        fails.append("solidify dropped a pixel that was in the grown mask")
    if int(sol.sum()) < int(lacy.sum()):
        fails.append("solidify removed mask pixels")

    # radius 0 is a no-op, so the old behaviour stays reachable exactly.
    if not np.array_equal(np.asarray(mmps._solidify_mask(lacy.copy(), 0)), lacy):
        fails.append("radius 0 changed the mask; it must leave it as grown")

    # A thin process must survive rather than being swallowed or fattened away.
    proc = np.zeros((N, N), np.uint8)
    proc[20:100, 40:43] = 1
    proc_sol = np.asarray(mmps._solidify_mask(proc.copy(), 2))
    grew = int(proc_sol.sum()) - int(proc.sum())
    if grew > int(proc.sum()) * 0.5:
        fails.append(f"solidify at r=2 fattened a 3 px process by {grew} px; "
                     f"thin branches would stop being thin")

    # dtype and 'on' value must survive, or a 0/255 mask reads as empty
    m255 = (lacy * 255).astype(np.uint8)
    out255 = np.asarray(mmps._solidify_mask(m255, 2))
    if out255.dtype != m255.dtype or set(np.unique(out255)) - {0, 255}:
        fails.append(f"a 0/255 mask came back as {out255.dtype} with values "
                     f"{np.unique(out255)[:4]}")

    if mmps._solidify_mask(np.zeros((N, N), np.uint8), 2).any():
        fails.append("an empty mask came back non-empty")

    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        sys.exit(1)
    print("OK: the single view outlines the mask like the grid, and painting "
          "still gets a fill")


if __name__ == '__main__':
    main()
